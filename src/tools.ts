import {
  ApprovalCallback,
  AgentBlitAgentConfig,
  ElicitationCallback,
  ElicitationRequest,
  ElicitationResult,
  OpenAIToolCall,
  ToolDefinition,
  ToolHandler,
  ToolOptions,
} from "./types.js";
import {
  functionToToolSchema,
  getToolMetadata,
  jsonDumpsSafe,
  readStdinLine,
  setToolMetadata,
} from "./utils.js";

export function tool(options: ToolOptions): <T extends ToolHandler>(fn: T) => T;
export function tool<T extends ToolHandler>(fn: T, options?: ToolOptions): T;
export function tool<T extends ToolHandler>(arg1: ToolOptions | T, arg2?: ToolOptions) {
  if (typeof arg1 === "function") {
    return setToolMetadata(arg1, arg2 ?? {}) as T;
  }
  return (fn: T): T => setToolMetadata(fn, arg1) as T;
}

export class ToolRegistry {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeout: number;
  private readonly remote = new Map<string, ToolDefinition>();
  private readonly custom = new Map<string, ToolDefinition>();

  constructor(options: { baseUrl: string; apiKey: string; timeout?: number }) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
  }

  register(fn: ToolHandler): void {
    const metadata = getToolMetadata(fn);
    const name = metadata?.name ?? fn.name;
    const description = metadata?.description ?? "";
    const permissionMode = metadata?.permissionMode ?? "always_allow";
    const inputSchema = functionToToolSchema(fn, metadata?.inputSchema);
    this.custom.set(name, {
      name,
      description,
      inputSchema,
      permissionMode,
      handler: fn,
    });
  }

  private applyRemoteTools(tools: Array<Record<string, unknown>>): void {
    this.remote.clear();
    for (const toolItem of tools) {
      const type = String(toolItem.type ?? "");
      if (type !== "function") {
        continue;
      }
      const fn = (toolItem.function as Record<string, unknown> | undefined) ?? {};
      const name = String(fn.name ?? "");
      if (!name) {
        continue;
      }
      this.remote.set(name, {
        name,
        description: String(fn.description ?? ""),
        inputSchema: (fn.parameters as Record<string, unknown>) ?? {
          type: "object",
          properties: {},
        },
        permissionMode:
          String(toolItem.permission_mode ?? "always_allow") === "needs_approval"
            ? "needs_approval"
            : "always_allow",
      });
    }
  }

  async refreshRemote(): Promise<AgentBlitAgentConfig> {
    const url = `${this.baseUrl}/api/1.0/agent`;
    const response = await fetch(url, {
      method: "GET",
      headers: { "X-API-Key": this.apiKey },
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!response.ok) {
      throw new Error(
        `AgentBlit request failed for GET '${url}': ${response.status} ${response.statusText}`,
      );
    }

    const data = (await response.json()) as AgentBlitAgentConfig;
    this.applyRemoteTools(data.tools);
    return data;
  }

  private merged(): Map<string, ToolDefinition> {
    const merged = new Map(this.remote);
    for (const [name, def] of this.custom.entries()) {
      merged.set(name, def);
    }
    return merged;
  }

  getDefinition(name: string): ToolDefinition | undefined {
    return this.merged().get(name);
  }

  toOpenAITools(): Array<Record<string, unknown>> {
    return [...this.merged().values()].map((definition) => ({
      type: "function",
      function: {
        name: definition.name,
        description: definition.description,
        parameters: definition.inputSchema,
      },
    }));
  }

  private async ensureApproval(
    definition: ToolDefinition,
    toolName: string,
    args: Record<string, unknown>,
    approvalCallback?: ApprovalCallback,
  ): Promise<boolean> {
    if (definition.permissionMode !== "needs_approval") {
      return true;
    }
    if (approvalCallback) {
      return approvalCallback(toolName, args);
    }
    const answer = await readStdinLine(
      `Approve tool "${toolName}" with args ${jsonDumpsSafe(args)}? [y/N]: `,
    );
    return ["y", "yes"].includes(answer.toLowerCase());
  }

  private parseToolArguments(
    argumentsJson: string,
  ): { ok: true; args: Record<string, unknown> } | { ok: false; error: string } {
    try {
      const args = argumentsJson
        ? (JSON.parse(argumentsJson) as Record<string, unknown>)
        : {};
      return { ok: true, args };
    } catch (error) {
      return { ok: false, error: `Invalid JSON arguments: ${String(error)}` };
    }
  }

  async execute(
    toolCallId: string,
    toolName: string,
    argumentsJson: string,
    approvalCallback?: ApprovalCallback,
    elicitationCallback?: ElicitationCallback,
  ): Promise<string> {
    const results = await this.executeBatch(
      [{ id: toolCallId, function: { name: toolName, arguments: argumentsJson } }],
      approvalCallback,
      elicitationCallback,
    );
    return results.get(toolCallId) ?? jsonDumpsSafe({ error: "No result for tool_call_id" });
  }

  /**
   * Executes multiple tool calls in one round.
   * Custom tools (inline handlers) run sequentially; remote tools are sent in a
   * single POST to `/api/1.0/tools/call` so the server can reuse one MCP session.
   *
   * When a remote tool responds with `elicitation_required`, the SDK calls
   * `elicitationCallback` to collect user input, then retries the call with
   * `elicitation_result`. This loop continues until the tool resolves or the
   * user declines/cancels.
   */
  async executeBatch(
    toolCalls: Array<{ id: string; function: { name: string; arguments: string } }>,
    approvalCallback?: ApprovalCallback,
    elicitationCallback?: ElicitationCallback,
  ): Promise<Map<string, string>> {
    const results = new Map<string, string>();
    const remotePending: Array<{
      toolCall: { id: string; function: { name: string; arguments: string } };
      args: Record<string, unknown>;
    }> = [];

    for (const toolCall of toolCalls) {
      const definition = this.getDefinition(toolCall.function.name);
      if (!definition) {
        results.set(
          toolCall.id,
          jsonDumpsSafe({ error: `Unknown tool: ${toolCall.function.name}` }),
        );
        continue;
      }

      const parsed = this.parseToolArguments(toolCall.function.arguments);
      if (!parsed.ok) {
        results.set(toolCall.id, jsonDumpsSafe({ error: parsed.error }));
        continue;
      }
      const args = parsed.args;

      if (definition.handler) {
        const result = await this.executeCustomTool(
          definition,
          toolCall.function.name,
          args,
          approvalCallback,
        );
        results.set(toolCall.id, result);
        continue;
      }

      if (!(await this.ensureApproval(definition, toolCall.function.name, args, approvalCallback))) {
        results.set(
          toolCall.id,
          jsonDumpsSafe({ error: "User denied approval for this tool call." }),
        );
        continue;
      }

      remotePending.push({ toolCall, args });
    }

    if (remotePending.length > 0) {
      // Run elicitation-aware HTTP round-trips for each remote tool individually
      // so we can interleave elicitation without blocking unrelated calls.
      const remoteResults = await Promise.all(
        remotePending.map(({ toolCall, args }) =>
          this.executeRemoteToolWithElicitation(
            toolCall,
            args,
            elicitationCallback,
          ),
        ),
      );
      for (const { id, result } of remoteResults) {
        results.set(id, result);
      }
    }

    return results;
  }

  /**
   * Executes one remote tool call, handling the `elicitation_required` retry loop.
   * Each iteration posts to `/api/1.0/tools/call`; if the server returns
   * `elicitation_required`, we call `elicitationCallback`, attach the result, and
   * retry. The loop exits when the server returns a final result or error, or when
   * the user declines/cancels.
   */
  private async executeRemoteToolWithElicitation(
    toolCall: { id: string; function: { name: string; arguments: string } },
    args: Record<string, unknown>,
    elicitationCallback?: ElicitationCallback,
  ): Promise<{ id: string; result: string }> {
    const url = `${this.baseUrl}/api/1.0/tools/call`;
    let pendingElicitationResult: ElicitationResult | undefined;

    // Max elicitation rounds per call to prevent infinite loops.
    const MAX_ELICITATION_ROUNDS = 10;

    for (let round = 0; round <= MAX_ELICITATION_ROUNDS; round += 1) {
      const payload: Record<string, unknown> = {
        tool_calls: [
          {
            id: toolCall.id,
            type: "function",
            function: {
              name: toolCall.function.name,
              arguments: jsonDumpsSafe(args),
            },
          },
        ] satisfies OpenAIToolCall[],
      };

      if (pendingElicitationResult) {
        payload.elicitation_result = pendingElicitationResult;
      }

      let response: Response;
      try {
        response = await fetch(url, {
          method: "POST",
          headers: {
            "X-API-Key": this.apiKey,
            "Content-Type": "application/json",
          },
          body: jsonDumpsSafe(payload),
          signal: AbortSignal.timeout(this.timeout),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { id: toolCall.id, result: jsonDumpsSafe({ error: message }) };
      }

      if (!response.ok) {
        return {
          id: toolCall.id,
          result: jsonDumpsSafe({
            error: `AgentBlit request failed for POST '${url}': ${response.status} ${response.statusText}`,
          }),
        };
      }

      const data = (await response.json()) as {
        results?: Array<{
          tool_call_id?: string;
          result?: unknown;
          error?: unknown;
          elicitation_required?: ElicitationRequest;
        }>;
      };

      const row = (data.results ?? []).find((r) => r.tool_call_id === toolCall.id);
      if (!row) {
        return {
          id: toolCall.id,
          result: jsonDumpsSafe({ error: "No result for tool_call_id" }),
        };
      }

      // Server is requesting user input before it can complete the tool call.
      if (row.elicitation_required) {
        if (!elicitationCallback) {
          // No callback — cancel immediately so the connector can handle it gracefully.
          pendingElicitationResult = { action: "cancel" };
          continue;
        }

        const elicitResult = await elicitationCallback(
          toolCall.function.name,
          row.elicitation_required,
        );

        if (elicitResult.action !== "accept") {
          // User declined or cancelled — return the result of the final (non-elicitation) call.
          pendingElicitationResult = elicitResult;
          continue;
        }

        pendingElicitationResult = elicitResult;
        continue;
      }

      // Final result or error.
      if (typeof row.error !== "undefined") {
        return { id: toolCall.id, result: jsonDumpsSafe({ error: row.error }) };
      }
      if (typeof row.result !== "undefined") {
        return { id: toolCall.id, result: jsonDumpsSafe(row.result) };
      }
      return {
        id: toolCall.id,
        result: jsonDumpsSafe({ error: "Missing result or error for tool_call_id" }),
      };
    }

    return {
      id: toolCall.id,
      result: jsonDumpsSafe({ error: "Exceeded max elicitation rounds" }),
    };
  }

  private async executeCustomTool(
    definition: ToolDefinition,
    toolName: string,
    args: Record<string, unknown>,
    approvalCallback?: ApprovalCallback,
  ): Promise<string> {
    if (!(await this.ensureApproval(definition, toolName, args, approvalCallback))) {
      console.error("User denied approval for this tool call.", toolName, args);
      return jsonDumpsSafe({ error: "User denied approval for this tool call." });
    }

    if (!definition.handler) {
      return jsonDumpsSafe({ error: `Tool has no handler: ${toolName}` });
    }

    try {
      let result: unknown;
      try {
        result = await definition.handler(...Object.values(args));
      } catch {
        result = await definition.handler(args);
      }
      return jsonDumpsSafe(result);
    } catch (error) {
      return jsonDumpsSafe({ error: error instanceof Error ? error.message : String(error) });
    }
  }
}
