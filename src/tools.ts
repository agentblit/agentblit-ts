import {
  ApprovalCallback,
  AgentBlitAgentConfig,
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

  async execute(
    toolCallId: string,
    toolName: string,
    argumentsJson: string,
    approvalCallback?: ApprovalCallback,
  ): Promise<string> {
    const definition = this.getDefinition(toolName);
    if (!definition) {
      return jsonDumpsSafe({ error: `Unknown tool: ${toolName}` });
    }
    let args: Record<string, unknown>;
    try {
      args = argumentsJson ? (JSON.parse(argumentsJson) as Record<string, unknown>) : {};
    } catch (error) {
      return jsonDumpsSafe({ error: `Invalid JSON arguments: ${String(error)}` });
    }

    if (!(await this.ensureApproval(definition, toolName, args, approvalCallback))) {
      console.error("User denied approval for this tool call.", toolName, args);
      return jsonDumpsSafe({ error: "User denied approval for this tool call." });
    }

    if (definition.handler) {
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

    const payload = {
      tool_calls: [
        {
          id: toolCallId,
          type: "function",
          function: {
            name: toolName,
            arguments: jsonDumpsSafe(args),
          },
        },
      ] satisfies OpenAIToolCall[],
    };
    if (!toolCallId.trim()) {
      return jsonDumpsSafe({ error: "Missing tool_call_id" });
    }
    const url = `${this.baseUrl}/api/1.0/tools/call`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "X-API-Key": this.apiKey,
        "Content-Type": "application/json",
      },
      body: jsonDumpsSafe(payload),
      signal: AbortSignal.timeout(this.timeout),
    });
    if (!response.ok) {
      throw new Error(
        `AgentBlit request failed for POST '${url}': ${response.status} ${response.statusText}`,
      );
    }
    const data = (await response.json()) as {
      results?: Array<{
        tool_call_id?: string;
        result?: unknown;
        error?: unknown;
      }>;
    };
    const result = (data.results ?? []).find((item) => item.tool_call_id === toolCallId);
    if (!result) {
      return jsonDumpsSafe({ error: "No result for tool_call_id" });
    }
    if (typeof result.error !== "undefined") {
      return jsonDumpsSafe({ error: result.error });
    }
    if (typeof result.result !== "undefined") {
      return jsonDumpsSafe(result.result);
    }
    return jsonDumpsSafe({ error: "Missing result or error for tool_call_id" });
  }
}
