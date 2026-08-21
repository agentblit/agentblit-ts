import OpenAI from "openai";
import { randomUUID } from "node:crypto";
import { ChatMemory } from "./memory.js";
import {
  AgentRunInput,
  AgentConfig,
  AgentOptions,
  ChatMessage,
  ChatMessageContent,
  ElicitationRequest,
  ElicitationResult,
  OpenAIInputContentPart,
  OpenAIToolCall,
  ToolHandler,
} from "./types.js";
import { ToolRegistry } from "./tools.js";
import { DebugLogger, jsonDumpsSafe, nowIsoTimestamp, randomId } from "./utils.js";

const VENDOR_BASE_URLS: Record<string, string> = {
  openai: "https://api.openai.com/v1",
  anthropic: "https://api.anthropic.com/v1",
  gemini: "https://generativelanguage.googleapis.com/v1beta/openai",
  openrouter: "https://openrouter.ai/api/v1",
};

const DEFAULT_TOOL_USAGE_INSTRUCTION =
  "Use tools when they help answer accurately.";

/** Matches POST /api/events/batch `MAX_BATCH` on the AgentBlit API. */
const MAX_EVENTS_PER_BATCH = 100;

function resolveVendorAndModel(modelInput: string): { vendor: string; model: string } {
  const model = modelInput.trim();
  if (!model || !model.includes("/")) {
    throw new Error(
      "model must be in the format 'vendor/model', for example 'openai/gpt-4o-mini'.",
    );
  }
  const [vendorRaw, ...rest] = model.split("/");
  const vendor = (vendorRaw ?? "").trim().toLowerCase();
  const providerModel = rest.join("/").trim();
  if (!vendor || !providerModel) {
    throw new Error(
      "model must be in the format 'vendor/model', for example 'openai/gpt-4o-mini'.",
    );
  }
  return { vendor, model: providerModel };
}

function resolveLlmUrl(vendor: string): string {
  const url = VENDOR_BASE_URLS[vendor];
  if (!url) {
    const supported = Object.keys(VENDOR_BASE_URLS).sort().join(", ");
    throw new Error(`Unsupported model vendor '${vendor}'. Supported vendors: ${supported}.`);
  }
  return url;
}

function formatMessagesForSummary(messages: ChatMessage[]): string {
  return messages
    .map((message, index) => {
      if (message.role === "assistant" && message.tool_calls) {
        return `[${index}] assistant (tool_calls): ${jsonDumpsSafe(message.tool_calls)}`;
      }
      return `[${index}] ${message.role}: ${serializeContentForSummary(message.content)}`;
    })
    .join("\n");
}

function serializeContentForSummary(content: ChatMessageContent | undefined): string {
  if (content === null || content === undefined) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  const parts: string[] = [];
  for (const part of content) {
    if (part.type === "text") {
      parts.push(part.text);
      continue;
    }
    if (part.type === "image_url") {
      parts.push(`[image: ${part.image_url.url}]`);
      continue;
    }
    if (part.type === "file") {
      parts.push(`[file: ${part.file.file_id ?? part.file.filename ?? "inline_data"}]`);
      continue;
    }
    parts.push("[unsupported_content_part]");
  }
  return parts.join(" ").trim();
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function validateInputContentParts(parts: OpenAIInputContentPart[]): OpenAIInputContentPart[] {
  for (const part of parts) {
    if (part.type === "text") {
      if (typeof part.text !== "string") {
        throw new Error("Text content part must include a string 'text' field.");
      }
      continue;
    }
    if (part.type === "image_url") {
      if (!part.image_url || typeof part.image_url.url !== "string") {
        throw new Error(
          "Image content part must include image_url.url as a string.",
        );
      }
      continue;
    }
    if (part.type === "file") {
      const file = part.file;
      if (!file || (!file.file_id && !file.file_data)) {
        throw new Error(
          "File content part must include at least one of file.file_id or file.file_data.",
        );
      }
      continue;
    }
    throw new Error(
      "Unsupported content part type. Supported types: text, image_url, file.",
    );
  }
  return parts;
}

function normalizeRunInput(userInput: AgentRunInput): ChatMessageContent {
  if (typeof userInput === "string") {
    return userInput;
  }
  if (Array.isArray(userInput)) {
    return validateInputContentParts(userInput);
  }
  if (isObject(userInput)) {
    const { content } = userInput;
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return validateInputContentParts(content as OpenAIInputContentPart[]);
    }
  }
  throw new Error(
    "run input must be a string, an array of OpenAI content parts, or an object with a content field.",
  );
}

interface EventPayload {
  id: string;
  session_id: string;
  agent_id: string;
  timestamp: string;
  type: string;
  data: unknown;
  tokens: number;
  latency_ms: number;
}

export class Agent {
  vendor = "";
  model = "";
  llmUrl = "";
  agentId = "";
  readonly sessionId: string;
  config: AgentConfig | null = null;
  readonly memory: ChatMemory;

  private client!: OpenAI;
  private readonly tools: ToolRegistry;
  private systemPrompt = "";
  private readonly timeout: number;
  private readonly debug: DebugLogger;
  private readonly approvalCallback?: AgentOptions["approvalCallback"];
  private readonly elicitationCallback?: AgentOptions["elicitationCallback"];
  private readonly maxToolRounds: number;
  private readonly eventBaseUrl: string;
  private readonly eventApiKey: string;
  private readonly pendingCustomEvents: EventPayload[] = [];
  /** Events for the in-flight `run()`; summarization appends here when set. */
  private activeRunEvents: EventPayload[] | null = null;
  private agentInitSent = false;
  private toolsSignature?: string;
  private initialized = false;

  private readonly llmApiKey: string;
  private readonly modelOverride?: string;
  private readonly debugEnabled: boolean;
  private readonly maxHistory: number;

  constructor(options: AgentOptions) {
    const agentblitApiKey = options.agentblitApiKey.trim();
    if (!agentblitApiKey) {
      throw new Error("agentblitApiKey is required");
    }
    const maxHistory = options.maxHistory ?? 10;
    if (maxHistory < 1) {
      throw new Error("maxHistory must be at least 1");
    }
    const maxToolRounds = options.maxToolRounds ?? 25;
    if (maxToolRounds < 1) {
      throw new Error("maxToolRounds must be at least 1");
    }

    const timeoutSeconds = options.timeout ?? 30;
    const timeoutMs = timeoutSeconds * 1000;
    const agentblitUrl = (options.agentblitUrl ?? "https://console.agentblit.com").replace(/\/$/, "");

    this.llmApiKey = options.apiKey;
    const modelOverride = options.model?.trim();
    this.modelOverride = modelOverride || undefined;
    this.maxHistory = maxHistory;
    this.debugEnabled = Boolean(options.debug);
    this.timeout = timeoutMs;
    this.approvalCallback = options.approvalCallback;
    this.elicitationCallback = options.elicitationCallback;
    this.maxToolRounds = maxToolRounds;
    this.sessionId = options.sessionId?.trim() || randomUUID();
    this.eventBaseUrl = agentblitUrl;
    this.eventApiKey = agentblitApiKey;
    this.debug = new DebugLogger(this.debugEnabled);

    this.tools = new ToolRegistry({
      baseUrl: agentblitUrl,
      apiKey: agentblitApiKey,
      timeout: timeoutMs,
      sessionId: this.sessionId,
    });
    for (const customTool of options.customTools ?? []) {
      this.registerTool(customTool);
    }
    this.memory = new ChatMemory({
      maxHistory,
      summarizeFn: (older) => this.summarizeOlderMessages(older),
    });
  }

  private async ensureInitialized(): Promise<void> {
    const remote = await this.tools.refreshRemote();
    if (this.initialized) {
      return;
    }
    const modelInput = this.modelOverride ?? remote.model;
    if (!modelInput.trim()) {
      throw new Error(
        "No model configured; set model in AgentBlit or pass model in AgentOptions.",
      );
    }
    this.agentId = remote.id.trim();
    if (!this.agentId) {
      throw new Error(
        "AgentBlit agent response is missing id; check your AgentBlit API key and agent configuration.",
      );
    }
    const { vendor, model } = resolveVendorAndModel(modelInput);
    const llmUrl = resolveLlmUrl(vendor);
    this.vendor = vendor;
    this.model = model;
    this.llmUrl = llmUrl;
    this.systemPrompt = remote.system_prompt.trim() || DEFAULT_TOOL_USAGE_INSTRUCTION;
    this.client = new OpenAI({
      apiKey: this.llmApiKey,
      baseURL: llmUrl,
      timeout: this.timeout,
    });
    this.config = Object.freeze({
      model: this.model,
      vendor: this.vendor,
      llmUrl: this.llmUrl,
      agentblitUrl: this.eventBaseUrl,
      systemPrompt: this.systemPrompt,
      maxHistory: this.maxHistory,
      debug: this.debugEnabled,
      timeout: this.timeout / 1000,
      agentId: this.agentId,
      sessionId: this.sessionId,
    });
    this.initialized = true;
  }

  registerTool(fn: ToolHandler): void {
    this.tools.register(fn);
  }

  track(eventType: string, properties: Record<string, unknown>): void {
    this.pendingCustomEvents.push(this.makeEvent({ eventType, data: properties }));
  }

  private makeEvent(input: {
    eventType: string;
    data: unknown;
    tokens?: number;
    latencyMs?: number;
  }): EventPayload {
    return {
      id: randomId("evt"),
      session_id: this.sessionId,
      agent_id: this.agentId,
      timestamp: nowIsoTimestamp(),
      type: input.eventType,
      data: input.data,
      tokens: Math.max(0, Math.trunc(input.tokens ?? 0)),
      latency_ms: Math.max(0, Math.trunc(input.latencyMs ?? 0)),
    };
  }

  private async flushEvents(events: EventPayload[]): Promise<void> {
    if (events.length === 0) {
      return;
    }
    const url = `${this.eventBaseUrl}/api/events/batch`;
    for (let offset = 0; offset < events.length; offset += MAX_EVENTS_PER_BATCH) {
      const chunk = events.slice(offset, offset + MAX_EVENTS_PER_BATCH);
      this.debug.log("Event batch send start url=%s count=%s", url, chunk.length);
      try {
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "X-API-Key": this.eventApiKey,
            "Content-Type": "application/json",
          },
          body: jsonDumpsSafe({ events: chunk }),
          signal: AbortSignal.timeout(this.timeout),
        });
        if (!response.ok) {
          const body = (await response.text()).slice(0, 1000);
          this.debug.log(
            "Failed to send events batch status=%s body=%s",
            response.status,
            body,
          );
          continue;
        }
        this.debug.log(
          "Event batch send success status=%s count=%s",
          response.status,
          chunk.length,
        );
      } catch (error) {
        this.debug.log("Failed to send events batch: %s", String(error));
      }
    }
  }

  private extractLlmEventMessages(messages: ChatMessage[]): ChatMessage[] {
    if (messages.length === 0) {
      return [];
    }
    const summaryMessage = messages.find(
      (message) =>
        message.role === "system" &&
        typeof message.content === "string" &&
        message.content.startsWith("Summary of earlier conversation:"),
    );
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage) {
      return [];
    }
    if (!summaryMessage) {
      return [lastMessage];
    }
    if (summaryMessage === lastMessage) {
      return [summaryMessage];
    }
    return [summaryMessage, lastMessage];
  }

  private async buildAgentLoopErrorData(error: unknown): Promise<Record<string, unknown>> {
    const data: Record<string, unknown> = {
      error: error instanceof Error ? error : String(error),
    };
    try {
      data.messages = await this.memory.buildMessagesForLLM(this.systemPrompt);
    } catch (memoryError) {
      data.messages = this.memory.messages;
      data.messages_build_error = String(memoryError);
    }
    try {
      data.tools = this.tools.toOpenAITools();
    } catch (toolsError) {
      data.tools = [];
      data.tools_error = String(toolsError);
    }
    return data;
  }

  private async summarizeOlderMessages(older: ChatMessage[]): Promise<string> {
    const text = formatMessagesForSummary(older);
    this.debug.log("Summarizing %s older messages", older.length);
    const startedAt = Date.now();
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        {
          role: "system",
          content:
            "Summarize the conversation excerpt below for use as memory. Preserve key facts, decisions, and tool outcomes. Be concise.",
        },
        { role: "user", content: text },
      ],
      temperature: 0.2,
    });
    const usage = response.usage as Record<string, unknown> | undefined;
    const totalTokens =
      typeof usage?.total_tokens === "number" ? usage.total_tokens : 0;
    const content = response.choices[0]?.message?.content?.trim() ?? "";
    this.activeRunEvents?.push(
      this.makeEvent({
        eventType: "llm_call",
        data: {
          source: "memory_summarize",
          request: { model: this.model, stream: false },
          response: {
            content,
            ...(usage ? { usage } : {}),
          },
        },
        tokens: totalTokens,
        latencyMs: Date.now() - startedAt,
      }),
    );
    return content;
  }

  async *run(userInput: AgentRunInput): AsyncGenerator<string> {
    const events: EventPayload[] = [...this.pendingCustomEvents];
    this.pendingCustomEvents.length = 0;
    this.activeRunEvents = events;
    try {
      const userContent = normalizeRunInput(userInput);
      this.memory.append({ role: "user", content: userContent });
      await this.ensureInitialized();
      const openaiTools = this.tools.toOpenAITools();
      const toolsSignature = jsonDumpsSafe(openaiTools);
      if (!this.agentInitSent) {
        events.push(
          this.makeEvent({
            eventType: "agent_init",
            data: { system_prompt: this.systemPrompt, tools: openaiTools },
          }),
        );
        this.agentInitSent = true;
        this.toolsSignature = toolsSignature;
      } else if (toolsSignature !== this.toolsSignature) {
        events.push(this.makeEvent({ eventType: "tools_updated", data: { tools: openaiTools } }));
        this.toolsSignature = toolsSignature;
      }
      events.push(
        this.makeEvent({
          eventType: "user_prompt",
          data: { request: { message: userContent }, response: null },
        }),
      );

      let finishedNormally = false;
      for (let round = 0; round < this.maxToolRounds; round += 1) {
        const messages = await this.memory.buildMessagesForLLM(this.systemPrompt);
        this.debug.logLLMRequest(this.model, messages.length, openaiTools.length);
        const llmRequest = {
          model: this.model,
          messages: messages as any,
          stream: true as const,
          ...(openaiTools.length > 0 ? { tools: openaiTools } : {}),
          ...(this.vendor === "openai" || this.vendor === "openrouter"
            ? { stream_options: { include_usage: true } }
            : {}),
        };

        const llmEventRequestData = {
          model: this.model,
          messages: this.extractLlmEventMessages(messages) as any,
          stream: true,
        };

        const startedAt = Date.now();
        const stream = (await this.client.chat.completions.create(
          llmRequest as any,
        )) as unknown as AsyncIterable<any>;

        const contentParts: string[] = [];
        const toolCallsMap = new Map<
          number,
          { id: string; name: string; arguments: string }
        >();
        let finishReason: string | null = null;
        let llmTotalTokens = 0;
        let llmUsage: Record<string, unknown> | null = null;

        for await (const chunk of stream) {
          if (chunk.usage) {
            llmUsage = chunk.usage as Record<string, unknown>;
            if (typeof chunk.usage.total_tokens === "number") {
              llmTotalTokens = chunk.usage.total_tokens;
            }
          }
          const choice = chunk.choices?.[0];
          if (!choice) {
            continue;
          }
          if (choice.finish_reason) {
            finishReason = choice.finish_reason;
          }
          const delta = choice.delta;
          if (!delta) {
            continue;
          }
          if (delta.content) {
            contentParts.push(delta.content);
            yield delta.content;
          }
          for (const toolCall of delta.tool_calls ?? []) {
            const index = toolCall.index ?? 0;
            const existing = toolCallsMap.get(index) ?? { id: "", name: "", arguments: "" };
            if (toolCall.id) {
              existing.id = toolCall.id;
            }
            if (toolCall.function?.name) {
              existing.name = toolCall.function.name;
            }
            if (toolCall.function?.arguments) {
              existing.arguments += toolCall.function.arguments;
            }
            toolCallsMap.set(index, existing);
          }
        }

        const assistantContent = contentParts.join("") || null;
        const llmLatencyMs = Date.now() - startedAt;
        const toolCalls = [...toolCallsMap.entries()]
          .sort(([a], [b]) => a - b)
          .map(([, call]): OpenAIToolCall => ({
            id: call.id,
            type: "function",
            function: {
              name: call.name,
              arguments: call.arguments || "{}",
            },
          }));

        events.push(
          this.makeEvent({
            eventType: "llm_call",
            data: {
              request: llmEventRequestData,
              response: {
                finish_reason: finishReason,
                content: assistantContent,
                tool_calls: toolCalls,
                ...(llmUsage ? { usage: llmUsage } : {}),
              },
            },
            tokens: llmTotalTokens,
            latencyMs: llmLatencyMs,
          }),
        );

        if (finishReason === "tool_calls" || toolCalls.length > 0) {
          this.debug.logToolCalls(toolCalls);
          this.memory.append({
            role: "assistant",
            content: assistantContent,
            tool_calls: toolCalls,
          });

          // Wrap elicitationCallback to emit elicitation_request / elicitation_response
          // events so analytics can separate user think-time from tool execution time.
          const instrumentedElicitationCallback = this.elicitationCallback
            ? async (toolName: string, request: ElicitationRequest): Promise<ElicitationResult> => {
                const elicitStartedAt = Date.now();
                events.push(
                  this.makeEvent({
                    eventType: "elicitation_request",
                    data: { tool_name: toolName, request },
                  }),
                );
                const result = await this.elicitationCallback!(toolName, request);
                events.push(
                  this.makeEvent({
                    eventType: "elicitation_response",
                    data: { tool_name: toolName, result },
                    latencyMs: Date.now() - elicitStartedAt,
                  }),
                );
                return result;
              }
            : undefined;

          const batchStartedAt = Date.now();
          const toolResults = await this.tools.executeBatch(
            toolCalls,
            this.approvalCallback,
            instrumentedElicitationCallback,
          );
          const batchLatencyMs = Date.now() - batchStartedAt;

          for (const toolCall of toolCalls) {
            const result =
              toolResults.get(toolCall.id) ??
              jsonDumpsSafe({ error: "No result for tool_call_id" });
            this.debug.logToolResult(toolCall.id, true, result);
            let parsedResponse: unknown;
            try {
              parsedResponse = JSON.parse(result);
            } catch {
              parsedResponse = { raw: result };
            }
            events.push(
              this.makeEvent({
                eventType: "tool_call",
                data: {
                  request: toolCall,
                  response: parsedResponse,
                },
                latencyMs: batchLatencyMs,
              }),
            );
            this.memory.append({
              role: "tool",
              tool_call_id: toolCall.id,
              content: result,
            });
          }
          continue;
        }

        this.memory.append({
          role: "assistant",
          content: assistantContent ?? "",
        });
        finishedNormally = true;
        break;
      }

      if (!finishedNormally) {
        throw new Error(
          `Exceeded maxToolRounds (${this.maxToolRounds}); increase maxToolRounds or simplify the task.`,
        );
      }
    } catch (error) {
      events.push(
        this.makeEvent({
          eventType: "agent_loop_error",
          data: await this.buildAgentLoopErrorData(error),
        }),
      );
      throw error;
    } finally {
      this.debug.log("Queueing event flush count=%s", events.length);
      try {
        await this.flushEvents(events);
      } finally {
        this.activeRunEvents = null;
      }
    }
  }

  /**
   * Runs the agent to completion and returns the full assistant text (concatenated stream chunks).
   * Same behavior as iterating {@link Agent.run}; use when you do not need incremental output.
   */
  async runSync(userInput: AgentRunInput): Promise<string> {
    let text = "";
    for await (const chunk of this.run(userInput)) {
      text += chunk;
    }
    return text;
  }
}
