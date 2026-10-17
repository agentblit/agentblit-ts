import process from "node:process";
import { randomUUID } from "node:crypto";
import type {
  ChatMessage,
  OpenAIInputContentPart,
  ToolHandler,
  ToolOptions,
} from "./types.js";

const LOGGER_PREFIX = "[agentblit]";

export class DebugLogger {
  constructor(private readonly enabled: boolean) {}

  log(message: string, ...args: unknown[]): void {
    if (!this.enabled) {
      return;
    }
    console.debug(`${LOGGER_PREFIX} ${message}`, ...args);
  }

  logLLMRequest(model: string, messageCount: number, toolCount: number): void {
    this.log("LLM request model=%s messages=%s tools=%s", model, messageCount, toolCount);
  }

  logToolCalls(calls: unknown[]): void {
    this.log("Tool calls: %s", jsonDumpsSafe(calls));
  }

  logToolResult(toolCallId: string, ok: boolean, preview: string): void {
    this.log("Tool result id=%s ok=%s preview=%s", toolCallId, ok, preview.slice(0, 500));
  }
}

export function jsonDumpsSafe(obj: unknown): string {
  return JSON.stringify(obj, (_key, value) => {
    if (value instanceof Error) {
      return {
        name: value.name,
        message: value.message,
        stack: value.stack,
      };
    }
    return value;
  });
}

/** Strip large base64 image payloads from tool results before logging/events. */
export function redactToolResultForLog(value: unknown): unknown {
  if (!value || typeof value !== "object") return value;
  if (Array.isArray(value)) {
    return value.map((item) => redactToolResultForLog(item));
  }
  const record = value as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(record)) {
    if (
      (key === "data" || key === "image_base64") &&
      typeof item === "string" &&
      item.length > 200
    ) {
      out[key] = `[base64 ${item.length} chars]`;
      continue;
    }
    out[key] = redactToolResultForLog(item);
  }
  return out;
}

/** Redact data-URL images in chat messages before logging/events. */
export function redactChatMessageForLog(message: ChatMessage): ChatMessage {
  const content = message.content;
  if (!Array.isArray(content)) return message;
  return {
    ...message,
    content: content.map((part) => {
      if (part.type !== "image_url") return part;
      const url = part.image_url.url;
      if (!url.startsWith("data:") || url.length < 200) return part;
      return {
        type: "image_url" as const,
        image_url: {
          ...part.image_url,
          url: `[data-url ${url.length} chars]`,
        },
      };
    }),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export type ParsedToolResultMedia = {
  /** String content for the `tool` role (Chat Completions requires string). */
  toolContent: string;
  /** Follow-up user message with images (vision). Null if none. */
  imageMessage: OpenAIInputContentPart[] | null;
};

/**
 * Split a tools/call JSON result into a string tool message plus optional
 * vision follow-up. Chat Completions tool roles are string-only; screenshots
 * are attached as the next user message so the model can see them.
 */
export function parseToolResultMedia(resultJson: string): ParsedToolResultMedia {
  let parsed: unknown;
  try {
    parsed = JSON.parse(resultJson) as unknown;
  } catch {
    return { toolContent: resultJson, imageMessage: null };
  }
  if (!isRecord(parsed) || !Array.isArray(parsed.content)) {
    return { toolContent: resultJson, imageMessage: null };
  }

  const textParts: string[] = [];
  const imageParts: OpenAIInputContentPart[] = [];
  for (const block of parsed.content) {
    if (!isRecord(block) || typeof block.type !== "string") continue;
    if (block.type === "text" && typeof block.text === "string") {
      textParts.push(block.text);
      continue;
    }
    if (block.type === "image" && typeof block.data === "string" && block.data) {
      const mime =
        typeof block.mimeType === "string" && block.mimeType
          ? block.mimeType
          : "image/jpeg";
      imageParts.push({
        type: "image_url",
        image_url: { url: `data:${mime};base64,${block.data}`, detail: "high" },
      });
    }
  }

  if (imageParts.length === 0) {
    return { toolContent: resultJson, imageMessage: null };
  }

  const summary = textParts.join("\n").trim() || "Screenshot attached.";
  return {
    toolContent: jsonDumpsSafe({ ok: true, text: summary, image: true }),
    imageMessage: [{ type: "text", text: summary }, ...imageParts],
  };
}

function extractParamNames(fn: ToolHandler): string[] {
  const source = fn.toString();
  const match = source.match(/\(([^)]*)\)/);
  if (!match) {
    return [];
  }
  const rawParams = match[1] ?? "";
  return rawParams
    .split(",")
    .map((segment) => segment.trim())
    .filter(Boolean)
    .map((segment) => segment.replace(/=[\s\S]*$/, "").trim())
    .map((segment) => segment.replace(/^\.{3}/, ""))
    .filter((segment) => segment !== "this");
}

export function functionToToolSchema(fn: ToolHandler, explicitSchema?: Record<string, unknown>): Record<string, unknown> {
  if (explicitSchema) {
    return explicitSchema;
  }
  const properties = Object.fromEntries(
    extractParamNames(fn).map((name) => [name, { type: "string" }]),
  );
  const required = Object.keys(properties);
  return {
    type: "object",
    properties,
    ...(required.length > 0 ? { required } : {}),
  };
}

export const TOOL_META = Symbol.for("agentblit.tool.meta");

export interface ToolMetadata extends Required<Pick<ToolOptions, "name" | "description" | "permissionMode">> {
  inputSchema?: Record<string, unknown>;
}

export function setToolMetadata(fn: ToolHandler, options: ToolOptions): ToolHandler {
  const metadata: ToolMetadata = {
    name: options.name ?? fn.name,
    description: options.description ?? "",
    permissionMode: options.permissionMode ?? "always_allow",
    inputSchema: options.inputSchema,
  };
  (fn as ToolHandler & { [TOOL_META]?: ToolMetadata })[TOOL_META] = metadata;
  return fn;
}

export function getToolMetadata(fn: ToolHandler): ToolMetadata | undefined {
  return (fn as ToolHandler & { [TOOL_META]?: ToolMetadata })[TOOL_META];
}

export function nowIsoTimestamp(): string {
  return new Date().toISOString();
}

export function randomId(prefix: string): string {
  return `${prefix}_${randomUUID().replace(/-/g, "")}`;
}

export function readStdinLine(promptText: string): Promise<string> {
  return new Promise((resolve) => {
    process.stdout.write(promptText);
    process.stdin.resume();
    process.stdin.once("data", (chunk) => {
      resolve(String(chunk).trim());
    });
  });
}
