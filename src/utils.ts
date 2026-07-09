import process from "node:process";
import { randomUUID } from "node:crypto";
import { ToolHandler, ToolOptions } from "./types.js";

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
