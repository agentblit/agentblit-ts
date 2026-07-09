export type JSONSchema = Record<string, unknown>;

export type ToolPermissionMode = "always_allow" | "needs_approval";

export type ToolHandler = (...args: any[]) => unknown | Promise<unknown>;

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: JSONSchema;
  permissionMode: ToolPermissionMode;
  outputSchema?: JSONSchema;
  handler?: ToolHandler;
}

export interface AgentBlitAgentConfig {
  id: string;
  name: string;
  model: string;
  system_prompt: string;
  tools: Array<Record<string, unknown>>;
}

export interface AgentConfig {
  model: string;
  vendor: string;
  llmUrl: string;
  agentblitUrl: string;
  systemPrompt: string;
  maxHistory: number;
  debug: boolean;
  timeout: number;
  agentId: string;
  sessionId: string;
}

export type ApprovalCallback = (
  toolName: string,
  args: Record<string, unknown>,
) => Promise<boolean>;

/**
 * A flat JSON Schema property as defined in the MCP elicitation spec
 * (`requestedSchema.properties[key]`). Only top-level primitive fields are supported —
 * no nesting. A `string` field with `enum` renders as a multiple-choice picker.
 */
export interface ElicitFieldSchema {
  type: "string" | "number" | "integer" | "boolean";
  title?: string;
  description?: string;
  /** When present on a `string` field, the host should render a choice picker. */
  enum?: string[];
  minimum?: number;
  maximum?: number;
  default?: string | number | boolean | string[];
}

/** Mirrors MCP `ElicitRequest.params` (form mode). */
export interface ElicitationRequest {
  /** Human-readable message describing what information is needed. */
  message: string;
  /**
   * Optional JSON Schema subset for the fields to collect.
   * When absent, the host should present a plain confirm/decline prompt.
   */
  requestedSchema?: {
    type: "object";
    properties: Record<string, ElicitFieldSchema>;
    required?: string[];
  };
}

/** Mirrors MCP `ElicitResult`. */
export interface ElicitationResult {
  /** "accept" — user filled the form; "decline" / "cancel" — user refused. */
  action: "accept" | "decline" | "cancel";
  /** Field values, keyed by property name. Present only when `action === "accept"`. */
  content?: Record<string, string | number | boolean | string[]>;
}

/**
 * Called by the SDK when a remote tool (MCP or HTTP connector) requests user
 * input mid-execution. The host application must present the form/picker to
 * the user and resolve with an `ElicitationResult`.
 *
 * Return `{ action: "cancel" }` if no UI is available.
 */
export type ElicitationCallback = (
  toolName: string,
  request: ElicitationRequest,
) => Promise<ElicitationResult>;

export type SummarizeFn = (messages: ChatMessage[]) => Promise<string>;

export interface OpenAITextContentPart {
  type: "text";
  text: string;
}

export interface OpenAIImageUrlContentPart {
  type: "image_url";
  image_url: {
    url: string;
    detail?: "auto" | "low" | "high" | string;
  };
}

export interface OpenAIFileContentPart {
  type: "file";
  file: {
    file_id?: string;
    file_data?: string;
    filename?: string;
  };
}

export type OpenAIInputContentPart =
  | OpenAITextContentPart
  | OpenAIImageUrlContentPart
  | OpenAIFileContentPart;

export type ChatMessageContent = string | OpenAIInputContentPart[] | null;

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: ChatMessageContent;
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
}

export type AgentRunInput =
  | string
  | OpenAIInputContentPart[]
  | {
      content: string | OpenAIInputContentPart[];
    };

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface AgentOptions {
  apiKey: string;
  agentblitApiKey: string;
  agentblitUrl?: string;
  /** When set, overrides the model from `GET /api/1.0/agent`. */
  model?: string;
  maxHistory?: number;
  debug?: boolean;
  timeout?: number;
  approvalCallback?: ApprovalCallback;
  /**
   * Called when a remote tool requests user input during execution (MCP or HTTP
   * connector elicitation). The host application should present the prompt to the
   * user and return an `ElicitationResult`. When absent, elicitation requests are
   * automatically cancelled so the connector can handle the absent-user case.
   */
  elicitationCallback?: ElicitationCallback;
  customTools?: ToolHandler[];
  maxToolRounds?: number;
}

export interface ToolOptions {
  name?: string;
  description?: string;
  permissionMode?: ToolPermissionMode;
  inputSchema?: JSONSchema;
}
