import type { ChatMessage, SummarizeFn } from "./types.js";

const SUMMARY_PREFIX = "Summary of earlier conversation:\n";
/** Soft cap on tool output size kept in older history before summarizing. */
const TOOL_OUTPUT_MAX_CHARS = 2_000;
/** Keep about this fraction of the budget as recent verbatim messages. */
const KEEP_RATIO = 0.3;

export type ChatMemoryOptions = {
  /**
   * Approximate token budget for conversation memory (not including the
   * system prompt). When exceeded, older turns are summarized in place.
   * Default: 100_000.
   */
  contextBudgetTokens?: number;
  summarizeFn?: SummarizeFn;
};

export class ChatMemory {
  private readonly messagesStore: ChatMessage[] = [];
  private readonly contextBudgetTokens: number;
  private readonly summarizeFn?: SummarizeFn;

  constructor(options?: ChatMemoryOptions) {
    const budget = options?.contextBudgetTokens ?? 100_000;
    if (budget < 1_000) {
      throw new Error("contextBudgetTokens must be at least 1000");
    }
    this.contextBudgetTokens = budget;
    this.summarizeFn = options?.summarizeFn;
  }

  get messages(): ChatMessage[] {
    return [...this.messagesStore];
  }

  append(message: ChatMessage): void {
    this.messagesStore.push(message);
  }

  extend(messages: ChatMessage[]): void {
    this.messagesStore.push(...messages);
  }

  /** Replace the entire message store (e.g. restore a persisted thread). */
  replace(messages: ChatMessage[]): void {
    this.messagesStore.length = 0;
    this.messagesStore.push(...repairToolPairing(messages));
  }

  clear(): void {
    this.messagesStore.length = 0;
  }

  /** Drop prior computer-screenshot follow-ups (not user-uploaded images). */
  dropScreenshotFollowUps(): void {
    for (let i = this.messagesStore.length - 1; i >= 0; i -= 1) {
      const message = this.messagesStore[i];
      if (!message || message.role !== "user" || !Array.isArray(message.content)) {
        continue;
      }
      const text = message.content.find((part) => part.type === "text");
      const hasImage = message.content.some((part) => part.type === "image_url");
      if (
        !hasImage ||
        !text ||
        text.type !== "text" ||
        !/Screenshot \d+x\d+/.test(text.text)
      ) {
        continue;
      }
      this.messagesStore.splice(i, 1);
    }
  }

  /**
   * Compact in place if over budget, then return system prompt + memory.
   * Compaction persists in the store so hosts that save memory keep the summary.
   */
  async buildMessagesForLLM(systemPrompt: string): Promise<ChatMessage[]> {
    await this.compactIfNeeded();
    return [{ role: "system", content: systemPrompt }, ...this.messagesStore];
  }

  private async compactIfNeeded(): Promise<void> {
    this.messagesStore.splice(
      0,
      this.messagesStore.length,
      ...repairToolPairing(this.messagesStore),
    );

    if (estimateTokens(this.messagesStore) <= this.contextBudgetTokens) {
      return;
    }

    const keepTokens = Math.max(
      1,
      Math.floor(this.contextBudgetTokens * KEEP_RATIO),
    );
    let keepStart = snapToSafeCut(
      this.messagesStore,
      findKeepStart(this.messagesStore, keepTokens),
    );

    // If the recent window alone fills the budget, keep only the last turn.
    if (keepStart <= 0 && this.messagesStore.length > 1) {
      keepStart = snapToSafeCut(
        this.messagesStore,
        this.messagesStore.length - 1,
      );
    }
    if (keepStart <= 0) {
      this.messagesStore.splice(
        0,
        this.messagesStore.length,
        ...repairToolPairing(shrinkToolOutputs(this.messagesStore)),
      );
      return;
    }

    const { priorSummary, rest } = splitPriorSummary(
      this.messagesStore.slice(0, keepStart),
    );
    const recent = this.messagesStore.slice(keepStart);
    const toSummarize = shrinkToolOutputs(rest);

    let summaryText: string;
    if (!this.summarizeFn) {
      summaryText = `[Earlier conversation truncated: ${toSummarize.length} message(s).]`;
    } else {
      const withPrior = priorSummary
        ? [
            {
              role: "system" as const,
              content: `${SUMMARY_PREFIX}${priorSummary}`,
            },
            ...toSummarize,
          ]
        : toSummarize;
      summaryText = await this.summarizeFn(withPrior);
    }

    this.messagesStore.splice(
      0,
      this.messagesStore.length,
      {
        role: "system",
        content: `${SUMMARY_PREFIX}${summaryText}`,
      },
      ...recent,
    );
    this.messagesStore.splice(
      0,
      this.messagesStore.length,
      ...repairToolPairing(this.messagesStore),
    );
  }
}

function estimateTokens(messages: ChatMessage[]): number {
  try {
    return Math.ceil(JSON.stringify(messages).length / 4);
  } catch {
    return messages.length * 500;
  }
}

function isToolResult(message: ChatMessage): boolean {
  return message.role === "tool";
}

function toolCallIds(message: ChatMessage): string[] {
  if (message.role !== "assistant" || !Array.isArray(message.tool_calls)) {
    return [];
  }
  return message.tool_calls
    .map((call) => call.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
}

/**
 * Walk from the end until the recent window is about `keepTokens`, then return
 * the start index of that window.
 */
function findKeepStart(messages: ChatMessage[], keepTokens: number): number {
  let used = 0;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    used += estimateTokens([messages[i]!]);
    if (used >= keepTokens) return i;
  }
  return 0;
}

/**
 * A cut must not start on a `tool` result (orphan), and must not leave an
 * assistant `tool_calls` message without its results in the kept window.
 */
function snapToSafeCut(messages: ChatMessage[], start: number): number {
  let i = Math.max(0, Math.min(start, messages.length));
  while (i < messages.length && isToolResult(messages[i]!)) {
    i += 1;
  }
  // If the message before `i` is an assistant with tool_calls, include it so
  // its results (which follow) stay paired.
  while (i > 0 && toolCallIds(messages[i - 1]!).length > 0) {
    i -= 1;
  }
  return i;
}

function splitPriorSummary(messages: ChatMessage[]): {
  priorSummary?: string;
  rest: ChatMessage[];
} {
  const first = messages[0];
  if (
    first?.role === "system" &&
    typeof first.content === "string" &&
    first.content.startsWith(SUMMARY_PREFIX)
  ) {
    return {
      priorSummary: first.content.slice(SUMMARY_PREFIX.length),
      rest: messages.slice(1),
    };
  }
  return { rest: messages };
}

function shrinkToolOutputs(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((message) => {
    if (message.role !== "tool" || typeof message.content !== "string") {
      return message;
    }
    if (message.content.length <= TOOL_OUTPUT_MAX_CHARS) return message;
    return {
      ...message,
      content: `[tool output omitted, ${message.content.length} chars]`,
    };
  });
}

/**
 * Drop orphan tool results and stale unfinished tool_calls so the provider
 * never sees a broken tool pair (the HTTP 400 case).
 *
 * Trailing incomplete assistant tool_calls are kept: interrupt/resume leaves
 * pending calls (and any same-batch results already written) at the end of
 * memory, and {@link ChatMemory.replace} must restore that state intact.
 */
function repairToolPairing(messages: ChatMessage[]): ChatMessage[] {
  let lastTurnIndex = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const role = messages[i]?.role;
    if (role === "user" || role === "assistant") {
      lastTurnIndex = i;
      break;
    }
  }

  const declared = new Set<string>();
  for (const message of messages) {
    for (const id of toolCallIds(message)) declared.add(id);
  }

  const answered = new Set<string>();
  for (const message of messages) {
    if (message.role === "tool" && typeof message.tool_call_id === "string") {
      answered.add(message.tool_call_id);
    }
  }

  const out: ChatMessage[] = [];
  for (let i = 0; i < messages.length; i += 1) {
    const message = messages[i];
    if (!message) continue;

    if (isToolResult(message)) {
      const id = message.tool_call_id;
      if (typeof id !== "string" || !declared.has(id)) continue;
      out.push(message);
      continue;
    }

    const ids = toolCallIds(message);
    if (ids.length === 0) {
      out.push(message);
      continue;
    }

    // Fully answered tool round.
    if (ids.every((id) => answered.has(id))) {
      out.push(message);
      continue;
    }

    // Pending interrupt / partial batch at the end of memory — keep it so
    // resume() and getPendingToolCalls() still see the open tool_calls.
    if (i === lastTurnIndex) {
      out.push(message);
      continue;
    }

    // Stale incomplete round earlier in history (host forgot to close
    // dangling calls before a new user turn): drop it; orphan results for
    // those ids are removed in the filter below.
  }

  // Recompute declared from kept assistants and drop any leftover orphans.
  const keptDeclared = new Set<string>();
  for (const message of out) {
    for (const id of toolCallIds(message)) keptDeclared.add(id);
  }
  return out.filter((message) => {
    if (!isToolResult(message)) return true;
    return (
      typeof message.tool_call_id === "string" &&
      keptDeclared.has(message.tool_call_id)
    );
  });
}
