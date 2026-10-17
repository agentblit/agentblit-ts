import { ChatMessage, SummarizeFn } from "./types.js";

export class ChatMemory {
  private readonly messagesStore: ChatMessage[] = [];
  private readonly maxHistory: number;
  private readonly summarizeFn?: SummarizeFn;

  constructor(options?: { maxHistory?: number; summarizeFn?: SummarizeFn }) {
    const maxHistory = options?.maxHistory ?? 10;
    if (maxHistory < 1) {
      throw new Error("maxHistory must be at least 1");
    }
    this.maxHistory = maxHistory;
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

  async buildMessagesForLLM(systemPrompt: string): Promise<ChatMessage[]> {
    const out: ChatMessage[] = [{ role: "system", content: systemPrompt }];
    if (this.messagesStore.length <= this.maxHistory) {
      out.push(...this.messagesStore);
      return out;
    }

    const older = this.messagesStore.slice(0, -this.maxHistory);
    const recent = this.messagesStore.slice(-this.maxHistory);
    let summaryText: string;
    if (!this.summarizeFn) {
      summaryText = `[Earlier conversation truncated: ${older.length} message(s) not shown. Provide summarizeFn on Agent to compress older context with the LLM.]`;
    } else {
      summaryText = await this.summarizeFn(older);
    }
    out.push({
      role: "system",
      content: `Summary of earlier conversation:\n${summaryText}`,
    });
    out.push(...recent);
    return out;
  }
}
