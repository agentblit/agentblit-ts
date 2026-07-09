# AgentBlit TypeScript SDK

Build LLM agents with:
- AgentBlit remote tools
- optional local TypeScript tools
- streaming output
- approval-gated actions
- short-term memory with automatic summarization

## Install

```bash
npm install agentblit
```

## Quick Start

```ts
import { Agent, tool } from "agentblit";

const add = tool({
  name: "local_add",
  description: "Add two integers.",
  inputSchema: {
    type: "object",
    properties: {
      a: { type: "integer" },
      b: { type: "integer" },
    },
    required: ["a", "b"],
  },
})((a: number, b: number) => a + b);

const agent = new Agent({
  apiKey: process.env.LLM_API_KEY ?? "",
  agentblitApiKey: process.env.AGENTBLIT_API_KEY ?? "",
  customTools: [add],
});

for await (const chunk of agent.run("What is 200 + 30? Use tools if needed.")) {
  process.stdout.write(chunk);
}
process.stdout.write("\n");
```

## Multimodal Input (OpenAI Content Parts)

`agent.run(...)` accepts:
- a plain string (existing behavior), or
- OpenAI-style content parts (`text`, `image_url`, `file`), or
- an object with a `content` field containing either of the above.

```ts
const input = [
  { type: "text", text: "What do you see in this image?" },
  {
    type: "image_url",
    image_url: { url: "https://example.com/sample.png", detail: "auto" },
  },
];

for await (const chunk of agent.run(input)) {
  // Output stream stays text chunks.
  process.stdout.write(chunk);
}
```

---

## 1) Agent Initialization Configs

`new Agent({...})` accepts:

| Key | Type | Required | Default | Values / Format | Use Case |
|---|---|---|---|---|---|
| `apiKey` | `string` | Yes | - | LLM provider API key | Send model requests |
| `agentblitApiKey` | `string` | Yes | - | AgentBlit API key (`X-API-Key`) | Load agent config + remote tools + analytics events |
| `agentblitUrl` | `string` | No | `https://console.agentblit.com` | Valid base URL | Self-hosted or staging AgentBlit |
| `maxHistory` | `number` | No | `5` | Integer `>= 1` | Keep recent messages before summarization |
| `maxToolRounds` | `number` | No | `25` | Integer `>= 1` | Limit LLM-tool loop iterations |
| `debug` | `boolean` | No | `false` | `true` or `false` | Enable verbose SDK logs |
| `timeout` | `number` (seconds) | No | `30` | Positive number | Control HTTP timeout for LLM + AgentBlit |
| `approvalCallback` | `async (toolName, args) => boolean` | No | - | Async function | Approve/reject `needs_approval` tools |
| `customTools` | `ToolHandler[]` | No | `[]` | Local tool functions | Register app-specific local tools |

---

## 2) All SDK Features

- Vendor routing for `openai`, `anthropic`, `gemini`, and `openrouter`
- Remote AgentBlit agent config and tools (`GET /api/1.0/agent`, `POST /api/1.0/tools/call`)
- Local tools via `tool(...)` and `customTools` / `registerTool(...)`
- Approval-gated tools (`needs_approval`) via callback or terminal prompt
- OpenAI-native multimodal inputs (`text`, `image_url`, `file`)
- Streaming text responses with `for await (const chunk of agent.run(...))`
- Multi-round tool calling in one `run()`
- Memory summarization using `maxHistory`
- Default system guidance to use tools and available long-term memory tools (`retrieveRelevantMemory`, `updateMemory`) when needed
- Automatic analytics batching (`agent_init`, `user_prompt`, `llm_call`, `tool_call`, `tools_updated`, `agent_loop_error`)
- Custom analytics via `agent.track(eventType, properties)`

---

## 3) Examples

### Example 1: Basic Streaming Agent

```ts
const agent = new Agent({
  apiKey: process.env.LLM_API_KEY ?? "",
  agentblitApiKey: process.env.AGENTBLIT_API_KEY ?? "",
});

for await (const chunk of agent.run("Summarize what AgentBlit does in 2 lines.")) {
  process.stdout.write(chunk);
}
```

Model, system prompt, agent id, and remote tools are loaded from AgentBlit on the first `run()`.

### Example 2: Approval-Gated Tool

```ts
const agent = new Agent({
  apiKey: process.env.LLM_API_KEY ?? "",
  agentblitApiKey: process.env.AGENTBLIT_API_KEY ?? "",
  approvalCallback: async (toolName, args) => {
    console.log("Approve tool call?", toolName, args);
    return toolName !== "delete_production_data";
  },
});
```

## Repo Examples

- [`examples/basic-agent.ts`](examples/basic-agent.ts) - interactive terminal chat with remote tools.
- [`examples/custom-tools.ts`](examples/custom-tools.ts) - local TypeScript tools + AgentBlit tools.
- [`examples/multimodal-input.ts`](examples/multimodal-input.ts) - OpenAI-style multimodal input with text stream output.
