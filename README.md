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
| `maxHistory` | `number` | No | `10` | Integer `>= 1` | Keep recent messages before summarization |
| `maxToolRounds` | `number` | No | `25` | Integer `>= 1` | Limit LLM-tool loop iterations |
| `debug` | `boolean` | No | `false` | `true` or `false` | Enable verbose SDK logs |
| `model` | `string` | No | AgentBlit agent model | `vendor/model`, e.g. `openai/gpt-4o-mini` | Override the model loaded from AgentBlit |
| `sessionId` | `string` | No | random UUID | Non-empty string | Reuse a session on tool calls and analytics events |
| `timeout` | `number` (seconds) | No | `30` | Positive number | Control HTTP timeout for LLM + AgentBlit |
| `approvalCallback` | `async (toolName, args) => boolean` | No | terminal prompt | Async function | Approve/reject `needs_approval` tools |
| `elicitationCallback` | `async (toolName, request) => ElicitationResult` | No | auto-cancel | Async function | Answer mid-tool user-input requests from MCP or HTTP connectors |
| `customTools` | `ToolHandler[]` | No | `[]` | Local tool functions | Register app-specific local tools |

---

## 2) All SDK Features

- Vendor routing for `openai`, `anthropic`, `gemini`, and `openrouter`
- Remote AgentBlit agent config and tools (`GET /api/1.0/agent`, `POST /api/1.0/tools/call`)
- Local tools via `tool(...)` and `customTools` / `registerTool(...)`, with optional `inputSchema`
- Approval-gated tools (`needs_approval`) via callback or terminal prompt
- Mid-execution elicitation (`elicitation_required`) with retry until accept, decline, or cancel
- OpenAI-native multimodal inputs (`text`, `image_url`, `file`)
- Streaming text responses with `for await (const chunk of agent.run(...))`
- Non-streaming completion via `await agent.runSync(...)`
- Multi-round tool calling in one `run()`
- Memory summarization using `maxHistory`
- Default system guidance to use tools and available long-term memory tools (`retrieveRelevantMemory`, `updateMemory`) when needed
- Automatic analytics batching (`agent_init`, `user_prompt`, `llm_call`, `tool_call`, `elicitation_request`, `elicitation_response`, `tools_updated`, `agent_loop_error`)
- Custom analytics via `agent.track(eventType, properties)`
- Optional `sessionId` sent with `/api/1.0/tools/call` and analytics events

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

### Example 3: Elicitation and `runSync`

Remote MCP or HTTP tools can pause and ask the host for user input. Return
`{ action: "cancel" }` if no UI is available. When `elicitationCallback` is
omitted, the SDK cancels those requests automatically.

```ts
const agent = new Agent({
  apiKey: process.env.LLM_API_KEY ?? "",
  agentblitApiKey: process.env.AGENTBLIT_API_KEY ?? "",
  sessionId: "user-42",
  elicitationCallback: async (toolName, request) => {
    console.log(toolName, request.message);
    return { action: "accept", content: { city: "Berlin" } };
  },
});

const text = await agent.runSync("Look up the weather using tools.");
console.log(text);
```

## Repo Examples

- [`examples/basic-agent.ts`](examples/basic-agent.ts) - interactive terminal chat with remote tools.
- [`examples/custom-tools.ts`](examples/custom-tools.ts) - local TypeScript tools + AgentBlit tools.
- [`examples/multimodal-input.ts`](examples/multimodal-input.ts) - OpenAI-style multimodal input with text stream output.
- [`examples/elicitation.ts`](examples/elicitation.ts) - mid-tool elicitation callback and `runSync`.
