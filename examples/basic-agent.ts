/**
 * Interactive AgentBlit agent example (remote tools + optional local tool).
 *
 * Prerequisites:
 *   - Node.js 18+
 *   - Environment variables (see below)
 *
 * Run from repo root:
 *   npm install
 *   npm run example
 *
 * Or:
 *   npx tsx examples/basic-agent.ts
 *
 * Environment:
 *   LLM_API_KEY        — required (LLM provider API key for the selected model vendor)
 *   AGENTBLIT_API_KEY  — required (X-API-Key for AgentBlit)
 *   AGENTBLIT_URL      — optional (default https://console.agentblit.com)
 */

import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { Agent, tool } from "../src/index.js";

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

async function main(): Promise<void> {
  const apiKey = process.env.LLM_API_KEY?.trim();
  const agentblitApiKey = process.env.AGENTBLIT_API_KEY?.trim();

  if (!apiKey) {
    console.error("Set LLM_API_KEY to your LLM provider API key.");
    process.exit(1);
  }
  if (!agentblitApiKey) {
    console.error("Set AGENTBLIT_API_KEY to your AgentBlit API key.");
    process.exit(1);
  }

  const agent = new Agent({
    apiKey,
    agentblitApiKey,
    maxHistory: 5,
    debug: process.env.DEBUG === "1" || process.env.DEBUG === "true",
    timeout: Number(process.env.TIMEOUT_SECONDS ?? "30") || 30,
    customTools: [add],
  });

  console.log("session_id:", agent.sessionId);
  console.log("Type a message and press Enter. Commands: exit, quit. Ctrl+D to leave.");
  console.log("---");

  const rl = readline.createInterface({ input, output });

  try {
    for (; ;) {
      let line: string;
      try {
        line = await rl.question("\nYou: ");
      } catch {
        break;
      }

      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      const cmd = trimmed.toLowerCase();
      if (cmd === "exit" || cmd === "quit") {
        break;
      }

      process.stdout.write("Assistant: ");
      try {
        for await (const chunk of agent.run(trimmed)) {
          process.stdout.write(chunk);
        }
      } catch (err) {
        console.error("\n", err);
      }
      process.stdout.write("\n");
    }
  } finally {
    rl.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
