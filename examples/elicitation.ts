/**
 * Example: answer mid-tool elicitation requests, then wait for the full reply.
 *
 * Environment variables: same as basic-agent.ts (LLM_API_KEY, AGENTBLIT_*, etc.).
 */

import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { Agent, type ElicitationRequest, type ElicitationResult } from "../src/index.js";

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    console.error(`Missing environment variable ${name}.\n  export ${name}=your-key-here`);
    process.exit(1);
  }
  return value;
}

async function elicitationCallback(
  toolName: string,
  request: ElicitationRequest,
): Promise<ElicitationResult> {
  const rl = readline.createInterface({ input, output });
  try {
    console.log(`\n[${toolName}] ${request.message}`);
    const schema = request.requestedSchema;
    if (!schema) {
      const answer = (await rl.question("Accept? [y/N]: ")).trim().toLowerCase();
      return answer === "y" || answer === "yes"
        ? { action: "accept", content: {} }
        : { action: "decline" };
    }
    const content: Record<string, string> = {};
    for (const [fieldName, field] of Object.entries(schema.properties)) {
      const label = field.title ?? fieldName;
      content[fieldName] = (await rl.question(`${label}: `)).trim();
    }
    return { action: "accept", content };
  } finally {
    rl.close();
  }
}

async function main(): Promise<void> {
  const agent = new Agent({
    apiKey: requireEnv("LLM_API_KEY"),
    agentblitApiKey: requireEnv("AGENTBLIT_API_KEY"),
    agentblitUrl: process.env.AGENTBLIT_URL?.trim() || "https://console.agentblit.com",
    sessionId: process.env.AGENTBLIT_SESSION_ID?.trim() || undefined,
    elicitationCallback,
    debug: true,
  });
  const prompt =
    process.env.AGENTBLIT_PROMPT?.trim() || "Use tools if you need extra input from me.";
  const text = await agent.runSync(prompt);
  process.stdout.write(`${text}\n`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
