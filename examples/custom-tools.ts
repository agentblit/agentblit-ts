/**
 * Example: register local TypeScript functions as tools alongside AgentBlit tools.
 *
 * Environment variables: same as basic-agent.ts (LLM_API_KEY, AGENTBLIT_*, etc.).
 */

import { Agent, tool } from "../src/index.js";

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    console.error(`Missing environment variable ${name}.\n  export ${name}=your-key-here`);
    process.exit(1);
  }
  return value;
}

const greet = tool({
  description: "Return a friendly greeting for a name.",
})((name: string) => `Hello, ${name}!`);

const doubleValue = tool({
  name: "double",
  description: "Double an integer.",
  permissionMode: "always_allow",
})((x: number) => x * 2);

async function main(): Promise<void> {
  const llmKey = requireEnv("LLM_API_KEY");
  const agentblitKey = requireEnv("AGENTBLIT_API_KEY");

  const agent = new Agent({
    apiKey: llmKey,
    agentblitUrl: process.env.AGENTBLIT_URL?.trim() || "https://console.agentblit.com",
    agentblitApiKey: agentblitKey,
    maxHistory: 5,
    debug: true,
    customTools: [greet, doubleValue],
  });

  // Or register after construction:
  // agent.registerTool(greet);

  const prompt =
    process.env.AGENTBLIT_PROMPT?.trim() ||
    "Use the greet tool for the name Ada, then double 21 with double.";

  for await (const chunk of agent.run(prompt)) {
    process.stdout.write(chunk);
  }
  process.stdout.write("\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
