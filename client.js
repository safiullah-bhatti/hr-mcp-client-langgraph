// hr-mcp-client-langchain/client.js — ONE client, both providers,
// now driven by a LangGraph graph instead of a hand-rolled while-loop.
//
// What changed vs. the previous version, and why:
//   - The MCP connection, tool discovery, and the OpenAI-shape tool
//     wrapping are UNCHANGED — LangGraph doesn't touch any of that,
//     it's still the same `bindTools()` call from before.
//   - The `while (response.tool_calls.length > 0) { ... }` loop is
//     GONE. In its place: build the graph from graph.js, then a
//     single `graph.invoke(...)` call. The loop still effectively
//     happens — it's just expressed as edges in graph.js now instead
//     of a JS while-loop, which is what makes the corrective-RAG
//     retry cycle (see graph.js) practical to add without turning
//     this file into a mess of counters and branches.
//
// Run (identical to before):
//   node client.js "How many leave days do I have left?"
//   MODEL_PROVIDER=gemini node client.js "..."   (defaults to claude)
//
// Requires (in THIS project's own .env):
//   ANTHROPIC_API_KEY  (if using claude)
//   GEMINI_API_KEY     (if using gemini)
//   MCP_SERVER_URL     (defaults to http://localhost:3000/mcp)

import "dotenv/config";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage } from "@langchain/core/messages";
import { buildGraph } from "./graph.js";
import { ChatOllama } from "@langchain/ollama";
import Setting from "./settings.js";

const MCP_SERVER_URL = Setting.MCP.SERVER;
const PROVIDER = Setting.MODEL.PROVIDER.toLowerCase();

// The one place that knows the two providers are different at all —
// unchanged from before.
function getChatModel() {
  if (PROVIDER === "gemini") {
    console.log("[CLIENT] Using provider: gemini");
    return new ChatGoogleGenerativeAI({
      apiKey: Setting.GEMINI.API_KEY,
      model: Setting.GEMINI.MODEL_NAME,
    });
  }
  if (PROVIDER === "ollama") {
    //const model = process.env.OLLAMA_MODEL || "llama3.1:8b";
    const model = Setting.OLLAMA.MODEL_NAME;
    const baseUrl = Setting.OLLAMA.BASE_URL;
    console.log(`[CLIENT] Using provider: ollama (model=${model}, baseUrl=${baseUrl})`);
    return new ChatOllama({
      model,
      baseUrl,
    });
  }
  console.log("[CLIENT] Using provider: claude");
  return new ChatAnthropic({
    apiKey: Setting.ANTHROPIC.API_KEY,
    model: Setting.ANTHROPIC.MODEL_NAME,
    maxTokens: 300,
  });
}

async function main() {
  const userMessage = process.argv.slice(2).join(" ");
  if (!userMessage) {
    console.log('Usage: node client.js "your question here"');
    console.log(`       MODEL_PROVIDER=${Setting.MODEL.PROVIDER} node client.js "your question here"`);
    return;
  }

  console.log(`\n[CLIENT] User message: "${userMessage}"`);

  console.log(`[CLIENT] Connecting to MCP server at ${MCP_SERVER_URL} ...`);
  const transport = new StreamableHTTPClientTransport(new URL(MCP_SERVER_URL));
  const mcpClient = new Client({ name: "hr-chatbot-client-langgraph", version: "1.0.0" });
  await mcpClient.connect(transport);
  console.log("[CLIENT] Connected to MCP server");

  const { tools: mcpTools } = await mcpClient.listTools();
  console.log(
    `[CLIENT] Discovered ${mcpTools.length} tools: ${mcpTools.map((t) => t.name).join(", ")}`
  );

  // Same OpenAI-shape wrapping as before — bindTools() accepts this
  // as-is for both ChatAnthropic and ChatGoogleGenerativeAI. See the
  // previous version's comment here for why this exact shape was
  // chosen; nothing about it changes for LangGraph.
  const langchainTools = mcpTools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
    },
  }));

  // rawModel: used by graph.js's rewriteQuery node for a plain,
  // tool-free reformulation call.
  // boundModel: the same model with tools bound, used by the agent
  // node — exactly what `model` was in the old while-loop version.
  const rawModel = getChatModel();
  const boundModel = rawModel.bindTools(langchainTools);

  const graph = buildGraph({ boundModel, rawModel, mcpClient });

  console.log("[CLIENT] Running graph...");
  const finalState = await graph.invoke(
    { messages: [new HumanMessage(userMessage)] },
    // Belt-and-suspenders cap on total graph steps — the
    // rewriteQuery/tools/agent cycle is already bounded by
    // MAX_RETRIES_PER_CALL in graph.js, but this guards against any
    // other runaway loop (e.g. the model repeatedly re-requesting
    // tools) the way you'd want a max-iterations guard on any agent
    // loop, hand-rolled or not.
    { recursionLimit: 25 }
  );

  const finalMessage = finalState.messages[finalState.messages.length - 1];
  console.log(`\n[CLIENT] Final answer to user:\n${finalMessage.content}\n`);

  await mcpClient.close();
}

main().catch((err) => {
  console.error("[CLIENT] Error:", err);
  process.exit(1);
});
