// hr-mcp-client-langchain/graph.js
//
// ---- Why this file exists (read this before the code) ----
//
// The old client.js had a `while (response.tool_calls.length > 0)` loop:
// call the model, if it asked for tools run them, feed results back,
// repeat. That loop ALREADY supports everything in your first two
// bullet points:
//   - multiple tools registered (going 6 -> 7 tools, or 7 -> 20, is
//     just a bigger `mcpTools` array — no loop change needed)
//   - "tool call -> llm -> decides to call ANOTHER tool" — that's
//     just another lap of the same while loop, already works today
//
// So "more docs in more vector DBs" and "the LLM chaining several
// tool calls in a row" are NOT reasons to reach for LangGraph. A
// flat while-loop (which is exactly what LangGraph's own prebuilt
// `createReactAgent` compiles down to internally) already handles
// both.
//
// What a flat while-loop genuinely can't express cleanly is a
// CONTROL decision that ISN'T "did the LLM decide to call a tool" —
// it's "did the SYSTEM decide the tool result was good enough to
// use". That's the one real trigger for LangGraph in this codebase:
// corrective retrieval. server.js's RAG tools now return a
// `topScore` alongside their matches (see server.js). This graph
// adds a node that:
//
//   agent -> [wants a RAG tool] -> tools -> topScore too low? ->
//   rewriteQuery (LLM reformulates the query, tool is re-called
//   with it, capped at MAX_RETRIES) -> back into tools -> agent
//
// That's a genuine CYCLE with its own piece of state (a retry count
// per tool call, not just "the message list so far") and a
// conditional edge that depends on data the LLM never sees directly
// (topScore). Modelling that as a bare while-loop means hand-rolling
// exactly what LangGraph already gives you: named nodes, explicit
// edges, a typed state object, and a `recursionLimit` safety cap
// instead of an ad hoc counter. That's the whole case for it here —
// not "more tools", not "more DBs", specifically this.
//
// Everything below is intentionally still small: 3 nodes, one cycle.
// If you later add a 4th domain, a supervisor node that routes to a
// domain-specific sub-agent, or persistence/human-approval before a
// tool with side effects runs, those are all additional nodes/edges
// on this SAME graph — the shape doesn't change, it just grows.

import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";

// Tune to your embedding model/data — cosine similarity below this is
// treated as "weak enough to try reformulating the query once".
const RELEVANCE_THRESHOLD = 0.55;
const MAX_RETRIES_PER_CALL = 1;

// Only the RAG tools carry a topScore worth checking. The 4 "live
// data" tools (get_leave_balance, etc.) always pass straight through.
const RAG_TOOL_NAMES = new Set([
  "search_hr_policy",
  "search_engineering_practices",
  "search_admin_docs",
]);

// ---- Graph state ----
// `messages` is the same running transcript the old while-loop kept
// in a plain array — same idea, just with LangGraph's built-in
// "append, don't replace" reducer.
// `retryCounts` is the NEW piece: a { [tool_call_id]: number } map
// that a flat while-loop had nowhere natural to live. It's keyed
// per-call so two different RAG calls in the same turn are retried
// independently.
const GraphState = Annotation.Root({
  messages: Annotation({
    reducer: (current, update) => current.concat(update),
    default: () => [],
  }),
  retryCounts: Annotation({
    reducer: (current, update) => ({ ...current, ...update }),
    default: () => ({}),
  }),
  // Transient: calls waiting on a query-rewrite retry. Fully replaced
  // each time a node touches it (not accumulated like messages).
  pendingRewrites: Annotation({
    reducer: (_current, update) => update,
    default: () => [],
  }),
});

// Parses a RAG tool's JSON envelope back into { topScore, text }.
// Non-RAG tools (or a malformed RAG response) fall back to treating
// the raw text as-is with a topScore high enough to never retry.
function parseToolResult(toolName, rawText) {
  if (!RAG_TOOL_NAMES.has(toolName)) {
    return { topScore: 1, text: rawText };
  }
  try {
    const parsed = JSON.parse(rawText);
    const text = parsed.matches
      .map((m) => `[Source: ${m.source}]\n${m.text}`)
      .join("\n\n---\n\n");
    return { topScore: parsed.topScore ?? 0, text };
  } catch {
    // Valid, non-error response that still wasn't parseable JSON —
    // shouldn't happen given server.js's contract, but don't crash
    // the graph over it. Loud on purpose: this means server.js and
    // graph.js have drifted out of sync somehow, and staying silent
    // here is exactly what hid the bug you just ran into.
    console.log(
      `[CLIENT] [graph:tools] WARNING: "${toolName}" returned non-JSON, non-error text — treating as pass-through: ${rawText}`
    );
    return { topScore: 1, text: rawText };
  }
}

export function buildGraph({ boundModel, rawModel, mcpClient }) {
  // ---- Node: agent ----
  // Identical job to the old `response = await model.invoke(messages)`
  // call — the model sees the full transcript and either answers or
  // requests tool calls.
  async function agentNode(state) {
    console.log("[CLIENT] [graph:agent] invoking model...");
    const response = await boundModel.invoke(state.messages);
    console.log(
      `[CLIENT] [graph:agent] model requested ${response.tool_calls?.length || 0} tool call(s)`
    );
    return { messages: [response] };
  }

  // ---- Node: tools ----
  // Runs every requested tool call (in parallel, unlike the old
  // sequential for-loop — there's no reason two independent MCP
  // calls need to be awaited one at a time). RAG calls whose
  // topScore is below threshold AND still have retries left are
  // held back from `messages` and queued into `pendingRewrites`
  // instead of being answered immediately.
  async function toolsNode(state) {
    const lastMessage = state.messages[state.messages.length - 1];
    const calls = lastMessage.tool_calls || [];

    const settledMessages = [];
    const pendingRewrites = [];

    await Promise.all(
      calls.map(async (call) => {
        console.log(
          `[CLIENT] [graph:tools] calling "${call.name}" with args ${JSON.stringify(call.args)}`
        );
        const result = await mcpClient.callTool({ name: call.name, arguments: call.args });
        const rawText = result.content.map((c) => c.text).join(" ");

        if (result.isError) {
          // The MCP tool handler threw (e.g. an embeddings-API call
          // failed). This is NOT "weak retrieval" — don't run it
          // through parseToolResult/the retry logic at all, just
          // surface it plainly so it's visible instead of silently
          // misread as a fine, never-retry result.
          console.log(`[CLIENT] [graph:tools] "${call.name}" call FAILED: ${rawText}`);
          settledMessages.push(
            new ToolMessage({
              content: `Tool error (not a retrieval-quality issue): ${rawText}`,
              tool_call_id: call.id,
              name: call.name,
            })
          );
          return;
        }

        const { topScore, text } = parseToolResult(call.name, rawText);
        console.log(`[CLIENT] [graph:tools] "${call.name}" topScore=${topScore}`);

        const retriesSoFar = state.retryCounts[call.id] || 0;
        const weakAndRetryable =
          RAG_TOOL_NAMES.has(call.name) &&
          topScore < RELEVANCE_THRESHOLD &&
          retriesSoFar < MAX_RETRIES_PER_CALL;

        if (weakAndRetryable) {
          console.log(
            `[CLIENT] [graph:tools] "${call.name}" result is weak (score ${topScore} < ${RELEVANCE_THRESHOLD}), queuing a query rewrite (retry ${retriesSoFar + 1}/${MAX_RETRIES_PER_CALL})`
          );
          pendingRewrites.push({ call, weakText: text, weakScore: topScore });
        } else {
          settledMessages.push(
            new ToolMessage({ content: text, tool_call_id: call.id, name: call.name })
          );
        }
      })
    );

    return { messages: settledMessages, pendingRewrites };
  }

  // ---- Node: rewriteQuery ----
  // For each call that came back weak: ask the (unbound) model to
  // reformulate just that query, re-call the SAME tool with the new
  // query, bump that call's retry count, and push the final
  // ToolMessage — whatever it ends up being — onto the transcript.
  // Capped at MAX_RETRIES_PER_CALL so this can't loop forever even if
  // every reformulation is still weak.
  async function rewriteQueryNode(state) {
    const rewrites = state.pendingRewrites || [];
    const newMessages = [];
    const retryUpdates = {};

    for (const { call, weakText, weakScore } of rewrites) {
      const originalQuery = call.args?.query ?? "";
      console.log(
        `[CLIENT] [graph:rewriteQuery] reformulating query for "${call.name}": "${originalQuery}"`
      );

      const rewriteResponse = await rawModel.invoke([
        new SystemMessage(
          "You reformulate a weak internal-document search query into a better one. " +
            "Reply with ONLY the new search query text, nothing else."
        ),
        new HumanMessage(
          `Original query: "${originalQuery}"\n` +
            `Best match found had low relevance (score ${weakScore.toFixed(3)}).\n` +
            `Give a broader or differently-worded query likely to retrieve a relevant document.`
        ),
      ]);
      const newQuery = rewriteResponse.content.toString().trim();
      console.log(`[CLIENT] [graph:rewriteQuery] retrying "${call.name}" with: "${newQuery}"`);

      const result = await mcpClient.callTool({ name: call.name, arguments: { query: newQuery } });
      const rawText = result.content.map((c) => c.text).join(" ");

      if (result.isError) {
        console.log(`[CLIENT] [graph:rewriteQuery] retry call FAILED: ${rawText}`);
        newMessages.push(
          new ToolMessage({
            content: `Tool error on retry (not a retrieval-quality issue): ${rawText}`,
            tool_call_id: call.id,
            name: call.name,
          })
        );
        retryUpdates[call.id] = (state.retryCounts[call.id] || 0) + 1;
        continue;
      }

      const { topScore, text } = parseToolResult(call.name, rawText);
      console.log(`[CLIENT] [graph:rewriteQuery] retry topScore=${topScore}`);

      // Whether or not the retry actually improved things, this is
      // the last attempt (MAX_RETRIES_PER_CALL=1) — hand the best
      // text we have back to the agent. If it's still weak, say so
      // explicitly so the model can caveat its final answer instead
      // of stating a low-confidence match as fact.
      const finalText =
        topScore < RELEVANCE_THRESHOLD
          ? `${text}\n\n[Note: retrieval confidence is still low after a reformulated query — treat this as tentative.]`
          : text;

      newMessages.push(new ToolMessage({ content: finalText, tool_call_id: call.id, name: call.name }));
      retryUpdates[call.id] = (state.retryCounts[call.id] || 0) + 1;
    }

    return { messages: newMessages, retryCounts: retryUpdates, pendingRewrites: [] };
  }

  // ---- Conditional edges ----
  function routeAfterAgent(state) {
    const last = state.messages[state.messages.length - 1];
    if (last instanceof AIMessage && last.tool_calls && last.tool_calls.length > 0) {
      return "tools";
    }
    return END;
  }

  function routeAfterTools(state) {
    return state.pendingRewrites && state.pendingRewrites.length > 0 ? "rewriteQuery" : "agent";
  }

  const graph = new StateGraph(GraphState)
    .addNode("agent", agentNode)
    .addNode("tools", toolsNode)
    .addNode("rewriteQuery", rewriteQueryNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", routeAfterAgent, { tools: "tools", [END]: END })
    .addConditionalEdges("tools", routeAfterTools, { rewriteQuery: "rewriteQuery", agent: "agent" })
    .addEdge("rewriteQuery", "agent");

  // No checkpointer wired in here — this demo is one query, one
  // process run, so there's nothing to resume. Adding
  // `.compile({ checkpointer })` with e.g. LangGraph's
  // MemorySaver/SqliteSaver is the natural next step once client.js
  // needs to remember a conversation across multiple CLI invocations,
  // or to pause before a tool with side effects (a future
  // "submit_leave_request" tool, say) for human approval.
  return graph.compile();
}
