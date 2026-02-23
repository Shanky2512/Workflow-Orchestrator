# Langfuse vs OpenTelemetry — Why Both?

**OpenTelemetry (OTel) = The transport layer.** It's the industry-standard protocol for collecting traces, spans, and metrics. It handles how telemetry data flows between services and across threads.

**Langfuse = The AI-specific intelligence layer.** It sits on top of OTel and adds what OTel can't do natively:

```
┌───────────────────────────────────┬────────────┬────────────────────┐
│            Capability             │ OTel alone │ Langfuse (on OTel) │
├───────────────────────────────────┼────────────┼────────────────────┤
│ Distributed tracing               │ Yes        │ Yes (inherits)     │
├───────────────────────────────────┼────────────┼────────────────────┤
│ HTTP/thread span collection       │ Yes        │ Yes (inherits)     │
├───────────────────────────────────┼────────────┼────────────────────┤
│ LLM token counting                │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ LLM cost tracking                 │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ Prompt/completion capture         │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ Prompt injection scoring          │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ Fine-tuning dataset export        │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ LangChain/CrewAI callbacks        │ No         │ Yes                │
├───────────────────────────────────┼────────────┼────────────────────┤
│ Visual trace replay (time-travel) │ No         │ Yes                │
└───────────────────────────────────┴────────────┴────────────────────┘
```

**In one line:**
- OTel = plumbing (moves telemetry data, propagates context across threads)
- Langfuse = brain (understands LLM calls, costs, prompts, and gives you a UI to debug agent runs)

Langfuse SDK v3 is built on OTel internally — initializing Langfuse auto-creates the OTel TracerProvider. So they're not two separate systems; Langfuse is the AI-aware layer that rides on OTel's infrastructure.

❯ and what about for our codebase

## In EchoAI's Codebase Specifically

**The problem:** EchoAI has a double thread-hop execution path:

FastAPI (async) → asyncio.to_thread() → LangGraph.invoke() → CrewAI crew.kickoff() → ThreadPoolExecutor → Tool execution

### OTel solves EchoAI's threading problem:

```
┌───────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│   OTel Instrumentor   │                                             What it does in EchoAI                                             │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ThreadingInstrumentor │ Carries trace context across the 2 thread hops (FastAPI→LangGraph→CrewAI) — without this, spans become orphans │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FastAPIInstrumentor   │ Auto-traces every /api/workflow/execute HTTP request as the root span                                          │
├───────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ CrewAIInstrumentor    │ Auto-traces internal CrewAI mechanics (agent delegation, tool selection, LLM calls inside crews)               │
└───────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Langfuse solves EchoAI's AI debugging problem:

```
    1. CallbackHandler on compiled_graph.invoke()
        Captures every LangGraph node
        Captures every LLM call
        Captures every tool call
        Works automatically

    2. @observe on CrewAI adapter
        Traces the handoff between LangGraph nodes → CrewAI crews
        Covers hierarchical, parallel, and sequential execution

    3. session_id=run_id
        Groups all spans of one workflow run together
        Allows full workflow replay step-by-step

    4. audit_log_input()
        Detects prompt injection
        Runs before the graph executes

    5. save_completion_for_finetuning()
        Saves agent outputs
        Stores them in Langfuse Datasets
        Enables future fine-tuning of your own models

    6. Cost tracking
        Tracks token usage per workflow run
        Calculates cost per run
```

**In one line for EchoAI:**
- OTel = glues the trace together across FastAPI → LangGraph → CrewAI → Tools (thread-safe context propagation)
- Langfuse = shows you what each agent did, what it cost, whether the prompt was safe, and lets you replay/export it
