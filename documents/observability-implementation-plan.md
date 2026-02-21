# Implementation Plan: Observability for EchoAI (LangGraph + CrewAI)

## 1. Goal
Implement deep, persistent observability and tracing for the EchoAI platform, capable of visualizing:
1.  **LangGraph State Transitions**: How the workflow state evolves between nodes.
2.  **Agent-to-Agent Handoffs**: Interaction between Manager and Worker agents (CrewAI).
3.  **Hybrid Execution**: Correlating the high-level LangGraph flow with the low-level CrewAI execution steps.

**Selected Stack**: **Langfuse SDK v3** (Self-Hosted/Cloud) — built natively on **OpenTelemetry**.

## 2. Architecture Analysis

EchoAI uses a **LangGraph-First** architecture:
-   **Orchestrator**: LangGraph (`apps/workflow/runtime/executor.py`) manages the global state and node transitions.
-   **Compiler**: `apps/workflow/designer/compiler.py` builds the `StateGraph` and compiles it.
-   **Executor**: CrewAI (`apps/workflow/crewai_adapter.py`) is invoked *inside* LangGraph node functions.
-   **Entry Point**: FastAPI app at `apps/gateway/main.py`, run via `uvicorn apps.gateway.main:app`.
-   **Current Tracing**: `apps/workflow/runtime/transparency.py` provides ephemeral, in-memory step tracking for the UI via WebSockets.
-   **Existing Stubs**: `echolib/adapters.py:OTelLogger` and `apps/workflow/runtime/telemetry.py:WorkflowTelemetry` are placeholder classes (no real OTel calls).

**Strategy**:
We will **NOT** replace `transparency.py`. It is optimized for real-time UI feedback.
We will **AUGMENT** the system with Langfuse for persistent, historical debugging and engineering analytics.

### Key Architectural Insight: Langfuse SDK v3 IS OpenTelemetry
Langfuse SDK v3 (GA since June 2025) is a thin layer on top of the official OTel SDK. When initialized:
1. It auto-creates/attaches a `LangfuseSpanProcessor` to the global `TracerProvider`.
2. **Any** OTel-instrumented library (FastAPI, CrewAI, threading) automatically sends spans to Langfuse — zero additional wiring.
3. The `CallbackHandler` converts LangChain/LangGraph callback events into OTel spans internally.

This means we do NOT configure Langfuse and OTel separately — initializing Langfuse IS the OTel setup.

### Thread Boundary Challenge
EchoAI has a **double thread hop** in its execution path:
```
FastAPI async handler
  → asyncio.to_thread(executor.execute_workflow)        # Thread hop 1
    → compiled_graph.invoke()
      → CrewAI node function → crew.kickoff()
        → DynamicCrewAITool._run()
          → ThreadPoolExecutor → asyncio.run(executor.invoke())  # Thread hop 2
```
OTel context (`contextvars`) does NOT propagate across threads by default. We solve this with `ThreadingInstrumentor` (Phase 2), which patches `threading.Thread`, `Timer`, and `ThreadPoolExecutor` to carry context.

---

## 3. Implementation Steps

### Phase 1: Dependencies & Configuration

#### 1.1 Dependencies
Add to `echoAI-Production/requirements.txt`:
```text
langfuse>=3.0.0
opentelemetry-instrumentation-fastapi
opentelemetry-instrumentation-threading
openinference-instrumentation-crewai>=0.1.19
```

**Note**: `opentelemetry-api` and `opentelemetry-sdk` are already in requirements.txt. The `langfuse>=3.0.0` SDK depends on them and will use them automatically. We do NOT need `opentelemetry-instrumentation-langchain` — the Langfuse `CallbackHandler` already handles LangChain/LangGraph tracing.

#### 1.2 Configuration
**Target File**: `echolib/config.py`

Add to the existing `Settings` class:
```python
class Settings(BaseModel):
    # ... existing settings ...

    # Langfuse Observability
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = "https://cloud.langfuse.com"  # or self-hosted URL
    LANGFUSE_TRACING_ENABLED: bool = True
    LANGFUSE_SAMPLE_RATE: float = 1.0  # 1.0 = trace everything, 0.1 = 10% sampling
    LANGFUSE_BLOCKED_SCOPES: str = ""  # comma-separated, e.g. "sqlalchemy,psycopg"
```

#### 1.3 Environment Variables
Add to `.env.example`:
```bash
# Observability (Langfuse)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_BLOCKED_SCOPES=
```

---

### Phase 2: Core Initialization & Thread Context Propagation

**Target File**: `apps/gateway/main.py` (inside the existing `lifespan` context manager)

This is the single initialization point. Langfuse SDK v3 auto-sets up OTel, so all other instrumentors attach to the same global `TracerProvider`.

```python
from langfuse import Langfuse, get_client
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from echolib.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing startup (database_lifespan, cache_lifespan) ...

    # --- Observability Initialization ---
    langfuse_client = None
    if settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
        # Parse blocked scopes
        blocked = [s.strip() for s in settings.LANGFUSE_BLOCKED_SCOPES.split(",") if s.strip()]

        # 1. Initialize Langfuse (auto-creates OTel TracerProvider + LangfuseSpanProcessor)
        langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            base_url=settings.LANGFUSE_BASE_URL,
            blocked_instrumentation_scopes=blocked or None,
            sample_rate=settings.LANGFUSE_SAMPLE_RATE,
        )
        langfuse_client.auth_check()

        # 2. Thread context propagation (CRITICAL for asyncio.to_thread + ThreadPoolExecutor)
        ThreadingInstrumentor().instrument()

        # 3. FastAPI auto-instrumentation (HTTP request spans → Langfuse)
        FastAPIInstrumentor.instrument_app(app)

        # 4. CrewAI auto-instrumentation (agent/tool spans → Langfuse)
        CrewAIInstrumentor().instrument()

    yield

    # --- Shutdown ---
    if langfuse_client:
        langfuse_client.flush()
        langfuse_client.shutdown()
    # ... existing shutdown ...
```

**Why this order matters**:
1. Langfuse first — creates the global `TracerProvider` that everything else attaches to.
2. ThreadingInstrumentor second — patches thread classes so all subsequent threaded work carries context.
3. FastAPI/CrewAI instrumentors — attach to the already-configured provider.

---

### Phase 3: LangGraph Instrumentation

**Target File**: `apps/workflow/runtime/executor.py`

Inject the `CallbackHandler` at the `invoke()` call site. This is the single point where all LangGraph execution passes through.

```python
from langfuse.langchain import CallbackHandler
from echolib.config import settings

class WorkflowExecutor:

    def execute_workflow(self, workflow_id, execution_mode, version=None, input_payload=None):
        # ... existing setup (load workflow, compile, build initial_state) ...

        config = {"configurable": {"thread_id": run_id}}

        # --- Langfuse Tracing ---
        if settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
            langfuse_handler = CallbackHandler()
            config["callbacks"] = [langfuse_handler]
            config["metadata"] = {
                "langfuse_session_id": str(run_id),
                "langfuse_user_id": str(input_payload.get("user_id", "system")),
                "langfuse_tags": [
                    f"workflow:{workflow_id}",
                    f"mode:{execution_mode}",
                ],
            }

        final_state = compiled_graph.invoke(initial_state, config)
        # ... existing post-processing ...
```

Apply the same pattern to `execute_workflow_hitl()` and `resume_workflow()`.

**What this captures automatically**:
- Each LangGraph node execution as a span (on_chain_start/end callbacks)
- All LLM calls within nodes as generation spans (tokens, cost, latency)
- Tool calls made by LangChain agents
- Errors and exceptions

---

### Phase 4: CrewAI Adapter Instrumentation

**Target File**: `apps/workflow/crewai_adapter.py`

The `CrewAIInstrumentor` from Phase 2 handles standard CrewAI internals. But EchoAI's custom `DynamicCrewAITool` wrappers and node functions need explicit `@observe` decorators as a reliable layer.

```python
from langfuse import observe

class CrewAIAdapter:

    @observe(name="crewai_create_hierarchical_node")
    def create_hierarchical_crew_node(self, master_agent_config, sub_agent_configs, delegation_strategy="auto"):
        # ... existing logic ...

        @observe(name="crewai_hierarchical_execution")
        def hierarchical_node(state):
            # ... existing crew.kickoff() logic ...
            return state

        return hierarchical_node

    @observe(name="crewai_create_parallel_node")
    def create_parallel_crew_node(self, agent_configs, aggregation_strategy="merge"):
        # ... existing logic ...

        @observe(name="crewai_parallel_execution")
        def parallel_node(state):
            # ... existing crew.kickoff() logic ...
            return state

        return parallel_node

    @observe(name="crewai_create_sequential_node")
    def create_sequential_agent_node(self, agent_config):
        # ... existing logic ...

        @observe(name="crewai_sequential_execution")
        def sequential_agent_node(state):
            # ... existing crew.kickoff() logic ...
            return state

        return sequential_agent_node
```

**Why both auto-instrumentation AND decorators**:
- `CrewAIInstrumentor` captures internal Crew mechanics (agent delegation, tool selection, LLM calls within CrewAI).
- `@observe` decorators capture EchoAI's custom wrapper functions (node creation, state extraction, the handoff between LangGraph and CrewAI).
- Together they produce the full trace: `LangGraph Node → CrewAI Adapter → Crew Execution → Agent → Tool → LLM Call`.

---

### Phase 5: State Transition Tracing

LangGraph nodes are arbitrary callables (not LangChain Chains). The `CallbackHandler` captures LLM calls and chain events but does NOT automatically log the *state dict* changing between nodes.

**Approach**: Add a lightweight wrapper in the compiler that logs state snapshots.

**Target File**: `apps/workflow/designer/compiler.py`

```python
from langfuse import observe, get_client

def _wrap_node_with_tracing(self, node_fn, node_name: str):
    """Wrap a LangGraph node function to trace state transitions."""

    @observe(name=f"node:{node_name}")
    def traced_node(state):
        langfuse = get_client()
        # Log input state keys (not full values — can be large)
        langfuse.update_current_observation(
            input={"state_keys": list(state.keys()), "node": node_name},
        )
        result = node_fn(state)
        langfuse.update_current_observation(
            output={"modified_keys": list(result.keys()) if isinstance(result, dict) else "non-dict"},
        )
        return result

    return traced_node
```

Use this wrapper in `_create_node_for_type()` when building the graph:
```python
# In compile methods, after creating node_fn:
if settings.LANGFUSE_TRACING_ENABLED:
    node_fn = self._wrap_node_with_tracing(node_fn, agent_id)
graph.add_node(agent_id, node_fn)
```

**What this captures**: For each node transition, you see the state keys entering and the keys modified — enough for debugging without serializing large state objects.

---

### Phase 6: Existing Stub Cleanup

#### 6.1 `echolib/adapters.py` — `OTelLogger`
Currently a stub that wraps Python `logging`. Two options:
- **Option A (Recommended)**: Refactor to use a real OTel-backed logger. Since Langfuse SDK v3 sets up the global TracerProvider, we can create spans from `OTelLogger.trace()`:
  ```python
  from opentelemetry import trace

  class OTelLogger(ILogger):
      def __init__(self):
          self._tracer = trace.get_tracer("echoai.logger")

      def trace(self, span_name: str, ctx: dict) -> None:
          with self._tracer.start_as_current_span(span_name) as span:
              for k, v in ctx.items():
                  span.set_attribute(k, str(v))
  ```
- **Option B**: Remove `OTelLogger` entirely and replace DI registrations with direct Langfuse `@observe` usage. Simpler, but requires touching every container.py.

#### 6.2 `apps/workflow/runtime/telemetry.py` — `WorkflowTelemetry`
This stub has `workflow_span`, `agent_span`, `tool_span` context managers that are all no-ops (timing fields are `None`). Since our Phase 3-5 instrumentation covers all these concerns, **delete this file** and remove its imports. It adds confusion with no value.

---

### Phase 7: Production Hardening

#### 7.1 Sampling
Control via `LANGFUSE_SAMPLE_RATE` (configured in Phase 1):
- **Development/Staging**: `1.0` (trace everything)
- **Production**: `0.1` to `0.3` (10-30%) depending on volume

#### 7.2 Sensitive Data
For workflows processing sensitive data, disable input/output capture on specific observations:
```python
@observe(capture_input=False, capture_output=False)
def process_sensitive_data(state):
    ...
```

#### 7.3 Noise Filtering
Use `LANGFUSE_BLOCKED_SCOPES` to suppress high-volume, low-value spans:
```bash
LANGFUSE_BLOCKED_SCOPES=sqlalchemy,psycopg,asyncpg
```

**Warning**: Filtering scopes can break parent-child relationships in traces, creating orphaned observations. Only block scopes that produce noise unrelated to AI workflow debugging.

#### 7.4 DI Container Registration
Register the Langfuse client as a singleton in the DI system so it can be injected into services that need explicit trace manipulation:
```python
# In apps/gateway/main.py or a shared container
from langfuse import get_client
# get_client() already returns a singleton — just use it directly.
# No need for a custom DI wrapper unless other services need it injected.
```

---

## 4. Verification Plan

### 4.1 Smoke Test
```python
# Quick script to verify Langfuse connectivity
from langfuse import get_client
langfuse = get_client()
assert langfuse.auth_check(), "Langfuse auth failed"
print("Langfuse connected successfully")
```

### 4.2 Integration Test
Create `tests/test_observability.py`:
1. Execute a simple sequential workflow via the API.
2. Wait for `langfuse.flush()`.
3. Query the Langfuse API for the trace by `session_id` (= `run_id`).
4. Assert the trace contains expected spans.

### 4.3 Manual Verification
Execute a "Research" workflow (Hierarchical Crew) and verify in Langfuse Trace View:
```
Root Trace: Workflow Run (session_id=run_id, user_id=user_id)
├── Span: node:researcher (state_keys in, modified_keys out)
│   ├── Span: crewai_hierarchical_execution
│   │   ├── Span: CrewAI Manager Agent (auto-instrumented)
│   │   │   ├── Generation: LLM Call (model, tokens, cost)
│   │   │   └── Span: Tool Call: DynamicCrewAITool
│   │   └── Span: CrewAI Worker Agent (auto-instrumented)
│   │       └── Generation: LLM Call
├── Span: node:writer
│   └── ...
└── HTTP Span: POST /api/workflow/execute (FastAPI auto-instrumented)
```

---

## 5. File Change Summary

| File | Change Type | Description |
|---|---|---|
| `requirements.txt` | Modify | Add `langfuse>=3.0.0`, OTel instrumentors |
| `echolib/config.py` | Modify | Add Langfuse settings to `Settings` class |
| `apps/gateway/main.py` | Modify | Initialize Langfuse + instrumentors in `lifespan` |
| `apps/workflow/runtime/executor.py` | Modify | Add `CallbackHandler` + metadata to `invoke()` config |
| `apps/workflow/crewai_adapter.py` | Modify | Add `@observe` decorators on node factory methods |
| `apps/workflow/designer/compiler.py` | Modify | Add `_wrap_node_with_tracing()` for state transition logging |
| `echolib/adapters.py` | Modify | Wire `OTelLogger` to real OTel tracer |
| `apps/workflow/runtime/telemetry.py` | Delete | Remove unused stub (replaced by Langfuse instrumentation) |
| `tests/test_observability.py` | Create | Integration test for trace verification |
| `.env.example` | Modify | Add Langfuse environment variables |

---

## 6. Dependency & Compatibility Matrix

| Package | Version | Purpose | Compatibility |
|---|---|---|---|
| `langfuse` | `>=3.0.0` | Core SDK (OTel-native) | Python 3.10+ |
| `opentelemetry-instrumentation-fastapi` | latest | Auto-trace HTTP requests | FastAPI 0.115+ |
| `opentelemetry-instrumentation-threading` | latest | Context propagation across threads | Python 3.10+ |
| `openinference-instrumentation-crewai` | `>=0.1.19` | Auto-trace CrewAI internals | CrewAI 1.0+, Python 3.10-3.13 |
| `opentelemetry-api` | `>=1.20.0` | Already in requirements.txt | -- |
| `opentelemetry-sdk` | `>=1.20.0` | Already in requirements.txt | -- |

**Not needed** (removed from original plan):
- `opentelemetry-instrumentation-langchain` — Langfuse `CallbackHandler` already covers LangChain/LangGraph.
- `langfuse<3.0` — v2 import paths (`langfuse.callback`, `langfuse.decorators`) are deprecated.

---

## 7. Implementation Order

1. **Phase 1**: Dependencies + Config (no runtime changes, safe to merge first)
2. **Phase 2**: Gateway initialization (enables tracing globally, but no traces emitted yet without Phase 3+)
3. **Phase 3**: Executor instrumentation (first traces appear in Langfuse)
4. **Phase 4**: CrewAI adapter decorators (enriches traces with CrewAI detail)
5. **Phase 5**: State transition tracing (adds node-level state visibility)
6. **Phase 6**: Stub cleanup (housekeeping, no functional change)
7. **Phase 7**: Production hardening (tuning before production rollout)

Each phase is independently deployable and does not break existing functionality.

---

## 8. What Is NOT Changed
- `apps/workflow/runtime/transparency.py` — untouched, continues handling real-time WebSocket UI updates.
- All existing `execute_workflow()`, `crew.kickoff()`, and route handler logic — untouched.
- DI container registrations (except removing `WorkflowTelemetry` stub references).
- Database schema, migrations, API contracts — untouched.

---

## 9. References
- [Langfuse Python SDK v3 Docs](https://langfuse.com/docs/observability/sdk/overview)
- [Langfuse LangChain/LangGraph Integration](https://langfuse.com/integrations/frameworks/langchain)
- [Langfuse OpenTelemetry Integration](https://langfuse.com/integrations/native/opentelemetry)
- [Langfuse SDK v2→v3 Upgrade Path](https://langfuse.com/docs/observability/sdk/python/upgrade-path)
- [Langfuse LangGraph Cookbook](https://langfuse.com/guides/cookbook/integration_langgraph)
- [OpenTelemetry FastAPI Instrumentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html)
- [OpenTelemetry Threading Instrumentation](https://pypi.org/project/opentelemetry-instrumentation-threading/)
- [OpenInference CrewAI Instrumentation](https://pypi.org/project/openinference-instrumentation-crewai/)
