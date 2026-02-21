# Implementation Plan: A2UI Integration (Unified Transparency)

> **STATUS**: Original plan (Sections 0–6) has been audited and corrected.
> The **authoritative plan** is in **Section 9 onwards**.
> Original sections are preserved for historical reference with correction notes inline.

---

## 0. Strategic Comparison: A2UI vs. Current `transparency.py`

### 0.1 Architecture Logic & Reasoning

**The Goal**: Make the **"Execution Progress"** (currently visible in the Workflow IDE) available inside the **App Manager Chat**.

**Current State (Siloed)**
-   **Workflow IDE**: Provides real-time visibility — events flow from `crewai_adapter.py` → `event_publisher.py` → `ws_manager.py` → WebSocket to IDE clients.
-   **App Manager Chat**: A "Black Box". User types -> Waits -> Gets final response. The rich execution details are lost.

**The A2UI Solution (Unified Protocol)**
**A2UI** acts as a standardized **UI Protocol** between the Backend Agents and any Frontend (IDE or Chat).

1.  **Workflow**: instead of raw WebSocket-only events, emit events to **any subscriber** — WebSocket OR `asyncio.Queue`.
2.  **App Manager**: The chat stream becomes a **Universal Renderer** via SSE, consuming A2UI JSONL components.
    -   The **exact same execution steps** from the IDE appear in the Chat bubble.

### 0.2 Why A2UI?
-   **Consistency**: A single source of truth for "How execution looks".
-   **Decoupling**: The backend controls the visualization. Adding a new step type doesn't require a frontend deploy.
-   **App Manager Ready**: HTTP-based App Manager maps naturally to SSE + A2UI JSONL.

---

## 1. Goal
Implement a **"Cool Looking"** Agent-to-UI experience, specifically focused on **visualizing the execution steps** of the backend agents in real-time, across the full pipeline: Prompt Enhancement → Orchestrator Planning → Skill Execution (Workflow/Agent).

---

## 2. Original Protocol Schema *(Superseded — see Section 9.1)*

> ⚠️ **NOTE**: The custom component schema below (`step_list`, `status_card`) does NOT align with
> the official A2UI v0.8 spec. The corrected schema (Section 9.1) uses the proper
> `surfaceUpdate / dataModelUpdate / beginRendering` message types.

### Core Components (Original — Do Not Use)
```python
class ComponentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    STEP_LIST = "step_list"  # non-standard
    STATUS_CARD = "status_card"  # non-standard
    JSON_VIEW = "json_view"

class A2UIEvent(BaseModel):
    type: str = "update"  # non-standard
    target_id: str
    payload: BaseComponent
```

---

## 3. Workflow Engine Analysis (`apps/workflow`) — Corrected

### Real Transparency Chain (Confirmed from Codebase)

```
crewai_adapter.py  (sync LangGraph node function)
  → step_tracker.start_step()              [in-memory only — transparency.py]
  → publish_step_started_sync()            [event_publisher.py]
    → ws_manager.buffer_and_send_sync(run_id, event)
      → asyncio.run_coroutine_threadsafe(...)   [thread-safe bridge — ALREADY SOLVED]
        → WebSocket.send_json()            [Workflow IDE clients only]
```

### Key Facts
-   `transparency.py` is **in-memory state only** — it emits NOTHING. No WebSocket calls.
-   **`crewai_adapter.py`** is at `apps/workflow/crewai_adapter.py` (NOT in `runtime/`).
-   The actual emission layer is `apps/workflow/runtime/event_publisher.py`.
-   Thread-safety (sync LangGraph → async loop) is **already solved** in `ws_manager.buffer_and_send_sync()` via `asyncio.run_coroutine_threadsafe()`.

### run_id Flow (Critical Architectural Link)
```
SkillExecutor._execute_workflow_hitl_aware()
  → input_payload = {"user_input": ..., "message": ...}   ← run_id NOT injected yet
  → asyncio.to_thread(executor.execute_workflow_hitl, ..., input_payload, ...)
    → WorkflowExecutor: run_id = input_payload.get("run_id") or new_id("run_")
    → initial_state["run_id"] = run_id
    → compiled_graph.invoke(initial_state, ...)
      → LangGraph nodes (crewai_adapter) receive state
        → crewai_adapter._get_run_id(state)  ← reads state["run_id"]
          → ws_manager.buffer_and_send_sync(run_id, event)
```

**Implication**: If we pre-generate a `run_id`, register an `asyncio.Queue` in `ws_manager` under that `run_id` BEFORE calling the executor, then inject it into `input_payload["run_id"]` — every crewai_adapter event auto-routes to the queue. Zero changes to crewai_adapter needed.

### Original "Implementation Hooks" *(Superseded)*
> ⚠️ The original plan proposed modifying `transparency.py` to support event listeners
> and modifying `crewai_adapter.py` to emit tool start/end events.
> This is **NOT needed for the bridge**. The bridge is handled via ws_manager queue subscription.
> Tool-level granularity is a future enhancement, not a Phase 1 requirement.

---

## 4. App Manager Analysis (`apps/appmgr`) — Corrected

### Real Architecture (Confirmed)

**`apps/appmgr/orchestrator/pipeline.py` — `OrchestrationPipeline.run()`**
-   15-step **blocking** pipeline. Returns `Dict[str, Any]`. No streaming path exists.
-   Steps: load app → session → pre-guardrails → HITL check → persona → **enhance prompt** → RAG → skill manifest → history → **orchestrator LLM** → HITL response → **execute skills** → post-guardrails → save messages → save trace.
-   Steps 6 (enhance), 10 (orchestrator LLM), 12 (execute skills) are the "expensive" stages — all fully opaque to users currently.

**`apps/appmgr/orchestrator/skill_executor.py` — `SkillExecutor`**
-   `_execute_workflow_hitl_aware()` runs via `asyncio.to_thread()`. Does NOT pass `run_id` to executor.
-   No callback/listener mechanism exists.

**`apps/appmgr/orchestrator/agent_executor.py` — `StandaloneAgentExecutor`**
-   100% independent of `crewai_adapter.py`. Imports `crewai` directly.
-   **Zero transparency hooks** — no step_tracker calls, no event_publisher calls, no ws_manager integration.
-   Runs via `asyncio.to_thread(crew.kickoff)`.

**`apps/appmgr/routes.py`**
-   Only `POST /{id}/chat` blocking endpoint exists. No SSE/streaming route.

### Original Bridge Strategy *(Superseded)*
> ⚠️ Original plan proposed `event_listener: Callable[[Dict], Awaitable[None]]` argument on
> `WorkflowExecutor.execute_workflow`. This is WRONG — async callables cannot be safely called
> from sync LangGraph threads. The correct solution is `asyncio.Queue` subscribers in `ws_manager`.
> The original plan also missed the need to stream pipeline-level stages (enhancer, orchestrator).

---

## 5. Original Execution Plan *(Superseded — see Section 9.4)*

> The original 3-phase plan (Core Protocol → Workflow Upgrade → App Manager Integration)
> has been replaced by the corrected 6-phase plan in Section 9.4.
> The original phases are preserved here only for reference.

### Original Phase 1: Core Protocol
1.  Create `echolib/a2ui/models.py`.

### Original Phase 2: Workflow Upgrade
1.  Modify `apps/workflow/runtime/transparency.py` to support "Local Event Listeners".
2.  Update `apps/workflow/crewai_adapter.py` to emit Tool Start/End events.

### Original Phase 3: App Manager Integration
1.  Refactor `apps/appmgr/orchestrator/pipeline.py` to support Streaming.
2.  Refactor `apps/appmgr/orchestrator/skill_executor.py` to bridge Workflow events.
3.  Implement `POST /api/applications/{id}/chat/stream`.

---

## 6. Original Verification *(Superseded — see Section 9.5)*

---

## 7. Q&A: Is A2UI a good option or my current `transparency.py` is good?

**Answer**:
Your current `transparency.py` is **NOT good enough** for your goal of displaying execution steps in the **App Manager**. It is currently designed as a closed loop for the Workflow IDE only.

**Why A2UI is the BETTER option:**
1.  **Universal Visibility**: `transparency.py` events are stuck in the Workflow Engine. To get them into the App Manager App, you *have* to stream them out. A2UI provides the standard protocol/format for that stream.
2.  **Frontend Scalability**: If you stick with raw events, you have to write complex parsing logic in the App Manager frontend. A2UI tells the frontend explicitly what to render.
3.  **Unified Experience**: A2UI ensures that the "cool steps" you see in the IDE look *exactly the same* in the App Manager chat.

**Conclusion**: Keep `transparency.py` as the *in-memory state engine*, route events to `asyncio.Queue` subscribers via `ws_manager`, and speak **A2UI JSONL** as the wire format.

---

## 8. Alternatives to A2UI (Market Search)

1.  **Vercel AI SDK (RSC)**: Tied to Next.js/React Server Components. Bad for Python/FastAPI.
2.  **Chainlit**: Takes over the entire UI. Great for prototypes, too restrictive for a custom platform.
3.  **AG-UI / Adaptive Cards (Microsoft)**: Enterprise look-and-feel. Hard to style modernly.

**Why custom A2UI pattern is best for EchoAI**: No heavy framework needed. Lightweight `StepList` schema gives 100% control over aesthetics without SDK bloat.

---

---

# CORRECTED IMPLEMENTATION PLAN
## (Authoritative — Based on Full Codebase Audit)

---

## 9. Corrected Architecture Overview

### 9.0 What the Original Plan Got Wrong

| Original Claim | Reality Found | Impact |
|---|---|---|
| `transparency.py` emits WebSocket events | In-memory state only. Zero I/O. | Don't touch transparency.py for the bridge |
| `crewai_adapter.py` is in `apps/workflow/runtime/` | Lives at `apps/workflow/crewai_adapter.py` | Correct file path when modifying |
| Thread-safety sync→async is unsolved | Already solved in `ws_manager.buffer_and_send_sync()` via `asyncio.run_coroutine_threadsafe()` | No new bridge needed |
| `event_listener: Callable[..., Awaitable[None]]` bridge | Async callables unsafe from sync threads | Use `asyncio.Queue` in ws_manager instead |
| Only Workflow events need streaming | Prompt Enhancer, Orchestrator LLM, Guardrails, Agents are all completely dark | Pipeline-level events are the biggest gap |
| `StandaloneAgentExecutor` has transparency | Zero hooks — completely independent | Needs separate treatment (Phase 5) |

### 9.1 Target Architecture

```
[POST /api/applications/{id}/chat/stream]   (SSE Route — NEW)
  │
  ├─ generates:  run_id = new_id("run_")
  ├─ creates:    sse_queue = asyncio.Queue(maxsize=500)
  ├─ registers:  ws_manager.subscribe_queue(run_id, sse_queue)
  │
  ├─ asyncio.create_task(pipeline.run_stream(run_id, sse_queue, ...))
  │
  └─ EventSourceResponse(sse_generator(sse_queue))
       └─ yields JSONL lines from sse_queue as SSE data: text/event-stream


[OrchestrationPipeline.run_stream()]   (NEW METHOD alongside existing run())
  │
  ├─ Emit: A2UI surface init → sse_queue
  ├─ Stage: pre-guardrails → emit step running/completed to sse_queue
  ├─ Stage: prompt enhancement → emit step running/completed to sse_queue
  ├─ Stage: RAG retrieval → emit step running/completed to sse_queue
  ├─ Stage: orchestrator planning → emit step running/completed+plan to sse_queue
  ├─ Stage: skill dispatch → for each skill, emit step running to sse_queue
  │          inject run_id into input_payload["run_id"]
  │
  │          [ws_manager now has sse_queue subscribed to run_id]
  │          [crewai_adapter events auto-route to sse_queue — no code change needed]
  │
  ├─ Stage: post-guardrails → emit step running/completed to sse_queue
  ├─ Emit: final output A2UI dataModelUpdate → sse_queue
  └─ Put None sentinel → sse_queue (signals end of stream)


[ws_manager.py — MINIMAL CHANGE]
  _queue_subscribers: Dict[str, Set[asyncio.Queue]]  ← NEW field
  subscribe_queue(run_id, queue)                      ← NEW
  unsubscribe_queue(run_id, queue)                    ← NEW
  buffer_and_send_sync() → also: q.put_nowait(event) for queue_subscribers[run_id]
  broadcast()            → also: q.put_nowait(event) for queue_subscribers[run_id]


[StandaloneAgentExecutor — MINIMAL CHANGE]
  execute(sse_queue: Optional[asyncio.Queue] = None)
  emits A2UI step events directly to sse_queue (no ws_manager — no run_id concept)
```

### 9.2 A2UI Wire Format (Official v0.8 Spec)

Using proper A2UI JSONL transport: `surfaceUpdate` → `dataModelUpdate` → `beginRendering`.

#### Surface Initialization (emitted once per chat request, immediately on connection)
```jsonl
{"type":"surfaceUpdate","surfaceId":"exec_{run_id}","components":[
  {"id":"root","type":"Column","children":{"explicitList":["header","steps","output"]}},
  {"id":"header","type":"Text","content":{"path":"/title"},"hint":"h3"},
  {"id":"steps","type":"List","children":{"template":{"dataPath":"/steps","componentType":"Card","childIds":["step-icon","step-label","step-detail","step-status"]}}},
  {"id":"step-icon","type":"Icon","name":{"path":"/steps[*]/icon"}},
  {"id":"step-label","type":"Text","content":{"path":"/steps[*]/label"},"hint":"h5"},
  {"id":"step-detail","type":"Text","content":{"path":"/steps[*]/detail"},"hint":"caption"},
  {"id":"step-status","type":"Text","content":{"path":"/steps[*]/status"},"hint":"caption"},
  {"id":"output","type":"Card","children":{"explicitList":["output-text"]}},
  {"id":"output-text","type":"Text","content":{"path":"/final_output"},"hint":"body"}
]}
{"type":"dataModelUpdate","surfaceId":"exec_{run_id}","data":{"title":"Processing your request...","steps":[],"final_output":""}}
{"type":"beginRendering","surfaceId":"exec_{run_id}","rootComponentId":"root"}
```

#### Step Progress Update (one per pipeline stage change, re-sends full steps array)
```jsonl
{"type":"dataModelUpdate","surfaceId":"exec_{run_id}","data":{"steps":[
  {"id":"step_guardrails","icon":"shield","label":"Checking guardrails","detail":"","status":"completed"},
  {"id":"step_enhance","icon":"auto_fix_high","label":"Enhancing prompt","detail":"Analyzing intent...","status":"running"},
  {"id":"step_rag","icon":"search","label":"Retrieving context","detail":"","status":"pending"},
  {"id":"step_plan","icon":"account_tree","label":"Planning execution","detail":"","status":"pending"},
  {"id":"step_execute","icon":"play_circle","label":"Executing skills","detail":"","status":"pending"}
]}}
```

#### Final Output Update
```jsonl
{"type":"dataModelUpdate","surfaceId":"exec_{run_id}","data":{"final_output":"The agent completed the task. Here are the results..."}}
```

#### Step Status Values
| Status | Icon (Material) | Meaning |
|---|---|---|
| `pending` | `radio_button_unchecked` | Not started |
| `running` | `sync` (animated) | In progress |
| `completed` | `check_circle` | Done successfully |
| `failed` | `error` | Errored |
| `interrupted` | `pause_circle` | HITL pause |
| `skipped` | `skip_next` | Bypassed |

### 9.3 Files Changed vs. Untouched

#### Files That WILL Change
| File | Change | Risk |
|---|---|---|
| `echolib/a2ui/__init__.py` | NEW — module init | None |
| `echolib/a2ui/models.py` | NEW — Pydantic models for A2UI messages | None |
| `echolib/a2ui/builder.py` | NEW — `A2UIStreamBuilder` helper | None |
| `apps/workflow/runtime/ws_manager.py` | ADD `_queue_subscribers`, 2 new methods, 2 call sites | Low — additive only |
| `apps/appmgr/orchestrator/pipeline.py` | ADD `run_stream()` method alongside existing `run()` | Low — existing `run()` untouched |
| `apps/appmgr/orchestrator/skill_executor.py` | PASS `sse_queue` and `run_id` down to executors | Low |
| `apps/appmgr/orchestrator/agent_executor.py` | ADD optional `sse_queue` param + direct queue emission | Low |
| `apps/appmgr/routes.py` | ADD new SSE route — existing blocking route untouched | None |

#### Files That WILL NOT Change
| File | Reason |
|---|---|
| `apps/workflow/runtime/transparency.py` | In-memory state tracker — bridge happens via ws_manager, not here |
| `apps/workflow/runtime/event_publisher.py` | Emission layer is correct — ws_manager changes propagate automatically |
| `apps/workflow/crewai_adapter.py` | Events auto-route to queues via ws_manager — no adapter changes needed |
| `apps/workflow/runtime/executor.py` | run_id injection happens in input_payload upstream — executor already reads it |
| `apps/appmgr/orchestrator/pipeline.py` existing `run()` | Preserved as-is — only new method added |
| `apps/appmgr/routes.py` existing `POST /{id}/chat` | Preserved as-is — new SSE route added alongside |

### 9.4 Corrected Execution Plan (6 Phases)

#### Phase 1 — A2UI Protocol Models
**Create**: `echolib/a2ui/__init__.py`, `echolib/a2ui/models.py`, `echolib/a2ui/builder.py`

`models.py` — Pydantic models:
-   `BoundValue` — `literalString: Optional[str]`, `path: Optional[str]`
-   `ChildrenSpec` — `explicitList: Optional[list[str]]`, `template: Optional[TemplateSpec]`
-   `Component` — `id: str`, `type: str`, `children: Optional[ChildrenSpec]`, `content: Optional[BoundValue]`, `hint: Optional[str]`, `name: Optional[BoundValue]`, `label: Optional[BoundValue]`, `action: Optional[ActionDef]`
-   `SurfaceUpdateMessage` — `type: Literal["surfaceUpdate"]`, `surfaceId: str`, `components: list[Component]`
-   `DataModelUpdateMessage` — `type: Literal["dataModelUpdate"]`, `surfaceId: str`, `data: dict`
-   `BeginRenderingMessage` — `type: Literal["beginRendering"]`, `surfaceId: str`, `rootComponentId: str`

`builder.py` — `A2UIStreamBuilder`:
-   `build_initial_surface(run_id: str, title: str) -> list[str]` — returns list of JSONL lines
-   `step_update(run_id: str, steps: list[dict]) -> str` — returns single JSONL line
-   `final_output_update(run_id: str, text: str) -> str` — returns single JSONL line
-   All methods return serialized JSON strings, ready for `queue.put_nowait(line)`

#### Phase 2 — WebSocket Manager Queue Extension
**File**: `apps/workflow/runtime/ws_manager.py`

Additions (additive only, nothing removed or changed in existing methods except two call sites):

```python
# New field on WebSocketManager
_queue_subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

def subscribe_queue(self, run_id: str, queue: asyncio.Queue) -> None:
    self._queue_subscribers[run_id].add(queue)

def unsubscribe_queue(self, run_id: str, queue: asyncio.Queue) -> None:
    self._queue_subscribers[run_id].discard(queue)
    if not self._queue_subscribers[run_id]:
        del self._queue_subscribers[run_id]

# Inside buffer_and_send_sync() — after existing WebSocket send block:
for q in list(self._queue_subscribers.get(run_id, set())):
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass  # slow consumer — drop silently

# Inside broadcast() — after existing WebSocket send block:
for q in list(self._queue_subscribers.get(run_id, set())):
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass
```

#### Phase 3 — Pipeline Streaming Method
**File**: `apps/appmgr/orchestrator/pipeline.py`

Add `run_stream(self, sse_queue: asyncio.Queue, run_id: str, ...)` alongside existing `run()`.

The method mirrors `run()` step-by-step but:
1.  At start: puts A2UI `build_initial_surface(run_id, app.name)` lines to `sse_queue`
2.  Before each expensive stage: puts step `status=running` update to `sse_queue`
3.  After each stage: puts step `status=completed` or `status=failed` update to `sse_queue`
4.  Before calling `skill_executor.execute_plan()`:
    -   Calls `ws_manager.subscribe_queue(run_id, sse_queue)`
    -   Injects `run_id` into the input payload passed to executor
5.  After `execute_plan()` returns: calls `ws_manager.unsubscribe_queue(run_id, sse_queue)`
6.  Emits final output A2UI update
7.  Puts `None` sentinel to `sse_queue` to signal end

Pipeline stages and their A2UI step IDs:
| Stage | step_id | icon | label |
|---|---|---|---|
| pre-guardrails | `step_guardrails` | `shield` | Checking guardrails |
| prompt enhancement | `step_enhance` | `auto_fix_high` | Enhancing prompt |
| RAG retrieval | `step_rag` | `search` | Retrieving context |
| orchestrator planning | `step_plan` | `account_tree` | Planning execution |
| skill N dispatch | `step_skill_{N}` | `play_circle` | Running: {skill_name} |
| post-guardrails | `step_post_guard` | `verified` | Validating output |

#### Phase 4 — SSE Route
**File**: `apps/appmgr/routes.py`

Add `POST /api/applications/{app_id}/chat/stream` with `EventSourceResponse`:

```python
@router.post("/{app_id}/chat/stream")
async def chat_stream(app_id: str, request: ChatRequest, db: AsyncSession = Depends(get_db)):
    run_id = new_id("run_")
    sse_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def sse_generator():
        try:
            while True:
                item = await asyncio.wait_for(sse_queue.get(), timeout=120.0)
                if item is None:  # sentinel — stream ended
                    break
                yield {"data": item}
        except asyncio.TimeoutError:
            yield {"data": json.dumps({"type": "error", "message": "Stream timeout"})}
        finally:
            ws_manager.unsubscribe_queue(run_id, sse_queue)  # disconnect cleanup

    asyncio.create_task(
        pipeline.run_stream(
            sse_queue=sse_queue,
            run_id=run_id,
            db=db,
            app_id=app_id,
            user_input=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
    )

    return EventSourceResponse(sse_generator(), media_type="text/event-stream")
```

#### Phase 5 — Agent Transparency
**File**: `apps/appmgr/orchestrator/agent_executor.py`

`StandaloneAgentExecutor.execute()` gains `sse_queue: Optional[asyncio.Queue] = None`.

Emit directly to queue (not via ws_manager — standalone agents have no run_id):
-   Before `_run_crew()`: put step `status=running, label=f"Running agent: {agent_name}", icon=smart_toy`
-   On success: put step `status=completed, detail=output[:200]`
-   On exception: put step `status=failed, detail=str(exc)`

`SkillExecutor` must pass `sse_queue` when calling `agent_executor.execute()`.

#### Phase 6 — Verification Checklist

1.  **Workflow path**: POST `/chat/stream` → trigger workflow skill → confirm per-LangGraph-node step updates arrive from crewai_adapter via ws_manager queue subscription
2.  **Agent path**: POST `/chat/stream` → trigger standalone agent → confirm agent start/complete steps appear
3.  **Pipeline stages**: Pre-guardrails, enhancement, planning, post-guardrails all appear as distinct steps in correct order before workflow steps
4.  **SSE disconnect**: Client closes mid-stream → `unsubscribe_queue` fires in `finally` → no dangling queue, no memory leak
5.  **Concurrent sessions**: Two users, same app, different sessions → each gets their own `run_id` and queue, no cross-talk
6.  **Fallback / error**: Orchestrator returns `fallback_message` → final output step shows fallback text, surface closes cleanly
7.  **HITL interrupt**: Workflow hits HITL node → step shows `status=interrupted`, SSE stream stays open, resume path emits `status=resumed`
8.  **Queue full**: Slow SSE consumer — `put_nowait` drops silently, no exception propagation to LangGraph thread
9.  **No regression**: Existing `POST /{id}/chat` blocking endpoint continues to work unchanged

### 9.5 Dependency and Import Notes

-   `EventSourceResponse` comes from `sse_starlette.sse` — verify it is in requirements.txt
-   `asyncio.Queue` is stdlib — no new dependencies for ws_manager changes
-   `echolib/a2ui/` is a new internal module — ensure it is importable from both `apps/appmgr` and `apps/workflow` (check `echolib` package structure)
-   `ws_manager` singleton is imported in routes.py already for HITL WS — same import pattern for `subscribe_queue`/`unsubscribe_queue`

### 9.6 Open Questions (Resolve Before Coding Phase 3)

1.  **`new_id` location**: Where is `new_id("run_")` defined? Confirm import path in `pipeline.py` context.
2.  **`SkillExecutor.execute_plan()` signature**: Does it accept `run_id` today, or does it need to be added?
3.  **SSE auth**: Does the existing `POST /{id}/chat` route use auth middleware? The new stream route must match.
4.  **Frontend contract**: What format does the frontend SSE client expect? Standard `data: ...\n\n` or custom headers?
5.  **HITL during streaming**: When the workflow hits HITL, should the SSE stream stay open indefinitely or close with a resume token?

---

## 10. Implementation Order Summary

```
Phase 1: echolib/a2ui/models.py + builder.py           [No dependencies]
Phase 2: ws_manager.py queue subscriber support         [No dependencies]
Phase 3: pipeline.py run_stream() method                [Depends on Phase 1 + 2]
Phase 4: routes.py SSE endpoint                         [Depends on Phase 3]
Phase 5: agent_executor.py sse_queue param              [Depends on Phase 1, parallel with Phase 3]
Phase 6: Verification                                   [Depends on Phases 1–5]
```

Phases 1 and 2 can start in parallel.
Phase 5 can start in parallel with Phase 3.
Phase 4 must wait for Phase 3.

---

## 11. Frontend: `workflow_builder_ide.html` Changes

**File**: `C:\Users\Shashank Singh\Desktop\Production-Echo\frontend\workflow_builder_ide.html`

### 11.1 Impact Analysis

| Backend Change | Impact on IDE HTML |
|---|---|
| `ws_manager.py` queue subscribers | Zero — WebSocket events to IDE are format-identical |
| New `POST /api/applications/{id}/chat/stream` SSE route | New — IDE has no consumer yet; needs A2UI SSE client |
| `OrchestrationPipeline.run_stream()` | Zero direct impact |
| `echolib/a2ui/` models + builder | The JSONL format those produce is what the SSE client must parse |
| `agent_executor.py` A2UI hooks | Zero direct impact on IDE |

### 11.2 What MUST NOT Change

- `connectExecutionWebSocket()` — still used for Workflow IDE execution via WebSocket
- `handleExecutionEvent()` — handles raw `run_started / step_started / step_completed / run_completed` events; untouched
- `pollForExecutionResult()` — fallback polling; untouched
- All existing `.execution-step`, `.chat-system-message` CSS — untouched; new classes only added
- `sendChatMessage()` — continues to use `/workflows/chat/send` (workflow-specific endpoint)

### 11.3 Required Additions

#### A. Google Material Icons CDN (`<head>`)
A2UI spec uses Material Icons names (`shield`, `auto_fix_high`, `account_tree`, `play_circle`, `smart_toy`, `sync`, `check_circle`, `error`, `pause_circle`, etc.) for step icons. Add to `<head>`:
```html
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
```

#### B. New CSS Rules
Add after existing `.execution-step.failed` block:
```css
/* A2UI step status variants */
.execution-step.pending {
    background: #f8fafc;
    border-left: 3px solid #cbd5e1;
    color: #94a3b8;
}
.execution-step.interrupted {
    background: #fffbeb;
    border-left: 3px solid #f59e0b;
    animation: pulse-step 1.5s ease-in-out infinite;
}
.execution-step.skipped {
    background: #f8fafc;
    border-left: 3px solid #94a3b8;
    opacity: 0.6;
}
/* A2UI surface container */
.a2ui-surface {
    border: 1px solid #e0e7ff;
    border-radius: 8px;
    padding: 10px;
    background: #fafbff;
    margin: 6px 0;
}
.a2ui-step-icon {
    font-family: 'Material Icons';
    font-size: 16px;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.a2ui-step-icon.running { color: #6366f1; animation: spin 0.8s linear infinite; }
.a2ui-step-icon.completed { color: #10b981; }
.a2ui-step-icon.failed { color: #ef4444; }
.a2ui-step-icon.pending { color: #94a3b8; }
.a2ui-step-icon.interrupted { color: #f59e0b; }
.a2ui-step-icon.skipped { color: #94a3b8; }
.a2ui-final-output {
    margin-top: 8px;
    padding: 10px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-size: 12px;
    color: #334155;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.5;
}
```

#### C. New Vue Reactive State (inside `setup()`)
```javascript
// A2UI SSE State
const a2uiSurfaces = ref({});          // surfaceId → { components, dataModel, ready }
const a2uiActiveSteps = ref([]);       // derived: steps array from active surface
const a2uiFinalOutput = ref('');       // derived: /final_output from active surface
const a2uiTitle = ref('');            // derived: /title from active surface
const a2uiStreamActive = ref(false);   // true while SSE stream is open
const a2uiAbortController = ref(null); // AbortController for current stream
const a2uiActiveSurfaceId = ref(null); // currently active surfaceId
```

#### D. `processA2UILine(jsonLine)` function
```javascript
const processA2UILine = (jsonLine) => {
    try {
        const msg = JSON.parse(jsonLine);
        if (!msg.type || !msg.surfaceId) return;

        const sid = msg.surfaceId;

        if (msg.type === 'surfaceUpdate') {
            if (!a2uiSurfaces.value[sid]) {
                a2uiSurfaces.value[sid] = { components: {}, dataModel: {}, ready: false };
            }
            // Store components by id
            (msg.components || []).forEach(c => {
                a2uiSurfaces.value[sid].components[c.id] = c;
            });
        } else if (msg.type === 'dataModelUpdate') {
            if (!a2uiSurfaces.value[sid]) {
                a2uiSurfaces.value[sid] = { components: {}, dataModel: {}, ready: false };
            }
            // Merge data into dataModel
            Object.assign(a2uiSurfaces.value[sid].dataModel, msg.data || {});
            // Sync derived refs if this is the active surface
            if (sid === a2uiActiveSurfaceId.value) {
                const dm = a2uiSurfaces.value[sid].dataModel;
                if (dm.steps !== undefined) a2uiActiveSteps.value = dm.steps || [];
                if (dm.final_output !== undefined) a2uiFinalOutput.value = dm.final_output || '';
                if (dm.title !== undefined) a2uiTitle.value = dm.title || '';
            }
        } else if (msg.type === 'beginRendering') {
            if (a2uiSurfaces.value[sid]) {
                a2uiSurfaces.value[sid].ready = true;
            }
            a2uiActiveSurfaceId.value = sid;
            // Sync derived refs on first render
            const dm = (a2uiSurfaces.value[sid] || {}).dataModel || {};
            a2uiActiveSteps.value = dm.steps || [];
            a2uiFinalOutput.value = dm.final_output || '';
            a2uiTitle.value = dm.title || '';
        } else if (msg.type === 'deleteSurface') {
            if (a2uiSurfaces.value[sid]) {
                delete a2uiSurfaces.value[sid];
            }
            if (sid === a2uiActiveSurfaceId.value) {
                a2uiActiveSurfaceId.value = null;
                a2uiActiveSteps.value = [];
                a2uiFinalOutput.value = '';
            }
        }
    } catch (e) {
        console.warn('[A2UI] Failed to parse JSONL line:', jsonLine, e);
    }
};
```

#### E. `connectA2UIStream(appId, message, sessionId)` function
Uses `fetch` + `ReadableStream` (NOT `EventSource` — EventSource only supports GET, this endpoint is POST):
```javascript
const connectA2UIStream = async (appId, message, sessionId = null) => {
    // Cancel any existing stream
    if (a2uiAbortController.value) {
        a2uiAbortController.value.abort();
    }

    // Reset state
    a2uiSurfaces.value = {};
    a2uiActiveSteps.value = [];
    a2uiFinalOutput.value = '';
    a2uiTitle.value = '';
    a2uiActiveSurfaceId.value = null;
    a2uiStreamActive.value = true;

    const controller = new AbortController();
    a2uiAbortController.value = controller;

    try {
        const response = await fetch(`${API_BASE_URL}/api/applications/${appId}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
            signal: controller.signal,
        });

        if (!response.ok) {
            throw new Error(`SSE stream failed: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line in buffer

            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('data: ')) {
                    const jsonLine = trimmed.slice(6);  // strip 'data: ' prefix
                    if (jsonLine && jsonLine !== '[DONE]') {
                        processA2UILine(jsonLine);
                    }
                }
            }
        }
    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error('[A2UI] Stream error:', e);
        }
    } finally {
        a2uiStreamActive.value = false;
        a2uiAbortController.value = null;
    }
};

const disconnectA2UIStream = () => {
    if (a2uiAbortController.value) {
        a2uiAbortController.value.abort();
        a2uiAbortController.value = null;
    }
    a2uiStreamActive.value = false;
};
```

#### F. Vue Template — A2UI Surface Block (in the Chat tab, below the existing `executionSteps` block)
Add this block **after** the existing `<div v-if="executionSteps.length > 0 && isExecutionInProgress">` block:
```html
<!-- A2UI Live Execution Surface (from SSE stream endpoint) -->
<div v-if="a2uiActiveSurfaceId || a2uiStreamActive" class="mt-2 a2ui-surface">
    <div class="text-[10px] font-semibold text-indigo-600 mb-2 flex items-center gap-1">
        <span class="material-icons" style="font-size:13px;">smart_toy</span>
        {{ a2uiTitle || 'Agent Execution' }}
        <span v-if="a2uiStreamActive" class="ml-auto">
            <span class="w-3 h-3 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin inline-block"></span>
        </span>
    </div>
    <!-- Step List -->
    <div v-if="a2uiActiveSteps.length > 0" class="space-y-1">
        <div v-for="step in a2uiActiveSteps" :key="step.id"
            :class="['execution-step', step.status]">
            <div class="a2ui-step-icon" :class="step.status">
                <span class="material-icons" style="font-size:16px;">{{ step.icon || 'circle' }}</span>
            </div>
            <div class="step-content">
                <div class="step-name">{{ step.label }}</div>
                <div v-if="step.detail" class="step-type">{{ step.detail }}</div>
            </div>
            <div class="text-[9px] font-semibold ml-auto" :class="{
                'text-indigo-500': step.status === 'running',
                'text-emerald-600': step.status === 'completed',
                'text-red-500': step.status === 'failed',
                'text-amber-500': step.status === 'interrupted',
                'text-slate-400': step.status === 'pending' || step.status === 'skipped'
            }">{{ step.status }}</div>
        </div>
    </div>
    <!-- Final Output Card -->
    <div v-if="a2uiFinalOutput" class="a2ui-final-output">
        {{ a2uiFinalOutput }}
    </div>
</div>
```

#### G. Expose new refs/functions in the `return` block of `setup()`
Add to the existing return object:
```javascript
// A2UI SSE
a2uiSurfaces,
a2uiActiveSteps,
a2uiFinalOutput,
a2uiTitle,
a2uiStreamActive,
a2uiActiveSurfaceId,
connectA2UIStream,
disconnectA2UIStream,
processA2UILine,
```

### 11.4 Files Changed

| File | Change |
|---|---|
| `frontend/workflow_builder_ide.html` | ADD Material Icons CDN; ADD CSS for pending/interrupted/skipped/a2ui-surface; ADD Vue state + A2UI processor + SSE client; ADD A2UI step-list template block |

### 11.5 What Stays Unchanged

- `connectExecutionWebSocket()` and all WebSocket event handlers
- All existing `.execution-step` CSS variants
- `sendChatMessage()` — still uses `/workflows/chat/send` for workflow chat
- All other tabs (Nodes, Agents), canvas, inspector, HITL panels
