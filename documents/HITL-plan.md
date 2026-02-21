# HITL Node & Celery Integration Plan

## Goal Description
Implement a production-grade Human-in-the-Loop (HITL) system using **Celery** for async orchestration and **LangGraph** for workflow management. The system must support **Approve**, **Reject**, **Edit** (Smart Replay), and **Defer** actions. This involves introducing Celery, upgrading state persistence to PostgreSQL, and fixing the current HITL node logic to correctly pause execution.

## User Review Required
> [!IMPORTANT]
> **Celery Introduction**: This plan introduces Celery using **Memcached** as the broker (leveraging existing infrastructure: `memcached://localhost:11211`).
> **State Persistence Upgrade**: Switching from `MemorySaver` to `PostgresSaver` is critical for pausing workflows.
> **Breaking Change**: The current `_create_agent_node` logic for HITL will be modified to support actual interruption.

## Proposed Changes

### Infrastructure Layer
#### [NEW] `celery_app.py`
- Initialize Celery application.
- Configure Broker: `memcached://localhost:11211` (using `pymemcache`).
- Configure Backend: Database (via `sqlalchemy`).
- Define task queues (default, workflows).

#### [NEW] `apps/workflow/tasks.py`
- `execute_workflow_task(run_id, workflow_id, input_data)`: Async wrapper for workflow execution.
- `resume_workflow_task(run_id, action, payload)`: Task to resume execution after human input.

### Runtime Layer (`apps/workflow/runtime/`)
#### [MODIFY] `executor.py`
- **Replace `checkpointer=MemorySaver()` with specific `PostgresSaver`** (using `echolib` DB connection).
- Update `execute_workflow` to support running in "background" (via Celery) vs "foreground".
- Add method to `resume_workflow(run_id, command)`.

#### [MODIFY] `hitl.py`
- Enhance `HITLManager` to integrate with the new persistent Checkpointer.
- Add `resume_workflow` trigger logic.

### Designer Layer (`apps/workflow/designer/`)
#### [MODIFY] `compiler.py`
- Update `_create_agent_node` (HITL logic):
    - **Extract UI config fields**:
        - `title`: Review title.
        - `message`: Instructions for the human.
        - `priority`: Task priority (Low/Medium/High/Critical).
        - `allowed_decisions`: List of enabled actions (Approve, Reject, Edit, Defer).
    - Pass these fields into `HITLManager.request_approval` context.
    - **Enforcement**: The `HITLManager` or API layer must validate that the user's action is actually enabled in `allowed_decisions`.
    - Interrupt graph using LangGraph native interruption (raise `node_interrupt` or similar).

### API Layer
#### [NEW] `apps/workflow/routes/hitl_routes.py` (or update `routes.py`)
- `POST /hitl/decide`: Endpoint for user actions.
    - **Approve**: Resumes execution with current state.
    - **Reject**:
        - Terminates workflow immediately.
        - Sets run status to `FAILED` or `REJECTED`.
        - Returns "Execution rejected by human" message.
    - **Edit (Smart Replay)**:
        - **Intelligent Resolution**:
            - **Scenario 1 (Future only)**: If edit only affects downstream → Resume next node.
            - **Scenario 2 (Immediate prev)**: If edit corrects immediate output → Re-execute previous node + continue.
            - **Scenario 3 (Chain correction)**: If edit changes data source (e.g., "use fresh MCP data") → Re-execute from that upstream node.
            - **Scenario 4 (Root change)**: If edit changes original goal → Restart workflow with **merged context**:
                - Original User Input
                - + Edit Prompt (Feedback)
                - + API/MCP Results (optionally cached or re-fetched)
                - The system MUST inject the edit prompt as "feedback" or "instruction" to prevent repeating the error.
        - **Implementation**:
            - `EditResolutionEngine`: Analyzes edit prompt vs node graph to determine `restart_from_node_id`.
            - `WorkflowReplayEngine`: triggering `resume_workflow_task(restart_from=..., context_merge=True)`.
    - **Defer**:
        - Updates HITL state to `DEFERRED`.
        - Does NOT resume workflow.
        - Celery task remains terminated (state is safe in DB).
        - Worker is released.

## Verification Plan

### Automated Tests
- **Unit Tests**: Test `HITLManager` state transitions and `EditResolutionEngine` logic.
- **Integration Tests**:
    1. Start workflow -> Celery Task -> Pauses at HITL.
    2. Check DB for paused state and Checkpoint.
    3. **Test Reject**: Call API -> Verify workflow status changes to REJECTED and execution stops.
    4. **Test Smart Edit**:
        - Case A: "Fix typo" -> Resumes next node.
        - Case B: "Retry analysis" -> Re-executes previous node.
        - Case C: "Change goal" -> Restarts workflow.
    5. **Test Defer**: Call API -> Verify status is DEFERRED and no execution happens.
    6. **Test Approve**: Call API -> Verify Celery Task resumes and completes.

### Manual Verification
- Deploy (ensure Memcached is running).
- Run "Human Review" workflow from UI.
- Verify "Waiting" status and UI fields (Title, Priority).
- Test all allowed decisions.


===========================================================================
==================Current Code Analysis (Post-Audit)========================

> **CONSTRAINT**: No refactoring or revamping of existing code. All changes are **additive only** — new functions, new classes, new files.

### Audit Corrections Applied

| # | Original Issue | Correction |
|---|----------------|------------|
| 1 | Plan said MemorySaver is in `executor.py` | **Actually in `compiler.py`** (5+ locations: lines 302, 471, 583, 787, 1287). We will NOT touch existing MemorySaver. Instead, add a **new wrapper/factory** that provides `PostgresSaver` for HITL-enabled workflows alongside the existing `MemorySaver` for non-HITL flows. |
| 2 | Memcached as Celery broker | **Celery does not support Memcached as broker**. Using **Redis** instead (open-source, free, already in `requirements.txt` line 64). |
| 3 | Missing `node_mapper.py` | Must add HITL config extraction (`hitl_config`) in `_convert_node_to_agent`. |
| 4 | No execution history for Smart Replay | Must add per-node execution history tracking (`{node_id, input, output}`) to state. |
| 5 | HITL routing implicit | Add explicit `elif "hitl"` branch in `_create_node_for_type`. |
| 6 | Current HITL doesn't pause | Add `NodeInterrupt`/`interrupt()` call after `request_approval`. |
| 7 | File persistence won't work across workers | DB persistence must be done BEFORE Celery integration. |

---

### Files To Create (NEW)

1. **`celery_app.py`** — Celery app init with **Redis** broker (`redis://localhost:6379/0`), DB backend.
2. **`apps/workflow/tasks.py`** — `execute_workflow_task`, `resume_workflow_task`.
3. **`apps/workflow/runtime/execution_history.py`** — Per-node input/output tracker for Smart Replay.
4. **`apps/workflow/runtime/edit_resolution.py`** — `EditResolutionEngine` + `WorkflowReplayEngine`.
5. **`apps/workflow/runtime/checkpointer.py`** — `PostgresSaver` wrapper (new file, does NOT modify existing `MemorySaver` usage).

### Files To Modify (ADDITIVE ONLY)

6. **`apps/workflow/designer/compiler.py`**
   - **DO NOT** touch existing `MemorySaver()` calls.
   - **ADD**: New method `_get_checkpointer(workflow)` that returns `PostgresSaver` if workflow has HITL nodes, else returns `MemorySaver()` (existing behavior preserved).
   - **ADD**: New method `_create_hitl_node(node_id, node_config)` — dedicated HITL node creation with `interrupt()` call.
   - **ADD**: Explicit `elif "hitl"` branch in `_create_node_for_type` (line ~2568).

7. **`apps/workflow/runtime/executor.py`**
   - **DO NOT** touch existing `execute_workflow` logic.
   - **ADD**: New method `resume_workflow(run_id, action, payload)` to handle HITL resume.
   - **ADD**: New method `execute_workflow_async(...)` that wraps execution via Celery task.

8. **`apps/workflow/runtime/hitl.py`**
   - **DO NOT** remove existing file-based persistence.
   - **ADD**: New class `HITLDBManager(HITLManager)` that extends `HITLManager` with DB persistence.
   - **ADD**: New method `trigger_resume(run_id, action)` to dispatch Celery resume task.

9. **`apps/workflow/visualization/node_mapper.py`**
   - **ADD**: HITL config extraction block inside `_convert_node_to_agent` (after existing MCP/API/Code blocks):
     ```python
     agent["metadata"]["hitl_config"] = {
         "title": config.get("title", ""),
         "message": config.get("message", ""),
         "priority": config.get("priority", "medium"),
         "allowed_decisions": config.get("allowed_decisions", ["approve","reject","edit","defer"])
     }
     ```
   - **ADD**: Promote to `agent["config"]` for HITL type (same pattern as API/Code/MCP fix).

10. **`apps/workflow/routes.py`**
    - **ADD**: New route `POST /workflows/hitl/decide` at the bottom of the file.
    - Existing routes are untouched.

### Dependency Changes

- **`requirements.txt`**: Uncomment `redis>=5.0.0` (line 64). Add `celery>=5.4.0`.

Yes, all 7 issues are fixed in the updated 
HITL-plan.md
. Here's the confirmation:

#	Issue	Status	How It's Fixed in Plan
1	MemorySaver location	✅ Fixed	Plan now says "DO NOT touch existing MemorySaver() calls" in 
compiler.py
. Adds new _get_checkpointer(workflow) factory method alongside it.
2	Memcached as broker	✅ Fixed	Changed to Redis (redis://localhost:6379/0). Uncomment redis>=5.0.0 in 
requirements.txt
.
3	Missing 
node_mapper.py
✅ Fixed	Added as item #9 — HITL config extraction (hitl_config) in 
_convert_node_to_agent
.
4	No execution history	✅ Fixed	New file apps/workflow/runtime/execution_history.py added (item #3 in new files).
5	HITL routing implicit	✅ Fixed	Explicit elif "hitl" branch to be added in 
_create_node_for_type
.
6	Pause doesn't work	✅ Fixed	New dedicated _create_hitl_node() method with interrupt() call.
7	File persistence	✅ Fixed	New HITLDBManager(HITLManager) class extends existing manager with DB persistence. Marked as prerequisite before Celery.
Key constraint honored: All changes are additive only — no existing code is modified or refactored.


===========================================================================
## UI Field Functional Data Flow (HITL Node Configuration)
===========================================================================

Each field from the HITL node config panel must be **fully functional** end-to-end:

### Field-by-Field Breakdown

| UI Field | Stored In | Consumed By | Functional Effect |
|----------|-----------|-------------|-------------------|
| **Review title** | `hitl_config.title` | API response → Reviewer UI | Displayed as header to the human reviewer when workflow pauses |
| **Message to reviewer** | `hitl_config.message` | API response → Reviewer UI | Instructions/context shown to the reviewer |
| **Priority** (Low/Medium/High/Critical) | `hitl_config.priority` | `HITLContext` → sorting/filtering | Controls urgency — used for notification priority and queue ordering |
| **Allowed Decisions** (checkboxes) | `hitl_config.allowed_decisions` | `POST /hitl/decide` enforcement | Only checked actions are permitted. Unchecked actions are **rejected by API**. |

### End-to-End Data Flow

```
Step 1: Frontend Canvas → User fills Review Title, Message, Priority, checks Allowed Decisions
         ↓ (SAVE CONFIG click)

Step 2: node_mapper._convert_node_to_agent()
         → Extracts hitl_config = { title, message, priority, allowed_decisions }
         → Stores in agent["metadata"]["hitl_config"]
         → Promotes to agent["config"] (same pattern as API/Code/MCP nodes)
         ↓

Step 3: Workflow saved to DB/JSON with hitl_config embedded in agent definition
         ↓

Step 4: compiler._create_hitl_node(node_id, node_config)
         → Reads hitl_config from node_config["config"] or node_config["metadata"]["hitl_config"]
         → Passes title, message, priority, allowed_decisions to HITLManager.request_approval(context)
         ↓

Step 5: HITLManager.request_approval()
         → Creates HITLCheckpoint with full context
         → Stores HITLContext (includes title, message, priority, allowed_decisions)
         → Sets state = WAITING_FOR_HUMAN
         ↓

Step 6: Graph PAUSES via interrupt()
         → Celery worker released
         → State persisted in DB (PostgresSaver)
         ↓

Step 7: GET /workflows/hitl/status/{run_id}
         → Returns { title, message, priority, allowed_decisions, state_snapshot }
         → Frontend renders: title as header, message as body, only enabled decision buttons
         ↓

Step 8: Reviewer clicks one of the allowed actions (e.g., "Approve")
         ↓

Step 9: POST /workflows/hitl/decide
         → Validates: action ∈ allowed_decisions (if Defer unchecked and user sends Defer → 403 Forbidden)
         → Records HITLDecision (audit trail)
         → Triggers resume_workflow_task or terminates (based on action)
```

### Priority Behavior Detail

| Priority | Effect |
|----------|--------|
| **Low** | Normal queue position, no notification urgency |
| **Medium** | Default, standard processing |
| **High** | Elevated in review queue, may trigger immediate notification |
| **Critical** | Top of queue, triggers urgent notification (email/webhook if configured) |


===========================================================================
## CRITICAL: Post-Audit Issue Fixes (A, C, D, G)
===========================================================================

> **SCOPE CLARIFICATION**: This HITL Node system is a **completely separate system** from
> the existing HITL used in prompt-based agent/workflow creation (`apps/appmgr/orchestrator/hitl.py`).
> That file handles conversation clarification flows (awaiting_input → awaiting_clarification).
> **Do NOT touch, import from, or conflate logic with `apps/appmgr/orchestrator/hitl.py`.**
> The HITL Node system lives entirely under `apps/workflow/runtime/hitl.py` and its new extensions.

---

### Issue A: `allowed_decisions` Field Mismatch

**Problem**:
The existing `node_mapper.py:469-476` stores HITL config as boolean flags:
```python
# CURRENT CODE (node_mapper.py line 469-476)
agent["metadata"]["hitl_config"] = {
    "title": config.get("title", ""),
    "message": config.get("message", ""),
    "priority": config.get("priority", "medium"),
    "allowEdit": config.get("allowEdit", True),    # <-- boolean
    "allowDefer": config.get("allowDefer", False)   # <-- boolean
}
```
But the plan references `allowed_decisions: ["approve","reject","edit","defer"]` (a list).
These are two different shapes. The plan's item #9 would overwrite existing storage format,
breaking any workflows already saved with the boolean schema.

**Solution**: Derive `allowed_decisions` at runtime. Keep existing booleans. Add a new
helper function that converts the boolean flags into the list format at consumption time.
This is additive — existing saved workflows continue to load correctly.

**Implementation (additive only)**:

1. **DO NOT modify** `node_mapper.py:469-476` (existing HITL config extraction).

2. **ADD** a new top-level utility function in `apps/workflow/runtime/hitl.py`:
```python
def derive_allowed_decisions(hitl_config: dict) -> list[str]:
    """
    Derive allowed_decisions list from hitl_config.

    Supports BOTH formats:
    - New format: hitl_config has "allowed_decisions" list directly
    - Legacy format: hitl_config has "allowEdit" / "allowDefer" booleans

    Approve and Reject are ALWAYS allowed (non-negotiable safety constraint).

    Args:
        hitl_config: The hitl_config dict from node metadata

    Returns:
        List of allowed action strings, e.g. ["approve", "reject", "edit", "defer"]
    """
    # If new format exists, use it directly
    if "allowed_decisions" in hitl_config:
        decisions = list(hitl_config["allowed_decisions"])
        # Enforce: approve and reject are always present
        for required in ("approve", "reject"):
            if required not in decisions:
                decisions.append(required)
        return decisions

    # Legacy boolean format → derive list
    decisions = ["approve", "reject"]  # Always allowed
    if hitl_config.get("allowEdit", True):
        decisions.append("edit")
    if hitl_config.get("allowDefer", False):
        decisions.append("defer")
    return decisions
```

3. **Consume** `derive_allowed_decisions()` in:
   - `_create_hitl_node()` in compiler.py (when building interrupt payload)
   - `POST /hitl/decide` route (when validating user's action)
   - `GET /hitl/status/{run_id}` route (when returning allowed actions to frontend)

4. **Frontend backward compatibility**: The `_extract_node_config` method at
   `node_mapper.py:1080-1081` already does `config.update(metadata.get("hitl_config") or {})`
   which will include the boolean fields. Frontend sees `allowEdit: true` and renders
   the Edit button. No frontend change required for legacy workflows.

5. **Forward compatibility**: If the frontend is later updated to send
   `allowed_decisions: ["approve","reject","edit"]` directly, the `derive_allowed_decisions()`
   function handles it transparently (checks for the key first).

**Validation rule for API enforcement**:
```python
# In POST /hitl/decide handler:
allowed = derive_allowed_decisions(hitl_config)
if action not in allowed:
    raise HTTPException(
        status_code=403,
        detail=f"Action '{action}' is not allowed. Permitted: {allowed}"
    )
```

---

### Issue C: HITLManager Stateless Across Requests

**Problem**:
Every route in `routes.py` instantiates `HITLManager()` fresh:
```python
# routes.py line 1402, 1469, 1584, etc.
hitl = HITLManager()  # NEW instance every request!
```
This means the in-memory `_checkpoints`, `_contexts`, `_states` dicts are rebuilt
from disk files on every single API call. This is:
- **Slow**: Disk I/O on every request
- **Race-prone**: Two concurrent requests could read stale state
- **Incompatible with DB persistence**: The new `HITLDBManager` must be a singleton

Meanwhile, `container.py:51` already creates a singleton:
```python
_hitl = HITLManager()
container.register('workflow.hitl', lambda: _hitl)
```
But the routes ignore it and create their own instances.

**Solution**: Use the DI container singleton everywhere. Routes must resolve
`workflow.hitl` from the container instead of instantiating directly.

**Implementation**:

1. **Modify `routes.py`**: Replace all `HITLManager()` direct instantiation with
   container resolution. Add a helper function at the top of the HITL routes section:

```python
# Add near the top of routes.py, alongside existing svc() helper (line 60-61)
def hitl_manager():
    """Get the singleton HITLManager (or HITLDBManager) from DI container."""
    return container.resolve('workflow.hitl')
```

2. **Replace every occurrence** in HITL routes (lines 1400-1705):
```python
# BEFORE (broken - creates new instance):
hitl = HITLManager()

# AFTER (correct - uses singleton):
hitl = hitl_manager()
```

   Affected routes (all in `routes.py`):
   - `get_hitl_status` (line ~1402)
   - `get_hitl_context` (line ~1431)
   - `approve_hitl` (line ~1469)
   - `reject_hitl` (line ~1515)
   - `modify_hitl` (line ~1565)
   - `defer_hitl` (line ~1618)
   - `list_pending_hitl_reviews` (line ~1663)
   - `get_hitl_decisions` (line ~1695)

3. **Update `container.py`** when `HITLDBManager` is introduced:
```python
# container.py — swap singleton to DB-backed manager (additive change)
# BEFORE:
# _hitl = HITLManager()

# AFTER:
from .runtime.hitl import HITLDBManager
_hitl = HITLDBManager()  # Extends HITLManager, adds DB persistence
container.register('workflow.hitl', lambda: _hitl)
```
   Because `HITLDBManager` extends `HITLManager`, all existing code that calls
   `.approve()`, `.reject()`, `.get_status()`, etc. works unchanged.

4. **Remove the `from .runtime.hitl import HITLManager` inside each route function body**.
   The import is no longer needed inline since the container handles it.

**Why this matters for the HITL Node**:
- The `_create_hitl_node()` in compiler.py will call `hitl.request_approval()` during
  graph execution. If that's a different instance than what the routes use, the route
  handlers will never find the checkpoint. The singleton guarantees one source of truth.
- When we move to `HITLDBManager`, the singleton ensures all reads/writes go through
  the same DB-backed instance with connection pooling.

---

### Issue D: `executor.py` Must Handle `interrupt()` Return Value

**Problem**:
The current executor at `executor.py:300` runs the graph synchronously:
```python
final_state = compiled_graph.invoke(initial_state, config)
```
When LangGraph hits an `interrupt()` call inside a node, `invoke()` returns immediately
with partial state that includes an `__interrupt__` field. The current code then treats
this as a completed execution and returns `status: "completed"` — which is wrong.

Per LangGraph docs (https://docs.langchain.com/oss/python/langgraph/interrupts):
- `interrupt()` pauses the graph and saves state via the checkpointer
- `invoke()` returns with `__interrupt__` in the result containing the interrupt payload
- To resume: call `graph.invoke(Command(resume=<value>), config)` with the **same thread_id**

**Solution**: Add interrupt-aware execution logic. The executor must detect `__interrupt__`
in the result and return `status: "interrupted"` instead of `"completed"`.

**Implementation (additive only — DO NOT modify existing `execute_workflow`)**:

1. **ADD** new method `execute_workflow_hitl` to `WorkflowExecutor` class in `executor.py`:

```python
def execute_workflow_hitl(
    self,
    workflow_id: str,
    execution_mode: str,
    version: str | None = None,
    input_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute a workflow with HITL interrupt support.

    Same as execute_workflow but:
    1. Detects __interrupt__ in invoke() return value
    2. Returns status="interrupted" with interrupt payload
    3. Stores the compiled_graph reference for later resume

    This method is used ONLY for workflows that contain HITL nodes.
    Non-HITL workflows continue using execute_workflow() unchanged.
    """
    # ... (same workflow loading logic as execute_workflow) ...

    try:
        # Run the compiled graph
        config = {"configurable": {"thread_id": run_id}}
        result = compiled_graph.invoke(initial_state, config)

        # CHECK: Did the graph pause at an interrupt?
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"]
            # Extract the interrupt payload (HITL context)
            # interrupt_data is a list of Interrupt objects
            interrupt_payload = None
            if interrupt_data and len(interrupt_data) > 0:
                interrupt_payload = interrupt_data[0].value

            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "interrupted",          # <-- NOT "completed"
                "execution_mode": execution_mode,
                "interrupt": interrupt_payload,    # HITL context for frontend
                "output": result,
                "messages": result.get("messages", [])
            }

        # Normal completion (no interrupt)
        return {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "completed",
            "execution_mode": execution_mode,
            "output": result,
            "messages": result.get("messages", [])
        }

    except Exception as e:
        # ... (same error handling as execute_workflow) ...
```

2. **ADD** new method `resume_workflow` to `WorkflowExecutor` class:

```python
def resume_workflow(
    self,
    workflow_id: str,
    run_id: str,
    action: str,
    payload: dict[str, Any] | None = None,
    execution_mode: str = "draft",
    version: str | None = None,
) -> dict[str, Any]:
    """
    Resume a workflow that was paused at a HITL interrupt.

    Uses LangGraph's Command(resume=...) to continue execution
    from the exact point where interrupt() was called.

    The resume value becomes the return value of interrupt() inside
    the HITL node function, so the node can read the human's decision.

    Args:
        workflow_id: Workflow identifier
        run_id: The original run_id (used as thread_id for checkpoint lookup)
        action: Human decision ("approve", "reject", "edit", "defer")
        payload: Additional data (edit content, rationale, etc.)
        execution_mode: "draft", "test", or "final"
        version: Version (for final mode)

    Returns:
        Execution result (may be another interrupt or final completion)
    """
    from langgraph.types import Command

    # Re-compile the graph (needed to get the same graph structure)
    # The checkpointer will restore state from the DB using thread_id=run_id
    workflow = self._load_workflow(workflow_id, execution_mode, version)
    agent_defs = self._load_agent_defs(workflow)
    compiled_graph = self.compiler.compile_to_langgraph(workflow, agent_defs)

    # Build resume command
    resume_value = {
        "action": action,
        "payload": payload or {},
    }

    config = {"configurable": {"thread_id": run_id}}

    try:
        # Resume from interrupt — Command(resume=...) becomes the return
        # value of the interrupt() call inside the HITL node
        result = compiled_graph.invoke(Command(resume=resume_value), config)

        # Check if we hit ANOTHER interrupt (chained HITL nodes)
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"]
            interrupt_payload = None
            if interrupt_data and len(interrupt_data) > 0:
                interrupt_payload = interrupt_data[0].value

            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "interrupted",
                "interrupt": interrupt_payload,
                "output": result,
                "messages": result.get("messages", [])
            }

        return {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "completed",
            "output": result,
            "messages": result.get("messages", [])
        }

    except Exception as e:
        return {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "status": "failed",
            "error": str(e),
            "output": {}
        }
```

3. **ADD** helper methods `_load_workflow` and `_load_agent_defs` to factor out
   the loading logic from `execute_workflow` (extract, don't modify):

```python
def _load_workflow(self, workflow_id: str, mode: str, version: str | None = None) -> dict:
    """Load workflow definition by mode. Extracted from execute_workflow."""
    if mode == "draft":
        try:
            return self.storage.load_workflow(workflow_id=workflow_id, state="draft")
        except FileNotFoundError:
            return self.storage.load_workflow(workflow_id=workflow_id, state="temp")
    elif mode == "test":
        return self.storage.load_workflow(workflow_id=workflow_id, state="temp")
    elif mode == "final":
        if not version:
            versions = self.storage.list_versions(workflow_id)
            if not versions:
                raise ValueError(f"No final versions for '{workflow_id}'")
            version = versions[-1]
        return self.storage.load_workflow(workflow_id=workflow_id, state="final", version=version)
    raise ValueError(f"Invalid mode: {mode}")

def _load_agent_defs(self, workflow: dict) -> dict:
    """Load agent definitions from workflow. Extracted from execute_workflow."""
    agent_defs = {}
    for entry in workflow.get("agents", []):
        if isinstance(entry, dict):
            aid = entry.get("agent_id")
            if aid:
                agent_defs[aid] = entry
        elif isinstance(entry, str):
            agent = self.agent_registry.get_agent(entry)
            if agent:
                agent_defs[entry] = agent
    for aid in agent_defs:
        normalize_agent_config(agent_defs[aid])
    return agent_defs
```

4. **Update `_execute_workflow_with_transparency` in `routes.py`** (the background task):
   When a workflow has HITL nodes, call `execute_workflow_hitl` instead of `execute_workflow`.
   Detect `status == "interrupted"` and handle accordingly:

```python
# In _execute_workflow_with_transparency (routes.py line ~970):
result = await asyncio.to_thread(
    executor.execute_workflow_hitl,  # <-- use HITL-aware method
    workflow_id=session.workflow_id,
    execution_mode=session.workflow_mode,
    version=session.workflow_version,
    input_payload={...}
)

if result.get("status") == "interrupted":
    # Workflow paused at HITL node — notify frontend via WebSocket
    interrupt_payload = result.get("interrupt", {})
    if settings.transparency_enabled:
        # Emit a "hitl_waiting" event so the frontend shows the review UI
        await publish_hitl_waiting(run_id, interrupt_payload)
    return  # Do NOT add assistant message — workflow is paused
```

5. **Decision on when to use `execute_workflow_hitl` vs `execute_workflow`**:
   The caller (routes.py background task) should check if the workflow has HITL nodes:
```python
has_hitl = any(
    (a.get("metadata", {}).get("node_type") == "HITL")
    for a in workflow.get("agents", [])
    if isinstance(a, dict)
)
if has_hitl:
    result = executor.execute_workflow_hitl(...)
else:
    result = executor.execute_workflow(...)  # Existing path unchanged
```

**Key LangGraph API references** (from official docs):
```python
# Imports needed:
from langgraph.types import interrupt, Command

# Inside HITL node function:
human_decision = interrupt(hitl_payload)  # Pauses graph, returns to caller
# When resumed, human_decision = {"action": "approve", "payload": {...}}

# To resume from API handler:
result = compiled_graph.invoke(Command(resume=resume_value), config)
```

---

### Issue G: PostgresSaver Requires `psycopg` v3 (Not `psycopg2-binary`)

**Problem**:
`requirements.txt:65` has `psycopg2-binary>=2.9` which is the **psycopg2** adapter.
LangGraph's `PostgresSaver` requires **psycopg v3** (the modern async-capable driver)
via the `langgraph-checkpoint-postgres` package. These are different packages:
- `psycopg2-binary` = psycopg2 (legacy, sync-only, C adapter)
- `psycopg[binary]` = psycopg3 (modern, async+sync, pure Python + C speedups)

Per LangGraph docs: `pip install "psycopg[binary,pool]" langgraph-checkpoint-postgres`

The existing `echolib/database.py` uses `asyncpg` for SQLAlchemy async engine.
This is fine — `asyncpg` and `psycopg` can coexist. The PostgresSaver uses its own
direct connection via `psycopg`, not through SQLAlchemy.

**Solution**: Add the correct packages to `requirements.txt`. Keep `psycopg2-binary`
(other things may depend on it). Add `psycopg` and `langgraph-checkpoint-postgres` as
new entries.

**Implementation**:

1. **ADD to `requirements.txt`** (under the "Database & Caching" section):
```
# LangGraph HITL Checkpoint Persistence (requires psycopg v3)
langgraph-checkpoint-postgres>=2.0.0
psycopg[binary,pool]>=3.1.0
```

2. **Keep existing** `psycopg2-binary>=2.9` — it's used by `scripts/setup_postgres.py`
   and possibly other components. Do NOT remove it.

3. **Connection string format for PostgresSaver**:
   The existing `echolib/config.py:108-110` has:
   ```python
   database_url = 'postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai'
   ```
   PostgresSaver needs a **plain psycopg** connection string (no SQLAlchemy prefix):
   ```
   postgresql://echoai:echoai_dev@localhost:5432/echoai
   ```

4. **ADD** connection string derivation in `apps/workflow/runtime/checkpointer.py`:
```python
"""
HITL Checkpoint Persistence via LangGraph PostgresSaver.

Provides a PostgresSaver instance for HITL-enabled workflows.
Non-HITL workflows continue using MemorySaver (unchanged).

This module does NOT modify any existing MemorySaver usage in compiler.py.
"""
from echolib.config import settings


def get_hitl_checkpointer_uri() -> str:
    """
    Derive a plain PostgreSQL URI for PostgresSaver from the existing
    echolib database_url config.

    echolib uses: postgresql+asyncpg://user:pass@host:port/db
    PostgresSaver needs: postgresql://user:pass@host:port/db

    Returns:
        Plain PostgreSQL connection string
    """
    db_url = settings.database_url
    # Strip SQLAlchemy async driver prefix
    if "+asyncpg" in db_url:
        return db_url.replace("+asyncpg", "")
    if "+psycopg" in db_url:
        return db_url.replace("+psycopg", "")
    return db_url


def get_postgres_checkpointer():
    """
    Create and return a PostgresSaver instance for HITL checkpointing.

    MUST be used as a context manager:
        with get_postgres_checkpointer() as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)

    First-time setup: call checkpointer.setup() to create checkpoint tables.
    This is idempotent and safe to call multiple times.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    db_uri = get_hitl_checkpointer_uri()
    return PostgresSaver.from_conn_string(db_uri)
```

5. **Usage in `compiler.py`** (the `_get_checkpointer` factory method):
```python
def _get_checkpointer(self, workflow: dict):
    """
    Return the appropriate checkpointer for this workflow.

    - Workflows with HITL nodes → PostgresSaver (persistent, survives restarts)
    - All other workflows → MemorySaver() (existing behavior, unchanged)

    Args:
        workflow: Workflow definition dict

    Returns:
        A LangGraph checkpointer instance
    """
    from langgraph.checkpoint.memory import MemorySaver

    # Check if workflow has any HITL nodes
    has_hitl = False
    for agent in workflow.get("agents", []):
        if isinstance(agent, dict):
            node_type = agent.get("metadata", {}).get("node_type", "")
            if node_type == "HITL":
                has_hitl = True
                break

    if not has_hitl:
        return MemorySaver()  # Existing behavior preserved

    # HITL workflow → use PostgresSaver
    from apps.workflow.runtime.checkpointer import get_postgres_checkpointer
    return get_postgres_checkpointer()
```

6. **One-time setup**: Add a startup hook or migration step to create checkpoint tables:
```python
# In app startup (e.g., main.py or container.py):
# This is idempotent — safe to run on every startup
from apps.workflow.runtime.checkpointer import get_postgres_checkpointer
with get_postgres_checkpointer() as cp:
    cp.setup()  # Creates langgraph checkpoint tables if they don't exist
```

---

## Separation of Concerns: HITL Node vs HITL Prompt System

> **CRITICAL FOR IMPLEMENTING AGENT**: These are two COMPLETELY different systems.
> Never mix imports, classes, or logic between them.

| Aspect | HITL Node (This Plan) | HITL Prompt System (Existing) |
|--------|----------------------|-------------------------------|
| **File** | `apps/workflow/runtime/hitl.py` | `apps/appmgr/orchestrator/hitl.py` |
| **Purpose** | Pause workflow graph at a dedicated HITL node | Handle conversation clarification in app chat |
| **Trigger** | `interrupt()` call inside LangGraph node | Orchestrator sets `needs_clarification=true` |
| **State Machine** | RUNNING → WAITING_FOR_HUMAN → APPROVED/REJECTED/MODIFIED/DEFERRED | awaiting_input → awaiting_clarification → awaiting_input |
| **Persistence** | PostgresSaver (LangGraph checkpoint) + HITLDBManager | SQLAlchemy session `context_data` |
| **Resume** | `Command(resume=...)` via LangGraph | Merged prompt re-sent to orchestrator |
| **Routes** | `/workflows/hitl/*` | Handled internally by app orchestrator |
| **Container Key** | `workflow.hitl` | N/A (used internally by orchestrator) |

**Rules for the implementing agent**:
1. NEVER import from `apps.appmgr.orchestrator.hitl` in any workflow HITL code
2. NEVER import from `apps.workflow.runtime.hitl` in any appmgr code
3. The `HITLManager` class in `apps/workflow/runtime/hitl.py` is the ONLY class to extend
4. The `HITLDBManager` subclass goes in the SAME file (`apps/workflow/runtime/hitl.py`)
5. All new HITL node routes use `container.resolve('workflow.hitl')`, never direct instantiation

---

## Updated Dependency Changes Summary

**`requirements.txt` — Full list of changes**:
```
# UNCOMMENT (line 64):
redis>=5.0.0

# ADD (new entries):
celery>=5.4.0
langgraph-checkpoint-postgres>=2.0.0
psycopg[binary,pool]>=3.1.0

# KEEP UNCHANGED:
psycopg2-binary>=2.9   # Used by setup scripts, NOT by PostgresSaver
langgraph>=0.2.0       # Already present
sqlalchemy[asyncio]>=2.0.35  # Already present
asyncpg>=0.30.0        # Already present (echolib database engine)
```

---

## Updated Files To Modify (Revised)

Incorporating fixes A, C, D, G into the original file list:

| # | File | Changes | Issues Fixed |
|---|------|---------|-------------|
| 5 | `apps/workflow/runtime/checkpointer.py` [NEW] | `get_postgres_checkpointer()`, `get_hitl_checkpointer_uri()` | G |
| 6 | `apps/workflow/designer/compiler.py` | ADD `_get_checkpointer()`, `_create_hitl_node()`, `elif "hitl"` branch | G (checkpointer factory) |
| 7 | `apps/workflow/runtime/executor.py` | ADD `execute_workflow_hitl()`, `resume_workflow()`, `_load_workflow()`, `_load_agent_defs()` | D |
| 8 | `apps/workflow/runtime/hitl.py` | ADD `derive_allowed_decisions()`, `HITLDBManager` class | A, C |
| 9 | `apps/workflow/visualization/node_mapper.py` | **NO CHANGE** (existing HITL config at lines 469-476 is correct as-is) | A (keep booleans) |
| 10 | `apps/workflow/routes.py` | ADD `hitl_manager()` helper, replace all `HITLManager()` with `hitl_manager()`, handle `status=="interrupted"` in background task | C, D |
| 11 | `apps/workflow/container.py` | Swap `HITLManager()` → `HITLDBManager()` when DB persistence is added | C |
| 12 | `requirements.txt` | Add `langgraph-checkpoint-postgres`, `psycopg[binary,pool]`, uncomment `redis` | G |


===========================================================================
## Implementation Progress
===========================================================================

| # | Task | Status | File | Notes |
|---|------|--------|------|-------|
| 1 | requirements.txt | Done | `requirements.txt` | Uncommented redis>=5.0.0. Added celery>=5.4.0, langgraph-checkpoint-postgres>=2.0.0, psycopg[binary,pool]>=3.1.0. Kept psycopg2-binary unchanged. |
| 2 | checkpointer.py | Done | `apps/workflow/runtime/checkpointer.py` | New file. get_hitl_checkpointer_uri() strips +asyncpg from echolib DB URL. get_postgres_checkpointer() returns PostgresSaver. setup_checkpoint_tables() for idempotent table creation. |
| 3 | execution_history.py | Done | `apps/workflow/runtime/execution_history.py` | New file. Per-node input/output tracker stored in state["_execution_history"]. Functions: record_node_execution, get_execution_history, get_node_history, get_last_node_output, get_executed_node_ids. Safe serialization with truncation. |
| 4 | edit_resolution.py | Done | `apps/workflow/runtime/edit_resolution.py` | New file. EditResolutionEngine with resolve() method handles 4 scenarios: FUTURE_ONLY, IMMEDIATE_PREV, CHAIN_CORRECTION, ROOT_CHANGE. WorkflowReplayEngine builds replay payload with context merge and dispatches Celery tasks. |
| 5 | celery_app.py | Done | `celery_app.py` (project root) | New file. Celery app with Redis broker (redis://localhost:6379/0), DB backend (db+postgresql://). Task routing: workflow tasks to "workflows" queue. Configured for HITL: acks_late=True, prefetch_multiplier=1. |
| 6 | tasks.py | Done | `apps/workflow/tasks.py` | New file. execute_workflow_task delegates to executor.execute_workflow_hitl(). resume_workflow_task uses executor.resume_workflow(). Both resolve executor from DI container. |
| 7 | hitl.py (modify) | Done | `apps/workflow/runtime/hitl.py` | Added derive_allowed_decisions() supporting both boolean and list formats. Added HITLDBManager(HITLManager) subclass with DB persistence layer and trigger_resume() method for Celery dispatch. All existing code untouched. |
| 8 | compiler.py (modify) | Done | `apps/workflow/designer/compiler.py` | Added _get_checkpointer(workflow) factory returning PostgresSaver for HITL workflows, MemorySaver otherwise. Added _create_hitl_node() with LangGraph interrupt() call, transparency support, hitl_config extraction. Added explicit elif "hitl" branch in _create_node_for_type. Existing code untouched. |
| 9 | executor.py (modify) | Done | `apps/workflow/runtime/executor.py` | Added _load_workflow() and _load_agent_defs() helpers. Added execute_workflow_hitl() with __interrupt__ detection returning status="interrupted". Added resume_workflow() using Command(resume=value). Existing execute_workflow() untouched. |
| 10 | routes.py (modify) | Done | `apps/workflow/routes.py` | Added hitl_manager() helper using DI container. Replaced all 8 HITLManager() direct instantiations with hitl_manager() calls. Added POST /workflows/hitl/decide unified route with allowed_decisions validation and Celery dispatch. Updated _execute_workflow_with_transparency to detect HITL workflows, use execute_workflow_hitl, and handle interrupted status. |
| 11 | container.py (modify) | Done | `apps/workflow/container.py` | Imported HITLDBManager. Changed _hitl = HITLManager() to _hitl = HITLDBManager(). Drop-in replacement since HITLDBManager extends HITLManager. |

**Implemented by**: backend-python-dev agent
**Date**: 2026-02-17
