# BUG: API & Code & MCP Node Config Not Reaching Compiler

## Status: FIXED (API, Code, MCP) | PENDING (Self-Review, HITL, Map, Loop, Conditional)

**Fixed on:** 2026-02-16
**Files modified:**
- `apps/workflow/visualization/node_mapper.py` — `normalize_agent_config()` utility + top-level `config` promotion + MCP field expansion
- `apps/workflow/designer/compiler.py` — metadata fallback for pre-fix workflows
- `apps/workflow/runtime/executor.py` — normalization before compilation
- `echolib/repositories/workflow_repo.py` — normalization before DB write

---

## Problem Statement

When a workflow is saved from the frontend canvas and then executed, the **API node**, **Code node**, and **MCP node** fail with:

```
API node 'API Request' failed: ConnectorConfig base_url — Value error, Base URL must start with http:// or https:// [input_value='']
Code node 'Code Execution' failed: No code provided for Code node execution.
```

The user **has configured** both nodes on the frontend canvas (URL for API, Python code for Code), but the data is **not reaching the compiler** at execution time.

---

## Root Cause Analysis

### The Data Flow

```
Frontend Canvas → Save API (routes.py /canvas/save)
  → node_mapper.map_frontend_to_backend() → _convert_node_to_agent()
  → Filesystem JSON + Database JSONB (dual-write)
  → Executor loads from filesystem → builds agent_defs from embedded agents
  → compiler.compile_to_langgraph(workflow, agent_defs)
  → _create_node_for_type() dispatches to _create_api_node / _create_code_node / _create_mcp_node
```

### Where It Breaks: Two-layer mismatch

**Layer 1 — Storage location:** `node_mapper` stores config in `agent["metadata"]["*_config"]`, but the compiler reads from `agent.get("config", agent)`. No top-level `"config"` key exists.

**Layer 2 — MCP field names:** Even if promoted, MCP config has `serverName`/`toolName` but the compiler expects `endpoint_url`/`method`/`headers`/`connector_id`.

### The Mismatch Table (before fix)

| Node Type | Where node_mapper stored it | Where compiler looks for it | Result |
|-----------|----------------------------|---------------------------|--------|
| Code | `agent["metadata"]["code_config"]["code"]` | `agent.get("config", agent).get("code")` | `""` |
| API | `agent["metadata"]["api_config"]["url"]` | `agent.get("config", agent).get("url")` | `None` |
| MCP | `agent["metadata"]["mcp_config"]["serverName"]` | `agent.get("config", agent).get("endpoint_url")` | `None` |

---

## Fix Applied — Three layers of defense

### Layer 1: `node_mapper.py` — Source-of-truth fix (new saves)

#### 1a: `normalize_agent_config()` utility function (line 12-51)

Reusable, idempotent function that ensures any agent dict has a top-level `"config"` key derived from its `metadata.*_config`. Handles all 8 node types. Used by executor and repository.

```python
def normalize_agent_config(agent: Dict[str, Any]) -> Dict[str, Any]:
    if agent.get("config") and isinstance(agent["config"], dict) and len(agent["config"]) > 0:
        return agent
    metadata = agent.get("metadata") or {}
    node_type = metadata.get("node_type", "")
    if node_type == "Code":
        agent["config"] = metadata.get("code_config", {})
    elif node_type == "API":
        agent["config"] = metadata.get("api_config", {})
    elif node_type == "MCP":
        agent["config"] = metadata.get("mcp_config", {})
    # ... + Conditional, Loop, HITL, Map, Self-Review
    return agent
```

#### 1b: Expanded MCP metadata fields (line ~485-495)

The `mcp_config` now stores both frontend fields AND compiler-expected fields.

#### 1c: Top-level `"config"` key promotion in `_convert_node_to_agent()` (line ~511-519)

After all node-type metadata blocks, promotes `metadata.*_config` to `agent["config"]` for Code, API, MCP.

### Layer 2: `compiler.py` — Runtime fallback (old workflows)

Each node creator method falls back to `metadata.*_config` when the essential fields are missing from `config`:

- `_create_api_node()` (line 1722-1724): checks `url`, `base_url`, `connector_id`
- `_create_code_node()` (line 2029-2031): checks `code`
- `_create_mcp_node()` (line 1888-1890): checks `endpoint_url`, `connector_id`, `serverName`

### Layer 3: Executor + Repository — Persistent normalization

#### `executor.py` (line 210-213)

Normalizes ALL agent dicts immediately after loading, before compilation. Ensures the compiler always sees correct data regardless of source (filesystem, DB, or registry).

```python
from apps.workflow.visualization.node_mapper import normalize_agent_config

# After building agent_defs, before compilation:
for agent_id in agent_defs:
    normalize_agent_config(agent_defs[agent_id])
```

#### `workflow_repo.py` (line 128-131)

Normalizes embedded agents before every DB write via `save_with_agents()`. This means re-saving any old workflow automatically fixes the database JSONB.

```python
from apps.workflow.visualization.node_mapper import normalize_agent_config

# Before bulk_upsert:
for agent_entry in agents:
    if isinstance(agent_entry, dict):
        normalize_agent_config(agent_entry)
```

### How the three layers interact

| Scenario | Layer 1 (node_mapper) | Layer 2 (compiler) | Layer 3 (executor/repo) |
|----------|----------------------|---------------------|------------------------|
| New save from canvas | Sets `config` at save time | Not triggered | Normalizes before DB write |
| Run old workflow (no re-save) | N/A (saved before fix) | Fallback finds metadata | Executor normalizes before compile |
| Re-save old workflow | `_convert_node_to_agent` adds config | Not triggered | Fixes DB JSONB on write |
| Agent loaded from registry (string ID) | Depends on registration time | Fallback handles it | Executor normalizes |

---

## Remaining Work (TODO)

Compiler fallback for these node types (when their compiler methods are implemented):

| Node Type | Metadata Key | Compiler Method | normalize_agent_config | Compiler Fallback |
|-----------|-------------|-----------------|----------------------|-------------------|
| Code | `code_config` | `_create_code_node` | DONE | DONE |
| API | `api_config` | `_create_api_node` | DONE | DONE |
| MCP | `mcp_config` | `_create_mcp_node` | DONE | DONE |
| Self-Review | `review_config` | `_create_self_review_node` | DONE (in normalize fn) | PENDING |
| HITL | `hitl_config` | `_create_agent_node` | DONE (in normalize fn) | PENDING |
| Conditional | `branches` | `_create_conditional_node` | DONE (in normalize fn) | PENDING |
| Loop | `loop_config` | BFS traversal | DONE (in normalize fn) | PENDING |
| Map | `map_config` | BFS traversal | DONE (in normalize fn) | PENDING |

---

## Verification

After the fix:
1. **Run existing workflow without re-saving** — executor normalizes + compiler fallback handles it
2. **Re-save workflow from canvas** — node_mapper sets `config`, DB gets normalized on write
3. Confirm API node receives the URL and makes the HTTP call
4. Confirm Code node receives the code and executes it
5. Confirm MCP node receives endpoint_url/connector_id and connects
6. Confirm the full pipeline Start -> API -> Code -> Agent -> End completes with real data
