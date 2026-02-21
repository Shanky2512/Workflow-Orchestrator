# Comprehensive Implementation Plan: Universal Integration Layer (API & MCP)

## 1. Executive Summary
This plan delivers a high-value **Universal Integration Layer** for the EchoAI workflow engine. It enables users to connect workflows to the outside world with the same ease and power found in platforms like **Leah AI** and **UnifyApps**.

**Core Philosophy:**
*   **Unified Experience:** Whether connecting to a local file system via MCP or a remote Salesforce API, the user experience is identical: drag, drop, configure.
*   **Zero-Friction Ad-hoc:** Users can "just call an API" without registering a connector first. This is the "Builder Mode".
*   **Robust Managed Connectors:** Reusable, secure connections with auth lifecycle management. This is the "Enterprise Mode".

## 2. Technical Requirements & Standards

### 2.1 Technology Stack
*   **Backend Framework:** **FastAPI** (Async, Type-safe, High performance).
*   **Workflow Engine:** **LangGraph** (Stateful, observable, graph-based execution).
*   **Data Validation:** **Pydantic V2** (Strict schema enforcement for connector configs).
*   **Orchestration:** Python 3.10+ AsyncIO.

### 2.2 Integration Standards (The "UnifyApps" Bar)
To meet the user's high standards, our implementation must support:
1.  **Dynamic Schema Resolution:** The system must understand what inputs an API needs (Headers, Body, Params) and expose them clearly.
2.  **Secret Management:** Secrets (API Keys, Bearer Tokens) must never be logged or exposed in client-side state.
3.  **Unified Error Handling:** A 429 from OpenAI and a connection error from a local MCP server should both surface as a standardized `WorkflowError` event.

### 2.3 Connector Logic Constraint
> [!IMPORTANT]
> **Constraint:** We will **NOT** modify the internal logic of `echolib.services.ConnectorManager` or its underlying classes (`APIConnector`, `MCPConnector`). We will **consume** these classes as strict dependencies, ensuring stability of the core connector codebase.

## 3. Detailed Architecture

### 3.1 The Two Modes of Execution

#### A. Managed Mode (The "Store" Experience)
*   **User Action:** User selects "Jira" from a palette of pre-configured tools.
*   **Configuration:** The node only stores `{ "connector_id": "jira-prod-01", "operation": "create_issue" }`.
*   **Runtime:** The system looks up the secure configuration from the DB and executes it.

#### B. Ad-hoc Mode (The "Builder" Experience)
*   **User Action:** User drags a generic "HTTP Request" or "MCP Call" node.
*   **Configuration:** The node stores the *full* configuration: `{ "url": "https://api.weather.gov...", "method": "GET", "auth": { ... } }`.
*   **Runtime:** The system instantiates a **Ephemeral Connector** in memory just for this single execution. It is disposed of immediately after.

### 3.2 Workflow Compiler Logic (`WorkflowCompiler`)

The compiler is the brain that converts the JSON definition into executable LangGraph code. It needs precise logic to handle these two modes.

**`_create_api_node` Logic:**

> [!IMPORTANT]
> `APIConnector.invoke()` is **synchronous**. Since our workflow nodes run inside an `async def`, we **must** wrap it with `asyncio.to_thread()` to avoid blocking the event loop.

```python
import asyncio

def _create_api_node(self, node_id: str, config: Dict[str, Any]):
    async def execute(state):
        connector_id = config.get("connector_id")
        
        if connector_id:
            # MANAGED MODE
            # APIConnector.invoke() is SYNC → wrap in thread
            api_manager = ConnectorManager().get_manager("api")
            result = await asyncio.to_thread(
                api_manager.invoke, connector_id, config.get("payload")
            )
        else:
            # AD-HOC MODE
            temp_config = ConnectorConfig(
                name=f"transient_{node_id}",
                base_url=config.get("base_url") or config.get("url"),
                auth=config.get("auth_config") or config.get("authentication")
            )
            connector = ConnectorFactory.create(temp_config)
            
            # HTTPConnector.execute() is SYNC → wrap in thread
            # Full signature: execute(method, endpoint, headers, query_params, body)
            result = await asyncio.to_thread(
                connector.execute,
                method=config.get("method", "GET"),
                endpoint=config.get("endpoint", "/"),
                headers=config.get("headers"),
                query_params=config.get("query_params"),
                body=config.get("body")
            )
            
        return {"result": result}
    return execute
```

**`_create_mcp_node` Logic:**

> [!IMPORTANT]
> `MCPConnector.invoke_async()` is **async**. There is **no sync** `invoke()` on MCP. Call it directly with `await`.

```python
def _create_mcp_node(self, node_id: str, config: Dict[str, Any]):
    async def execute(state):
        connector_id = config.get("connector_id")
        
        if connector_id:
            # MANAGED MODE
            # MCPConnector.invoke_async() is ASYNC → await directly
            mcp_manager = ConnectorManager().get_manager("mcp")
            result = await mcp_manager.invoke_async(
                connector_id, payload=config.get("payload")
            )
        else:
            # AD-HOC MODE
            from echolib.Get_connector.Get_MCP.http_script import HTTPMCPConnector
            # Construct ephemeral MCP connector and test (async)
            connector = HTTPMCPConnector(**config)
            result = await connector.test(payload=config.get("payload"))
            
        return {"result": result}
    return execute
```

### 3.3 Tool Registry Sync Logic (`ToolRegistry`)

To make connectors available to **Agents** (e.g., "AI, please check Jira"), we must synchronize them into the Tool system.

**The Sync Algorithm:**

> [!IMPORTANT]
> **Verified API:** Both `APIConnector.list()` (line 3314) and `MCPConnector.list()` (line 2257) exist and return `{"success": True, "count": N, "connectors": [...]}`.

1.  **Query**: Fetch all connectors via:
    ```python
    cm = ConnectorManager()
    api_result = cm.api.list()   # Returns {"success": True, "connectors": [...]}
    mcp_result = cm.mcp.list()   # Returns {"success": True, "connectors": [...]}
    all_connectors = api_result["connectors"] + mcp_result["connectors"]
    ```
2.  **Transform**: Convert each Connector definition into a `ToolDef`.
    *   *Input Schema*: Derived from the connector's `creation_payload.input_schema` or `example_payload`.
    *   *System Prompt*: "This tool allows you to interact with [Name]. Use it to [Description]."
3.  **Register**: Save these as `ToolDef` objects in the `ToolRegistry` with `type="api_connector"` or `type="mcp_connector"`.

## 4. Implementation Steps

### Phase 1: Workflow Runtime (The Engine)
*   [ ] **API Node Support**: Implement `_create_api_node` in `apps/workflow/designer/compiler.py`.
    *   Managed path: `await asyncio.to_thread(api_manager.invoke, connector_id, payload)` (sync → thread)
    *   Ad-hoc path: `ConnectorFactory.create(ConnectorConfig(...)).execute(method, endpoint, headers, query_params, body)` (sync → thread)
    *   *Verification*: Test with both a registered connector ID and a raw URL config.
*   [ ] **MCP Node Support**: Implement `_create_mcp_node` in `apps/workflow/designer/compiler.py`.
    *   Managed path: `await mcp_manager.invoke_async(connector_id, payload)` (native async)
    *   Ad-hoc path: `HTTPMCPConnector(**config)` + `await connector.test(payload)` (native async)
    *   *Verification*: Test with a local MCP server configuration.
*   [ ] **Code Node Support**: Implement `_create_code_node` for `type="Code"` execution.
*   [ ] **Self-Review Node Support**: Implement `_create_self_review_node` for `type="Self-Review"` validation.
*   [ ] **Dispatcher Routing**: Add node-type routing in the compiler's node iteration loop.
    *   The compiler does **NOT** have a `_create_agent_node` method. Routing must be added where nodes are iterated (inside `_compile_sequential`, `_compile_parallel`, `_compile_hybrid`).

### Phase 2: Tool System (The Bridge)
*   [ ] **Fix Sync Logic**: Modify `apps/tool/registry.py` → `sync_connectors_as_tools`.
    *   *Detail*: Replace `connector_manager.list()` with `cm.api.list()` + `cm.mcp.list()` aggregation.
    *   Both return `{"success": True, "connectors": [...]}` — iterate over `result["connectors"]`.
*   [ ] **Auto-Sync**: Ensure this sync happens on application startup so agents always have fresh tools.

### Phase 3: Agent Designer (The Brain)
*   [ ] **Tool Selection**: Update `apps/agent/designer/agent_designer.py` → `_select_tools_for_agent`.
    *   *Detail*: Update the search query to include `type:api_connector` and `type:mcp_connector`.

### Phase 4: Database Dual-Write Persistence (Multi-User Support)

> [!IMPORTANT]
> **Problem:** `ToolRegistry.register()` currently persists tools **only to filesystem** via `ToolStorage` (JSON files in `apps/storage/tools/`). The existing `ToolRepository` (PostgreSQL, `tools` table) is defined with full CRUD, user-scoping, GIN indexes, and soft-delete — but is **never used**.
>
> **Solution:** Add a dual-write path in `ToolRegistry.register()` so that every tool is persisted to **both** filesystem (backward-compatible) and database (multi-user, queryable).

**Key Constraint:** Do NOT remove or replace `ToolStorage` — add the DB write alongside it.

#### Files to modify:

##### [MODIFY] `apps/tool/registry.py` — `ToolRegistry.__init__()`
*   Accept optional `tool_repo: Optional[ToolRepository] = None` parameter — **direct import**, not DI (class is stateless).
*   Store as `self._tool_repo: Optional[ToolRepository]`.
*   Import `ToolRepository` directly from `echolib.repositories.tool_repo`.

##### [MODIFY] `apps/tool/container.py`
*   Pass `ToolRepository()` instance to `ToolRegistry` during construction.

##### [MODIFY] `apps/tool/registry.py` — `ToolRegistry.register()`
*   Add `user_id: Optional[str] = None` parameter. Defaults to `SYSTEM_USER_ID` when `None` (startup sync, CLI).
*   After `self.storage.save_tool(tool)` succeeds, schedule non-blocking DB write via `_schedule_db_write()`.
*   **Async Bridge (corrected):** `register()` is sync but runs inside FastAPI's event loop. `run_until_complete()` would crash. Use `loop.create_task()` for fire-and-forget:
    ```python
    def _schedule_db_write(self, coro):
        """Non-blocking, fail-safe DB write from sync context."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._safe_db_op(coro))
        except RuntimeError:
            # No event loop (startup/CLI) — spawn one
            threading.Thread(
                target=lambda: asyncio.run(self._safe_db_op(coro)),
                daemon=True
            ).start()

    async def _safe_db_op(self, coro):
        try:
            await coro
        except Exception as e:
            logger.warning(f"DB dual-write failed (non-blocking): {e}")
    ```
*   **UUID Conversion (corrected):** `Tool.tool_id` is `UUID`, but `ToolDef.tool_id` is `str` (e.g. `"tool_api_jira"`). Use deterministic UUID5:
    ```python
    import uuid
    TOOL_NS = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000000"

    def _tool_id_to_uuid(tool_id_str: str) -> uuid.UUID:
        return uuid.uuid5(TOOL_NS, tool_id_str)
    ```
*   **Field Mapping (corrected):** `ToolDef` has no `to_dict()` — use Pydantic V2 `model_dump()`:
    ```python
    tool_data = {
        "tool_id": _tool_id_to_uuid(tool.tool_id),  # UUID, not str
        "name": tool.name,
        "description": tool.description,
        "tool_type": tool.tool_type.value,
        "definition": tool.model_dump(),              # model_dump(), not to_dict()
        "status": tool.status or "active",
        "version": tool.version or "1.0",
        "tags": tool.tags or [],
    }
    ```
*   **Upsert Strategy:** `BaseRepository` has no native upsert. Use check-then-act with EAFP:
    ```python
    existing = await self._tool_repo.get_by_id(db, str(tool_uuid), user_id)
    if existing:
        await self._tool_repo.update(db, str(tool_uuid), user_id, tool_data)
    else:
        await self._tool_repo.create(db, user_id, tool_data)
    ```
*   Wrap in try/except — DB failure must NOT block registration (filesystem is primary).

##### [MODIFY] `apps/tool/registry.py` — `sync_connectors_as_tools()`
*   For idempotent re-sync: when a tool already exists in cache but the connector definition has changed, update both filesystem and DB.
*   Tag synced tools with `"synced"` so DB queries can filter synced vs. manually created tools.
*   Uses `SYSTEM_USER_ID` for all startup-synced tools (no HTTP context at boot).

##### [MODIFY] `apps/tool/registry.py` — `delete()`
*   Add `user_id: Optional[str] = None` parameter. Defaults to `SYSTEM_USER_ID`.
*   After filesystem delete succeeds, schedule non-blocking DB soft-delete via `_schedule_db_write()`.
*   Convert `tool_id` string to UUID via `_tool_id_to_uuid()` before calling `ToolRepository.delete()`.

#### DB Schema Compatibility (verified):
*   `tool_type` column has `CheckConstraint`: `IN ('local', 'mcp', 'api', 'crewai', 'custom')` — **`'api'` and `'mcp'` are already valid**.
*   `definition` is `JSONB` — stores full `ToolDef` dict, no schema change needed.
*   `tags` is `ARRAY(String)` with GIN index — supports `"synced"` tag filtering.
*   `user_id` is FK to `users` — enables multi-user tool isolation.
*   **No migration required.**

#### Verification:
*   Create a tool via `sync_connectors_as_tools` → verify entry exists in both `apps/storage/tools/tool_api_*.json` and `SELECT * FROM tools WHERE tool_type = 'api'`.
*   Delete a tool → verify soft-deleted in DB and removed from filesystem.
*   Two users syncing the same connector → verify each gets their own DB row with different `user_id`.

## 5. Verification & Quality Assurance

### Manual Verification Script (`verify_integration_layer.py`)
We will create a comprehensive script that simulates the user experience code-side:
1.  **Scenario A (The "Leah" Flow)**:
    *   Create a registered "Jira" connector (mocked).
    *   Create a workflow referencing it.
    *   Execute and assert valid result.
2.  **Scenario B (The "Unify" Builder Flow)**:
    *   Create a workflow with a raw "GET https://httpbin.org/get" node.
    *   Execute and assert valid result.
3.  **Scenario C (The Agent Flow)**:
    *   Ask Agent Designer: "Create a support agent that can check Jira".
    *   Assert that the "Jira" tool is selected.

This structured approach ensures we deliver the high-quality, flexible integration layer you requested.

## 6. Connector API Reference & Integration Logic

This section details how the workflow runtime interacts with the Connector system, based on the **MCP Connector API** documentation (`documents/connector_help/MCP Connector API.md`).

### 6.1 Invocation Strategy (Managed Mode)

> [!WARNING]
> **Async/Sync Mismatch:** `APIConnector.invoke()` is **synchronous**, while `MCPConnector.invoke_async()` is **asynchronous**. The compiler must handle both correctly.

#### MCP Connectors (Async)
*   **Endpoint Reference**: `POST /connectors/mcp/invoke`
*   **Internal Call**: `await ConnectorManager().mcp.invoke_async(connector_id, payload)`
*   **Payload Structure**:
    ```json
    {
      "connector_id": "mcp_...",
      "payload": { ... }
    }
    ```
*   **Behavior**: If `payload` is omitted, the system uses the validated `example_payload` stored with the connector.

#### API Connectors (Sync → Thread)
*   **Internal Call**: `await asyncio.to_thread(ConnectorManager().api.invoke, connector_id, payload)`
*   **Returns**: `{"success": bool, "status_code": int, "data": Any, "error": dict|None, "elapsed_seconds": float}`

### 6.2 Ad-hoc Strategy (Builder Mode)
Ad-hoc nodes mimic the **Create Connector** payload structure to instantiate ephemeral connectors.

#### For API Nodes (HTTPConnector)
*   **Construction**: Build `ConnectorConfig` from node data, then `ConnectorFactory.create(config)` → `HTTPConnector`
*   **Field Mapping**:
    *   `node.config.url` → `ConnectorConfig.base_url`
    *   `node.config.authentication` → `ConnectorConfig.auth` (dict with `type`, `token`, etc.)
    *   `node.config.method` → passed to `execute(method=...)`
*   **Execution**: `HTTPConnector.execute(method, endpoint, headers, query_params, body)` → returns `ExecuteResponse`
*   **Note**: This is **synchronous** — wrap with `asyncio.to_thread()`

#### For MCP Nodes (HTTPMCPConnector)
*   **Reference**: `POST /connectors/mcp/create` (Payload definition)
*   **Construction**: `HTTPMCPConnector(**normalized_config)` (see `MCPConnector.create()` for normalization)
*   **Execution**: `await connector.test(payload)` — this is **async**, no thread wrapping needed.
*   **Key**: We call `.test()` instead of `.save()` to execute without persisting.

### 6.3 Supported Authentication Types
Our implementation respects the strict auth types defined in the content:
*   `bearer`: Simple token auth.
*   `api_key`: Query or Header based injection.
*   `oauth2`: Full flow (handled via pre-configured tokens for Ad-hoc).
*   `custom_header`: For specialized internal systems.

## 7. Workflow Example Analysis (Proposal Generation)

This section contains an **Illustrative Example** of a full workflow JSON.
> [!NOTE]
> This JSON represents the *concept* of the workflow structure. It helps visualize how nodes connect. The actual runtime execution will parse this structure and map it to our internal API/Node logic as described in Section 3.

```json
{
  "title": "Proposal Generation Workflow",
  "description": "End-to-end RFP response with requirements analysis, solution design, and pricing optimization",
  "nodes": [
    {
      "id": 5000,
      "type": "Start",
      "name": "RFP Received",
      "icon": "▶️",
      "color": "#10b981",
      "x": 100,
      "y": 300,
      "status": "idle",
      "config": {
        "triggerType": "email",
        "inputVariables": [
          {"name": "rfp_document", "type": "string", "required": true},
          {"name": "prospect_name", "type": "string", "required": true},
          {"name": "deadline", "type": "string", "required": true},
          {"name": "opportunity_value", "type": "number", "required": true}
        ]
      }
    },
    {
      "id": 5001,
      "type": "Agent",
      "name": "RFP Analyzer",
      "icon": "🔶",
      "color": "#f59e0b",
      "x": 350,
      "y": 300,
      "status": "idle",
      "config": {
        "prompt": "Analyze RFP document comprehensively. Extract: Scope and objectives, Specific requirements (functional, technical, compliance), Evaluation criteria and weights, Timeline and milestones, Budget constraints, Mandatory vs. optional requirements, Submission requirements. Create structured requirement matrix.",
        "model": {
          "provider": "anthropic",
          "modelName": "claude-opus-4-20250514",
          "displayName": "Claude Opus 4"
        },
        "tools": ["Document Analysis"],
        "maxIterations": 5,
        "temperature": 0.2
      }
    },
    {
      "id": 5002,
      "type": "Agent",
      "name": "Win Probability Scorer",
      "icon": "🔶",
      "color": "#f59e0b",
      "x": 600,
      "y": 300,
      "status": "idle",
      "config": {
        "prompt": "Assess win probability and bid/no-bid decision factors. Evaluate: Alignment with firm capabilities, Relationship with prospect, Competitive positioning, Resource availability, Margin potential, Strategic value. Calculate win probability score (0-100). Recommend bid strategy.",
        "model": {
          "provider": "anthropic",
          "modelName": "claude-sonnet-4-5-20250514",
          "displayName": "Claude Sonnet 4.5"
        },
        "tools": ["Code Execution"],
        "temperature": 0.3
      }
    },
    {
      "id": 5003,
      "type": "Conditional",
      "name": "Pursue Opportunity?",
      "icon": "🔀",
      "color": "#8b5cf6",
      "x": 850,
      "y": 300,
      "status": "idle",
      "config": {
        "branches": [
          {"type": "if", "condition": "win_probability < 30 AND opportunity_value < 500000", "targetNodeId": 5004},
          {"type": "else", "condition": "", "targetNodeId": 5005}
        ]
      }
    },
    {
      "id": 5004,
      "type": "API",
      "name": "Decline & Notify",
      "icon": "🌐",
      "color": "#3b82f6",
      "x": 1100,
      "y": 150,
      "status": "idle",
      "config": {
        "method": "POST",
        "url": "https://api.crm.com/v1/opportunities/decline",
        "authentication": "bearer"
      }
    },
    {
      "id": 5005,
      "type": "Agent",
      "name": "Solution Designer",
      "icon": "🔶",
      "color": "#f59e0b",
      "x": 1100,
      "y": 400,
      "status": "idle",
      "config": {
        "prompt": "Design tailored solution addressing all RFP requirements. Create: Solution architecture and approach, Methodology and deliverables, Team composition and qualifications, Project plan with phases, Risk mitigation strategies, Innovation and value-adds. Map solution to evaluation criteria.",
        "model": {
          "provider": "anthropic",
          "modelName": "claude-opus-4-20250514",
          "displayName": "Claude Opus 4"
        },
        "tools": [],
        "maxIterations": 5,
        "temperature": 0.4
      }
    },
    {
      "id": 5006,
      "type": "API",
      "name": "Query Historical Projects",
      "icon": "🌐",
      "color": "#3b82f6",
      "x": 1350,
      "y": 300,
      "status": "idle",
      "config": {
        "method": "GET",
        "url": "https://api.projectdb.com/v1/similar-projects",
        "authentication": "api_key"
      }
    },
    {
      "id": 5007,
      "type": "Code",
      "name": "Pricing Calculator",
      "icon": "💻",
      "color": "#0ea5e9",
      "x": 1600,
      "y": 300,
      "status": "idle",
      "config": {
        "language": "python",
        "code": "# Calculate competitive pricing\nimport numpy as np\n\n# Resource-based costing\nhours_by_role = {'partner': 100, 'manager': 300, 'consultant': 800}\nrates = {'partner': 450, 'manager': 275, 'consultant': 175}\n\ncost = sum(hours_by_role[role] * rates[role] for role in hours_by_role)\nmargin = 0.25\nbase_price = cost / (1 - margin)\n\n# Adjust for win probability and competitive factors\nadjustment = 0.9 if win_probability < 50 else 1.0\nfinal_price = base_price * adjustment\n\nprint(f'Recommended price: ${final_price:,.0f}')",
        "packages": ["numpy"]
      }
    },
    {
      "id": 5008,
      "type": "Agent",
      "name": "Proposal Writer",
      "icon": "🔶",
      "color": "#f59e0b",
      "x": 1850,
      "y": 300,
      "status": "idle",
      "config": {
        "prompt": "Generate compelling proposal document. Structure: Executive Summary (value proposition, key differentiators), Understanding (restate client needs and challenges), Approach & Methodology, Team & Qualifications, Project Plan & Timeline, Pricing & Commercial Terms, Case Studies & References, Terms & Conditions. Use persuasive, client-focused language. Highlight competitive advantages.",
        "model": {
          "provider": "anthropic",
          "modelName": "claude-opus-4-20250514",
          "displayName": "Claude Opus 4"
        },
        "tools": ["Document Generation"],
        "temperature": 0.5
      }
    },
    {
      "id": 5009,
      "type": "Self-Review",
      "name": "Compliance Check",
      "icon": "✅",
      "color": "#06b6d4",
      "x": 2100,
      "y": 300,
      "status": "idle",
      "config": {
        "reviewPrompt": "Verify proposal meets all RFP requirements and submission guidelines. Check: All mandatory requirements addressed, Format and structure compliance, Page limits respected, Required attachments included, Signature and certification pages, Pricing format matches requirements.",
        "minConfidence": 0.9
      }
    },
    {
      "id": 5010,
      "type": "HITL",
      "name": "Partner Final Review",
      "icon": "👤",
      "color": "#06b6d4",
      "x": 2350,
      "y": 300,
      "status": "idle",
      "config": {
        "reviewTitle": "Proposal Final Approval",
        "priority": "high",
        "message": "Review proposal quality, pricing strategy, and competitive positioning",
        "allowedDecisions": ["approve", "request_revisions"],
        "notificationChannels": ["email"]
      }
    },
    {
      "id": 5011,
      "type": "API",
      "name": "Submit Proposal",
      "icon": "🌐",
      "color": "#3b82f6",
      "x": 2600,
      "y": 300,
      "status": "idle",
      "config": {
        "method": "POST",
        "url": "https://api.proposalportal.com/v1/submit",
        "authentication": "oauth2"
      }
    },
    {
      "id": 5012,
      "type": "End",
      "name": "Proposal Submitted",
      "icon": "⏹️",
      "color": "#64748b",
      "x": 2850,
      "y": 300,
      "status": "idle",
      "config": {
        "outputVariables": [
          {"name": "proposal_id", "type": "string"},
          {"name": "final_price", "type": "number"},
          {"name": "submission_timestamp", "type": "string"}
        ]
      }
    }
  ],
  "connections": [
    {"id": 6000, "from": 5000, "to": 5001},
    {"id": 6001, "from": 5001, "to": 5002},
    {"id": 6002, "from": 5002, "to": 5003},
    {"id": 6003, "from": 5003, "to": 5004},
    {"id": 6004, "from": 5003, "to": 5005},
    {"id": 6005, "from": 5004, "to": 5012},
    {"id": 6006, "from": 5005, "to": 5006},
    {"id": 6007, "from": 5006, "to": 5007},
    {"id": 6008, "from": 5007, "to": 5008},
    {"id": 6009, "from": 5008, "to": 5009},
    {"id": 6010, "from": 5009, "to": 5010},
    {"id": 6011, "from": 5010, "to": 5011},
    {"id": 6012, "from": 5011, "to": 5012}
  ]
}
```

### 7.1 Ad-hoc API Nodes (The "Builder" Logic)
The user's workflow contains specific nodes that require on-the-fly execution without prior registration.

*   **Node 5004 (Decline & Notify):**
    *   *Type:* `API`
    *   *Config:* `{"method": "POST", "url": "https://api.crm.com/v1/opportunities/decline", "authentication": "bearer"}`
    *   *Plan Mapping:* This confirms the need for our **Ad-hoc Mode**. Our `WorkflowCompiler` will instantiate an ephemeral `APIConnector` using this exact config structure.

*   **Node 5006 (Query Historical Projects):**
    *   *Type:* `API`
    *   *Config:* `{"method": "GET", "url": "...", "authentication": "api_key"}`
    *   *Plan Mapping:* Validates support for `api_key` authentication in Ad-hoc mode.

*   **Node 5011 (Submit Proposal):**
    *   *Type:* `API`
    *   *Config:* `{"method": "POST", "url": "...", "authentication": "oauth2"}`
    *   *Plan Mapping:* Validates need for robust OAuth2 token handling even in Ad-hoc scenarios.

### 7.2 Agent with Tools (The "Intelligence" Logic)

*   **Node 5001 (RFP Analyzer):**
    *   *Type:* `Agent`
    *   *Config:* `{"tools": ["Document Analysis"]}`
    *   *Plan Mapping:* This validates our **Tool Registry Sync**. Whether "Document Analysis" is a built-in tool or a custom "SharePoint Connector", our system will expose it uniformly to the Agent via the Registry.

### 7.3 Conclusion
The provided workflow example acts as a rigorous integration test case. Our "Universal Integration Layer" covers every requirement:
1.  **Ad-hoc Nodes**: Covered by Section 3.1.B.
2.  **Tool Usage**: Covered by Section 3.3.
3.  **Authentication**: Covered by Section 6.3.


## Current Problem:
  Here's the core problem and what needs to be fixed:    
                                                            
  When a user tries to create a new tool (e.g., "teams tool") from the agent builder frontend, the system loops on a hardcoded tool list (web
   search, file reader, code executor, etc.) and cannot:
  1. Dynamically fetch all available tools from the ToolRegistry (which includes connector-synced tools)
  2. Create a new tool from an existing connector during the agent-building flow

  Backend Gap

  The backend AgentDesigner (apps/agent/designer/agent_designer.py:29) has a static AVAILABLE_TOOLS list:
  AVAILABLE_TOOLS = ["tool_web_search", "tool_file_reader", "tool_code_generator", "tool_code_reviewer", "tool_calculator"]

  While Phase 4 added _search_connector_tools() for dynamic scoring, there's no API endpoint that returns the full available tools catalog   
  (built-in + connector-synced) in a format the frontend agent builder can consume for its tool selection UI.

  What's Needed

  1. New API endpoint (GET /agents/available-tools or similar) that returns all tools from ToolRegistry (built-in local tools + synced       
  connector tools) for the frontend's tool picker
  2. Make AVAILABLE_TOOLS dynamic — populate from ToolRegistry instead of a hardcoded list
  3. Add a "create tool from connector" flow — when the frontend user says "create me a teams tool", the backend should be able to generate a
   tool definition from an existing Teams connector and register it

  Frontend Reference Files

  - type-agent_builder.html — the agent builder UI (needs to call the new endpoint)
  - workflow_builder_ide.html — workflow builder (may also benefit)

  Now let me invoke the backend agent to implement this.

  ## Solution to this: 
  Changes Applied

  1. apps/agent/designer/agent_designer.py — 4 new methods added
  Method: get_available_tools()
  Lines: 463-501
  Purpose: Queries ToolRegistry for all tools (built-in + connector), falls back to static   
    list
  ────────────────────────────────────────
  Method: _build_tools_response_from_registry()
  Lines: 503-545
  Purpose: Transforms List[ToolDef] → frontend-friendly response with source:
    "builtin"|"connector"
  ────────────────────────────────────────
  Method: _build_tools_response_from_static()
  Lines: 547-582
  Purpose: Fallback response from hardcoded AVAILABLE_TOOLS
  ────────────────────────────────────────
  Method: create_tool_from_connector()
  Lines: 584-739
  Purpose: Creates a new tool from an existing connector (API/MCP), registers it in
    ToolRegistry
  ────────────────────────────────────────
  Method: _infer_json_type()
  Lines: 741-757
  Purpose: Helper for schema inference from example payloads
  2. apps/agent/routes.py — 2 new endpoints added
  Endpoint: GET /agents/available-tools
  Purpose: Returns full tool catalog (built-in + connector-synced) for the frontend tool     
    picker
  ────────────────────────────────────────
  Endpoint: POST /agents/create-tool-from-connector
  Purpose: Creates a tool from a registered connector — solves the "create me a teams tool"  
    use case
  Both endpoints are placed before the GET /{agent_id} catch-all route to prevent FastAPI    
  path collision.

  What This Solves

  - Before: Frontend showed a hardcoded 7-tool list, couldn't create new tools from
  connectors
  - After: Frontend can call GET /agents/available-tools to dynamically fetch all available  
  tools (built-in + any connector-synced tools), and POST /agents/create-tool-from-connector 
  to create a new tool from a Teams (or any other) connector on the fly

  ## How to Use

  1. Frontend Tool Picker
  Call GET /agents/available-tools to populate the tool selection dropdown in the agent builder
  ────────────────────────────────────────
  2. Create Tool from Connector
  When the user wants to create a tool from a connector (e.g., "create a teams tool"), send a POST request to:
  POST /agents/create-tool-from-connector
  Body:
  {
    "connector_id": "teams-connector-id",
    "tool_name": "Teams Calendar Tool",
    "tool_description": "Manages Teams meetings and calendars"
  }
  ────────────────────────────────────────
  3. Fallback
  If the registry is empty or unavailable, the API still returns the 7 built-in tools so the agent builder never breaks.

## Future Requirement:
  One Minor Note

  The existing POST /tools/register and DELETE /tools/{tool_id} routes don't currently pass user_id to the registry (they call
  get_registry().register(tool) and get_registry().delete(tool_id) without it). Since user_id defaults to SYSTEM_USER_ID, this works fine    
  today. When you later want user-scoped tools, you'd update these routes to inject user: UserContext = Depends(get_current_user) and pass   
  user.user_id — but that's a future concern, not part of this plan.
