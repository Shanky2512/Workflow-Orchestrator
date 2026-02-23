# EchoAI Database Migration & Multi-Tenancy Plan

## Document Information
- **Created**: 2026-02-01
- **Last Updated**: 2026-02-03
- **Status**: Planning Phase (Revised v2)
- **Author**: Claude Code Analysis
- **Related Progress**: `documents/db_progress.md`
- **Related Cache Plan**: `documents/cache_plan.md`
- **Related Update Plan**: `documents/update_workflow_plan.md`

---

## 0. CRITICAL REVISION (2026-02-03)

### 0.1 Schema Simplification

The workflow table schema has been simplified. Instead of extracting many columns from the JSON, we now:
- Store **complete workflow JSON as-is** in `definition` JSONB column
- Only extract **key fields** as separate columns for indexing/querying
- When retrieved, JSON is **directly usable** by executor/CrewAI (no reconstruction)

**Simplified Schema:**
```sql
CREATE TABLE workflows (
    -- Key fields (for indexing/querying only)
    workflow_id     UUID PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    name            VARCHAR(255),
    status          VARCHAR(20) DEFAULT 'draft',

    -- Complete JSON (stored as-is, directly usable by CrewAI)
    definition      JSONB NOT NULL,

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deleted      BOOLEAN DEFAULT FALSE
);
```

**Retrieval Example:**
```sql
SELECT definition FROM workflows WHERE workflow_id = 'wf_xxx';
```

**Returns complete JSON (directly passable to executor):**
```json
{
  "workflow_id": "wf_xxx",
  "name": "RFP Response Workflow",
  "description": "...",
  "status": "draft",
  "version": "0.1",
  "execution_model": "hybrid",
  "agents": [{"agent_id": "agt_xxx", "name": "...", "role": "...", ...}],
  "connections": [{"from": "agt_1", "to": "agt_2"}],
  "state_schema": {"input": "string", "output": "string"},
  "human_in_loop": {"enabled": true, "review_points": ["agt_2"]},
  "metadata": {"created_at": "...", "created_by": "user_xxx", "canvas_layout": {...}}
}
```

### 0.2 Save Trigger Logic (IMPORTANT)

**NOT all saves go to database.** The trigger depends on the source action:

| Action | Source | Filesystem | Database |
|--------|--------|------------|----------|
| **Save Button** (workflow_builder_ide.html) | Draft folder | ✅ YES | ✅ YES |
| **Generate Workflow Button** | Temp folder | ✅ YES | ❌ NO |
| **Save Final** | Final folder | ✅ YES | ✅ YES |

**Flow Diagram:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    SAVE BUTTON (Draft)                          │
│                    workflow_builder_ide.html                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  POST /canvas/save    │
              │  save_as = "draft"    │
              └───────────┬───────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  Filesystem         │         │  PostgreSQL         │
│  draft/wf_xxx.json  │         │  workflows table    │
│  (full JSON)        │         │  definition = JSON  │
└─────────────────────┘         └─────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                GENERATE WORKFLOW BUTTON (Temp)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  POST /temp/save      │
              │  state = "temp"       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  Filesystem ONLY    │
              │  temp/wf_xxx.json   │
              └─────────────────────┘

              (NO database write)
```

### 0.3 Independent Storage (CRITICAL)

**Filesystem and Database are INDEPENDENT - no interrelation.**

| Storage | Purpose | Retention | Cleanup |
|---------|---------|-----------|---------|
| **Filesystem** | Working copy, import/export | Temporary | Can be deleted for space management |
| **Database** | Permanent storage, source of truth | Permanent | Never deleted (soft delete only) |

**Key Points:**
- Filesystem JSON can be cleaned up after some time (space management)
- Database JSON is permanent storage (source of truth for applications)
- They are independent copies - deleting filesystem does NOT affect database
- For application features, retrieve from database (permanent)
- For current working/export, use filesystem (temporary)

**Cleanup Strategy (Future):**
```
Filesystem cleanup job:
- Delete draft/*.json older than 30 days
- Delete temp/*.json older than 7 days
- final/*.json can be cleaned if DB has copy
- agents/*.json can be cleaned if DB has copy

Database:
- NEVER auto-delete
- is_deleted = TRUE for soft delete
- Permanent storage for application features
```

### 0.4 Agent Schema Simplification (Same Pattern as Workflow)

**Design Principle:** Store complete agent JSON as-is. Only extract key fields for indexing.

**Simplified Schema:**
```sql
CREATE TABLE agents (
    -- Key fields (for indexing/querying only)
    agent_id            UUID PRIMARY KEY,
    user_id             UUID NOT NULL REFERENCES users(user_id),
    name                VARCHAR(255) NOT NULL,

    -- Complete agent JSON (stored as-is, directly usable by CrewAI)
    definition          JSONB NOT NULL,

    -- Source tracking (which workflow last updated this agent)
    source_workflow_id  UUID,

    -- Timestamps
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deleted          BOOLEAN DEFAULT FALSE
);
```

**Removed columns (now in definition JSONB):**
- `role` - In JSON
- `description` - In JSON
- `icon` - In JSON
- `model_provider` - In JSON
- `model_name` - In JSON
- `tool_count` - In JSON

**Retrieval Example:**
```sql
SELECT definition FROM agents WHERE agent_id = 'agt_xxx';
```

**Returns complete JSON (directly passable to CrewAI/Agent Factory):**
```json
{
  "agent_id": "agt_xxx",
  "name": "Solution Designer",
  "role": "Autonomous AI agent",
  "description": "Complete the task as Solution Designer",
  "prompt": "Design tailored solution...",
  "icon": "🔶",
  "model": "claude-opus-4-20250514",
  "tools": [],
  "settings": {"temperature": 0.4, "max_token": 4000, ...},
  "permissions": {"can_call_agents": false, ...},
  "metadata": {"node_type": "Agent", "ui_layout": {...}, ...},
  "input_schema": [],
  "output_schema": []
}
```

### 0.5 Agent Independent Storage (Same as Workflow)

**Filesystem and Database agents are INDEPENDENT - no interrelation.**

| Storage | Location | Purpose | Retention |
|---------|----------|---------|-----------|
| **Filesystem** | `apps/storage/agents/*.json` | Working copy | Temporary (can be deleted) |
| **Database** | `agents` table | Source of truth | Permanent |

**Agent Save Flow (Current Phase):**
```
Agent Created/Updated
         │
         ▼
┌────────────────────┐
│  Save Agent Logic  │
└─────────┬──────────┘
          │
  ┌───────┴───────┐
  ▼               ▼
┌──────────┐  ┌──────────┐
│FILESYSTEM│  │ DATABASE │
│agt_xxx.  │  │ agents   │
│  json    │  │  table   │
└──────────┘  └──────────┘
     │              │
     │  INDEPENDENT │
     │  (no link)   │
     ▼              ▼
 Temporary      Permanent
 (cleanup)    (source of truth)
```

### 0.6 Future Route Migration (IMPORTANT)

**Current Phase (Dual-Write):**
- Routes READ from: Filesystem
- Routes WRITE to: Filesystem + Database (independent)

**Future Phase (DB-Only) - After Implementation:**
- Routes READ from: **Database ONLY**
- Routes WRITE to: **Database ONLY**
- Filesystem: **DEPRECATED** (can be removed entirely)

**Migration Path:**
```
PHASE 1 (Current):
┌─────────────────────────────────────────────────────────┐
│  Routes                                                  │
│    │                                                     │
│    ├── READ  → Filesystem (existing behavior)           │
│    │                                                     │
│    └── WRITE → Filesystem + Database (dual-write)       │
└─────────────────────────────────────────────────────────┘

PHASE 2 (Future - After DB Implementation):
┌─────────────────────────────────────────────────────────┐
│  Routes                                                  │
│    │                                                     │
│    ├── READ  → Database ONLY                            │
│    │                                                     │
│    └── WRITE → Database ONLY                            │
│                                                          │
│  Filesystem → DEPRECATED (cleanup for space)            │
└─────────────────────────────────────────────────────────┘
```

**What This Means:**
1. **Workflows**: Read/Write from `workflows` table only
2. **Agents**: Read/Write from `agents` table only
3. **Sessions**: Read/Write from `chat_sessions` table only
4. **Filesystem folders**: Can be deleted to free space (optional backup only)

### 0.4 Retrieval for Execution

When running a workflow, retrieve complete JSON from database:

```
User wants to run workflow
          │
          ▼
    Query Database
    SELECT definition FROM workflows WHERE workflow_id = ?
          │
          ▼
    Get complete JSON (as-is)
    {
      "workflow_id": "wf_xxx",
      "agents": [...],
      "connections": [...],
      ...
    }
          │
          ▼
    Pass directly to executor.execute_workflow()
          │
          ▼
    CrewAI receives same JSON structure

    NO RECONSTRUCTION NEEDED
```

---

## 1. EXECUTIVE SUMMARY

### 1.1 Objective
Transform EchoAI from a single-user, JSON-file-based prototype into a production-grade, multi-tenant AI workflow platform with:
- Strict user-level data isolation
- Persistent multi-chat sessions (ChatGPT-like experience)
- Real-time execution transparency via WebSocket
- PostgreSQL as the single source of truth
- Memcached as configurable session cache layer
- Dual-write strategy (filesystem + database) for workflows

### 1.2 Current State vs Target
| Aspect | Current | Target |
|--------|---------|--------|
| Storage | JSON files on filesystem | PostgreSQL + Filesystem (dual-write) |
| User Isolation | None (global access) | Strict per-user scoping |
| Authentication | JWT infrastructure exists, unused | JWT enforced on all endpoints |
| Sessions | Workflow-bound only | User-owned, multi-purpose |
| Chat Messages | Separate storage | Embedded JSON in session |
| Real-time Updates | None | WebSocket (ephemeral, no DB storage) |
| Session Caching | None | Memcached (configurable) |

### 1.3 Key Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | Local PostgreSQL (Docker) | Development flexibility, easy setup |
| Data Migration | Migrate existing JSON data | Preserve current workflows/agents |
| Authentication | Keep current JWT, enhance later | Faster implementation, SSO later |
| Real-time | WebSocket (ephemeral) | Per `transparent_plan.md`, no DB persistence |
| Chat Storage | Embedded JSON in session | LLM-friendly, simpler schema |
| Workflow Storage | Dual-write (file + DB) | Preserve import/export, enable DB queries |
| Agent Sync | UPSERT on workflow save | Same agent_id = same agent across updates |
| Session Cache | Memcached (configurable) | Fast LLM context retrieval |
| Tool Selection | Hybrid (IDs in session, config in separate table) | Flexibility for custom tools |

---

## 2. CURRENT SYSTEM ANALYSIS

### 2.1 Storage Layer (What Exists Today)

#### 2.1.1 Agent Storage
- **Location**: `apps/storage/agents/`
- **Format**: Individual JSON files per agent (`agt_<uuid>.json`)
- **Index**: `ai_agents.json` (master list for UI display)
- **Implementation**: `apps/agent/registry/registry.py` (AgentRegistry class)
- **Pattern**: In-memory cache + atomic file writes

```
apps/storage/agents/
├── ai_agents.json          # Master index
├── agt_02a1fff7333646a2.json
├── agt_0359b9ac2eeb4774.json
└── ... (50+ agent files)
```

#### 2.1.2 Workflow Storage
- **Location**: `apps/workflow/storage/workflows/`
- **Lifecycle States**: draft/, temp/, final/, archive/
- **Format**: `wf_<uuid>.<state>.json`
- **Implementation**: `apps/workflow/storage/filesystem.py` (WorkflowStorage class)
- **Key Insight**: Workflows contain **embedded agent definitions** (full copies, not references)

```
apps/workflow/storage/workflows/
├── draft/
│   ├── wf_20da2bedd561.draft.json
│   └── ...
├── temp/
│   └── ...
├── final/
│   └── wf_xxx.v1.0.json
└── archive/
```

#### 2.1.3 Session Storage
- **Location**: `apps/workflow/storage/sessions/`
- **Format**: `<session_uuid>.json`
- **Implementation**: `apps/workflow/runtime/chat_session.py` (ChatSessionManager class)
- **Binding**: Currently ONLY bound to workflow_id

#### 2.1.4 Tool Storage
- **Location**: `apps/storage/tools/`
- **Format**: Individual JSON + `tool_index.json`
- **Implementation**: `apps/tool/storage.py`

#### 2.1.5 HITL Checkpoint Storage
- **Location**: `apps/workflow/storage/hitl/`
- **Format**: `<run_id>_checkpoint.json`, `<run_id>_context.json`

### 2.2 Authentication (What Exists Today)

**File**: `echolib/security.py`

```python
# EXISTING IMPLEMENTATION
def create_token(sub: str, email: str, *, expires_minutes: int = 60) -> str
def decode_token(token: str) -> Optional[dict]
async def user_context(creds) -> UserContext  # FastAPI dependency

class UserContext(BaseModel):
    user_id: str
    email: str
```

**Current Status**:
- JWT token creation/validation: IMPLEMENTED
- HTTPBearer security scheme: IMPLEMENTED
- user_context dependency: IMPLEMENTED
- **Integration into routes: NOT DONE** (imported but not used)

### 2.3 Workflow JSON Structure (Reference)

Based on actual workflow data, workflows contain **embedded agent definitions**:

```json
{
  "workflow_id": "wf_0c16fe5da87046eaba1b77d84017fe35",
  "name": "RFP Response Workflow",
  "description": "Created from canvas on 2026-02-02",
  "status": "draft",
  "version": "0.1",
  "execution_model": "hybrid",
  "agents": [
    {
      "agent_id": "agt_88340f9ae971470cac811660ebf5a83f",
      "name": "RFP Received",
      "role": "Workflow entry point",
      "description": "Start node",
      "prompt": "",
      "icon": "▶️",
      "model": "claude-opus-4-20250514",
      "tools": [],
      "variables": [],
      "settings": {
        "temperature": 0.7,
        "max_token": 4000,
        "top_p": 0.9,
        "max_iteration": 5,
        "provider": "anthropic"
      },
      "permissions": {
        "can_call_agents": false,
        "allowed_agents": []
      },
      "metadata": {
        "node_type": "Start",
        "goal": "",
        "ui_layout": { "x": 100, "y": 300, "icon": "▶️", "color": "#10b981" },
        "created_at": "2026-02-02T16:53:08.084505"
      },
      "input_schema": ["rfp_document", "prospect_name", "deadline", "opportunity_value"],
      "output_schema": []
    }
    // ... more embedded agents
  ],
  "connections": [
    { "from": "agt_xxx", "to": "agt_yyy" }
  ],
  "state_schema": {
    "rfp_document": "string",
    "prospect_name": "string"
  },
  "metadata": {
    "created_by": "workflow_builder",
    "created_at": "2026-02-02T16:53:08.084505",
    "canvas_layout": { "width": 5000, "height": 5000 }
  },
  "human_in_loop": {
    "enabled": true,
    "review_points": ["agt_70da1fc282ce46fb84105ea87a727c8c"]
  }
}
```

---

## 3. GAP ANALYSIS

### 3.1 Data Isolation Gap

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Agents owned by user | No owner field | Add user_id to Agent model |
| Workflows owned by user | `created_by: "workflow_builder"` hardcoded | Add user_id, enforce in queries |
| Sessions owned by user | No owner field | Add user_id to ChatSession |
| Tools owned by user | No owner field | Add user_id to Tool |
| Query filtering | Returns all records | Filter WHERE user_id = ? |

### 3.2 Authentication Gap

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| JWT validation on requests | Middleware exists, not mounted | Mount in gateway |
| User extraction | Dependency exists, not used | Add to route parameters |
| 401 on invalid token | Only if dependency called | Enforce globally |

### 3.3 Storage Gap

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| PostgreSQL database | No database | Full implementation needed |
| Indexed queries | File scanning | Database indexes |
| Transactions | Atomic file writes | ACID transactions |
| Connection pooling | N/A | SQLAlchemy pool |

### 3.4 Session Model Gap

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| User-owned sessions | Workflow-bound | Decouple, add user_id |
| Multiple session types | Only workflow testing | Add context_type field |
| Session titles | None | Add title field |
| Session listing by user | List by workflow_id | List by user_id |
| Chat messages | Separate files | Embed as JSON in session |
| Tool selection per session | Not supported | Add selected_tool_ids array |

### 3.5 Execution Transparency Gap

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| Execution tracking | run_id generated only | Full execution table |
| Step-by-step events | None | **WebSocket only (ephemeral)** |
| Real-time updates | REST polling | WebSocket implementation |
| Status progression | Final status only | Event stream |

**Note**: Per `transparent_plan.md`, execution events are ephemeral (in-memory with 60s TTL). No `execution_events` table needed.

---

## 4. DATABASE SCHEMA DESIGN

### 4.1 Schema Overview (Revised)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     users       │     │     agents      │     │    workflows    │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ user_id (PK)    │◄────│ user_id (FK)    │     │ user_id (FK)    │────►│
│ email           │     │ agent_id (PK)   │     │ workflow_id (PK)│     │
│ display_name    │     │ name            │     │ name            │     │
│ created_at      │     │ definition      │     │ definition      │     │
│ metadata        │     │ (synced on save)│     │ (embedded agents)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                                │
        │               ┌─────────────────┐              │
        │               │  chat_sessions  │              │
        │               ├─────────────────┤              │
        └──────────────►│ user_id (FK)    │              │
                        │ session_id (PK) │              │
                        │ messages (JSONB)│ ◄── EMBEDDED │
                        │ selected_tool_ids│              │
                        │ context_id (FK) │──────────────┘
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐     ┌─────────────────┐
                        │session_tool_configs│   │   executions    │
                        ├─────────────────┤     ├─────────────────┤
                        │ session_id (FK) │     │ run_id (PK)     │
                        │ tool_id (FK)    │     │ workflow_id (FK)│
                        │ config_overrides│     │ user_id (FK)    │
                        └─────────────────┘     │ status          │
                                                └────────┬────────┘
                                                         │
                                                ┌────────▼────────┐
                                                │hitl_checkpoints │
                                                ├─────────────────┤
                                                │ checkpoint_id   │
                                                │ run_id (FK)     │
                                                │ state_snapshot  │
                                                └─────────────────┘

REMOVED TABLES:
- chat_messages (embedded in chat_sessions.messages)
- execution_events (ephemeral via WebSocket per transparent_plan.md)
```

### 4.2 Table Definitions

#### 4.2.1 users

Stores authenticated user accounts.

```sql
CREATE TABLE users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    display_name    VARCHAR(255),
    avatar_url      VARCHAR(500),
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at   TIMESTAMP WITH TIME ZONE,
    metadata        JSONB DEFAULT '{}',
    is_active       BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| user_id | UUID | Primary key, auto-generated |
| email | VARCHAR(255) | Unique email address |
| display_name | VARCHAR(255) | User's display name |
| avatar_url | VARCHAR(500) | Profile picture URL |
| created_at | TIMESTAMP | Account creation time |
| updated_at | TIMESTAMP | Last profile update |
| last_login_at | TIMESTAMP | Last successful login |
| metadata | JSONB | Extensible user preferences |
| is_active | BOOLEAN | Soft delete flag |

#### 4.2.2 agents (SIMPLIFIED - See Section 0.4)

Stores AI agent definitions. **Synced from workflow saves via UPSERT**.

**Design Principle:** Store complete agent JSON as-is. Only extract key fields for indexing.

```sql
CREATE TABLE agents (
    -- Key fields (for indexing/querying only)
    agent_id            UUID PRIMARY KEY,
    user_id             UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name                VARCHAR(255) NOT NULL,

    -- Complete agent JSON (stored as-is, directly usable by CrewAI)
    definition          JSONB NOT NULL,

    -- Source tracking (which workflow last updated this agent)
    source_workflow_id  UUID,

    -- Timestamps
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deleted          BOOLEAN DEFAULT FALSE,

    CONSTRAINT unique_agent_name_per_user UNIQUE (user_id, name)
);

CREATE INDEX idx_agents_user ON agents(user_id);
CREATE INDEX idx_agents_user_active ON agents(user_id) WHERE is_deleted = FALSE;
CREATE INDEX idx_agents_source_workflow ON agents(source_workflow_id);
CREATE INDEX idx_agents_name_search ON agents USING gin(to_tsvector('english', name));
```

**Removed Columns (now in definition JSONB):**
- `role` - Available in `definition->>'role'`
- `description` - Available in `definition->>'description'`
- `icon` - Available in `definition->>'icon'`
- `model_provider` - Available in `definition->'settings'->>'provider'`
- `model_name` - Available in `definition->>'model'`
- `tool_count` - Available via `jsonb_array_length(definition->'tools')`

**Note:** The `definition` JSONB column stores the EXACT same JSON that's embedded in workflows. No transformation. When retrieved, pass directly to CrewAI/Agent Factory.

**Sync Behavior**:
- When workflow is saved (draft/final), each embedded agent is UPSERTED by `agent_id`
- If `agent_id` exists → UPDATE the record
- If `agent_id` is new → INSERT new record
- `source_workflow_id` tracks which workflow last updated this agent

**Independent Storage:**
- Filesystem (`agents/*.json`) = Temporary working copy (can be deleted)
- Database (`agents` table) = Permanent source of truth
- No interrelation between them

**Future Route Migration:**
- Current: Routes read from filesystem, write to both
- Future: Routes read/write from DATABASE ONLY, filesystem deprecated

**Definition JSONB Structure** (complete agent JSON, directly usable):
```json
{
  "agent_id": "agt_xxx",
  "name": "RFP Analyzer",
  "role": "Autonomous AI agent",
  "description": "Complete the task as RFP Analyzer",
  "prompt": "Analyze RFP document comprehensively...",
  "icon": "🔶",
  "model": "claude-opus-4-20250514",
  "tools": [],
  "variables": [],
  "settings": {
    "temperature": 0.2,
    "max_token": 4000,
    "top_p": 0.9,
    "max_iteration": 5,
    "provider": "anthropic"
  },
  "permissions": {
    "can_call_agents": false,
    "allowed_agents": []
  },
  "metadata": {
    "node_type": "Agent",
    "goal": "Complete the task as RFP Analyzer",
    "ui_layout": { "x": 350, "y": 300, "icon": "🔶", "color": "#f59e0b" },
    "created_at": "2026-02-02T16:53:08.084505"
  },
  "input_schema": [],
  "output_schema": []
}
```

#### 4.2.3 workflows (SIMPLIFIED - See Section 0)

Stores workflow definitions with **complete JSON as-is** in definition column.

**Design Principle:** Store complete JSON directly usable by CrewAI/executor. Only extract key fields for indexing.

```sql
CREATE TABLE workflows (
    -- Key fields (for indexing/querying only)
    workflow_id     UUID PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name            VARCHAR(255) NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'draft',

    -- Complete workflow JSON (stored as-is, directly usable)
    definition      JSONB NOT NULL,

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deleted      BOOLEAN DEFAULT FALSE,

    CONSTRAINT valid_status CHECK (status IN ('draft', 'validated', 'final', 'archived'))
);

-- Note: 'temp' status NOT included - temp workflows go to filesystem only
CREATE INDEX idx_workflows_user ON workflows(user_id);
CREATE INDEX idx_workflows_user_status ON workflows(user_id, status);
CREATE INDEX idx_workflows_user_active ON workflows(user_id) WHERE is_deleted = FALSE;
CREATE INDEX idx_workflows_name_search ON workflows USING gin(to_tsvector('english', name));
```

**Note:** The `definition` JSONB column stores the EXACT same JSON that would be saved to filesystem. No transformation, no extraction. When retrieved, pass directly to executor.

**Selective Dual-Write Strategy** (See Section 0.2):
```
DRAFT SAVE (Save Button in workflow_builder_ide.html):
  1. Write JSON → filesystem (draft/ folder)            [EXISTING]
  2. Write JSON → workflows.definition (PostgreSQL)     [NEW]
  3. For each agent in workflow.agents[]:
     - UPSERT to agents table by agent_id               [NEW]

TEMP SAVE (Generate Workflow Button):
  1. Write JSON → filesystem (temp/ folder)             [EXISTING]
  2. NO DATABASE WRITE                                  [FILESYSTEM ONLY]

FINAL SAVE:
  1. Write JSON → filesystem (final/ folder)            [EXISTING]
  2. Write JSON → workflows.definition (PostgreSQL)     [NEW]
  3. Write JSON → workflow_versions table               [NEW]
  4. For each agent in workflow.agents[]:
     - UPSERT to agents table by agent_id               [NEW]
```

**Why temp doesn't go to DB:**
- Temp is for preview/testing before user commits
- Temp files are transient (can be cleaned up)
- Only committed workflows (draft/final) need permanent storage

**Definition JSONB Structure** (stores complete workflow with embedded agents):
```json
{
  "workflow_id": "wf_xxx",
  "name": "RFP Response Workflow",
  "description": "...",
  "status": "draft",
  "version": "0.1",
  "execution_model": "hybrid",
  "agents": [
    { "agent_id": "agt_xxx", "name": "...", "role": "...", ... }
  ],
  "connections": [
    { "from": "agt_1", "to": "agt_2" }
  ],
  "state_schema": { "input": "string", "output": "string" },
  "human_in_loop": { "enabled": true, "review_points": ["agt_2"] },
  "metadata": {
    "created_at": "...",
    "created_by": "user_xxx",
    "canvas_layout": { "width": 5000, "height": 5000 }
  }
}
```

#### 4.2.4 workflow_versions

Stores immutable workflow versions (final/archived states).

```sql
CREATE TABLE workflow_versions (
    id              SERIAL PRIMARY KEY,
    workflow_id     UUID NOT NULL REFERENCES workflows(workflow_id) ON DELETE CASCADE,
    version         VARCHAR(20) NOT NULL,
    definition      JSONB NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'final',

    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by      UUID REFERENCES users(user_id),
    notes           TEXT,

    CONSTRAINT unique_workflow_version UNIQUE (workflow_id, version),
    CONSTRAINT valid_version_status CHECK (status IN ('final', 'archived'))
);

CREATE INDEX idx_workflow_versions_workflow ON workflow_versions(workflow_id);
```

#### 4.2.5 chat_sessions (MODIFIED)

Stores user chat sessions with **embedded messages as JSON**.

```sql
CREATE TABLE chat_sessions (
    session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Session identity
    title           VARCHAR(255) NOT NULL DEFAULT 'New Chat',

    -- Context binding (flexible)
    context_type    VARCHAR(20) NOT NULL DEFAULT 'general',
    context_id      UUID,  -- workflow_id or agent_id depending on context_type

    -- State
    workflow_mode   VARCHAR(20),  -- For workflow contexts: draft/test/final
    workflow_version VARCHAR(20),

    -- EMBEDDED CHAT MESSAGES (LLM-friendly)
    messages        JSONB DEFAULT '[]',

    -- Tool selection (hybrid approach)
    selected_tool_ids UUID[] DEFAULT '{}',

    -- Session data
    context_data    JSONB DEFAULT '{}',
    variables       JSONB DEFAULT '{}',
    state_schema    JSONB DEFAULT '{}',

    -- Execution tracking
    run_ids         UUID[] DEFAULT '{}',

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Soft delete
    is_deleted      BOOLEAN DEFAULT FALSE,

    CONSTRAINT valid_context_type CHECK (context_type IN ('general', 'workflow', 'agent', 'workflow_design'))
);

CREATE INDEX idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_sessions_user_active ON chat_sessions(user_id, last_activity DESC) WHERE is_deleted = FALSE;
CREATE INDEX idx_sessions_context ON chat_sessions(context_type, context_id);
```

**Messages JSONB Structure**:
```json
{
  "messages": [
    {
      "id": "msg_abc123",
      "role": "user",
      "content": "Analyze this RFP document...",
      "timestamp": "2026-02-02T16:53:08.084505Z",
      "agent_id": null,
      "run_id": null,
      "metadata": {}
    },
    {
      "id": "msg_def456",
      "role": "assistant",
      "content": "I've analyzed the RFP. Here are the key requirements...",
      "timestamp": "2026-02-02T16:53:15.123456Z",
      "agent_id": "agt_80385fabf27b4422861b614cece7af26",
      "run_id": "run_xyz789",
      "metadata": {
        "tokens_used": 1250,
        "model": "claude-opus-4-20250514"
      }
    }
  ]
}
```

**Benefits of Embedded Messages**:
- Single query retrieves entire conversation
- LLM can receive whole chat context in one read
- Atomic session + messages save
- Simpler schema (no joins)
- Expected 50-200 messages per session (~200KB max)

**Context Types**:
| Type | Description | context_id |
|------|-------------|------------|
| `general` | Free-form chat, no binding | NULL |
| `workflow` | Testing a specific workflow | workflow_id |
| `agent` | Direct chat with an agent | agent_id |
| `workflow_design` | Iterating on workflow design | workflow_id |

#### 4.2.6 session_tool_configs (NEW)

Stores tool configuration overrides per session (hybrid approach).

```sql
CREATE TABLE session_tool_configs (
    session_id      UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    tool_id         UUID NOT NULL REFERENCES tools(tool_id) ON DELETE CASCADE,

    -- Configuration overrides
    config_overrides JSONB DEFAULT '{}',

    -- Metadata
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (session_id, tool_id)
);

CREATE INDEX idx_session_tool_configs_session ON session_tool_configs(session_id);
CREATE INDEX idx_session_tool_configs_tool ON session_tool_configs(tool_id);
```

**Hybrid Tool Storage Flow**:
```
1. User selects tools for session:
   chat_sessions.selected_tool_ids = [tool_1, tool_2]

2. User customizes tool config:
   INSERT INTO session_tool_configs (session_id, tool_id, config_overrides)
   VALUES (sess_xxx, tool_1, '{"max_results": 10}')

3. At runtime:
   SELECT t.*, stc.config_overrides
   FROM tools t
   LEFT JOIN session_tool_configs stc
     ON t.tool_id = stc.tool_id AND stc.session_id = ?
   WHERE t.tool_id = ANY(session.selected_tool_ids)
```

#### 4.2.7 tools

Stores tool definitions owned by users.

```sql
CREATE TABLE tools (
    tool_id         UUID PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    tool_type       VARCHAR(20) NOT NULL DEFAULT 'local',

    -- Full tool definition
    definition      JSONB NOT NULL,

    -- Status
    status          VARCHAR(20) DEFAULT 'active',
    version         VARCHAR(20) DEFAULT '1.0',

    -- Metadata
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_deleted      BOOLEAN DEFAULT FALSE,

    CONSTRAINT valid_tool_type CHECK (tool_type IN ('local', 'mcp', 'api', 'crewai', 'custom')),
    CONSTRAINT valid_tool_status CHECK (status IN ('active', 'deprecated', 'disabled'))
);

CREATE INDEX idx_tools_user ON tools(user_id);
CREATE INDEX idx_tools_user_active ON tools(user_id) WHERE is_deleted = FALSE AND status = 'active';
CREATE INDEX idx_tools_type ON tools(tool_type);
CREATE INDEX idx_tools_tags ON tools USING gin(tags);
```

#### 4.2.8 executions

Tracks workflow execution runs.

```sql
CREATE TABLE executions (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id     UUID NOT NULL REFERENCES workflows(workflow_id),
    user_id         UUID NOT NULL REFERENCES users(user_id),
    session_id      UUID REFERENCES chat_sessions(session_id),

    -- Execution info
    execution_mode  VARCHAR(20) NOT NULL,
    workflow_version VARCHAR(20),

    -- Status
    status          VARCHAR(30) NOT NULL DEFAULT 'queued',

    -- Input/Output
    input_payload   JSONB DEFAULT '{}',
    output          JSONB,
    error_message   TEXT,

    -- Timing
    started_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at    TIMESTAMP WITH TIME ZONE,

    -- Metrics
    duration_ms     INTEGER,
    agent_count     INTEGER,

    CONSTRAINT valid_exec_mode CHECK (execution_mode IN ('draft', 'test', 'final')),
    CONSTRAINT valid_exec_status CHECK (status IN (
        'queued', 'running', 'hitl_waiting', 'hitl_approved', 'hitl_rejected',
        'completed', 'failed', 'cancelled'
    ))
);

CREATE INDEX idx_executions_workflow ON executions(workflow_id);
CREATE INDEX idx_executions_user ON executions(user_id);
CREATE INDEX idx_executions_session ON executions(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_executions_status ON executions(status) WHERE status IN ('running', 'hitl_waiting');
CREATE INDEX idx_executions_user_recent ON executions(user_id, started_at DESC);
```

**Execution Status State Machine**:
```
queued → running → completed
              ↓          ↓
         hitl_waiting → hitl_approved → completed
              ↓
         hitl_rejected → completed (with rejection)
              ↓
            failed
              ↓
           cancelled
```

**Note**: Step-level execution events are handled by WebSocket (ephemeral, per `transparent_plan.md`). No `execution_events` table.

#### 4.2.9 hitl_checkpoints

Stores HITL checkpoint state for paused executions.

```sql
CREATE TABLE hitl_checkpoints (
    checkpoint_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES executions(run_id) ON DELETE CASCADE,

    -- State
    status          VARCHAR(30) NOT NULL DEFAULT 'waiting_for_human',
    blocked_at      UUID NOT NULL,  -- agent_id where paused

    -- Context snapshot
    agent_output    JSONB,
    tools_used      JSONB,
    execution_metrics JSONB,
    state_snapshot  JSONB,

    -- Decision tracking
    previous_decisions JSONB[] DEFAULT '{}',

    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at     TIMESTAMP WITH TIME ZONE,
    resolved_by     UUID REFERENCES users(user_id),
    resolution      VARCHAR(30),
    resolution_data JSONB,

    CONSTRAINT valid_hitl_status CHECK (status IN (
        'waiting_for_human', 'approved', 'rejected', 'modified', 'deferred'
    )),
    CONSTRAINT valid_resolution CHECK (resolution IN (
        'approve', 'reject', 'modify', 'defer', 'rerun'
    ) OR resolution IS NULL)
);

CREATE INDEX idx_hitl_run ON hitl_checkpoints(run_id);
CREATE INDEX idx_hitl_pending ON hitl_checkpoints(status) WHERE status = 'waiting_for_human';
```

### 4.3 Database Functions

#### 4.3.1 Updated At Trigger

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflows_updated_at BEFORE UPDATE ON workflows
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tools_updated_at BEFORE UPDATE ON tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_session_tool_configs_updated_at BEFORE UPDATE ON session_tool_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

#### 4.3.2 Session Last Activity Trigger

```sql
CREATE OR REPLACE FUNCTION update_session_last_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Triggered when messages JSONB is updated
    NEW.last_activity = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_session_activity BEFORE UPDATE OF messages ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_session_last_activity();
```

### 4.4 Initial Data Seeding

```sql
-- Default system user for migrated data
INSERT INTO users (user_id, email, display_name, metadata)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'admin@echoai.local',
    'System Admin',
    '{"role": "admin", "migrated": true}'
);
```

---

## 5. IMPLEMENTATION ARCHITECTURE

### 5.1 New Directory Structure

```
echoAI/
├── docker-compose.yml              # PostgreSQL + Memcached containers
├── alembic.ini                     # Alembic config
├── alembic/                        # Migrations
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 001_initial_schema.py
│       └── 002_seed_admin_user.py
│
├── echolib/
│   ├── database.py                 # SQLAlchemy setup
│   ├── cache.py                    # Memcached client (NEW)
│   ├── models/                     # ORM models
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user.py
│   │   ├── agent.py
│   │   ├── workflow.py
│   │   ├── session.py              # Includes embedded messages
│   │   ├── session_tool_config.py  # NEW
│   │   ├── tool.py
│   │   └── execution.py
│   ├── repositories/               # Data access layer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── user_repo.py
│   │   ├── agent_repo.py
│   │   ├── workflow_repo.py
│   │   ├── session_repo.py         # Handles embedded messages + cache
│   │   ├── tool_repo.py
│   │   └── execution_repo.py
│   ├── security.py                 # Add middleware
│   ├── config.py                   # Add DB + Memcached settings
│   └── websocket.py                # WebSocket manager (per transparent_plan.md)
│
├── apps/
│   ├── gateway/
│   │   └── main.py                 # Add auth middleware, WebSocket
│   ├── agent/
│   │   ├── registry/
│   │   │   └── registry.py         # REWRITE: Use repository
│   │   └── routes.py               # Add user_context
│   ├── workflow/
│   │   ├── storage/
│   │   │   └── filesystem.py       # MODIFY: Dual-write to DB
│   │   ├── runtime/
│   │   │   ├── chat_session.py     # REWRITE: Use repository + cache
│   │   │   └── executor.py         # Emit WebSocket events
│   │   └── routes.py               # Add user_context
│   ├── session/
│   │   └── routes.py               # Extend: User session APIs
│   └── tool/
│       └── storage.py              # REWRITE: Use repository
│
└── scripts/
    ├── migrate_json_to_db.py       # Data migration
    └── setup_db.py                 # Quick DB setup
```

### 5.2 Configuration Updates

**echolib/config.py additions**:
```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Database
    database_url: str = "postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Memcached (configurable)
    memcached_enabled: bool = False  # Disabled by default for dev
    memcached_hosts: str = "localhost:11211"
    memcached_ttl: int = 1800  # 30 minutes
    memcached_fallback: bool = True  # Fallback to PostgreSQL on failure

    # Auth
    auth_enforcement: str = "optional"  # "optional" | "required"

    # WebSocket
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
```

### 5.3 Repository Pattern

All data access goes through repositories with user scoping:

```python
class BaseRepository:
    async def create(self, user_id: str, data: dict) -> Model
    async def get_by_id(self, id: str, user_id: str) -> Optional[Model]
    async def list_by_user(self, user_id: str, limit: int, offset: int) -> List[Model]
    async def update(self, id: str, user_id: str, updates: dict) -> Model
    async def delete(self, id: str, user_id: str) -> bool
    async def exists(self, id: str, user_id: str) -> bool
```

**Key principle**: Every query includes `WHERE user_id = ?` for data isolation.

### 5.4 Workflow Save Logic (Dual-Write + Agent Sync)

```python
async def save_workflow(
    workflow_json: dict,
    filesystem_path: str,
    db_session: AsyncSession,
    user_id: str
) -> None:
    """Dual-write: filesystem + database + agent sync"""

    # 1. Write to filesystem (existing behavior - preserved)
    with open(filesystem_path, 'w') as f:
        json.dump(workflow_json, f)

    # 2. Write to PostgreSQL workflows table
    await upsert_workflow(db_session, user_id, workflow_json)

    # 3. Sync embedded agents to agents table
    for agent in workflow_json.get('agents', []):
        await upsert_agent(
            db_session,
            agent_id=agent['agent_id'],
            user_id=user_id,
            definition=agent,
            source_workflow_id=workflow_json['workflow_id']
        )

    await db_session.commit()
```

### 5.5 Session Cache Architecture

See `documents/cache_plan.md` for detailed Memcached implementation.

---

## 6. MIGRATION STRATEGY

### 6.1 Data Migration Script Logic

```
1. Connect to PostgreSQL
2. Create admin user (or use existing)
3. For each agent JSON file:
   a. Parse JSON
   b. Extract agent_id, name, role, description
   c. Store full JSON in definition column
   d. Set user_id = admin_user_id
   e. INSERT into agents table
   f. Log success/failure

4. For each workflow JSON file:
   a. Parse JSON
   b. Extract workflow_id, name, status, version
   c. Store full JSON in definition column (with embedded agents)
   d. Set user_id = admin_user_id
   e. INSERT into workflows table
   f. For each embedded agent, UPSERT to agents table
   g. If status = 'final', also insert into workflow_versions
   h. Log success/failure

5. For each session JSON file:
   a. Parse JSON
   b. Extract session_id, workflow_id, messages
   c. Set user_id = admin_user_id
   d. Determine context_type based on existing workflow_id
   e. Convert messages array to JSONB format
   f. INSERT into chat_sessions (with embedded messages)
   g. Log success/failure

6. For each tool JSON file:
   a. Parse JSON
   b. INSERT into tools table
   c. Log success/failure

7. Generate migration report
```

### 6.2 Rollback Strategy

If migration fails:
1. Database can be dropped and recreated
2. JSON files remain untouched (read-only migration)
3. Application can fall back to JSON storage via config flag

---

## 7. API CHANGES

### 7.1 New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sessions` | List user's chat sessions |
| POST | `/sessions` | Create new session |
| GET | `/sessions/{id}` | Get session with embedded messages |
| DELETE | `/sessions/{id}` | Delete session |
| PATCH | `/sessions/{id}/title` | Update session title |
| POST | `/sessions/{id}/tools` | Set selected tools for session |
| PUT | `/sessions/{id}/tools/{tool_id}/config` | Set tool config override |
| WS | `/ws/execution/{run_id}` | Real-time execution events |
| GET | `/executions` | List user's executions |
| GET | `/executions/{run_id}` | Get execution details |

### 7.2 Modified Endpoints

All existing endpoints add `user_id` filtering:

| Endpoint | Change |
|----------|--------|
| `GET /agents` | Filter by user_id |
| `POST /agents/*` | Set user_id on create |
| `GET /workflows/*` | Filter by user_id |
| `POST /workflows/*` | Set user_id on create, dual-write, agent sync |
| `GET /tools/*` | Filter by user_id |
| `POST /tools/*` | Set user_id on create |

### 7.3 Response Shape Compatibility

Existing response shapes preserved. New fields added:
- `user_id` added to agent/workflow/session responses
- Frontend can ignore new fields initially

---

## 8. TESTING STRATEGY

### 8.1 Unit Tests

| Component | Tests |
|-----------|-------|
| Repositories | CRUD operations with user scoping |
| Auth middleware | Token validation, user extraction |
| Session cache | Memcached hit/miss, fallback to DB |
| Workflow save | Dual-write, agent sync |

### 8.2 Integration Tests

| Test | Description |
|------|-------------|
| User isolation | User A cannot see User B's agents |
| Session persistence | Messages survive server restart |
| Agent sync | Workflow save updates agents table |
| Memcached fallback | Works when cache unavailable |
| Migration | JSON data correctly imported |

### 8.3 Manual Testing Checklist

- [ ] Create agent → appears in my list only
- [ ] Create workflow → appears in my list only
- [ ] Save workflow → agents table updated
- [ ] Reload workflow → same agent IDs preserved
- [ ] Start chat session → persists across page reload
- [ ] Messages embedded in session → single query retrieval
- [ ] Execute workflow → WebSocket shows live events
- [ ] Different user login → sees empty lists

---

## 9. RISK MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration data loss | Low | High | Keep JSON files, verify counts |
| Frontend breakage | Medium | Medium | Preserve response shapes |
| Performance regression | Low | Medium | Add indexes, Memcached cache |
| Auth breaking flows | Medium | High | Start with optional enforcement |
| Agent sync conflicts | Low | Medium | UPSERT by agent_id, last-write wins |
| Large message arrays | Low | Medium | Expected 50-200 messages max |

---

## 10. PHASE TIMELINE

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| 1 | Database foundation | None |
| 2 | Auth enforcement | Phase 1 |
| 3 | Repository layer | Phase 1 |
| 4 | Data migration | Phase 1, 3 |
| 5 | Session enhancement + Memcached | Phase 3, 4 |
| 6 | WebSocket transparency | Phase 3 |
| 7 | Testing & validation | Phase 4, 5, 6 |

---

## 11. SUCCESS CRITERIA

### 11.1 Functional

- [ ] All agents scoped to owner user
- [ ] All workflows scoped to owner user
- [ ] Workflow save syncs embedded agents to agents table
- [ ] Chat sessions persist with embedded messages
- [ ] Memcached caches hot sessions (when enabled)
- [ ] Execution events stream via WebSocket (ephemeral)
- [ ] Existing JSON data successfully migrated

### 11.2 Non-Functional

- [ ] API response time < 200ms for list operations
- [ ] Session retrieval < 50ms with Memcached hit
- [ ] WebSocket event latency < 100ms
- [ ] Zero data loss during migration
- [ ] Backward compatible API responses

---

## APPENDIX A: Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://echoai:echoai_dev@localhost:5432/echoai
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Memcached (see cache_plan.md for details)
MEMCACHED_ENABLED=false
MEMCACHED_HOSTS=localhost:11211
MEMCACHED_TTL=1800
MEMCACHED_FALLBACK=true

# Auth
AUTH_ENFORCEMENT=optional
JWT_SECRET=your-secret-key
JWT_ISSUER=echo
JWT_AUDIENCE=echo-clients

# WebSocket
WEBSOCKET_PING_INTERVAL=30
WEBSOCKET_PING_TIMEOUT=10
```

---

## APPENDIX B: Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: echoai-postgres
    environment:
      POSTGRES_USER: echoai
      POSTGRES_PASSWORD: echoai_dev
      POSTGRES_DB: echoai
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U echoai"]
      interval: 5s
      timeout: 5s
      retries: 5

  memcached:
    image: memcached:1.6-alpine
    container_name: echoai-memcached
    ports:
      - "11211:11211"
    command: memcached -m 256
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "11211"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

---

## APPENDIX C: Quick Reference

### Start Database & Cache
```bash
cd echoAI
docker-compose up -d
```

### Run Migrations
```bash
alembic upgrade head
```

### Migrate JSON Data
```bash
python scripts/migrate_json_to_db.py
```

### Start Server
```bash
uvicorn apps.gateway.main:app --reload --host 0.0.0.0 --port 8000
```

---

## APPENDIX D: Tables Summary

| Table | Status | Notes |
|-------|--------|-------|
| `users` | KEEP | User accounts |
| `agents` | KEEP | Synced from workflow saves |
| `workflows` | KEEP | Embedded agents, dual-write |
| `workflow_versions` | KEEP | Immutable snapshots |
| `chat_sessions` | MODIFIED | Embedded messages JSONB |
| `session_tool_configs` | NEW | Tool config overrides |
| `tools` | KEEP | User-owned tools |
| `executions` | KEEP | Run history |
| `hitl_checkpoints` | KEEP | HITL state |
| `chat_messages` | REMOVED | Embedded in session |
| `execution_events` | REMOVED | Ephemeral via WebSocket |
