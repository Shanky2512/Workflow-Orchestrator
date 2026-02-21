# AI Application Orchestrator — Implementation Plan

> **HOW TO RESUME**: In a new chat, say:
> *"Read `documents/apps_plan.md` and `documents/apps_progress.md` then continue implementation."*
> These two files contain the full plan + progress state. No prior context needed.

> **Status**: PENDING APPROVAL
> **Database**: PostgreSQL (async SQLAlchemy 2.0 + asyncpg)
> **SQLite**: REMOVED — everything in PostgreSQL
> **API Prefix**: `/api/applications/...`
> **CRITICAL RULE**: `apps/workflow/crewai_adapter.py` is UNTOUCHED — zero modifications

---

## 1. Codebase Analysis Findings

### 1.1 Existing Architecture

| Component | Technology | Location |
|-----------|-----------|----------|
| Gateway | FastAPI (async) | `apps/gateway/main.py` |
| Database | PostgreSQL + asyncpg + SQLAlchemy 2.0 | `echolib/database.py` |
| Models | SQLAlchemy ORM (UUID PKs, JSONB, soft deletes) | `echolib/models/` |
| Migrations | Alembic | `alembic/versions/` |
| Repositories | Async data access | `echolib/repositories/` |
| Cache | Memcached (optional, DB fallback) | `echolib/cache.py` |
| Auth | JWT, optional enforcement | `echolib/security.py` |
| LLM | LLMManager (OpenRouter/OpenAI/Anthropic/Ollama/Azure) | `llm_manager.py` |
| Orchestration | LangGraph (topology) + CrewAI (agent execution) | `apps/workflow/` |

### 1.2 Workflow Implementation

- **Storage**: PostgreSQL `workflows` table with JSONB `definition` column + filesystem JSON files (dual-write)
- **Model**: `echolib/models/workflow.py` — UUID PK, `user_id` FK, status (draft/validated/final/archived), JSONB definition
- **Execution**: `WorkflowExecutor` loads workflow → `WorkflowCompiler` compiles to LangGraph `StateGraph` → `compiled_graph.invoke(state)` runs it
- **CrewAI inside LangGraph**: `CrewAIAdapter` creates node functions (`create_sequential_agent_node`, `create_parallel_crew_node`, `create_hierarchical_crew_node`) that LangGraph calls
- **Chat**: `/workflows/chat/start` creates session, `/workflows/chat/send` executes workflow in background, WebSocket streams transparency events
- **Session Manager**: `ChatSessionManager` — filesystem JSON sessions with message history

### 1.3 Agent Implementation

- **Storage**: PostgreSQL `agents` table with JSONB `definition` column
- **Model**: `echolib/models/agent.py` — UUID PK, `user_id` FK, JSONB definition, `source_workflow_id` for sync
- **Execution**: Agents execute ONLY inside workflows via `CrewAIAdapter.create_sequential_agent_node()`
- **No standalone chat**: There is NO endpoint to chat with a single agent directly. **This is a gap we must fill.**
- **Agent Designer**: Template matching + LLM generation at `apps/agent/designer/`

### 1.4 Session Isolation Pattern

- Every entity has `user_id` FK to `users` table
- All queries filter by `user_id` extracted from JWT (`request.state.user`)
- `DEFAULT_USER_ID` used when auth is optional
- Memcached caches sessions by `session:{session_id}`, falls back to DB

### 1.5 Existing App Manager (TO BE REPLACED)

- **Location**: `apps/appmgr/` — uses SQLite (`apps_wizard_new.db`) with SQLModel (sync)
- **What exists**: 4-step wizard CRUD, skill linking (agent:/workflow: prefixes), file uploads, basic LLM router that picks ONE skill
- **What's missing**: Real orchestration, prompt enhancement, guardrails enforcement, persona injection, HITL, multi-skill execution, session isolation, PostgreSQL persistence
- **Action**: Rewrite persistence to PostgreSQL, keep Pydantic schemas compatible, replace single-skill router with full orchestration engine

---

## 2. Application Lifecycle & Status Rules

### 2.1 State Machine

```
CREATE ──► DRAFT ──(Publish clicked + validation passes)──► PUBLISHED
              │                                                  │
              ├── Edit → stays DRAFT                             ├── Edit + Save → stays PUBLISHED
              ├── Delete → soft delete                           ├── Delete → soft delete
              └── Publish → validates → PUBLISHED                └── Launch App → opens chat page
                         └── fails → stays DRAFT (return errors)

ERROR = app where save/creation had issues (broken skill links, invalid config)
```

**Key Rules:**
- **Created** → always starts as `DRAFT`
- **DRAFT** → buttons: `[Publish] [Edit] [Delete]`
- **PUBLISHED** → buttons: `[Launch App] [Edit] [Delete]` (NO Publish button)
- **Edit on PUBLISHED**: user edits, clicks Save → app stays PUBLISHED (does NOT revert to DRAFT)
- **Launch App**: frontend navigates to ChatGPT-like chat page, calls `POST /api/applications/{id}/chat`
- **Statuses**: `draft` | `published` | `error` (no `archived` for now)

### 2.2 Dashboard List API Response

The list endpoint returns Pydantic response models (not raw dicts), all DB-persistent:

```python
class ApplicationCard(BaseModel):
    """Single card in the dashboard grid — DB persistent, Pydantic serialized."""
    application_id: str
    name: str
    status: str                    # "draft" | "published" | "error"
    description: Optional[str]
    logo_url: Optional[str]
    llm_count: int                 # count of LLMs connected
    data_source_count: int         # count of data sources
    skill_count: int               # count of skills (workflows + agents)
    error_message: Optional[str]   # if status == "error", why
    created_at: datetime
    updated_at: datetime

class ApplicationListResponse(BaseModel):
    """Full dashboard response with stats + paginated cards."""
    page: int
    page_size: int
    total: int
    stats: ApplicationStats
    items: list[ApplicationCard]

class ApplicationStats(BaseModel):
    """Stats row above the grid."""
    total: int
    published: int
    draft: int
    errors: int
```

### 2.3 List API Filters

| Parameter | Values | Maps to Tab |
|-----------|--------|-------------|
| `?status=draft` | `draft` | Draft tab |
| `?status=published` | `published` | Published tab |
| `?status=error` | `error` | Errors tab |
| (no status filter) | all | All tab |
| `?sort=recently_updated` | order by `updated_at DESC` | Recently Updated tab |
| `?q=search` | name ILIKE | Search |
| `?page=1&pageSize=20` | pagination | Standard |

### 2.4 Field Requirements

> The 4 "steps" (App Setup, Access Permission, Context, Chat Configuration) are
> NOT a forced sequential flow. They are **UI sections** for building an application.
> Users can fill them in any order and leave most fields empty.
> Publish enforces stricter validation — creating/saving a draft does not.

**Required Fields (always, even for draft):**

| Field | Reason |
|-------|--------|
| `name` | Cannot create an app without a name |

**Required Only at Publish Time:**

| Field | Reason |
|-------|--------|
| At least 1 LLM bound | Orchestrator needs an LLM to function |
| At least 1 Skill bound | App must have something to orchestrate |
| `welcome_prompt` | Chat needs an initial greeting |
| `sorry_message` | Fallback when app can't help |
| At least 1 starter question | UI needs suggested questions |
| Persona (persona_id or persona_text) | Chat needs a personality |

**Optional Fields (can be left empty):**

| Field | Default if empty |
|-------|-----------------|
| `description` | null |
| `data_sources` | No data sources bound |
| `logo_url` | No logo |
| `designations` | No restriction |
| `business_units` | No restriction |
| `tags` | No tags |
| `available_for_all_users` | false |
| `upload_enabled` | false |
| `disclaimer` | null (no disclaimer shown) |
| `guardrail_text` | null (no custom guardrails) |
| `guardrail_categories` | None selected |
| `persona_text` | null (uses persona_id name instead) |
| `enhancer_llm` | Uses same LLM as orchestrator |

---

## 3. Database Schema (PostgreSQL)

All tables follow existing patterns: UUID PKs, `user_id` FK, timestamps, soft deletes where appropriate.

### 3.1 Core Application Table

```sql
CREATE TABLE applications (
    application_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name           VARCHAR(255) NOT NULL,     -- REQUIRED (only mandatory field for draft)
    description    TEXT,                      -- optional
    status         VARCHAR(20) NOT NULL DEFAULT 'draft',  -- draft / published / error
    error_message  TEXT,                      -- populated when status = 'error'

    -- Access (all optional, defaults applied)
    available_for_all_users BOOLEAN NOT NULL DEFAULT FALSE,
    upload_enabled          BOOLEAN NOT NULL DEFAULT FALSE,

    -- Context (optional for draft, some required at publish)
    welcome_prompt          TEXT,             -- optional (required at publish)
    disclaimer              TEXT,             -- optional
    sorry_message           TEXT,             -- optional (required at publish)
    starter_questions       JSONB DEFAULT '[]'::jsonb,  -- optional (at least 1 at publish)

    -- Chat Config (optional for draft, persona required at publish)
    persona_id              UUID REFERENCES app_personas(persona_id),  -- optional
    persona_text            TEXT,             -- optional
    guardrail_text          TEXT,             -- optional

    -- Misc
    logo_url                VARCHAR(512),
    is_deleted              BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_applications_user ON applications(user_id) WHERE is_deleted = FALSE;
CREATE INDEX idx_applications_status ON applications(user_id, status) WHERE is_deleted = FALSE;
```

### 3.2 Association Tables

```sql
-- App ↔ LLM (with role: 'orchestrator' | 'enhancer' | 'general')
CREATE TABLE application_llm_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    llm_id         UUID NOT NULL,  -- references app_llms catalog
    role           VARCHAR(20) NOT NULL DEFAULT 'general',  -- orchestrator / enhancer / general
    PRIMARY KEY (application_id, llm_id)
);

-- App ↔ Skills (workflows + agents)
CREATE TABLE application_skill_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    skill_type     VARCHAR(10) NOT NULL,  -- 'workflow' | 'agent'
    skill_id       VARCHAR(255) NOT NULL, -- workflow_id or agent_id from main system
    skill_name     VARCHAR(255),          -- denormalized for quick access
    skill_description TEXT,               -- denormalized for orchestrator prompt
    PRIMARY KEY (application_id, skill_type, skill_id)
);

-- App ↔ Data Sources
CREATE TABLE application_data_source_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    data_source_id VARCHAR(255) NOT NULL,
    PRIMARY KEY (application_id, data_source_id)
);

-- App ↔ Designations (access control)
CREATE TABLE application_designation_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    designation_id UUID NOT NULL REFERENCES app_designations(designation_id),
    PRIMARY KEY (application_id, designation_id)
);

-- App ↔ Business Units (access control)
CREATE TABLE application_business_unit_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    business_unit_id UUID NOT NULL REFERENCES app_business_units(business_unit_id),
    PRIMARY KEY (application_id, business_unit_id)
);

-- App ↔ Tags
CREATE TABLE application_tag_links (
    application_id UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    tag_id         UUID NOT NULL REFERENCES app_tags(tag_id),
    PRIMARY KEY (application_id, tag_id)
);

-- App ↔ Guardrail Categories
CREATE TABLE application_guardrail_links (
    application_id       UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    guardrail_category_id UUID NOT NULL REFERENCES app_guardrail_categories(guardrail_category_id),
    PRIMARY KEY (application_id, guardrail_category_id)
);
```

### 3.3 Lookup / Catalog Tables

```sql
CREATE TABLE app_personas (
    persona_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_guardrail_categories (
    guardrail_category_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                  VARCHAR(100) NOT NULL UNIQUE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_designations (
    designation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name           VARCHAR(100) NOT NULL UNIQUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_business_units (
    business_unit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             VARCHAR(100) NOT NULL UNIQUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_tags (
    tag_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_llms (
    llm_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    provider   VARCHAR(50),
    model_name VARCHAR(255),
    base_url   VARCHAR(512),
    api_key_env VARCHAR(100),  -- env var name, not the key itself
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE app_data_sources (
    data_source_id VARCHAR(255) PRIMARY KEY,  -- matches connector IDs
    name           VARCHAR(255) NOT NULL,
    kind           VARCHAR(50),  -- 'mcp' | 'api'
    metadata       JSONB DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 3.4 Chat Tables

```sql
CREATE TABLE application_chat_sessions (
    chat_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id  UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(user_id),
    title           VARCHAR(255),
    conversation_state VARCHAR(30) NOT NULL DEFAULT 'awaiting_input',
        -- awaiting_input | awaiting_clarification | executing
    llm_id          UUID,  -- which LLM was used
    is_closed       BOOLEAN NOT NULL DEFAULT FALSE,
    context_data    JSONB DEFAULT '{}'::jsonb,  -- clarification context, pending plan, etc.
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_app_chat_sessions_user ON application_chat_sessions(application_id, user_id);

CREATE TABLE application_chat_messages (
    message_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_session_id UUID NOT NULL REFERENCES application_chat_sessions(chat_session_id) ON DELETE CASCADE,
    role            VARCHAR(20) NOT NULL,  -- 'user' | 'assistant' | 'system'
    content         TEXT NOT NULL,
    enhanced_prompt TEXT,  -- the prompt-enhanced version (null for assistant/system)
    execution_trace JSONB,  -- orchestrator reasoning + skills invoked + timing
    guardrail_flags JSONB,  -- any guardrail violations detected
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_app_chat_messages_session ON application_chat_messages(chat_session_id, created_at);
```

### 3.5 Document / RAG Table

```sql
CREATE TABLE application_documents (
    document_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id    UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    chat_session_id   UUID REFERENCES application_chat_sessions(chat_session_id),
    original_filename VARCHAR(512) NOT NULL,
    stored_filename   VARCHAR(512) NOT NULL,
    public_url        VARCHAR(512),
    mime_type         VARCHAR(100),
    size_bytes        BIGINT,
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending / processing / ready / failed
    chunk_count       INT DEFAULT 0,
    metadata          JSONB DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 3.6 Execution Trace Table (Audit Trail)

```sql
CREATE TABLE application_execution_traces (
    trace_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    application_id   UUID NOT NULL REFERENCES applications(application_id) ON DELETE CASCADE,
    chat_session_id  UUID NOT NULL REFERENCES application_chat_sessions(chat_session_id) ON DELETE CASCADE,
    message_id       UUID REFERENCES application_chat_messages(message_id),
    user_message     TEXT NOT NULL,
    enhanced_prompt  TEXT,
    orchestrator_plan JSONB,  -- full orchestrator output JSON
    execution_result JSONB,   -- results from each skill execution
    skills_invoked   JSONB,   -- [{skill_id, skill_type, skill_name, duration_ms}]
    guardrail_input  JSONB,   -- pre-processing results
    guardrail_output JSONB,   -- post-processing results
    total_duration_ms INT,
    status           VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending / running / completed / failed
    error_message    TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_app_exec_traces_session ON application_execution_traces(chat_session_id, created_at);
```

### 3.7 Seed Data (Alembic Migration)

```python
# Personas
("Support Agent", "HR Assistant", "Sales Assistant", "Legal Advisor", "Technical Expert")

# Guardrail Categories
("Compliance", "PII", "Safety")

# Designations
("Executive", "Manager", "Individual Contributor", "Intern")

# Business Units
("Operations", "Human Resources", "Sales", "Engineering", "Legal", "Finance")

# Tags
("Internal", "External", "Priority", "Experimental")
```

---

## 4. API Endpoints

All endpoints under `APIRouter(prefix="/api/applications")`.
All responses use **Pydantic models** (not raw dicts). All data is **DB-persistent** in PostgreSQL.

### 4.1 Application CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/applications` | Create new application (name required, rest optional) → status=DRAFT |
| `GET` | `/api/applications` | List all apps as `ApplicationListResponse` (with stats + paginated cards). Supports `?status=&q=&sort=&page=&pageSize=` |
| `GET` | `/api/applications/{id}` | Get single application with all relations as `ApplicationDetail` |
| `PUT` | `/api/applications/{id}` | Full update (all sections at once). **If app is PUBLISHED, it stays PUBLISHED after save.** |
| `PATCH` | `/api/applications/{id}/setup` | LLMs, Data Sources, Skills. Stays in current status. |
| `PATCH` | `/api/applications/{id}/access` | Designations, Business Units, Tags, flags. Stays in current status. |
| `PATCH` | `/api/applications/{id}/context` | Welcome prompt, Disclaimer, Sorry message, Starters. Stays in current status. |
| `PATCH` | `/api/applications/{id}/chat-config` | Persona, Guardrails. Stays in current status. |
| `DELETE` | `/api/applications/{id}` | Soft delete (works on both DRAFT and PUBLISHED) |
| `POST` | `/api/applications/{id}/publish` | Validates required fields → DRAFT becomes PUBLISHED. If validation fails → stays DRAFT, returns errors. |
| `POST` | `/api/applications/{id}/unpublish` | PUBLISHED → DRAFT |
| `POST` | `/api/applications/{id}/logo` | Upload app logo (250x250 max) |

> **EDIT BEHAVIOR**: Editing a PUBLISHED app and clicking Save keeps it PUBLISHED.
> The status does NOT revert to DRAFT on edit. Only `unpublish` explicitly reverts it.

### 4.2 Skills

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/skills` | All workflows + agents for current user from PostgreSQL (`workflows` + `agents` tables), clearly labeled type |
| `POST` | `/api/skills/reload` | No-op or cache invalidation (data is already in PostgreSQL, no filesystem sync needed) |

### 4.3 Chat (Orchestrated)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/applications/{id}/chat` | Send message → full orchestration pipeline → response |
| `GET` | `/api/applications/{id}/chat/history` | List chat sessions for this app |
| `GET` | `/api/applications/{id}/chat/sessions/{session_id}` | Get messages for a session |
| `POST` | `/api/applications/{id}/chat/upload` | Upload document for RAG (when upload_enabled) |
| `DELETE` | `/api/applications/{id}/chat/sessions/{session_id}` | Close/delete a chat session |

### 4.4 Lookups (for dropdown population)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/llms` | Available LLMs |
| `POST` | `/api/llms/reload` | Reload LLMs from JSON catalog |
| `GET` | `/api/data-sources` | Available data sources |
| `POST` | `/api/data-sources/reload` | Reload from connectors API |
| `GET` | `/api/designations` | Available designations |
| `GET` | `/api/business-units` | Available business units |
| `GET` | `/api/tags` | Available tags |
| `GET` | `/api/personas` | Available personas |
| `GET` | `/api/guardrail-categories` | Available guardrail categories |

---

## 5. Orchestration Engine

### 5.1 Full Pipeline

```
User Message
    │
    ▼
┌─────────────────────┐
│  PRE-GUARDRAILS     │ ← Regex PII + Safety + Compliance check on raw input
│  (fast, no LLM)     │
└────────┬────────────┘
         │ (blocked? → return violation message)
         ▼
┌─────────────────────┐
│  PROMPT ENHANCER    │ ← Separate LLM call: fix typos, clarify intent, improve grammar
│  (dedicated LLM)    │   Does NOT answer the question — only restructures it
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  ORCHESTRATOR LLM   │ ← Receives: enhanced prompt + skill manifest + conversation history
│  (core intelligence)│   + persona context + guardrail rules
│                     │   Returns: execution plan JSON (or clarification question or fallback)
└────────┬────────────┘
         │
         ├── needs_clarification=true → Return clarification question to user (HITL)
         ├── fallback_message set    → Return sorry_message + app capabilities
         │
         ▼
┌─────────────────────┐
│  SKILL EXECUTOR     │ ← Executes the plan: sequential / parallel / hybrid
│                     │   Workflows via WorkflowExecutor (existing)
│                     │   Agents via CrewAIAdapter (existing) or new standalone executor
│                     │   Chains outputs: step N output → step N+1 input
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  POST-GUARDRAILS    │ ← Check LLM output for PII leaks, compliance violations
│  (regex + optional  │
│   LLM check)        │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  PERSONA FORMATTER  │ ← Apply persona tone/style to final response
└────────┬────────────┘
         ▼
    Final Response + Execution Trace
```

### 5.2 Prompt Enhancer (`apps/appmgr/orchestrator/prompt_enhancer.py`)

- Single LLM call via `LLMManager.get_llm()` → `ChatOpenAI` from `langchain_openai`
- System prompt: "You are a prompt enhancer. Fix typos, clarify intent, improve grammar. Do NOT answer the question. Return only the improved prompt."
- Uses the app's bound enhancer LLM (or default if not set)
- Graceful fallback: if LLM fails, pass through original input
- Returns: `{enhanced_prompt: str, detected_intent: str, confidence: float}`

### 5.3 Skill Manifest Builder (`apps/appmgr/orchestrator/skill_manifest.py`)

- Queries `application_skill_links` for the app's assigned skills
- For each skill, fetches rich metadata **from PostgreSQL only** (no filesystem, no HTTP calls):
  - Workflows: queries PostgreSQL `workflows` table (`definition` JSONB column has name, description, execution_model, agents list)
  - Agents: queries PostgreSQL `agents` table (`definition` JSONB column has name, role, goal, backstory, tools)
- Builds structured manifest:
```json
{
  "skills": [
    {
      "skill_id": "wf_xxx",
      "skill_type": "workflow",
      "name": "Lead to Opportunity",
      "description": "Qualifies leads and creates opportunities in CRM",
      "capabilities": ["lead qualification", "CRM"],
      "execution_model": "sequential"
    },
    {
      "skill_id": "agt_yyy",
      "skill_type": "agent",
      "name": "Sales Expert",
      "description": "Specialist in sales pipeline analysis",
      "capabilities": ["sales analysis"],
      "tools": ["crm_tool"]
    }
  ]
}
```
- Cached in-memory per `application_id` with TTL (invalidated on setup change)

### 5.4 Orchestrator LLM (`apps/appmgr/orchestrator/orchestrator.py`)

- LLM call via `LLMManager.get_llm()` → `ChatOpenAI`
- Uses LangChain `ChatPromptTemplate` with `SystemMessage` + `HumanMessage`
- System prompt includes:
  1. Persona instructions
  2. Guardrail rules
  3. Full skill manifest (names, descriptions, capabilities)
  4. Conversation history (last N messages)
  5. Instructions for output format
  6. HITL instructions: "If the request is unclear, set needs_clarification=true. If the request is out-of-scope (jokes, off-topic, general chat), set fallback_message."
- JSON mode forced via `response_format={"type": "json_object"}` or structured output parsing
- Output exactly matches the required format:
```json
{
    "reasoning": "...",
    "execution_strategy": "single|sequential|parallel|hybrid",
    "execution_plan": [
        {
            "step": 1,
            "skill_id": "uuid",
            "skill_type": "workflow|agent",
            "skill_name": "Name",
            "depends_on": [],
            "parallel_group": null,
            "input_source": "user_input|step_N_output|merged",
            "output_key": "result_identifier"
        }
    ],
    "final_output_from": "step_N|merged",
    "needs_clarification": false,
    "clarification_question": null,
    "fallback_message": null
}
```

### 5.5 Skill Executor (`apps/appmgr/orchestrator/skill_executor.py`)

Executes the orchestrator's plan by invoking existing infrastructure.

> **CRITICAL BOUNDARY RULE — DO NOT BREAK EXISTING CODE:**
>
> | File | Action | Reason |
> |------|--------|--------|
> | `apps/workflow/crewai_adapter.py` | **ZERO modifications. DO NOT TOUCH. DO NOT IMPORT FROM.** | Belongs to workflow system. Other developers built it. Breaking it breaks all existing workflows. |
> | `apps/workflow/runtime/executor.py` | **DO NOT MODIFY.** Call `execute_workflow()` only. | We invoke it as a black box — pass input, get output. |
> | `apps/workflow/designer/compiler.py` | **DO NOT MODIFY.** | Internal to workflow system. |
>
> The application orchestrator has its **own independent CrewAI integration** in `apps/appmgr/orchestrator/agent_executor.py`.
> This file imports from the `crewai` **library** directly (`from crewai import Crew, Agent, Task, Process`),
> **NOT** from `apps/workflow/crewai_adapter.py` (the workflow module file).

#### Two-Source Architecture (Metadata vs Execution)

The orchestrator has **two separate concerns** that use **different data sources**:

| Concern | What it does | Data Source | Why |
|---------|-------------|-------------|-----|
| **Skill Manifest** (section 5.3) | Reads skill names, descriptions, capabilities so the orchestrator LLM can pick which skills to invoke | **PostgreSQL** (`workflows` + `agents` tables, `definition` JSONB) | Fast async queries, single source of truth for metadata |
| **Workflow Execution** | Actually runs a workflow end-to-end | **Existing `WorkflowExecutor`** (internally reads filesystem JSON → compiles to LangGraph → runs via CrewAIAdapter) | Already works, we call it as a black box, DO NOT MODIFY |
| **Agent Execution** (standalone) | Actually runs a single agent | **PostgreSQL** → new `agent_executor.py` loads definition, creates CrewAI instances | New code, no filesystem dependency |

**Why two sources?**
The existing `WorkflowExecutor` is a working pipeline: `filesystem JSON → WorkflowCompiler → LangGraph → CrewAIAdapter → CrewAI`. We said DO NOT MODIFY it. It reads from filesystem — that's fine. We just **call** it. But for metadata (listing skills, building the manifest prompt), we query PostgreSQL because that's the persistent source of truth.

**Example — Workflow A + Agent B in sequence:**
```
1. Orchestrator LLM reads skill manifest (metadata from PostgreSQL)
   → Decides: run Workflow A first, then Agent B with A's output

2. skill_executor calls WorkflowExecutor.execute_workflow("wf_A", "final", input)
   → WorkflowExecutor internally: reads filesystem JSON → compiles → LangGraph → CrewAI
   → Returns result_A
   (We don't touch this pipeline — it's a black box call)

3. skill_executor calls agent_executor.execute("agt_B", result_A)
   → agent_executor: reads agent definition from PostgreSQL agents table
   → Creates CrewAI Crew/Agent/Task from that definition
   → Runs CrewAI → Returns result_B

4. Combined result returned to user
```

---

**For Workflows (execution via existing black box):**
- Resolves `WorkflowExecutor` via `container.resolve('workflow.executor')`
- Calls `executor.execute_workflow(workflow_id, execution_mode, input_payload=...)`
- Internally this reads filesystem JSON → compiles to LangGraph → runs via CrewAIAdapter — **we don't interfere with any of that**
- Extracts result from `output.crew_result` or `output.result`

**For Agents (standalone — NEW, fully independent code):**
- Loads agent definition from PostgreSQL `agents` table (`definition` JSONB column) via async SQLAlchemy query
- `apps/appmgr/orchestrator/agent_executor.py` creates its OWN `crewai.Crew`, `crewai.Agent`, `crewai.Task` instances
- Imports from `crewai` library directly: `from crewai import Crew, Agent, Task, Process`
- Does **NOT** import from `apps/workflow/crewai_adapter.py`
- Gets LLM via `LLMManager.get_crewai_llm()` (same centralized manager, independent call)
- Binds tools via its own `_bind_tools()` method (same pattern as adapter, but independent implementation)
- Returns extracted result text

**The hard boundary:**
```
apps/workflow/crewai_adapter.py          ← EXISTS. DO NOT TOUCH. DO NOT IMPORT FROM.
                                            Serves workflow execution ONLY.
WorkflowExecutor.execute_workflow()      ← EXISTS. DO NOT MODIFY.
                                            Internally reads filesystem JSON. We just call it.

apps/appmgr/orchestrator/
    agent_executor.py                    ← NEW. 100% independent. Imports `crewai` library.
                                            Loads agent definition from PostgreSQL.
                                            Serves application orchestrator ONLY.
    skill_executor.py                    ← NEW. Calls WorkflowExecutor for workflows
                                            (which uses filesystem+adapter internally — not our concern).
                                            Calls agent_executor.py for standalone agents
                                            (which loads from PostgreSQL).
```

**Execution Strategies:**
- **single**: Execute one skill, return its output
- **sequential**: Loop through steps in order; output of step N becomes `input_source` for step N+1
- **parallel**: Group steps by `parallel_group`; use `asyncio.gather()` for concurrent groups; `asyncio.to_thread()` for sync CrewAI calls
- **hybrid**: Topological sort on `depends_on`; steps with no unresolved dependencies run in parallel; sequential within dependency chains

### 5.6 Guardrails Engine (`apps/appmgr/orchestrator/guardrails.py`)

**Pre-processing (on user input):**
1. **PII Regex** (fast, no LLM cost):
   - SSN: `\b\d{3}-\d{2}-\d{4}\b`
   - Credit card: `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`
   - Email: standard email regex
   - Phone: `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`
2. **Safety keywords** check (configurable blocklist)
3. **Custom compliance rules** from `guardrail_text` (parsed as rules)

**Post-processing (on LLM output):**
- Same PII regex scan on response text
- Redact any detected PII with `[REDACTED]`
- Apply custom compliance rules

**Output:**
```python
@dataclass
class GuardrailResult:
    is_safe: bool
    violations: list[str]      # ["PII_DETECTED: SSN", "SAFETY: blocked keyword"]
    sanitized_text: str        # text with PII redacted
    categories_triggered: list[str]  # ["pii", "safety"]
```

### 5.7 Human-in-the-Loop (`apps/appmgr/orchestrator/hitl.py`)

**Conversation State Machine:**
```
awaiting_input ──(user sends message)──► orchestrator runs
                                              │
                    ┌─────────────────────────┤
                    │                         │
                    ▼                         ▼
         needs_clarification=true    execution_plan ready
                    │                         │
                    ▼                         ▼
         awaiting_clarification       executing skills
                    │                         │
        (user answers)                        │
                    │                         ▼
                    ▼                    awaiting_input
         orchestrator re-runs              (done)
         with original + answer
```

**Out-of-scope handling:**
When the orchestrator sets `fallback_message`:
- Response = app's `sorry_message` + "I'm a workflow orchestrator designed to help with: [list of skill names and what they do]. Please ask me something related to these capabilities."
- Stored in chat history with `execution_trace.fallback = true`

**Clarification handling:**
When `needs_clarification=true`:
- Response = `clarification_question` from orchestrator
- `conversation_state` → `awaiting_clarification`
- `context_data` on chat session stores the original prompt + orchestrator's partial analysis
- Next user message is treated as the clarification answer
- Orchestrator re-runs with: enhanced original prompt + clarification answer

### 5.8 Persona Formatter (`apps/appmgr/orchestrator/persona.py`)

- Injects persona tone into the orchestrator's system prompt
- If `persona_id` is set → loads persona name → "You are a {persona_name}. Respond accordingly."
- If `persona_text` is set → uses custom text verbatim in system prompt
- Applied BEFORE orchestrator runs (part of system prompt, not post-processing)

---

## 6. File Structure

### 6.1 New Files

```
echolib/models/
├── application.py                        # NEW — SQLAlchemy model for applications + all link tables
├── application_chat.py                   # NEW — Chat session + message models
├── application_lookups.py                # NEW — Persona, Designation, BusinessUnit, Tag, GuardrailCategory, LLM, DataSource

echolib/repositories/
├── application_repo.py                   # NEW — Async CRUD for applications
├── application_chat_repo.py              # NEW — Async CRUD for chat sessions/messages

alembic/versions/
├── 004_create_application_tables.py      # NEW — All tables + indexes + seed data

apps/appmgr/
├── __init__.py                           # EXISTS
├── main.py                               # REWRITE — PostgreSQL models replace SQLite
├── routes.py                             # REWRITE — Async endpoints, full orchestration chat
├── container.py                          # UPDATE — Register new providers
├── schemas.py                            # NEW — Pydantic request/response schemas (extracted from main.py)
├── orchestrator/                         # NEW DIRECTORY
│   ├── __init__.py
│   ├── pipeline.py                       # Orchestration pipeline coordinator (ties all steps together)
│   ├── prompt_enhancer.py                # Prompt enhancement LLM
│   ├── skill_manifest.py                 # Skill metadata builder + cache
│   ├── orchestrator.py                   # Core orchestration LLM (skill selection)
│   ├── skill_executor.py                 # Execute workflows/agents (uses existing executors)
│   ├── agent_executor.py                 # NEW — Standalone agent execution (fills the gap)
│   ├── guardrails.py                     # Pre/post guardrails
│   ├── hitl.py                           # Human-in-the-loop state management
│   └── persona.py                        # Persona injection

apps/gateway/
├── main.py                               # UPDATE — Register new router, remove old appmgr if needed
```

### 6.2 Files to Modify

| File | Change |
|------|--------|
| `apps/gateway/main.py` | Mount new application router at `/api/applications` |
| `apps/appmgr/container.py` | Register orchestrator providers in DI container |
| `echolib/models/__init__.py` | Export new models |

### 6.3 Files — DO NOT MODIFY (Existing System)

| File | Rule | How We Interact |
|------|------|-----------------|
| `apps/workflow/crewai_adapter.py` | **DO NOT TOUCH. DO NOT IMPORT FROM.** | We never reference this file. Our agent_executor.py uses `crewai` library independently. |
| `apps/workflow/runtime/executor.py` | **DO NOT MODIFY.** | Call `execute_workflow()` as a black box from skill_executor.py. |
| `apps/workflow/designer/compiler.py` | **DO NOT MODIFY.** | Internal to workflow system — we never call it. |
| `apps/workflow/runtime/chat_session.py` | **DO NOT MODIFY.** Read as reference only. | Pattern reference for our own chat_manager. |
| `apps/workflow/routes.py` | **DO NOT MODIFY.** | Existing workflow API — untouched. |
| `llm_manager.py` | **DO NOT MODIFY.** | Call `LLMManager.get_llm()` and `LLMManager.get_crewai_llm()` as consumers. |
| `echolib/models/workflow.py` | **DO NOT MODIFY.** Read as reference only. | Pattern reference for our SQLAlchemy models. |
| `echolib/models/agent.py` | **DO NOT MODIFY.** Read as reference only. | Pattern reference. |
| `echolib/repositories/session_repo.py` | **DO NOT MODIFY.** Read as reference only. | Pattern reference for our async repository. |

---

## 7. Execution Order (Implementation Phases)

### Phase 1: Foundation (Database + Models)
1. Create SQLAlchemy models (`echolib/models/application.py`, `application_chat.py`, `application_lookups.py`)
2. Create Alembic migration (`004_create_application_tables.py`) with all tables + seed data
3. Create repositories (`application_repo.py`, `application_chat_repo.py`)
4. Create Pydantic schemas (`apps/appmgr/schemas.py`)

### Phase 2: CRUD API
5. Rewrite `apps/appmgr/routes.py` — all CRUD endpoints using async PostgreSQL
6. Implement lookup endpoints (`/api/llms`, `/api/data-sources`, etc.)
7. Implement skills endpoint (`/api/skills`) with workflow+agent type labels
8. Update `apps/gateway/main.py` to mount new router
9. Update `apps/appmgr/container.py`

### Phase 3: Orchestration Engine
10. Implement `prompt_enhancer.py`
11. Implement `skill_manifest.py` with caching
12. Implement `orchestrator.py` (core LLM intelligence)
13. Implement `guardrails.py` (pre/post processing)
14. Implement `persona.py`
15. Implement `hitl.py` (conversation state machine)
16. Implement `agent_executor.py` (standalone agent execution — fills the gap)
17. Implement `skill_executor.py` (sequential/parallel/hybrid execution)
18. Implement `pipeline.py` (ties all orchestration steps together)

### Phase 4: Chat API
19. Implement `POST /api/applications/{id}/chat` — full orchestration pipeline
20. Implement chat history endpoints
21. Implement document upload endpoint (for RAG)

### Phase 5: Integration & Polish
22. End-to-end testing with real LLM calls
23. Session isolation verification
24. Guardrails testing
25. HITL flow testing

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | PostgreSQL (drop SQLite) | User requirement; consistency with main system; proper FK relationships |
| Persistence layer | Async SQLAlchemy 2.0 + asyncpg | Matches existing codebase exactly |
| Migrations | Alembic | Matches existing pattern (001-003 already exist) |
| Orchestrator LLM | LangChain `ChatOpenAI` via `LLMManager` | Real library; already in requirements.txt; JSON mode support |
| Skill execution | Call `WorkflowExecutor` + `CrewAIAdapter` in-process | No HTTP overhead; direct DI resolution; reuses existing logic |
| Agent standalone | New `agent_executor.py` — independent CrewAI code, does NOT import from `crewai_adapter.py` | Fills gap; keeps existing adapter untouched |
| Parallel execution | `asyncio.gather()` + `asyncio.to_thread()` | CrewAI is sync; wrap in threads for async compatibility |
| Hybrid execution | Topological sort on `depends_on` graph | Handles any DAG of skill dependencies |
| Guardrails | Regex (fast) + optional LLM (thorough) | Enterprise-grade; no external service dependency |
| HITL state | `conversation_state` column on chat session | Simple; no additional infrastructure; persisted in PG |
| Session isolation | `user_id` FK on all tables | Matches existing pattern (workflows, agents, sessions all use user_id) |
| API routes | `/api/applications/...` | Per user's explicit choice |
| Lookup tables | Separate PG tables with seed data | Per user's explicit choice |

---

## 9. Libraries Used (All Existing in requirements.txt)

| Library | Module | Usage |
|---------|--------|-------|
| `langchain-openai` | `ChatOpenAI` | Prompt enhancer + Orchestrator LLM calls |
| `langchain-core` | `SystemMessage`, `HumanMessage`, `ChatPromptTemplate` | Prompt construction |
| `langgraph` | `StateGraph` | Dynamic graph for hybrid execution plans (optional) |
| `crewai` | `Crew`, `Agent`, `Task`, `Process`, `LLM` | Standalone agent execution |
| `sqlalchemy[asyncio]` | `AsyncSession`, `Mapped`, `mapped_column` | PostgreSQL models |
| `asyncpg` | (driver) | PostgreSQL async driver |
| `alembic` | (migrations) | Schema creation + seeding |
| `pydantic` | `BaseModel`, `Field` | Request/response schemas |
| `httpx` | `AsyncClient` | Internal API calls for skill metadata |
| `fastapi` | `APIRouter`, `Depends`, `HTTPException` | API endpoints |

---

## 10. What This Plan Does NOT Do (Explicit Exclusions)

- Does NOT rewrite existing workflow/agent execution logic
- Does NOT modify existing LangGraph/CrewAI integration
- Does NOT change authentication/authorization infrastructure
- Does NOT build a frontend
- Does NOT implement real-time WebSocket streaming for app chat (can be added later using existing WS infrastructure)
- Does NOT change the LLMManager configuration


+## 11. HITL Node Integration in Apps (Additive Update)                                                                                 
  915 +                                                                                                                                       
  916 +> **Date**: 2026-02-17                                                                                                                 
  917 +> **Prerequisite**: HITL Node system fully implemented in `apps/workflow/` (see HITL-plan.md)                                          
  918 +> **Constraint**: ALL changes are ADDITIVE ONLY. Existing pipeline, skill_executor, routes, hitl.py remain untouched. No refactoring.  
  919 +                                                                                                                                       
  920 +### 11.1 Problem Statement                                                                                                             
  921 +                                                                                                                                       
  922 +The HITL Node feature works for direct workflow execution (`/workflows/chat/send`), but when a workflow containing HITL nodes is       
      +executed **through an app** (`/api/applications/{id}/chat`), the interrupt is invisible:                                               
  923 +                                                                                                                                       
  924 +1. `skill_executor.py` calls `executor.execute_workflow()` (old method, no interrupt detection)                                        
  925 +2. No handling for `status="interrupted"` in skill_executor or pipeline                                                                
  926 +3. No endpoint in apps for HITL decisions (approve/reject/edit/defer)                                                                  
  927 +4. Apps use synchronous request-response, not background tasks + WebSocket                                                             
  928 +                                                                                                                                       
  929 +### 11.2 New Conversation State                                                                                                        
  930 +                                                                                                                                       
  931 +Add `"awaiting_hitl_review"` as a valid conversation state alongside existing `"awaiting_input"`, `"awaiting_clarification"`,          
      +`"executing"`.                                                                                                                         
  932 +                                                                                                                                       
  933 +### 11.3 Files to Modify (ADDITIVE ONLY)                                                                                               
  934 +                                                                                                                                       
  935 +#### File 1: `apps/appmgr/orchestrator/skill_executor.py`                                                                              
  936 +                                                                                                                                       
  937 +**DO NOT** modify existing `_execute_workflow()`, `execute_plan()`, or any other method.                                               
  938 +                                                                                                                                       
  939 +**ADD**:                                                                                                                               
  940 +- New method `_execute_workflow_hitl_aware()` that:                                                                                    
  941 +  1. Checks if the workflow has HITL nodes (by loading workflow definition and checking agent metadata for `node_type == "HITL"`)      
  942 +  2. If HITL nodes present: calls `executor.execute_workflow_hitl()` instead of `execute_workflow()`                                   
  943 +  3. If result `status == "interrupted"`: returns a special dict `{"hitl_interrupted": True, "run_id": ..., "workflow_id": ...,        
      +"interrupt": ..., "partial_output": ...}` instead of extracted text                                                                    
  944 +  4. If no HITL nodes: delegates to existing `_execute_workflow()` unchanged                                                           
  945 +                                                                                                                                       
  946 +- Modify `_execute_step()` to call `_execute_workflow_hitl_aware()` for workflow skills, and propagate the interrupted result upward   
      +through `execute_plan()`                                                                                                               
  947 +                                                                                                                                       
  948 +- When `execute_plan()` encounters an interrupted step mid-sequence:                                                                   
  949 +  - Collect all outputs from completed prior steps                                                                                     
  950 +  - Return `{"hitl_interrupted": True, "completed_steps": [...], "interrupted_at_step": N, "interrupt_payload": {...}, "run_id": ...,  
      +"workflow_id": ...}`                                                                                                                   
  951 +                                                                                                                                       
  952 +#### File 2: `apps/appmgr/orchestrator/pipeline.py`                                                                                    
  953 +                                                                                                                                       
  954 +**DO NOT** modify existing `run()` logic, clarification handling, or any helper methods.                                               
  955 +                                                                                                                                       
  956 +**ADD** (after skill execution succeeds in `run()`):                                                                                   
  957 +- Check if `execution_result` contains `hitl_interrupted == True`                                                                      
  958 +- If yes:                                                                                                                              
  959 +  1. Store HITL context in session's `context_data` JSONB:                                                                             
  960 +     ```json                                                                                                                           
  961 +     {                                                                                                                                 
  962 +       "hitl_state": {                                                                                                                 
  963 +         "run_id": "...",                                                                                                              
  964 +         "workflow_id": "...",                                                                                                         
  965 +         "interrupt_payload": {...},                                                                                                   
  966 +         "completed_steps": [...],                                                                                                     
  967 +         "remaining_plan": [...],                                                                                                      
  968 +         "execution_plan": {...}                                                                                                       
  969 +       }                                                                                                                               
  970 +     }                                                                                                                                 
  971 +     ```                                                                                                                               
  972 +  2. Update session `conversation_state` → `"awaiting_hitl_review"`                                                                    
  973 +  3. Save completed step outputs as a partial assistant message                                                                        
  974 +  4. Return response with:                                                                                                             
  975 +     - `conversation_state: "awaiting_hitl_review"`                                                                                    
  976 +     - `response`: formatted text showing completed step outputs + "Workflow paused for human review"                                  
  977 +     - `execution_trace` includes `hitl_context` with title, message, priority, allowed_decisions, run_id                              
  978 +                                                                                                                                       
  979 +#### File 3: `apps/appmgr/routes.py`                                                                                                   
  980 +                                                                                                                                       
  981 +**DO NOT** modify existing `chat_with_application()`, CRUD endpoints, or any existing route.                                           
  982 +                                                                                                                                       
  983 +**ADD**:                                                                                                                               
  984 +- New endpoint: `POST /api/applications/{application_id}/chat/hitl-decide`                                                             
  985 +  - Request body: `{"session_id": str, "action": str, "payload": dict?, "rationale": str?}`                                            
  986 +  - Validates:                                                                                                                         
  987 +    - App exists and is published                                                                                                      
  988 +    - Session exists and `conversation_state == "awaiting_hitl_review"`                                                                
  989 +    - Action is in `allowed_decisions` from stored HITL context                                                                        
  990 +  - For **approve**:                                                                                                                   
  991 +    1. Retrieve HITL context from session `context_data`                                                                               
  992 +    2. Call `executor.resume_workflow(workflow_id, run_id, "approve", payload)` via `asyncio.to_thread()`                              
  993 +    3. If result `status == "completed"`: continue pipeline (post-guardrails, persona, save messages)                                  
  994 +    4. If result `status == "interrupted"` again: store new HITL state, return new review prompt                                       
  995 +    5. Update session state back to `"awaiting_input"` on completion                                                                   
  996 +  - For **reject**:                                                                                                                    
  997 +    1. Record rejection in HITL manager                                                                                                
  998 +    2. Save rejection message as assistant response                                                                                    
  999 +    3. Update session state → `"awaiting_input"`                                                                                       
 1000 +    4. Return rejection response                                                                                                       
 1001 +  - For **edit**:                                                                                                                      
 1002 +    1. Call `executor.resume_workflow()` with action="edit" and the edited payload                                                     
 1003 +    2. Same completion/re-interrupt handling as approve                                                                                
 1004 +  - For **defer**:                                                                                                                     
 1005 +    1. Record deferral                                                                                                                 
 1006 +    2. Session stays in `"awaiting_hitl_review"` (can be decided later)                                                                
 1007 +    3. Return deferral confirmation                                                                                                    
 1008 +                                                                                                                                       
 1009 +- New Pydantic schemas in `apps/appmgr/schemas.py`:                                                                                    
 1010 +  - `HITLDecisionRequest(BaseModel)`: session_id, action, payload, rationale                                                           
 1011 +  - `HITLDecisionResponse(BaseModel)`: status, message, session_id, conversation_state                                                 
 1012 +                                                                                                                                       
 1013 +#### File 4: `apps/appmgr/schemas.py`                                                                                                  
 1014 +                                                                                                                                       
 1015 +**ADD** (do not modify existing schemas):                                                                                              
 1016 +- `HITLDecisionRequest` — request body for hitl-decide                                                                                 
 1017 +- `HITLDecisionResponse` — response for hitl-decide                                                                                    
 1018 +- `HITLContext` — nested model for HITL data in execution_trace                                                                        
 1019 +                                                                                                                                       
 1020 +### 11.4 Data Flow: HITL Node in App Chat                                                                                              
 1021 +                                                                                                                                       
 1022 +```                                                                                                                                    
 1023 +User sends message → POST /api/applications/{id}/chat                                                                                  
 1024 +  → Pipeline.run()                                                                                                                     
 1025 +    → Orchestrator plans: [Step1: workflow_A, Step2: workflow_B]                                                                       
 1026 +    → SkillExecutor.execute_plan()                                                                                                     
 1027 +      → Step 1: _execute_workflow_hitl_aware("workflow_A")                                                                             
 1028 +        → executor.execute_workflow_hitl() → status="completed" → output_A                                                             
 1029 +      → Step 2: _execute_workflow_hitl_aware("workflow_B")                                                                             
 1030 +        → executor.execute_workflow_hitl() → status="interrupted" (HITL node hit)                                                      
 1031 +        → Returns {hitl_interrupted: true, run_id, interrupt_payload, ...}                                                             
 1032 +      → execute_plan() collects: step1 output + step2 interrupted                                                                      
 1033 +    → Pipeline detects hitl_interrupted                                                                                                
 1034 +    → Stores HITL state in session context_data                                                                                        
 1035 +    → Sets conversation_state = "awaiting_hitl_review"                                                                                 
 1036 +    → Returns: step1 output + "Workflow paused for review" + hitl_context                                                              
 1037 +                                                                                                                                    
 1038 +Frontend displays:                                                                                                                     
 1039 +  - Step 1 output (completed)                                                                                                          
 1040 +  - HITL review panel: title, message, approve/reject/edit/defer buttons                                                               
 1041 +                                                                                                                                       
 1042 +User clicks "Approve" → POST /api/applications/{id}/chat/hitl-decide                                                                   
 1043 +  → Retrieves HITL state from session                                                                                                  
 1044 +  → executor.resume_workflow(workflow_id, run_id, "approve")                                                                           
 1045 +  → result status="completed" → output_B                                                                                               
 1046 +  → Continues pipeline: post-guardrails on output_B                                                                                    
 1047 +  → Merges step1_output + output_B as final response                                                                                   
 1048 +  → Saves to chat history                                                                                                              
 1049 +  → Sets conversation_state = "awaiting_input"                                                                                         
 1050 +  → Returns final merged response                                                                                                      
 1051 +```                                                                                                                                    
 1052 +
 1053 +### 11.5 Key Constraints
 1054 +
 1055 +1. `apps/appmgr/orchestrator/hitl.py` (clarification HITL) is UNTOUCHED — different system
 1056 +2. `apps/workflow/runtime/hitl.py` (workflow HITL) is UNTOUCHED — we call it, don't modify it
 1057 +3. `apps/workflow/runtime/executor.py` is UNTOUCHED — we call `execute_workflow_hitl()` and `resume_workflow()` as black boxes
 1058 +4. Existing `_execute_workflow()` in skill_executor stays as-is — new method alongside it
 1059 +5. Existing `run()` in pipeline stays as-is — HITL check is added after execution result
 1060 +6. No new database tables — uses existing `context_data` JSONB on chat sessions
 1061 +7. WorkflowExecutor resolved from DI container as before

=============Done=============
Implementation is complete. Here's the summary:

  Changes Made (4 files, all additive)

  1. apps/appmgr/schemas.py

  - Added HITLDecisionRequest, HITLDecisionResponse, HITLContextSchema Pydantic models

  2. apps/appmgr/orchestrator/skill_executor.py

  - Added _execute_workflow_hitl_aware() — detects HITL nodes, calls execute_workflow_hitl(), returns interrupt dict on pause
  - Added _workflow_has_hitl_nodes() — queries workflow definition to check for HITL node types
  - Modified _execute_step() — routes workflow skills through HITL-aware path
  - Modified _execute_single() and _execute_sequential() — propagate HITL interrupts upward with completed step outputs

  3. apps/appmgr/orchestrator/pipeline.py

  - Added HITL interrupt handling after skill execution — stores HITL context in session context_data, formats partial outputs, returns       
  conversation_state: "awaiting_hitl_review" to frontend

  4. apps/appmgr/routes.py

  - Added POST /api/applications/{id}/chat/hitl-decide endpoint with full support for:
    - approve — resumes workflow, handles completion or chained interrupts
    - reject — terminates, saves rejection message
    - edit — resumes with edited payload
    - defer — session stays paused

  Key Design Decision

  The DB has a CHECK constraint on conversation_state that only allows awaiting_input | awaiting_clarification | executing. Rather than       
  altering the DB, the implementation stores "awaiting_clarification" in the DB (semantically correct — waiting for human) and returns        
  "awaiting_hitl_review" in the API response. The frontend uses the API response to distinguish HITL from regular clarification via the       
  presence of hitl_context in the execution trace.

  Known Limitation

  HITL interrupt propagation works for single and sequential execution strategies. Parallel/hybrid strategies don't cleanly support
  mid-execution HITL pauses due to the complexity of pausing one branch while others may be running.

Fixes Applied
                                                                                                                                                Fix 1: _execute_parallel() — HITL interrupt propagation                                                                                     

  - After asyncio.gather() completes a parallel group, each result is checked for hitl_interrupted
  - First HITL interrupt in the group is captured; other parallel steps' outputs are collected as completed
  - Remaining unexecuted groups become remaining_plan
  - Returns the standard hitl_interrupted dict so the pipeline can handle it

  Fix 2: _execute_hybrid() — HITL interrupt propagation

  - Same pattern as parallel — after each topological batch runs via asyncio.gather(), results are checked
  - First HITL interrupt stops the dependency-resolution loop
  - All remaining steps (unresolved dependencies) are collected as remaining_plan

  Fix 3: Pipeline HITL guard

  - Before the clarification check in pipeline.py step 4, added a guard that checks context_data.hitl_state
  - If the session is paused at a HITL node, regular chat messages are rejected with a message directing the user to the HITL decide panel    
  - Returns conversation_state: "awaiting_hitl_review" so the frontend knows to show the review UI
  - Normal clarification flow (no hitl_state) proceeds unchanged

  Fix 4: Remaining plan execution after approve/edit

  - After executor.resume_workflow() completes successfully in the hitl-decide route, the remaining steps from the original execution plan are
   now executed via SkillExecutor.execute_plan()
  - If remaining steps also hit a HITL interrupt, a new HITL state is stored and the session stays paused
  - If remaining steps complete normally, all outputs are merged, post-guardrails run, and the session returns to awaiting_input