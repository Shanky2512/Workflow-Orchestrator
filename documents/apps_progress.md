# AI Application Orchestrator -- Progress Tracker

> **HOW TO RESUME**: In a new chat, say:
> *"Read `documents/apps_plan.md` and `documents/apps_progress.md` then continue implementation."*

> **Last Updated**: 2026-02-10
> **Plan**: [apps_plan.md](./apps_plan.md)
> **Overall Status**: Phase 4 COMPLETE + Bug Fixes Applied + Field Usage Fixes Applied -- Ready for Phase 5 (Testing)
> **Key Decisions Made**:
> - PostgreSQL only (no SQLite)
> - API prefix: `/api/applications/...`
> - Lookup tables: separate PG tables with Alembic seed
> - `apps/workflow/crewai_adapter.py` -- DO NOT TOUCH, DO NOT IMPORT FROM
> - All existing workflow/agent files -- DO NOT MODIFY
> - Application orchestrator has its own independent CrewAI usage in `agent_executor.py`
> - Wizard fields are mostly OPTIONAL, only `name` required for draft, stricter validation at publish
> - Routes use `get_db_session()` async context manager (not Depends injection for DB sessions)
> - User auth via `Depends(get_current_user)` with fallback to `DEFAULT_USER_ID` for anonymous
> - Lookup and skills endpoints are separate `APIRouter` instances mounted in gateway

---

## Phase 1: Foundation (Database + Models)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | SQLAlchemy models -- `application.py` | DONE | Application model + 7 link tables (LLM, Skill, DataSource, Designation, BusinessUnit, Tag, Guardrail). UUID PKs, composite PKs on links, partial indexes, relationships with selectin loading. File: `echolib/models/application.py` |
| 2 | SQLAlchemy models -- `application_chat.py` | DONE | ApplicationChatSession (HITL state machine), ApplicationChatMessage, ApplicationDocument (RAG uploads), ApplicationExecutionTrace (audit trail). All with JSONB columns, proper FKs, indexes. File: `echolib/models/application_chat.py` |
| 3 | SQLAlchemy models -- `application_lookups.py` | DONE | AppPersona, AppGuardrailCategory, AppDesignation, AppBusinessUnit, AppTag (UUID PKs, unique names), AppLlm (UUID PK), AppDataSource (VARCHAR PK). File: `echolib/models/application_lookups.py`. Also updated `echolib/models/__init__.py` with all new exports. |
| 4 | Alembic migration -- `004_create_application_tables.py` | DONE | All 17 tables created in dependency order: 7 lookup tables, 1 core applications table, 7 link tables, 2 chat tables, 1 documents table, 1 execution traces table. Includes partial indexes, check constraints, updated_at triggers, and seed data for all lookup tables. File: `alembic/versions/004_create_application_tables.py` |
| 5 | Repository -- `application_repo.py` | DONE | Async CRUD for applications: create, get, list (paginated with status/search/sort filters), update, soft-delete, publish, unpublish, stats (conditional aggregation), and 7 sync methods for all link tables (LLM, skill, data source, designation, business unit, tag, guardrail). All queries filter is_deleted=FALSE. File: `echolib/repositories/application_repo.py`. Also updated `echolib/repositories/__init__.py` with new exports. |
| 6 | Repository -- `application_chat_repo.py` | DONE | Async CRUD for chat: create/get/list sessions, update conversation state, close session, add/list messages, create/update execution traces, create documents. File: `echolib/repositories/application_chat_repo.py` |
| 7 | Pydantic schemas -- `apps/appmgr/schemas.py` | DONE | 7 request schemas (ApplicationCreate, ApplicationSetupUpdate, ApplicationAccessUpdate, ApplicationContextUpdate, ApplicationChatConfigUpdate, ApplicationFullUpdate, ChatMessageRequest) and 12 response schemas (LlmLinkResponse, SkillLinkResponse, ApplicationCard, ApplicationDetail, ApplicationStats, ApplicationListResponse, ChatSessionResponse, ChatMessageResponse, ChatResponse, LookupResponse, SkillResponse). All response schemas use ConfigDict(from_attributes=True). File: `apps/appmgr/schemas.py` |

## Phase 2: CRUD API

| # | Task | Status | Notes |
|---|------|--------|-------|
| 8 | Rewrite `apps/appmgr/routes.py` -- CRUD | DONE | Complete rewrite from sync SQLite/SQLModel to async PostgreSQL. 12 endpoints: POST create, GET list (paginated + stats), GET detail, PUT full update, PATCH setup/access/context/chat-config, DELETE soft-delete, POST publish (with validation: LLM, skill, welcome_prompt, sorry_message, starters, persona), POST unpublish, POST logo upload. All use `async with get_db_session()` and `Depends(get_current_user)`. PUT preserves PUBLISHED status on edit. Publish returns structured errors. File: `apps/appmgr/routes.py` |
| 9 | Lookup endpoints (`/api/llms`, etc.) | DONE | 7 lookup GET endpoints on `lookup_router`: `/api/llms`, `/api/data-sources`, `/api/designations`, `/api/business-units`, `/api/tags`, `/api/personas`, `/api/guardrail-categories`. All return `list[LookupResponse]` with id+name. Simple SELECT * from lookup tables. Same file: `apps/appmgr/routes.py` |
| 10 | Skills endpoint (`/api/skills`) | DONE | GET `/api/skills` on `skills_router`. Queries both `workflows` table (status != archived, is_deleted = false) and `agents` table (is_deleted = false), filtered by user_id. Returns `list[SkillResponse]` with skill_type='workflow' or 'agent'. Extracts description from JSONB definition column. Same file: `apps/appmgr/routes.py` |
| 11 | Update `apps/gateway/main.py` | DONE | Added `lookup_router` and `skills_router` imports from `apps.appmgr.routes`. Added `app.include_router(app_lookup_router)` and `app.include_router(app_skills_router)` after existing `app_router` mount. All existing mounts preserved. File: `apps/gateway/main.py` |
| 12 | Update `apps/appmgr/container.py` | DONE | Registered `ApplicationRepository` as `app.repo` and `ApplicationChatRepository` as `app.chat_repo` in the DI container. Preserved existing `app.store` legacy registration. File: `apps/appmgr/container.py` |

## Phase 3: Orchestration Engine

| # | Task | Status | Notes |
|---|------|--------|-------|
| 13 | `prompt_enhancer.py` | DONE | Single LLM call via LLMManager.get_llm() -> ChatOpenAI. Uses LangChain SystemMessage/HumanMessage. System prompt instructs: fix typos, clarify intent, do NOT answer. Returns JSON with enhanced_prompt, detected_intent, confidence. Graceful fallback on LLM failure. asyncio.to_thread() for sync invoke. Supports enhancer_llm_config override. File: `apps/appmgr/orchestrator/prompt_enhancer.py` |
| 14 | `skill_manifest.py` | DONE | Queries application_skill_links + workflows/agents tables for rich metadata. Builds structured manifest dict with skill_id, skill_type, name, description, capabilities, execution_model (workflows), tools (agents). In-memory cache per application_id with configurable TTL (default 5min). invalidate()/invalidate_all() methods. Falls back to denormalized link data if DB record missing. File: `apps/appmgr/orchestrator/skill_manifest.py` |
| 15 | `orchestrator.py` | DONE | Core LLM intelligence via LLMManager.get_llm() -> ChatOpenAI. Full system prompt with persona, guardrails, skill manifest (with capabilities/tools/execution_model), conversation history (last 10). Forces JSON output with execution plan schema: reasoning, execution_strategy (single/sequential/parallel/hybrid), execution_plan steps with depends_on/parallel_group/input_source/output_key, needs_clarification, clarification_question, fallback_message. asyncio.to_thread() for sync invoke. Robust JSON parsing with fallback. File: `apps/appmgr/orchestrator/orchestrator.py` |
| 16 | `guardrails.py` | DONE | Pre/post processing engine. PII regex: SSN, credit card, email, phone. Safety keyword blocklist (configurable). Custom compliance rules from guardrail_text (BLOCK:, REGEX:, plain text). Pre-process: detect violations, return is_safe flag. Post-process: redact PII with [REDACTED]. GuardrailResult dataclass with is_safe, violations, sanitized_text, categories_triggered. All regex-based, no LLM cost. File: `apps/appmgr/orchestrator/guardrails.py` |
| 17 | `persona.py` | DONE | PersonaFormatter with build_persona_prompt(). Priority: persona_text (verbatim) > persona_id (load from app_personas table -> "You are a {name}") > empty string. Async DB query via safe_uuid + select. File: `apps/appmgr/orchestrator/persona.py` |
| 18 | `hitl.py` | DONE | HITLManager with handle_orchestrator_response(), get_clarification_context(), build_clarification_prompt(). States: awaiting_input -> executing -> awaiting_input (normal), awaiting_input -> awaiting_clarification -> awaiting_input (clarification). Stores original_prompt + partial_analysis in session context_data. Fallback handling returns action dict. Uses ApplicationChatRepository for state persistence. File: `apps/appmgr/orchestrator/hitl.py` |
| 19 | `agent_executor.py` | DONE | 100% INDEPENDENT from crewai_adapter.py -- imports crewai library directly (Crew, Agent, Task, Process). Loads agent definition from PostgreSQL agents table (JSONB). Creates CrewAI Agent from definition fields (name, role, goal, backstory). Own _bind_tools() implementation via DI container (tool.registry + tool.executor). Gets LLM via LLMManager.get_crewai_llm(). asyncio.to_thread(crew.kickoff). Graceful LLM empty response handling. Returns {result, agent_id, agent_name}. File: `apps/appmgr/orchestrator/agent_executor.py` |
| 20 | `skill_executor.py` | DONE | Executes orchestrator plans. For workflows: resolves WorkflowExecutor via DI container.resolve('workflow.executor'), calls execute_workflow() as black box via asyncio.to_thread(). For agents: calls StandaloneAgentExecutor.execute(). 4 execution strategies: single (one skill), sequential (chain output N -> input N+1), parallel (group by parallel_group, asyncio.gather), hybrid (topological sort on depends_on, independent steps in parallel, circular dependency detection). Returns {results, final_output, execution_log}. File: `apps/appmgr/orchestrator/skill_executor.py` |
| 21 | `pipeline.py` | DONE | Full orchestration coordinator -- OrchestrationPipeline.run() ties all 8 components together in 16 steps: load app, pre-guardrails, check HITL state, build persona, enhance prompt, build skill manifest, get history, call orchestrator LLM, handle HITL (clarify/fallback/execute), execute skills, post-guardrails, save messages, save execution trace, update session state. Creates chat session on first message. Validates app is published. Full audit trail via ApplicationExecutionTrace. File: `apps/appmgr/orchestrator/pipeline.py` |

## Phase 4: Chat API

| # | Task | Status | Notes |
|---|------|--------|-------|
| 22 | `POST /api/applications/{id}/chat` | DONE | Full pipeline endpoint. Validates app exists + is published (404/400). Instantiates OrchestrationPipeline() with no args. Calls pipeline.run(db, application_id, session_id, message, user_id) -- no llm_manager. Builds ChatResponse with ChatMessageResponse from pipeline result. Deferred import of OrchestrationPipeline to avoid circular imports. File: `apps/appmgr/routes.py` |
| 23 | Chat history endpoints | DONE | 3 endpoints: GET `/{id}/chat/history` lists sessions via ApplicationChatRepository.list_sessions() returning list[ChatSessionResponse] with message_count. GET `/{id}/chat/sessions/{session_id}` verifies ownership + app binding, returns list[ChatMessageResponse] via get_messages(). DELETE `/{id}/chat/sessions/{session_id}` verifies ownership + app binding, calls close_session(), returns 204. All scoped by user_id. File: `apps/appmgr/routes.py` |
| 24 | Document upload endpoint | DONE | POST `/{id}/chat/upload` accepts UploadFile + optional session_id query param. Validates app exists + upload_enabled (400 if not). Validates session if provided. Saves file to configurable UPLOADS_DIR (default: `apps/appmgr/uploads/`). Creates ApplicationDocument record via create_document(). Returns DocumentUploadResponse with document_id, filename, processing_status. New schema: DocumentUploadResponse added to schemas.py. File: `apps/appmgr/routes.py` |

## Bug Fixes (2026-02-10)

Applied after initial deployment testing revealed issues with empty payloads, empty lookups, and unused metadata fields.

| # | Issue | Severity | Status | Fix Applied |
|---|-------|----------|--------|-------------|
| BF-1 | `/api/llms` returns empty `[]` | P0 | FIXED | `app_llms` table had no seed data. Added 5 default LLMs (GPT-4o, GPT-4o Mini, Claude Sonnet, OpenRouter Free, Ollama Local) to Alembic migration `004`. Added `POST /api/llms/seed` endpoint for seeding if migration already ran. Files: `alembic/versions/004_create_application_tables.py`, `apps/appmgr/routes.py` |
| BF-2 | `/api/data-sources` returns empty `[]` | P0 | FIXED | `app_data_sources` table had no seed data. Added 3 default data sources (Local Filesystem, Web Search, REST API) to Alembic migration `004`. Added `POST /api/data-sources/seed` endpoint. Files: `alembic/versions/004_create_application_tables.py`, `apps/appmgr/routes.py` |
| BF-3 | `PATCH /setup` returns 422 | P0 | FIXED | `ApplicationSetupUpdate` schema used `List[...]` with `default_factory=list` which rejected `null` JSON values. Changed to `Optional[List[...]] = None`. Added null checks in `patch_setup()` route so null = "don't change". Files: `apps/appmgr/schemas.py`, `apps/appmgr/routes.py` |
| BF-4 | `ApplicationCreate` missing link fields | P0 | FIXED | `POST /api/applications` only accepted scalar fields. Added all 7 link fields to `ApplicationCreate` schema: `llm_links`, `skill_links`, `data_source_ids`, `designation_ids`, `business_unit_ids`, `tag_ids`, `guardrail_category_ids`. Added link syncing in `create_application` route after app row creation and before config JSONB rebuild. Files: `apps/appmgr/schemas.py`, `apps/appmgr/routes.py` |
| BF-5 | Chat always returns `needs_clarification` | P0 | FIXED | Cascading fix from BF-3/BF-4. With skills now linkable at creation time, the skill manifest is populated and the orchestrator LLM can produce real execution plans. |
| BF-6 | Guardrail categories hardcoded | P1 | FIXED | `_get_guardrail_category_names()` in pipeline.py had an empty loop and hardcoded `["PII", "Safety", "Compliance"]`. Changed to async method that queries `app_guardrail_categories` table by actual IDs from the app's guardrail links. File: `apps/appmgr/orchestrator/pipeline.py` |
| BF-7 | Missing persona templates | P1 | FIXED | Only 3 of 5 seeded personas had JSON templates. Created `legal_advisor.json` and `technical_expert.json` with full system prompt, domain context, tone/style, response format, and guardrail hints. Files: `echolib/persona_templates/legal_advisor.json`, `echolib/persona_templates/technical_expert.json` |
| BF-8 | Output formatting artifacts | P1 | FIXED | LLM/workflow output contained literal `\n` character sequences and "String\n" prefixes. Added output cleanup in `_execute_workflow()`: replace literal `\n` with real newlines, strip "String\n" prefix, collapse 3+ newlines to 2, strip whitespace. File: `apps/appmgr/orchestrator/skill_executor.py` |
| BF-9 | Tags not used in pipeline | P2 | FIXED | Tags were stored in link tables but never consumed. Added `_get_tag_names()` helper in pipeline.py, resolved names from DB, injected into orchestrator system prompt as `## Application Tags` section. Added `tag_names` resolved field to `ApplicationDetail` response. Files: `apps/appmgr/orchestrator/pipeline.py`, `apps/appmgr/orchestrator/orchestrator.py`, `apps/appmgr/schemas.py`, `apps/appmgr/routes.py` |
| BF-10 | Designations/BU not used in pipeline | P2 | FIXED | Designations and business units stored but never consumed. Added `_get_designation_names()` and `_get_business_unit_names()` helpers in pipeline.py, resolved names from DB, injected into orchestrator system prompt as `## Access Context` section. Added `designation_names` and `business_unit_names` resolved fields to `ApplicationDetail` response. Files: `apps/appmgr/orchestrator/pipeline.py`, `apps/appmgr/orchestrator/orchestrator.py`, `apps/appmgr/schemas.py`, `apps/appmgr/routes.py` |
| BF-11 | LLM name not visible in API responses | P2 | FIXED | `LlmLinkResponse` had `name` field but it was always null. Added `relationship` on `ApplicationLlmLink` to `AppLlm` (selectin, viewonly, explicit primaryjoin since no FK constraint). Updated `_app_to_detail` → async `_build_app_detail` to resolve LLM names. Added `app_llm_config` to chat execution trace. Files: `echolib/models/application.py`, `apps/appmgr/routes.py` |
| BF-12 | Commented OpenAI SDK alternative | — | DONE | Added commented-out OpenAI SDK code blocks to both `prompt_enhancer.py` and `orchestrator.py` as alternative to LangChain. User can uncomment to switch. Files: `apps/appmgr/orchestrator/prompt_enhancer.py`, `apps/appmgr/orchestrator/orchestrator.py` |

### Files Modified in Bug Fix Round

| File | Changes |
|------|---------|
| `alembic/versions/004_create_application_tables.py` | Added seed data for `app_llms` (5 LLMs) and `app_data_sources` (3 sources) |
| `apps/appmgr/schemas.py` | `ApplicationCreate`: added 7 link fields. `ApplicationSetupUpdate`: fields changed to Optional. `ApplicationDetail`: added `designation_names`, `business_unit_names`, `tag_names` |
| `apps/appmgr/routes.py` | `create_application`: syncs all 7 link tables at creation. `patch_setup`: null checks. `_app_to_detail` → async `_build_app_detail` with name resolution (all 10 call sites updated). Added `POST /api/llms/seed` and `POST /api/data-sources/seed` endpoints |
| `echolib/models/application.py` | `ApplicationLlmLink`: added `llm` relationship to `AppLlm` (selectin, viewonly) |
| `apps/appmgr/orchestrator/pipeline.py` | `_get_guardrail_category_names`: fixed to query DB. Added `_get_tag_names`, `_get_designation_names`, `_get_business_unit_names`. Tags/designations/BU resolved and passed to orchestrator. `app_llm_config` added to trace |
| `apps/appmgr/orchestrator/orchestrator.py` | System prompt template: added `{tags_section}` and `{access_context_section}`. `plan()` and `_build_system_prompt()`: added `app_tags`, `app_designations`, `app_business_units` params. Added commented OpenAI SDK block |
| `apps/appmgr/orchestrator/prompt_enhancer.py` | Added commented OpenAI SDK block |
| `apps/appmgr/orchestrator/skill_executor.py` | `_execute_workflow()`: added output cleanup (literal \n fix, String prefix strip, newline collapse) |
| `echolib/persona_templates/legal_advisor.json` | NEW — full persona template for Legal Advisor |
| `echolib/persona_templates/technical_expert.json` | NEW — full persona template for Technical Expert |

## Field Usage Fixes (2026-02-10)

Audit of all application fields to ensure every configuration field set during app creation is actively used during chat.

| # | Field | Gap | Status | Fix Applied |
|---|-------|-----|--------|-------------|
| FU-1 | `welcome_prompt` | Not sent when new session starts | FIXED | Auto-injected as first assistant message when a new chat session is created (`pipeline.py` step 2). Also returned in `session_metadata` on ChatResponse for frontend rendering. |
| FU-2 | `disclaimer` | Never shown to user or LLM | FIXED | (a) Injected into orchestrator system prompt as `## Disclaimer` section so LLM is aware of it. (b) Returned in `session_metadata` on ChatResponse for frontend to display as a banner. |
| FU-3 | `starter_questions` | Not returned in ChatResponse | FIXED | Returned in `session_metadata.starter_questions` on ChatResponse when new session is created. Frontend can render these as suggested question chips. |
| FU-4 | `name` | Not in orchestrator context | FIXED | Injected into orchestrator system prompt as `## Application Identity` section: "You are the AI assistant for the application **{name}**." |
| FU-5 | `description` | Not in orchestrator context | FIXED | Injected into `## Application Identity` section alongside name: "Application purpose: {description}". |
| FU-6 | `data_source_links` | Not passed to orchestrator | FIXED | Added `_get_data_source_names()` helper in pipeline.py. Resolved data source names from `app_data_sources` table and injected into orchestrator system prompt as `## Connected Data Sources` section. Also recorded in execution trace. |

### New Schema: `ChatSessionMetadata`

Added to `schemas.py` — populated only when a new session is created (first message without `session_id`):
```python
class ChatSessionMetadata(BaseModel):
    app_name: str
    app_description: Optional[str]
    app_logo_url: Optional[str]
    welcome_prompt: Optional[str]
    disclaimer: Optional[str]
    starter_questions: List[str]
```

### Files Modified in Field Usage Fix Round

| File | Changes |
|------|---------|
| `apps/appmgr/schemas.py` | Added `ChatSessionMetadata` schema. Added `session_metadata: Optional[ChatSessionMetadata]` to `ChatResponse`. |
| `apps/appmgr/routes.py` | Imported `ChatSessionMetadata`. Updated `chat_with_application` endpoint to build `ChatSessionMetadata` from pipeline result and pass to `ChatResponse`. |
| `apps/appmgr/orchestrator/orchestrator.py` | System prompt template: added `{app_identity_section}`, `{disclaimer_section}`, `{data_sources_section}` placeholders. `plan()`: added `app_name`, `app_description`, `app_disclaimer`, `app_data_sources` params. `_build_system_prompt()`: builds identity section ("You are the AI assistant for **{name}**"), disclaimer section, data sources section. Added instruction #10 about disclaimer compliance. |
| `apps/appmgr/orchestrator/pipeline.py` | Step 2: tracks `is_new_session` flag; auto-inserts `welcome_prompt` as first assistant message. Step 5a: added `_get_data_source_names()` call. Step 9: passes `app_name`, `app_description`, `app_disclaimer`, `app_data_sources` to orchestrator. Steps 10/11/16: all return paths include `session_metadata` dict when `is_new_session=True`. Added `_get_data_source_names()` static helper method. |

### Known Remaining Items

| Item | Status | Notes |
|------|--------|-------|
| LLM wiring (app's bound LLM used for processing) | DEFERRED | Orchestrator and prompt enhancer still use hardcoded module-level `ChatOpenAI` instances. The app's selected LLM is visible in API responses and chat traces but not used for actual LLM calls. Commented OpenAI SDK code provided for future switch. |
| Designation/BU access control enforcement | PARTIAL | Names resolved and injected into orchestrator context. No hard access control check (blocking unauthorized users) because `UserContext` has no designation/BU profile data to compare against. |
| `available_for_all_users` enforcement | DEFERRED | Flag is stored and returned in API responses but not enforced as access control during chat. Would require user profile data (designation/BU) to compare against, which `UserContext` doesn't provide. |

---

## Phase 5: Integration & Polish

| # | Task | Status | Notes |
|---|------|--------|-------|
| 25 | End-to-end testing | PENDING | Real LLM calls |
| 26 | Session isolation verification | PENDING | user_id filtering |
| 27 | Guardrails testing | PENDING | PII, Safety, Compliance |
| 28 | HITL flow testing | PENDING | Clarification + fallback |

---

## Blockers & Decisions Log

| Date | Item | Resolution |
|------|------|------------|
| 2026-02-07 | SQLite vs PostgreSQL | PostgreSQL -- user explicit requirement |
| 2026-02-07 | API route prefix | `/api/applications/...` -- user choice |
| 2026-02-07 | Lookup tables | Separate PG tables with Alembic seed -- user choice |
| 2026-02-07 | Standalone agent chat gap | Will build `agent_executor.py` wrapping CrewAIAdapter |
| 2026-02-08 | DB session pattern | `async with get_db_session()` context manager (not Depends injection) |
| 2026-02-08 | User ID resolution | `Depends(get_current_user)` + fallback to `DEFAULT_USER_ID` for anonymous |
| 2026-02-08 | Router structure | 3 routers: `router` (CRUD), `lookup_router` (catalogs), `skills_router` (skills) -- all in routes.py, separately mounted in gateway |
| 2026-02-08 | Orchestrator package | Created `apps/appmgr/orchestrator/` with `__init__.py` and 9 module files (Tasks 13-21) |
| 2026-02-08 | agent_executor independence | `agent_executor.py` imports `crewai` library directly, does NOT import from `crewai_adapter.py`. Has its own `_bind_tools()` and `_create_tool_wrapper()` implementations |
| 2026-02-08 | WorkflowExecutor resolution | `skill_executor.py` resolves `WorkflowExecutor` via `container.resolve('workflow.executor')`. Calls `execute_workflow()` as a synchronous black box wrapped in `asyncio.to_thread()` |
| 2026-02-08 | Chat endpoints on same router | All 5 chat endpoints added to the existing `router` (prefix `/api/applications`). No new router needed since paths are under `/{id}/chat/...`. Gateway already mounts this router. |
| 2026-02-08 | Pipeline import strategy | OrchestrationPipeline imported inside the endpoint function (deferred import) to avoid circular import issues at module load time |
| 2026-02-08 | Document upload directory | Configurable via `APP_UPLOADS_DIR` env var, default `apps/appmgr/uploads/`. Directory auto-created at module load. |
| 2026-02-08 | Full code audit performed | PASS -- All 24 files exist and are non-empty. All 25 API endpoints implemented. All cross-file signatures match (no llm_manager leakage, no crewai_adapter imports). Publish validation checks all 6 required fields. Status lifecycle correct. Gateway mounts all 3 routers. Container registers both repos. No bugs or missing pieces found. Progress doc task notes 13/15/19 still reference LLMManager but actual code correctly uses module-level LLM instances (notes are historical, code is correct). |
| 2026-02-10 | Bug fix round after deployment testing | 12 issues identified and fixed (BF-1 through BF-12). See Bug Fixes section above for full details. Key fixes: empty lookup tables seeded, ApplicationCreate accepts all fields at creation time, setup PATCH accepts null values, guardrail categories resolved from DB, missing persona templates added, tags/designations/BU injected into orchestrator context, LLM names visible in responses, output formatting artifacts cleaned. |
| 2026-02-10 | Field usage audit | 6 fields identified as stored-but-unused in chat: welcome_prompt, disclaimer, starter_questions, name, description, data_source_links. All fixed: injected into orchestrator context and/or returned in ChatResponse session_metadata. New `ChatSessionMetadata` schema added. |
