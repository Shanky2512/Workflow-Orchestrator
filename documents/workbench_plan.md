My Understanding of Your Requirement

  What exists today

  The platform (EchoAI) is a modular FastAPI application with this structure:

  apps/
    gateway/main.py       ← Single unified entry point, mounts all routers
    appmgr/               ← Application CRUD + orchestrated chat
      orchestrator/        ← Full pipeline: guardrails → prompt enhancer → LLM → skill executor → persona
    workflow/              ← Workflow designer, runtime, CrewAI adapter
    agent/                 ← Agent designer, factory, registry
    chat/                  ← Legacy chat orchestrator
    llm/ rag/ session/ connector/ tool/
    workbench/             ← Already exists with: container.py, routes.py, main.py, translation_service.py

  Normal Mode (Mode 1) works like this:
  POST /api/applications/{id}/chat
    → OrchestrationPipeline.run()
      → Pre-guardrails → Persona → Prompt Enhancer → Skill Manifest
      → Orchestrator LLM → HITL → Skill Executor → Post-guardrails → Save to DB

  Workbench today is minimal — it only has:
  - A TranslationService wrapping Azure Translator API
  - Two routes: POST /workbench/translate and POST /workbench/detect
  - Registered in DI as workbench.translation
  - Not mounted in the gateway — the workbench router is missing from gateway/main.py (lines 49-61 and 282-292 show all imported routers, workbench is absent)

  What you want

  Mode 2 (Workbench Mode) — a completely independent execution path:

  POST /api/workbench/chat → WorkbenchChatService → simple passthrough response
  POST /api/workbench/translate → TranslationService (existing)
  POST /api/workbench/detect → TranslationService (existing)

  Key architectural constraints:
  1. Zero orchestration — no pipeline, no prompt enhancer, no skill executor, no guardrails, no HITL, no CrewAI, no LangGraph
  2. Frontend does the LLM calls — backend is just a passthrough service, not an LLM caller
  3. Complete isolation — workbench code must never import from or depend on appmgr/orchestrator/, workflow/, agent entities, etc.
  4. Coexistence — both modes run simultaneously; mode selection happens on the frontend
  5. Extensible — designed so future workbench tools can be plugged in cleanly

  What needs to be built
  ┌─────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐     
  │         Deliverable         │                                                             Details                                                              │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ Restructure apps/workbench/ │ Move translation_service.py → services/translation_service.py, add services/chat_service.py, schemas.py, manager.py, __init__.py │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ WorkbenchChatService        │ Simple message-in/response-out service. No LLM call. Backend echoes/processes, frontend handles LLM                              │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ WorkbenchManager            │ Lifecycle manager exposing is_workbench_enabled(), handle_workbench_chat(), handle_translation(), handle_detection()             │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ Schemas                     │ Pydantic models for chat request/response (separate from appmgr/schemas.py)                                                      │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ Updated routes.py           │ Add POST /workbench/chat alongside existing translate/detect                                                                     │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ Updated container.py        │ Register workbench.chat and workbench.manager in DI                                                                              │     
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤     
  │ Gateway integration         │ Import workbench container + router in gateway/main.py, mount under /api prefix → endpoints become /api/workbench/*              │     
  └─────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘     
  What must NOT change

  - apps/appmgr/orchestrator/* (pipeline, orchestrator, skill_executor, etc.)
  - apps/workflow/* (runtime, crewai_adapter, executor)
  - llm_manager.py
  - apps/agent/*
  - The existing translation service behavior

  One thing I noticed

  The current workbench routes use prefix /workbench but the gateway doesn't mount them at all. Your requirement says endpoints should be /api/workbench/*. This means   
  when I mount in the gateway, I'll use prefix="/api" so the routes become /api/workbench/chat, /api/workbench/translate, /api/workbench/detect.



=================================================================================================================================================

## Here's a summary of what was implemented:                                                                                                                                                                                                                                                                               
  Implementation Complete                                                                                                                                                                                                                                                                                                                         
  Files written/updated:                                                                                                                                                   ┌────────────────────────────────┬─────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐                      │              File              │ Action  │                                               Purpose                                                │                      ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                      │ apps/workbench/__init__.py     │ Written │ Module docstring                                                                                     │                      ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                      │ apps/workbench/schema.py       │ Written │ WorkbenchChatRequest, WorkbenchChatResponse, WorkbenchStatusResponse                                 │                      ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                      │ apps/workbench/chat_service.py │ Written │ WorkbenchChatService — passthrough chat, generates session IDs, zero orchestrator deps               │                      ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                      │ apps/workbench/manager.py      │ Written │ WorkbenchManager — lifecycle coordinator, delegates to chat + translation services                   │                    
  ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ apps/workbench/container.py    │ Updated │ Registers workbench.chat, workbench.manager + existing workbench.translation                         │
  ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ apps/workbench/routes.py       │ Updated │ Added POST /workbench/chat, GET /workbench/status, cleaned up existing routes                        │
  ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ apps/workbench/main.py         │ Updated │ Proper DI trigger on import                                                                          │
  ├────────────────────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ apps/gateway/main.py           │ Updated │ 3 lines added — container import, router import, app.include_router(workbench_router, prefix="/api") │
  └────────────────────────────────┴─────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘
  Final API endpoints:

  POST /api/workbench/chat       → WorkbenchChatService (passthrough)
  POST /api/workbench/translate   → TranslationService (existing, unchanged)
  POST /api/workbench/detect      → TranslationService (existing, unchanged)
  GET  /api/workbench/health      → Health check
  GET  /api/workbench/status      → Workbench availability + service list

  Isolation guarantee:

  - Zero imports from appmgr/orchestrator/, workflow/, agent/, or llm_manager
  - Existing orchestrated mode (POST /api/applications/{id}/chat) is completely untouched