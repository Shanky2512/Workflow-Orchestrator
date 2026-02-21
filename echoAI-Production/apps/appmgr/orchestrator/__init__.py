"""
EchoAI Application Orchestrator Package

Phase 3 implementation of the orchestration engine.
Contains all components of the multi-step pipeline that powers
application chat: prompt enhancement, skill planning, guardrails,
persona injection, HITL management, skill execution, and the
top-level pipeline coordinator.

Modules:
    prompt_enhancer  -- LLM-based prompt improvement
    skill_manifest   -- Build skill metadata for orchestrator prompt
    orchestrator     -- Core LLM that selects and sequences skills
    guardrails       -- Pre/post-processing safety checks (regex + rules)
    persona          -- Persona injection into system prompts
    hitl             -- Human-in-the-loop conversation state management
    agent_executor   -- Standalone CrewAI agent execution (independent)
    skill_executor   -- Execute orchestrator plans (workflows + agents)
    pipeline         -- Top-level coordinator tying all steps together
"""
