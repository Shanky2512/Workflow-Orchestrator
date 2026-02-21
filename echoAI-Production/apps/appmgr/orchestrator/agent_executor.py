"""
EchoAI Orchestrator -- Standalone Agent Executor

Executes a single agent by loading its definition from the PostgreSQL
``agents`` table and running it through CrewAI.

IMPORTANT: This module is 100% INDEPENDENT of ``apps/workflow/crewai_adapter.py``.
    - Does NOT import from the adapter.
    - Imports ``crewai`` library directly: ``from crewai import Crew, Agent, Task, Process``
    - Uses a module-level CrewAI ``LLM`` instance (fill in placeholders before running)
    - Has its own ``_bind_tools()`` implementation

The pattern was *read* from ``crewai_adapter.py`` for reference but the code
here is a self-contained reimplementation for the application orchestrator's
standalone agent execution use-case.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from crewai import LLM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level CrewAI LLM instance -- fill in before running
# ------------------------------------------------------------------
# crewai_llm = LLM(
#     model="openai/",   # user will fill  (e.g. "openai/gpt-4o")
#     base_url="",       # user will fill
#     api_key="",        # user will fill
#     temperature=0.2,
# )

crewai_llm = LLM(
    model="liquid/lfm-2.5-1.2b-instruct:free",   # user will fill  (e.g. "openai/gpt-4o")
    base_url="https://openrouter.ai/api/v1",       # user will fill
    api_key="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",        # user will fill
    temperature=0.2,
)

class StandaloneAgentExecutor:
    """
    Executes a single agent from its PostgreSQL definition using CrewAI.
    """

    async def execute(
        self,
        db: AsyncSession,
        agent_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        sse_queue: Optional[asyncio.Queue] = None,
    ) -> Dict[str, Any]:
        """
        Load an agent definition from DB and execute it via CrewAI.

        Args:
            db: Async database session.
            agent_id: Agent UUID string.
            user_input: The task description / user input for the agent.
            context: Optional extra context dict (e.g. previous step output).
            sse_queue: Optional asyncio.Queue for A2UI SSE streaming.

        Returns:
            Dict with keys:
                result (str) -- the agent's output text
                agent_id (str) -- the agent UUID
                agent_name (str) -- the agent display name
        """
        from echolib.a2ui import A2UIStreamBuilder

        _builder = A2UIStreamBuilder()
        _run_id = f"agent_{agent_id}"

        async def _agent_emit(line: str) -> None:
            if sse_queue is not None:
                try:
                    sse_queue.put_nowait(line)
                except asyncio.QueueFull:
                    pass

        # 1. Load agent definition from DB
        agent_def = await self._load_agent_definition(db, agent_id)
        if agent_def is None:
            raise RuntimeError(f"Agent '{agent_id}' not found in database")

        definition = agent_def.get("definition", agent_def)

        # 2. Extract agent properties
        agent_name = definition.get("name", "Agent")
        agent_role = definition.get("role", "Assistant")
        agent_goal = definition.get("goal", f"Complete the assigned task: {agent_name}")
        agent_backstory = definition.get(
            "backstory",
            definition.get("description", f"Specialized {agent_role} agent"),
        )

        # 3. Build task description
        task_description = self._build_task_description(
            user_input, definition, context
        )

        # 4. Bind tools
        crewai_tools = self._bind_tools(definition)

        # 5. Emit running step to SSE queue
        step = {
            "id": f"step_agent_{agent_id[:8]}",
            "icon": "smart_toy",
            "label": f"Running agent: {agent_name}",
            "detail": "",
            "status": "running",
        }
        await _agent_emit(_builder.step_update(_run_id, [step]))

        # 6. Create CrewAI objects and run (uses module-level crewai_llm)
        try:
            result_text = await self._run_crew(
                agent_name=agent_name,
                agent_role=agent_role,
                agent_goal=agent_goal,
                agent_backstory=agent_backstory,
                crewai_tools=crewai_tools,
                crewai_llm=crewai_llm,
                task_description=task_description,
            )

            # Emit completed step
            step["status"] = "completed"
            await _agent_emit(_builder.step_update(_run_id, [step]))

        except Exception as exc:
            # Emit failed step
            step["status"] = "failed"
            step["detail"] = str(exc)[:120]
            await _agent_emit(_builder.step_update(_run_id, [step]))
            raise

        return {
            "result": result_text,
            "agent_id": agent_id,
            "agent_name": agent_name,
        }

    # ------------------------------------------------------------------
    # Load agent from DB
    # ------------------------------------------------------------------

    @staticmethod
    async def _load_agent_definition(
        db: AsyncSession, agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load agent row from the ``agents`` table and return its definition.
        """
        from echolib.models.agent import Agent as AgentModel
        from echolib.repositories.base import safe_uuid

        agt_uuid = safe_uuid(agent_id)
        if agt_uuid is None:
            return None

        stmt = select(AgentModel).where(
            AgentModel.agent_id == agt_uuid,
            AgentModel.is_deleted == False,  # noqa: E712
        )
        result = await db.execute(stmt)
        agent_row = result.scalar_one_or_none()

        if agent_row is None:
            return None

        # The definition JSONB contains the full agent config
        definition = agent_row.definition or {}
        # Ensure agent_id and name are present in the definition
        definition.setdefault("agent_id", str(agent_row.agent_id))
        definition.setdefault("name", agent_row.name)

        return {"definition": definition, "agent_id": str(agent_row.agent_id)}

    # ------------------------------------------------------------------
    # Task description builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_description(
        user_input: str,
        definition: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the CrewAI task description from user input and context."""
        parts = [f"USER REQUEST: {user_input}"]

        prompt = definition.get("prompt", "")
        if prompt:
            parts.append(f"AGENT INSTRUCTIONS: {prompt}")

        if context:
            prev_output = context.get("previous_output", "")
            if prev_output:
                parts.append(f"PREVIOUS STEP OUTPUT:\n{prev_output}")

        role = definition.get("role", "")
        if role:
            parts.append(f"YOUR ROLE: {role}")

        goal = definition.get("goal", "")
        if goal:
            parts.append(f"YOUR GOAL: {goal}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tool binding (independent implementation)
    # ------------------------------------------------------------------

    @staticmethod
    def _bind_tools(definition: Dict[str, Any]) -> List[Any]:
        """
        Bind tools to the agent based on its definition.

        Attempts to resolve tools from the DI container's tool registry.
        If the tool system is not available, returns an empty list
        (agent will run without tools).
        """
        tool_ids = definition.get("tools", [])
        if not tool_ids:
            return []

        crewai_tools: List[Any] = []

        try:
            from echolib.di import container
            tool_registry = container.resolve("tool.registry")
            tool_executor = container.resolve("tool.executor")
        except (KeyError, ImportError):
            logger.debug(
                "Tool system not available; agent will run without tools"
            )
            return []

        for tool_id in tool_ids:
            try:
                tool_def = tool_registry.get(tool_id)
                if tool_def is None:
                    # Try by name as fallback
                    tool_def = tool_registry.get_by_name(tool_id)
                if tool_def is None:
                    logger.warning("Tool '%s' not found in registry", tool_id)
                    continue

                crewai_tool = _create_tool_wrapper(tool_def, tool_executor)
                crewai_tools.append(crewai_tool)
                logger.info("Bound tool '%s' to standalone agent", tool_def.name)

            except Exception:
                logger.warning(
                    "Failed to bind tool '%s' to standalone agent", tool_id,
                    exc_info=True,
                )

        return crewai_tools

    # ------------------------------------------------------------------
    # CrewAI execution
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_crew(
        agent_name: str,
        agent_role: str,
        agent_goal: str,
        agent_backstory: str,
        crewai_tools: List[Any],
        crewai_llm,
        task_description: str,
    ) -> str:
        """
        Create a CrewAI Crew with a single agent and task, then execute.
        """
        from crewai import Crew, Agent, Task, Process

        agent = Agent(
            role=agent_role,
            goal=agent_goal,
            backstory=agent_backstory,
            tools=crewai_tools,
            allow_delegation=False,
            llm=crewai_llm,
            verbose=True,
        )

        task = Task(
            description=task_description,
            expected_output=f"Complete response from {agent_name} addressing the user's request",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        logger.info("Executing standalone agent: %s", agent_name)

        try:
            result = await asyncio.to_thread(crew.kickoff)
            output_text = result.raw if hasattr(result, "raw") else str(result)
        except (ValueError, AttributeError) as exc:
            error_msg = str(exc)
            if "None or empty" in error_msg or "'NoneType'" in error_msg:
                logger.warning(
                    "LLM returned empty response for agent '%s'. "
                    "Consider changing the model in agent_executor.py module-level crewai_llm.",
                    agent_name,
                )
                output_text = (
                    f"[Agent {agent_name} could not generate a response. "
                    f"The LLM model may not support the requested operation.]"
                )
            else:
                raise

        return output_text


# ---------------------------------------------------------------------------
# Tool wrapper factory (independent from crewai_adapter)
# ---------------------------------------------------------------------------

def _create_tool_wrapper(tool_def, executor) -> Any:
    """
    Create a CrewAI BaseTool wrapper from a ToolDef + ToolExecutor.

    This is a standalone implementation that mirrors the pattern in
    ``crewai_adapter.py`` but does NOT import from it.
    """
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field, create_model
    from typing import Optional as Opt

    captured_tool_def = tool_def
    captured_executor = executor

    # Build args schema from tool's input_schema
    input_schema = captured_tool_def.input_schema or {}
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields = {}
    for fname, fspec in properties.items():
        ftype = type_map.get(fspec.get("type", "string"), str)
        desc = fspec.get("description", "")
        default = fspec.get("default", ...)
        if fname in required_fields and default is ...:
            fields[fname] = (ftype, Field(..., description=desc))
        else:
            if default is ...:
                fields[fname] = (Opt[ftype], Field(None, description=desc))
            else:
                fields[fname] = (ftype, Field(default, description=desc))

    ArgsSchema = create_model("ToolArgsSchema", **fields)

    class WrappedTool(BaseTool):
        name: str = captured_tool_def.name
        description: str = captured_tool_def.description
        args_schema: type[BaseModel] = ArgsSchema

        def _run(self, **kwargs) -> str:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            captured_executor.invoke(
                                captured_tool_def.tool_id, kwargs
                            ),
                        )
                        result = future.result(timeout=60)
                except RuntimeError:
                    result = asyncio.run(
                        captured_executor.invoke(
                            captured_tool_def.tool_id, kwargs
                        )
                    )

                if result.success:
                    return json.dumps(result.output)
                else:
                    return json.dumps({"error": result.error, "tool_id": result.tool_id})
            except Exception as e:
                logger.error(
                    "Tool execution failed for %s: %s",
                    captured_tool_def.name, e,
                )
                return json.dumps({
                    "error": f"Tool execution failed: {str(e)}",
                    "tool_id": captured_tool_def.tool_id,
                })

    return WrappedTool()
