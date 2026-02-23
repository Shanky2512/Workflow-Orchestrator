"""
CrewAI Adapter Module

CRITICAL ARCHITECTURAL RULES:
============================
1. LangGraph OWNS: Workflow topology, execution order, branching, merging, state
2. CrewAI is ONLY invoked INSIDE LangGraph node functions
3. CrewAI HANDLES: Agent collaboration, delegation, parallelism WITHIN nodes
4. CrewAI NEVER: Controls graph traversal or state transitions
5. State flow: LangGraph state → CrewAI → LangGraph state

This adapter creates LangGraph node functions that execute CrewAI crews.
The node functions are called BY LangGraph, not the other way around.
"""

from typing import Dict, Any, List, Callable, Optional
import os
import asyncio
import json
import re
from datetime import datetime
import logging

from echolib.config import settings
from echolib.utils import new_id

# Langfuse observability — import @observe decorator for tracing.
# Wrapped in try/except so the module still loads if langfuse is not installed.
try:
    from langfuse import observe as _langfuse_observe
except ImportError:
    # Fallback: no-op decorator if langfuse is not available
    def _langfuse_observe(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(fn):
            return fn
        return decorator

logger = logging.getLogger(__name__)


def _is_transparency_enabled() -> bool:
    """Check if execution transparency is enabled."""
    return settings.transparency_enabled


def _get_run_id(state: Dict[str, Any]) -> str:
    """Extract run_id from state, or generate a placeholder."""
    return state.get("run_id", "unknown")


def _sanitize_input(state: Dict[str, Any], max_length: int = 200) -> Dict[str, Any]:
    """Create sanitized input summary (no CoT, truncated)."""
    user_input = (
        state.get("original_user_input")
        or state.get("user_input")
        or state.get("message")
        or ""
    )
    return {
        "user_input": user_input[:max_length] if user_input else "",
        "has_previous_context": bool(state.get("crew_result"))
    }


def _sanitize_output(output_text: str, tools: List = None, max_length: int = 500) -> Dict[str, Any]:
    """Create sanitized output summary (no CoT, truncated)."""
    return {
        "output_preview": output_text[:max_length] if output_text else "",
        "tools_used": [t.name for t in tools] if tools else []
    }


def _fire_and_forget(coro):
    """
    Schedule a coroutine for fire-and-forget execution.
    Works from both sync and async contexts.
    Non-blocking - does not wait for result.

    CRITICAL: When called from sync code inside an async context (common in LangGraph),
    we must use run_coroutine_threadsafe to properly schedule the task.
    """
    try:
        # First, try to get the running loop (only works from async context)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - create_task works directly
            loop.create_task(coro)
            logger.debug("Event scheduled via create_task (async context)")
            return
        except RuntimeError:
            # No running loop in current coroutine context - we're in sync code
            pass

        # We're in sync code - try to get the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("No event loop available for transparency event")
            return

        if loop.is_running():
            # Loop is running (in another call frame) but we're in sync code
            # Use run_coroutine_threadsafe which is designed for this exact case
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            # Don't wait for result - fire and forget
            logger.debug("Event scheduled via run_coroutine_threadsafe (sync context)")
        else:
            # Loop exists but not running - run synchronously
            loop.run_until_complete(coro)
            logger.debug("Event executed via run_until_complete")
    except Exception as e:
        # Log but don't fail - transparency is optional enhancement
        logger.warning(f"Failed to schedule transparency event: {e}")


class CrewAIAdapter:
    """
    Adapter for integrating CrewAI with LangGraph workflows.

    This class creates LangGraph-compatible node functions that execute
    CrewAI crews for agent collaboration within nodes.

    IMPORTANT: This adapter does NOT create graph structures. It creates
    node functions that LangGraph calls as part of ITS graph execution.
    """

    def __init__(self):
        """Initialize the CrewAI adapter."""
        # LLM caching is now handled by LLMManager
        pass

    # ========================================================================
    # TOOL BINDING METHODS
    # ========================================================================

    def _get_tool_executor(self):
        """
        Get ToolExecutor from DI container.

        Returns:
            ToolExecutor instance for invoking tools
        """
        from echolib.di import container
        return container.resolve('tool.executor')

    def _get_tool_registry(self):
        """
        Get ToolRegistry from DI container.

        Returns:
            ToolRegistry instance for looking up tool definitions
        """
        from echolib.di import container
        return container.resolve('tool.registry')

    def _create_crewai_tool_wrapper(self, tool_def, executor):
        """
        Create a CrewAI-compatible tool wrapper from a ToolDef.

        This method creates a dynamic CrewAI BaseTool subclass that wraps
        our ToolDef and uses the ToolExecutor to invoke it. CrewAI requires
        synchronous tool methods, so we handle the async executor here.

        Args:
            tool_def: ToolDef instance with tool metadata and execution config
            executor: ToolExecutor instance for tool execution

        Returns:
            CrewAI BaseTool instance that wraps the tool
        """
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field, create_model
        from typing import Optional, Any

        # Capture tool_def and executor in closure for the dynamic class
        captured_tool_def = tool_def
        captured_executor = executor

        # Create dynamic Pydantic model from tool's input_schema
        def create_args_schema(input_schema: dict):
            """Create a Pydantic model from JSON Schema."""
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            fields = {}
            for field_name, field_spec in properties.items():
                field_type = str  # Default to string

                # Map JSON Schema types to Python types
                json_type = field_spec.get("type", "string")
                if json_type == "string":
                    field_type = str
                elif json_type == "integer":
                    field_type = int
                elif json_type == "number":
                    field_type = float
                elif json_type == "boolean":
                    field_type = bool
                elif json_type == "array":
                    field_type = list
                elif json_type == "object":
                    field_type = dict

                # Get default and description
                default_value = field_spec.get("default", ...)
                description = field_spec.get("description", "")

                # If field is required and has no default, use ... (required)
                if field_name in required and default_value is ...:
                    fields[field_name] = (field_type, Field(..., description=description))
                else:
                    # Optional field or has default
                    if default_value is ...:
                        fields[field_name] = (Optional[field_type], Field(None, description=description))
                    else:
                        fields[field_name] = (field_type, Field(default_value, description=description))

            # Create the dynamic model
            return create_model("ToolArgsSchema", **fields)

        # Generate args_schema from tool's input_schema
        args_schema_class = create_args_schema(captured_tool_def.input_schema or {})

        class DynamicCrewAITool(BaseTool):
            """
            Dynamically created CrewAI tool wrapper.

            This class wraps an EchoAI ToolDef and executes it via ToolExecutor.
            """
            name: str = captured_tool_def.name
            description: str = captured_tool_def.description
            args_schema: type[BaseModel] = args_schema_class

            def _run(self, **kwargs) -> str:
                """
                Execute the tool synchronously.

                CrewAI expects sync methods, so we handle the async executor here
                by running it in an event loop.

                Args:
                    **kwargs: Tool input parameters

                Returns:
                    JSON string of tool output or error
                """
                try:
                    # ToolExecutor.invoke is async, so we need to run it
                    # Try to get running event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, need to handle carefully
                        # Create a new thread to run the async code
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(
                                asyncio.run,
                                captured_executor.invoke(captured_tool_def.tool_id, kwargs)
                            )
                            result = future.result(timeout=60)
                    except RuntimeError:
                        # No running loop, create new one
                        result = asyncio.run(
                            captured_executor.invoke(captured_tool_def.tool_id, kwargs)
                        )

                    # Check if execution was successful
                    if result.success:
                        # Serialize output to string for CrewAI
                        return json.dumps(result.output)
                    else:
                        # Return error message as JSON
                        return json.dumps({
                            "error": result.error,
                            "tool_id": result.tool_id
                        })

                except Exception as e:
                    # Handle any execution errors gracefully
                    logger.error(f"Tool execution failed for {captured_tool_def.name}: {e}")
                    return json.dumps({
                        "error": f"Tool execution failed: {str(e)}",
                        "tool_id": captured_tool_def.tool_id
                    })

        # Instantiate and return the tool
        return DynamicCrewAITool()

    def _bind_tools_to_agent(self, agent_config: Dict[str, Any]) -> List[Any]:
        """
        Bind tools to an agent based on its configuration.

        This method retrieves tool definitions from the registry and creates
        CrewAI-compatible tool wrappers for each one.

        Args:
            agent_config: Agent configuration dictionary with 'tools' key

        Returns:
            List of CrewAI tool instances ready for agent use
        """
        tool_ids = agent_config.get("tools", [])
        tool_ids = list(dict.fromkeys(tool_ids))  # deduplicate, preserve order
        crewai_tools = []

        if not tool_ids:
            logger.debug(f"Agent has no tools configured")
            return crewai_tools

        try:
            tool_registry = self._get_tool_registry()
            tool_executor = self._get_tool_executor()
        except KeyError as e:
            logger.warning(f"Tool system not available: {e}. Agent will run without tools.")
            return crewai_tools

        for tool_id in tool_ids:
            try:
                tool_def = tool_registry.get(tool_id)
                if tool_def:
                    # Create CrewAI-compatible tool wrapper
                    crewai_tool = self._create_crewai_tool_wrapper(tool_def, tool_executor)
                    crewai_tools.append(crewai_tool)
                    logger.info(f"Bound tool '{tool_def.name}' (id={tool_id}) to agent")
                else:
                    # Try finding by name as fallback (for frontend compatibility)
                    tool_def = tool_registry.get_by_name(tool_id)
                    if tool_def:
                        crewai_tool = self._create_crewai_tool_wrapper(tool_def, tool_executor)
                        crewai_tools.append(crewai_tool)
                        logger.info(f"Bound tool '{tool_def.name}' (by name lookup) to agent")
                    else:
                        logger.warning(f"Tool '{tool_id}' not found in registry, skipping")
            except Exception as e:
                # Log but don't fail - agent can still work without this tool
                logger.warning(f"Failed to bind tool '{tool_id}': {e}")

        logger.info(f"Successfully bound {len(crewai_tools)}/{len(tool_ids)} tools to agent")
        return crewai_tools

    # ========================================================================
    # HIERARCHICAL WORKFLOWS: Manager + Workers with Dynamic Delegation
    # ========================================================================

    @_langfuse_observe(name="crewai_create_hierarchical_node")
    def create_hierarchical_crew_node(
        self,
        master_agent_config: Dict[str, Any],
        sub_agent_configs: List[Dict[str, Any]],
        delegation_strategy: str = "dynamic"
    ) -> Callable:
        """
        Create a LangGraph node function that uses CrewAI for hierarchical coordination.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node as part of its graph execution
        - Inside this node, CrewAI Manager delegates to workers
        - Manager receives ORIGINAL user input to make delegation decisions
        - CrewAI returns results to LangGraph state
        - LangGraph decides what node to execute next

        Args:
            master_agent_config: Manager agent configuration
            sub_agent_configs: Worker agent configurations
            delegation_strategy: "dynamic" (manager decides) or "all" (invoke all)

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        @_langfuse_observe(name="crewai_hierarchical_execution")
        def hierarchical_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes CrewAI hierarchical crew.

            This function is CALLED BY LangGraph during graph execution.
            CrewAI handles agent collaboration WITHIN this function.
            """
            # === TRANSPARENCY: Setup ===
            run_id = _get_run_id(state)
            step_id = new_id("step_")
            step_name = master_agent_config.get("name", "Hierarchical Crew")
            worker_names = [w.get("name", f"Worker{i+1}") for i, w in enumerate(sub_agent_configs)]

            if _is_transparency_enabled():
                from .runtime.transparency import step_tracker, StepType
                from .runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync, publish_step_failed_sync

                input_summary = _sanitize_input(state)
                input_summary["manager"] = step_name
                input_summary["workers"] = worker_names
                input_summary["worker_count"] = len(sub_agent_configs)

                step_tracker.start_step(run_id, step_id, step_name, StepType.HIERARCHICAL_CREW, input_summary)
                publish_step_started_sync(run_id, step_id, step_name, StepType.HIERARCHICAL_CREW, input_summary)

            try:
                from crewai import Crew, Agent, Task, Process

                logger.info(f"Hierarchical Crew node executing with {len(sub_agent_configs)} workers")

                # FIXED: Extract ORIGINAL user input (preserved through workflow)
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("task_description")
                    or state.get("message")
                    or ""
                )

                # Get any previous context
                previous_context = state.get("crew_result", "")

                logger.info(f"Hierarchical execution for user request: {original_input[:100]}...")

                # Build comprehensive task description for manager
                task_parts = [f"USER REQUEST: {original_input}"]

                manager_prompt = master_agent_config.get("prompt", "")
                if manager_prompt:
                    task_parts.append(f"MANAGER INSTRUCTIONS: {manager_prompt}")

                if previous_context:
                    task_parts.append(f"PREVIOUS CONTEXT: {previous_context}")

                # List available workers for manager
                worker_descriptions = []
                for idx, wc in enumerate(sub_agent_configs):
                    worker_descriptions.append(
                        f"- {wc.get('name', f'Worker{idx+1}')}: {wc.get('role', 'Worker')} - {wc.get('goal', 'Complete assigned tasks')}"
                    )
                if worker_descriptions:
                    task_parts.append(f"AVAILABLE WORKERS:\n" + "\n".join(worker_descriptions))

                task_parts.append("Coordinate the workers to accomplish the user's request. Delegate tasks appropriately.")

                task_description = "\n\n".join(task_parts)
                expected_output = state.get("expected_output", "Completed results from hierarchical coordination addressing the user's request")

                # Bind tools to manager agent
                manager_tools = self._bind_tools_to_agent(master_agent_config)
                if manager_tools:
                    logger.info(f"Manager agent will use {len(manager_tools)} tool(s)")
                    # Add tool usage instruction to task description
                    tool_names = [t.name for t in manager_tools]
                    tool_instruction = (
                        f"AVAILABLE TOOLS: {', '.join(tool_names)}\n"
                        f"IMPORTANT: Use your available tools to enhance your response with real-time or external data. "
                        f"Combine tool results with your own knowledge for the best answer. "
                        f"Do not claim you cannot access information if you have tools that can help."
                    )
                    task_description = task_description + "\n\n" + tool_instruction

                # Create CrewAI Manager Agent (can delegate)
                manager = Agent(
                    role=master_agent_config.get("role", "Manager"),
                    goal=master_agent_config.get("goal") or f"Coordinate workers to accomplish: {original_input[:200]}",
                    backstory=master_agent_config.get("description", "Experienced manager coordinating specialized workers"),
                    tools=manager_tools,  # Pass bound tools to manager
                    allow_delegation=True,  # CRITICAL: Manager can delegate
                    llm=self._get_llm_for_agent(master_agent_config),
                    verbose=True
                )

                # Create CrewAI Worker Agents (cannot delegate)
                workers = []
                for worker_config in sub_agent_configs:
                    # Bind tools to each worker agent
                    worker_tools = self._bind_tools_to_agent(worker_config)
                    if worker_tools:
                        logger.info(f"Worker '{worker_config.get('name', 'Worker')}' will use {len(worker_tools)} tool(s)")

                    worker_goal = worker_config.get("goal") or f"Complete specialized task: {worker_config.get('name', 'work')}"
                    worker = Agent(
                        role=worker_config.get("role", "Worker"),
                        goal=worker_goal,
                        backstory=worker_config.get("description", "Specialized worker agent"),
                        tools=worker_tools,  # Pass bound tools to worker
                        allow_delegation=False,  # CRITICAL: Workers don't delegate (prevents loops)
                        llm=self._get_llm_for_agent(worker_config),
                        verbose=True
                    )
                    workers.append(worker)

                # Create main task for manager
                main_task = Task(
                    description=task_description,
                    expected_output=expected_output,
                    agent=manager  # Assigned to manager
                )

                # Create CrewAI Crew with HIERARCHICAL process
                crew = Crew(
                    agents=[manager] + workers,  # Manager first, then workers
                    tasks=[main_task],
                    process=Process.hierarchical,  # CRITICAL: Hierarchical delegation
                    manager_llm=self._get_llm_for_agent(master_agent_config),
                    verbose=True
                )

                # Execute crew with LLM failure handling (CrewAI handles delegation internally)
                logger.info(f"Executing hierarchical Crew with manager + {len(workers)} workers...")
                try:
                    result = crew.kickoff()
                    output_text = result.raw if hasattr(result, 'raw') else str(result)
                except (ValueError, AttributeError) as llm_error:
                    error_msg = str(llm_error)
                    if "None or empty" in error_msg or "'NoneType'" in error_msg:
                        logger.warning(
                            f"LLM returned empty response in hierarchical execution. "
                            f"Consider changing DEFAULT_MODEL in llm_manager.py to a model that supports tool calling."
                        )
                        # Return graceful fallback
                        return {
                            "hierarchical_output": f"[Hierarchical execution failed: LLM returned empty response]",
                            "crew_result": f"[Hierarchical execution incomplete due to LLM error]",
                            "original_user_input": original_input,
                            "messages": [{
                                "node": "hierarchical_crew",
                                "error": "LLM returned empty response",
                                "manager": master_agent_config.get("name", "Manager"),
                                "workers": [w.get("name", "Worker") for w in sub_agent_configs],
                                "timestamp": datetime.utcnow().isoformat()
                            }]
                        }
                    else:
                        raise

                # === TRANSPARENCY: Step completed ===
                if _is_transparency_enabled():
                    output_summary = _sanitize_output(output_text, manager_tools)
                    output_summary["manager"] = step_name
                    output_summary["workers"] = worker_names
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                # Extract state variables from hierarchical output
                extracted_vars = self._extract_state_variables(output_text, state)

                # Return only new data - LangGraph merges via state schema
                logger.info("Hierarchical Crew execution completed")
                return {
                    **extracted_vars,
                    "hierarchical_output": output_text,
                    "crew_result": output_text,
                    "_last_agent_output": output_text,
                    "original_user_input": original_input,  # Preserve for downstream
                    "messages": [{
                        "node": "hierarchical_crew",
                        "manager": master_agent_config.get("name", "Manager"),
                        "workers": [w.get("name", "Worker") for w in sub_agent_configs],
                        "original_request": original_input[:200] if original_input else "",
                        "output_preview": output_text[:500] if output_text else "",
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                # === TRANSPARENCY: Step failed ===
                if _is_transparency_enabled():
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                logger.error(f"CrewAI hierarchical execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fail fast - no silent errors
                raise RuntimeError(f"CrewAI hierarchical node failed: {e}")

        return hierarchical_node

    # ========================================================================
    # PARALLEL WORKFLOWS: Multiple Agents Executing Concurrently
    # ========================================================================

    @_langfuse_observe(name="crewai_create_parallel_node")
    def create_parallel_crew_node(
        self,
        agent_configs: List[Dict[str, Any]],
        aggregation_strategy: str = "combine"
    ) -> Callable:
        """
        Create a LangGraph node function for parallel agent execution.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node during graph execution
        - Inside this node, CrewAI executes ALL agents in a single Crew
        - Each agent gets a task with the ORIGINAL user input + their specific focus
        - Results are aggregated and returned to LangGraph state
        - True parallel context: all agents can see each other's work

        Args:
            agent_configs: List of agent configurations to run in parallel
            aggregation_strategy: How to merge results ("combine", "vote", "prioritize")

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        @_langfuse_observe(name="crewai_parallel_execution")
        def parallel_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes CrewAI agents in parallel.

            This function is CALLED BY LangGraph during graph execution.
            CrewAI handles parallel agent execution WITHIN this function.
            """
            # === TRANSPARENCY: Setup ===
            run_id = _get_run_id(state)
            step_id = new_id("step_")
            agent_names = [a.get("name", f"Agent{i+1}") for i, a in enumerate(agent_configs)]
            step_name = f"Parallel Crew ({len(agent_configs)} agents)"

            if _is_transparency_enabled():
                from .runtime.transparency import step_tracker, StepType
                from .runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync, publish_step_failed_sync

                input_summary = _sanitize_input(state)
                input_summary["agents"] = agent_names
                input_summary["agent_count"] = len(agent_configs)
                input_summary["aggregation_strategy"] = aggregation_strategy

                step_tracker.start_step(run_id, step_id, step_name, StepType.PARALLEL_CREW, input_summary)
                publish_step_started_sync(run_id, step_id, step_name, StepType.PARALLEL_CREW, input_summary)

            try:
                from crewai import Crew, Agent, Task, Process

                logger.info(f"Parallel Crew node executing with {len(agent_configs)} agents")

                # FIXED: Extract ORIGINAL user input (preserved from coordinator)
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("message")
                    or state.get("task_description")
                    or ""
                )

                # Also get any previous context
                previous_context = state.get("crew_result", "")

                logger.info(f"Parallel execution for user request: {original_input[:100]}...")

                # Create CrewAI agents (all work on the same user request)
                agents = []
                tasks = []
                individual_results = []

                for idx, agent_config in enumerate(agent_configs):
                    agent_name = agent_config.get("name", f"Agent{idx+1}")
                    agent_role = agent_config.get("role", f"Parallel Agent {idx+1}")
                    agent_goal = agent_config.get("goal", "")
                    agent_prompt = agent_config.get("prompt", "")

                    # Bind tools to parallel agent
                    agent_tools = self._bind_tools_to_agent(agent_config)
                    if agent_tools:
                        logger.info(f"Parallel agent '{agent_name}' will use {len(agent_tools)} tool(s)")

                    # Create agent
                    agent = Agent(
                        role=agent_role,
                        goal=agent_goal or f"Process: {agent_name}",
                        backstory=agent_config.get("description", f"Specialized {agent_role} agent"),
                        tools=agent_tools,  # Pass bound tools to agent
                        allow_delegation=False,  # Parallel agents work independently
                        llm=self._get_llm_for_agent(agent_config),
                        verbose=True
                    )
                    agents.append(agent)

                    # FIXED: Build comprehensive task description
                    task_parts = [f"USER REQUEST: {original_input}"]

                    if agent_prompt:
                        task_parts.append(f"YOUR SPECIFIC INSTRUCTIONS: {agent_prompt}")

                    if previous_context:
                        task_parts.append(f"PREVIOUS CONTEXT: {previous_context}")

                    task_parts.append(f"YOUR ROLE: {agent_role}")
                    task_parts.append(f"Focus on your specific expertise and provide a complete response.")

                    task_description = "\n\n".join(task_parts)

                    # Add tool usage instruction if agent has tools
                    if agent_tools:
                        tool_names = [t.name for t in agent_tools]
                        tool_instruction = (
                            f"AVAILABLE TOOLS: {', '.join(tool_names)}\n"
                            f"IMPORTANT: Use your available tools to enhance your response with real-time or external data. "
                            f"Combine tool results with your own knowledge for the best answer. "
                            f"Do not claim you cannot access information if you have tools that can help."
                        )
                        task_description = task_description + "\n\n" + tool_instruction

                    # Create task for this agent
                    task = Task(
                        description=task_description,
                        expected_output=f"Complete response from {agent_name} addressing the user request",
                        agent=agent
                    )
                    tasks.append(task)

                # Create crew with sequential process
                # Note: CrewAI's sequential here means tasks execute in order,
                # but each agent works on its own task independently
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )

                # Execute crew with LLM failure handling
                logger.info(f"Executing parallel Crew with {len(agents)} agents...")
                try:
                    result = crew.kickoff()
                except (ValueError, AttributeError) as llm_error:
                    error_msg = str(llm_error)
                    if "None or empty" in error_msg or "'NoneType'" in error_msg:
                        logger.warning(
                            f"LLM returned empty response in parallel execution. "
                            f"Consider changing DEFAULT_MODEL in llm_manager.py to a model that supports tool calling."
                        )
                        # Return graceful fallback
                        return {
                            "parallel_output": f"[Parallel execution failed: LLM returned empty response]",
                            "individual_outputs": [],
                            "crew_result": f"[Parallel execution incomplete due to LLM error]",
                            "original_user_input": original_input,
                            "messages": [{
                                "node": "parallel_crew",
                                "error": "LLM returned empty response",
                                "agents": [a.get("name", f"Agent{i+1}") for i, a in enumerate(agent_configs)],
                                "timestamp": datetime.utcnow().isoformat()
                            }]
                        }
                    else:
                        raise

                # Collect individual results from each task
                for task in crew.tasks:
                    if hasattr(task, 'output'):
                        output = task.output.raw if hasattr(task.output, 'raw') else str(task.output)
                        individual_results.append(output)

                # Aggregate results based on strategy
                if aggregation_strategy == "combine":
                    aggregated = self._combine_results(individual_results)
                elif aggregation_strategy == "vote":
                    aggregated = self._vote_on_results(individual_results)
                elif aggregation_strategy == "prioritize":
                    aggregated = individual_results[0] if individual_results else ""
                else:
                    aggregated = "\n\n".join(individual_results)

                # === TRANSPARENCY: Step completed ===
                if _is_transparency_enabled():
                    output_summary = {
                        "output_preview": aggregated[:500] if aggregated else "",
                        "agents": agent_names,
                        "result_count": len(individual_results),
                        "aggregation_strategy": aggregation_strategy
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                # Extract state variables from aggregated parallel output
                extracted_vars = self._extract_state_variables(aggregated, state)

                # Return only new data - LangGraph merges via state schema
                logger.info(f"Parallel Crew execution completed: {len(individual_results)} results aggregated")
                return {
                    **extracted_vars,
                    "parallel_output": aggregated,
                    "individual_outputs": individual_results,
                    "crew_result": aggregated,
                    "_last_agent_output": aggregated,
                    "messages": [{
                        "node": "parallel_crew",
                        "agents": [a.get("name", f"Agent{i+1}") for i, a in enumerate(agent_configs)],
                        "original_request": original_input[:200] if original_input else "",
                        "result_count": len(individual_results),
                        "aggregation_strategy": aggregation_strategy,
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                # === TRANSPARENCY: Step failed ===
                if _is_transparency_enabled():
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                logger.error(f"CrewAI parallel execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Fail fast
                raise RuntimeError(f"CrewAI parallel node failed: {e}")

        return parallel_node

    # ========================================================================
    # SEQUENTIAL WORKFLOWS: Single Agent with CrewAI
    # ========================================================================

    @_langfuse_observe(name="crewai_create_sequential_node")
    def create_sequential_agent_node(
        self,
        agent_config: Dict[str, Any]
    ) -> Callable:
        """
        Create a LangGraph node function for a single agent using CrewAI.

        ARCHITECTURE (FIXED):
        - LangGraph calls this node as part of sequential execution
        - Inside this node, CrewAI executes a single agent
        - Agent receives BOTH original user input AND previous agent output
        - Results returned to LangGraph state
        - LangGraph decides next node in sequence

        Args:
            agent_config: Configuration for the agent

        Returns:
            Callable node function compatible with LangGraph StateGraph
        """
        @_langfuse_observe(name="crewai_sequential_execution")
        def sequential_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            LangGraph node that executes a single CrewAI agent.

            This function is CALLED BY LangGraph during graph execution.
            """
            # === TRANSPARENCY: Setup ===
            run_id = _get_run_id(state)
            step_id = new_id("step_")
            step_name = agent_config.get("name", "Agent")

            if _is_transparency_enabled():
                from .runtime.transparency import step_tracker, StepType
                from .runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync, publish_step_failed_sync

                input_summary = _sanitize_input(state)
                input_summary["agent_name"] = step_name
                input_summary["agent_role"] = agent_config.get("role", "Processor")

                step_tracker.start_step(run_id, step_id, step_name, StepType.AGENT, input_summary)
                publish_step_started_sync(run_id, step_id, step_name, StepType.AGENT, input_summary)

            try:
                from crewai import Crew, Agent, Task, Process

                agent_name = agent_config.get("name", "Agent")
                agent_role = agent_config.get("role", "Processor")
                agent_goal = agent_config.get("goal", "")
                agent_description = agent_config.get("description", "")

                logger.info(f"Sequential node executing agent: {agent_name}")

                # --- FIXED: Extract ORIGINAL user input (preserved through workflow) ---
                original_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("message")
                    or state.get("task_description")
                    or state.get("question")
                    or state.get("input")
                    or ""
                )

                # Current user input (may differ from original in chat scenarios)
                current_input = (
                    state.get("user_input")
                    or state.get("message")
                    or original_input
                )

                logger.info(f"Agent {agent_name} processing request: {original_input[:100]}...")

                # Extract inputs from agent's input_schema
                input_schema = agent_config.get("input_schema", [])
                inputs = {key: state.get(key) for key in input_schema if state.get(key) is not None}

                # Get previous agent's output as context
                previous_output = state.get("crew_result", "")

                # Get parallel results if this agent is after a parallel section
                parallel_output = state.get("parallel_output", "")

                # --- Build comprehensive task description ---
                task_parts = []

                # ALWAYS include original user request first
                if original_input:
                    task_parts.append(f"ORIGINAL USER REQUEST: {original_input}")

                # Add agent's configured prompt/instructions
                # Guard: skip prompt if it looks like code (frontend bug where
                # Code node's code leaks into Agent node's prompt field)
                agent_prompt = agent_config.get("prompt", "")
                if agent_prompt and self._is_code_like_prompt(agent_prompt):
                    logger.warning(
                        f"Agent '{agent_name}' prompt looks like code, not instructions — skipping. "
                        f"Re-save the workflow to fix the prompt."
                    )
                    agent_prompt = ""
                if agent_prompt:
                    task_parts.append(f"YOUR INSTRUCTIONS: {agent_prompt}")

                # Add context from previous agents in the workflow
                if previous_output:
                    task_parts.append(f"PREVIOUS AGENT OUTPUT:\n{previous_output}")

                # Add parallel results if available (for post-merge agents)
                if parallel_output and parallel_output != previous_output:
                    task_parts.append(f"PARALLEL EXECUTION RESULTS:\n{parallel_output}")

                # Add extracted inputs if any
                if inputs:
                    task_parts.append(f"INPUT DATA: {inputs}")

                # Add role context
                task_parts.append(f"YOUR ROLE: {agent_role}")
                task_parts.append(f"YOUR GOAL: {agent_goal or 'Complete your assigned task based on the user request'}")

                # --- Structured output hint ---
                # If the workflow has state_schema variables, instruct the agent
                # to include them as explicit key-value pairs so downstream
                # conditional nodes can read them from state.
                state_schema = state.get("_state_schema", {})
                if state_schema:
                    schema_vars = [
                        k for k in state_schema
                        if not k.startswith("_")
                    ]
                    if schema_vars:
                        vars_list = ", ".join(
                            f"{k} ({state_schema[k]})" for k in schema_vars
                        )
                        task_parts.append(
                            f"IMPORTANT OUTPUT REQUIREMENT: At the end of your response, "
                            f"include a clearly labeled section with these variables as "
                            f"key-value pairs on separate lines:\n"
                            f"{chr(10).join(f'  {k}: <value>' for k in schema_vars)}\n"
                            f"Variables: {vars_list}"
                        )

                # Build final task description
                if task_parts:
                    task_description = "\n\n".join(task_parts)
                else:
                    # Fallback: use agent role/goal as the task
                    task_description = (
                        f"You are a {agent_role}. "
                        f"{agent_goal or agent_description or 'Complete your assigned task.'}"
                    )

                # --- Bind tools to agent ---
                # Get tools assigned to this agent and create CrewAI wrappers
                crewai_tools = self._bind_tools_to_agent(agent_config)
                if crewai_tools:
                    logger.info(f"Agent '{agent_name}' will use {len(crewai_tools)} tool(s)")
                    # Add tool usage instruction to task description
                    tool_names = [t.name for t in crewai_tools]
                    tool_instruction = (
                        f"AVAILABLE TOOLS: {', '.join(tool_names)}\n"
                        f"IMPORTANT: Use your available tools to enhance your response with real-time or external data. "
                        f"Combine tool results with your own knowledge for the best answer. "
                        f"Do not claim you cannot access information if you have tools that can help."
                    )
                    task_description = task_description + "\n\n" + tool_instruction

                # --- Create CrewAI agent ---
                agent = Agent(
                    role=agent_role,
                    goal=agent_goal or f"Complete task: {agent_name}",
                    backstory=agent_description or f"Specialized {agent_role} agent",
                    tools=crewai_tools,  # Pass bound tools to agent
                    allow_delegation=False,
                    llm=self._get_llm_for_agent(agent_config),
                    verbose=True
                )

                # Create task
                task = Task(
                    description=task_description,
                    expected_output=f"Complete response from {agent_name} addressing the user's request",
                    agent=agent
                )

                # Create crew with single agent
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )

                # Execute with LLM failure handling
                logger.info(f"Executing sequential agent: {agent_name}...")
                try:
                    result = crew.kickoff()
                    output_text = result.raw if hasattr(result, 'raw') else str(result)
                except (ValueError, AttributeError) as llm_error:
                    # Handle LLM returning None/empty responses
                    error_msg = str(llm_error)
                    if "None or empty" in error_msg or "'NoneType'" in error_msg:
                        logger.warning(
                            f"LLM returned empty response for agent {agent_name}. "
                            f"This often happens with models that don't support tool calling well. "
                            f"Consider changing DEFAULT_MODEL in llm_manager.py to a model like "
                            f"'mistralai/mistral-7b-instruct:free' or 'openai/gpt-3.5-turbo'"
                        )
                        # Provide graceful fallback response
                        output_text = (
                            f"[Agent {agent_name} could not generate a response. "
                            f"The LLM model may not support the requested operation. "
                            f"Previous context: {previous_output[:500] if previous_output else 'None'}]"
                        )
                    else:
                        # Re-raise unexpected errors
                        raise

                # --- Map output to state ---
                # Extract structured values from agent output text into
                # state variables defined in the workflow's state schema.
                # This replaces the naive approach that dumped raw prose
                # into every output_schema key.
                outputs = {}

                # Intelligent extraction from _state_schema
                extracted_vars = self._extract_state_variables(output_text, state)
                if extracted_vars:
                    outputs.update(extracted_vars)

                # Legacy fallback: if output_schema keys were explicitly
                # configured AND extraction did not cover them, fill them
                # with the full output text (preserves backward compat).
                output_schema = agent_config.get("output_schema", [])
                for key in output_schema:
                    if key not in outputs:
                        outputs[key] = output_text

                # Always store result for next agent to pick up
                outputs["crew_result"] = output_text
                # _last_agent_output: set ONLY by LLM agent nodes so the End node
                # can prefer this over raw API/MCP/Code responses when collecting
                # final output variables.
                outputs["_last_agent_output"] = output_text

                # Preserve original user input for downstream agents
                outputs["original_user_input"] = original_input

                # If this is the last agent (exit point), store as "result"
                if "result" in output_schema or agent_role in ("Workflow exit point", "Output"):
                    outputs["result"] = output_text

                # === TRANSPARENCY: Step completed ===
                if _is_transparency_enabled():
                    output_summary = _sanitize_output(output_text, crewai_tools)
                    output_summary["agent_name"] = agent_name
                    output_summary["agent_role"] = agent_role
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                # Return only new data - LangGraph merges via state schema
                logger.info(f"Sequential agent {agent_name} completed")
                return {
                    **outputs,
                    "messages": [{
                        "agent": agent_config.get("agent_id", "unknown"),
                        "name": agent_name,
                        "role": agent_role,
                        "tools_bound": len(crewai_tools),
                        "tool_names": [t.name for t in crewai_tools] if crewai_tools else [],
                        "original_request": original_input[:200] if original_input else "",
                        "had_previous_context": bool(previous_output),
                        "had_parallel_context": bool(parallel_output),
                        "output_preview": output_text[:500] if output_text else "",
                        "timestamp": datetime.utcnow().isoformat()
                    }]
                }

            except Exception as e:
                # === TRANSPARENCY: Step failed ===
                if _is_transparency_enabled():
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                logger.error(f"CrewAI sequential agent execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise RuntimeError(f"CrewAI sequential agent node failed: {e}")

        return sequential_agent_node

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    @staticmethod
    def _is_code_like_prompt(text: str) -> bool:
        """
        Detect if text looks like source code rather than natural language instructions.

        Used to guard against a frontend bug where the Code node's Python code
        leaks into the Agent node's prompt field.
        """
        if not text or len(text) < 10:
            return False

        lines = text.strip().splitlines()
        first_line = lines[0].strip() if lines else ""

        # Strong signals: starts with common code patterns
        if first_line.startswith(("import ", "from ", "def ", "class ", "#!")):
            return True

        # Count code-like lines vs total lines
        code_indicators = 0
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if (
                stripped.startswith(("import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "except", "return "))
                or "= " in stripped and not stripped.endswith(".")
                or stripped.endswith(":")
                or ".append(" in stripped
                or "print(" in stripped
            ):
                code_indicators += 1

        non_empty = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))
        if non_empty > 0 and code_indicators / non_empty > 0.5:
            return True

        return False

    def _get_llm_for_agent(self, agent_config: Dict[str, Any]):
        """
        Get LLM instance for agent using centralized LLM Manager.

        Supports per-agent LLM configuration with different providers.
        All LLM configuration is now centralized in llm_manager.py

        This method simply extracts agent preferences and delegates to LLMManager.
        """
        from llm_manager import LLMManager

        llm_config = agent_config.get("llm", {})

        # Extract agent's LLM preferences (or None to use defaults)
        provider = llm_config.get("provider")  # None = use LLMManager default
        model = llm_config.get("model")        # None = use LLMManager default
        temperature = llm_config.get("temperature")
        max_tokens = llm_config.get("max_tokens")

        # OVERRIDE FIX: Ignore provider/model from agent config if they conflict
        # Always use LLMManager defaults to avoid API key mismatches
        # Remove this override once all agent configs are cleaned up
        logger.info(f"Agent requested: provider={provider}, model={model}")
        logger.info(f"Using LLMManager defaults instead (configured in llm_manager.py)")
        provider = None  # Force use of LLMManager default
        model = None     # Force use of LLMManager default

        # Delegate to centralized LLM Manager (CrewAI-specific method)
        # CrewAI requires its own LLM class, not LangChain's ChatOpenAI
        try:
            llm = LLMManager.get_crewai_llm(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return llm

        except Exception as e:
            logger.error(f"Failed to get LLM for agent: {e}")
            logger.error("Check llm_manager.py configuration")
            raise

    def _combine_results(self, results: List[str]) -> str:
        """Combine multiple results by concatenating."""
        if not results:
            return ""
        return "\n\n---\n\n".join(results)

    def _vote_on_results(self, results: List[str]) -> str:
        """Select result that appears most frequently (simple voting)."""
        if not results:
            return ""

        # Count occurrences
        from collections import Counter
        counter = Counter(results)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # ========================================================================
    # STATE VARIABLE EXTRACTION FROM AGENT OUTPUT
    # ========================================================================

    @staticmethod
    def _coerce_value(raw_value: str, declared_type: str) -> Any:
        """
        Coerce a raw string value to the type declared in the state schema.

        Args:
            raw_value: The string value extracted from agent output.
            declared_type: The type string from state_schema (e.g. "number",
                           "string", "boolean", "integer", "float").

        Returns:
            The value coerced to the appropriate Python type.
        """
        if raw_value is None:
            return raw_value

        # If already the right type, return as-is
        if not isinstance(raw_value, str):
            return raw_value

        cleaned = raw_value.strip().strip('"').strip("'")
        dtype = declared_type.lower() if declared_type else "string"

        if dtype in ("number", "integer", "int"):
            try:
                # Try int first, then float
                if "." in cleaned:
                    return float(cleaned)
                return int(cleaned)
            except (ValueError, TypeError):
                # Try extracting a number from the string
                num_match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
                if num_match:
                    val = num_match.group()
                    return float(val) if "." in val else int(val)
                return cleaned

        if dtype == "float":
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                num_match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
                if num_match:
                    return float(num_match.group())
                return cleaned

        if dtype in ("boolean", "bool"):
            lower = cleaned.lower()
            if lower in ("true", "yes", "1"):
                return True
            if lower in ("false", "no", "0"):
                return False
            return cleaned

        if dtype in ("array", "list"):
            # Try JSON array first
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            # Try comma-separated values
            if "," in cleaned:
                items = [item.strip().strip('"').strip("'") for item in cleaned.split(",")]
                return [i for i in items if i]
            # Empty array indicators
            if cleaned.lower() in ("[]", "none", "null", "empty", "n/a", "0"):
                return []
            # Single value → single-element list (if non-empty)
            if cleaned:
                return [cleaned]
            return []

        if dtype in ("object", "dict"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return cleaned

        # Default: string -- return as-is
        return cleaned

    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        Normalize a key for fuzzy matching.

        Converts 'Overall Risk Score' -> 'overallriskscore' so it can be
        compared against 'risk_score' -> 'riskscore'.
        """
        return re.sub(r'[^a-z0-9]', '', key.lower())

    @staticmethod
    def _build_key_lookup(state_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Build a normalized-key -> original-key lookup from the state schema.

        This allows fuzzy matching of output keys to schema keys.
        For a schema key like 'risk_score', this produces entries:
            'riskscore' -> 'risk_score'
        """
        lookup: Dict[str, str] = {}
        for original_key in state_schema:
            normalized = CrewAIAdapter._normalize_key(original_key)
            lookup[normalized] = original_key
        return lookup

    @staticmethod
    def _match_key_to_schema(
        candidate_key: str,
        schema_lookup: Dict[str, str]
    ) -> Optional[str]:
        """
        Match a candidate key (from agent output) to a state schema key.

        Uses exact normalized match first, then substring containment.

        Args:
            candidate_key: The key found in agent output (e.g. 'Overall Risk Score').
            schema_lookup: Normalized-key -> original-key mapping.

        Returns:
            The matching state schema key, or None if no match.
        """
        normalized_candidate = CrewAIAdapter._normalize_key(candidate_key)
        if not normalized_candidate:
            return None

        # Exact normalized match
        if normalized_candidate in schema_lookup:
            return schema_lookup[normalized_candidate]

        # Substring containment: check if any schema key is contained in
        # the candidate or vice versa
        for normalized_schema_key, original_schema_key in schema_lookup.items():
            if len(normalized_schema_key) < 3:
                # Skip very short keys to avoid false positives
                continue
            if (normalized_schema_key in normalized_candidate
                    or normalized_candidate in normalized_schema_key):
                return original_schema_key

        return None

    @staticmethod
    def _extract_via_regex(
        text: str,
        state_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Extract state variable values from agent text output using regex patterns.

        Attempts extraction in the following order:
        1. JSON blocks (fenced or bare)
        2. Key-value patterns (colon-separated, equals-separated)
        3. Markdown table rows
        4. Markdown bold key patterns

        Args:
            text: The agent's raw text output.
            state_schema: The workflow state schema mapping variable names to types.

        Returns:
            Dict of extracted values keyed by state schema variable names.
        """
        if not text or not state_schema:
            return {}

        extracted: Dict[str, Any] = {}
        schema_lookup = CrewAIAdapter._build_key_lookup(state_schema)

        # ------------------------------------------------------------------
        # Strategy 1: JSON blocks (```json ... ``` or bare { ... })
        # ------------------------------------------------------------------
        json_patterns = [
            r'```json\s*\n?(.*?)\n?\s*```',    # fenced json
            r'```\s*\n?(\{.*?\})\n?\s*```',     # fenced bare object
        ]
        for pattern in json_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            schema_key = CrewAIAdapter._match_key_to_schema(
                                k, schema_lookup
                            )
                            if schema_key and schema_key not in extracted:
                                extracted[schema_key] = CrewAIAdapter._coerce_value(
                                    str(v) if not isinstance(v, str) else v,
                                    state_schema.get(schema_key, "string")
                                )
                except (json.JSONDecodeError, TypeError):
                    continue

        # Try bare JSON object (outermost braces)
        bare_json_match = re.search(r'\{[^{}]*\}', text)
        if bare_json_match:
            try:
                parsed = json.loads(bare_json_match.group())
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        schema_key = CrewAIAdapter._match_key_to_schema(
                            k, schema_lookup
                        )
                        if schema_key and schema_key not in extracted:
                            extracted[schema_key] = CrewAIAdapter._coerce_value(
                                str(v) if not isinstance(v, str) else v,
                                state_schema.get(schema_key, "string")
                            )
            except (json.JSONDecodeError, TypeError):
                pass

        # If JSON extraction found all schema keys, return early
        if len(extracted) == len(state_schema):
            return extracted

        # ------------------------------------------------------------------
        # Strategy 2: Key-value patterns in text lines
        # Matches patterns like:
        #   risk_score: 45
        #   Risk Score = 45
        #   **risk_score**: 45
        #   - Risk Score: 45
        #   Overall Risk Score: 45
        # ------------------------------------------------------------------
        # Pattern: optional markdown bold/list prefix, key, separator, value
        kv_pattern = re.compile(
            r'(?:^|\n)\s*'                         # line start
            r'(?:[-*]\s*)?'                         # optional list marker
            r'(?:\*{1,2})?'                         # optional bold markers
            r'([A-Za-z][A-Za-z0-9 _\-]*?)'         # key (capture group 1)
            r'(?:\*{1,2})?'                         # optional closing bold
            r'\s*[:=]\s*'                           # separator (: or =)
            r'(.+?)$',                              # value (capture group 2)
            re.MULTILINE
        )

        for match in kv_pattern.finditer(text):
            candidate_key = match.group(1).strip()
            candidate_value = match.group(2).strip()

            # Remove trailing markdown or punctuation artifacts
            candidate_value = re.sub(r'\*+$', '', candidate_value).strip()

            schema_key = CrewAIAdapter._match_key_to_schema(
                candidate_key, schema_lookup
            )
            if schema_key and schema_key not in extracted:
                extracted[schema_key] = CrewAIAdapter._coerce_value(
                    candidate_value,
                    state_schema.get(schema_key, "string")
                )

        # If key-value extraction found all schema keys, return early
        if len(extracted) == len(state_schema):
            return extracted

        # ------------------------------------------------------------------
        # Strategy 3: Markdown table rows
        # Matches: | Key | Value |
        # ------------------------------------------------------------------
        table_pattern = re.compile(
            r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|',
            re.MULTILINE
        )
        for match in table_pattern.finditer(text):
            candidate_key = match.group(1).strip()
            candidate_value = match.group(2).strip()

            # Skip table header separators (----)
            if re.match(r'^[-:]+$', candidate_key) or re.match(r'^[-:]+$', candidate_value):
                continue

            schema_key = CrewAIAdapter._match_key_to_schema(
                candidate_key, schema_lookup
            )
            if schema_key and schema_key not in extracted:
                extracted[schema_key] = CrewAIAdapter._coerce_value(
                    candidate_value,
                    state_schema.get(schema_key, "string")
                )

        return extracted

    @staticmethod
    def _extract_via_llm(
        text: str,
        state_schema: Dict[str, str],
        missing_keys: List[str]
    ) -> Dict[str, Any]:
        """
        Use a fast/cheap LLM call to extract remaining state variable values
        from agent output text.

        This is the fallback when regex extraction did not find values for
        all state schema keys.

        Args:
            text: The agent's raw text output.
            state_schema: The full state schema.
            missing_keys: List of schema keys that still need extraction.

        Returns:
            Dict of extracted values for the missing keys.
        """
        if not missing_keys or not text:
            return {}

        try:
            from llm_manager import LLMManager

            llm = LLMManager.get_llm(temperature=0.0, max_tokens=500)

            # Build the extraction prompt
            vars_description = "\n".join(
                f'  - "{k}" (type: {state_schema.get(k, "string")})'
                for k in missing_keys
            )

            prompt = (
                "Extract the following variable values from the text below. "
                "Return ONLY a valid JSON object with the variable names as keys "
                "and their extracted values. If a variable's value cannot be found, "
                "use null. Do not include any explanation or markdown formatting.\n\n"
                f"Variables to extract:\n{vars_description}\n\n"
                f"Text:\n{text[:3000]}\n\n"
                "JSON:"
            )

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the JSON from the response
            # Strip markdown fences if present
            cleaned = re.sub(r'^```(?:json)?\s*', '', response_text.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)

            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                return {}

            # Coerce and filter to only missing keys
            result: Dict[str, Any] = {}
            for key in missing_keys:
                if key in parsed and parsed[key] is not None:
                    result[key] = CrewAIAdapter._coerce_value(
                        str(parsed[key]) if not isinstance(parsed[key], str) else parsed[key],
                        state_schema.get(key, "string")
                    )

            return result

        except Exception as e:
            logger.warning(
                f"LLM fallback extraction failed: {e}. "
                f"State variables {missing_keys} will remain unset."
            )
            return {}

    @staticmethod
    def _extract_state_variables(
        output_text: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract state variable values from agent output text.

        This is the main entry point for state variable extraction.
        It tries regex-based extraction first (fast, no API cost), then
        falls back to an LLM call for any remaining unextracted variables.

        Only extracts values for variables defined in the workflow's
        state schema (stored in state["_state_schema"]).

        Args:
            output_text: The agent's raw text output.
            state: The current LangGraph state dict.

        Returns:
            Dict of extracted values keyed by state schema variable names.
            Only includes keys where a value was successfully extracted.
        """
        state_schema = state.get("_state_schema", {})
        if not state_schema or not output_text:
            return {}

        # Filter out internal/meta keys from the schema
        extraction_schema = {
            k: v for k, v in state_schema.items()
            if not k.startswith("_")
        }
        if not extraction_schema:
            return {}

        logger.info(
            f"Extracting state variables from agent output. "
            f"Schema keys: {list(extraction_schema.keys())}"
        )

        # Step 1: Regex-based extraction (fast, free)
        extracted = CrewAIAdapter._extract_via_regex(output_text, extraction_schema)
        logger.info(f"Regex extraction found: {list(extracted.keys())}")

        # Step 2: Identify missing keys
        missing_keys = [
            k for k in extraction_schema
            if k not in extracted
        ]

        # Step 3: LLM fallback for missing keys (only if regex missed some)
        if missing_keys:
            logger.info(
                f"Regex missed {len(missing_keys)} keys: {missing_keys}. "
                f"Attempting LLM fallback extraction."
            )
            llm_extracted = CrewAIAdapter._extract_via_llm(
                output_text, extraction_schema, missing_keys
            )
            if llm_extracted:
                logger.info(f"LLM extraction found: {list(llm_extracted.keys())}")
                extracted.update(llm_extracted)

        if extracted:
            logger.info(f"Final extracted state variables: {extracted}")
        else:
            logger.info("No state variables could be extracted from agent output.")

        return extracted

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def validate_no_orchestration_in_crewai(crew_config: Dict[str, Any]) -> bool:
        """
        Validate that CrewAI configuration doesn't contain orchestration logic.

        This prevents architectural violations where CrewAI tries to control
        graph-level decisions.
        """
        # Check for forbidden patterns
        forbidden_patterns = [
            "next_node",
            "graph.add_edge",
            "workflow_control",
            "decide_next",
            "routing_logic"
        ]

        config_str = str(crew_config).lower()
        for pattern in forbidden_patterns:
            if pattern in config_str:
                raise ValueError(
                    f"CrewAI configuration contains orchestration logic: '{pattern}'. "
                    f"CrewAI must only handle agent collaboration, not workflow control."
                )

        return True


# ============================================================================
# UTILITY FUNCTIONS FOR LANGGRAPH INTEGRATION
# ============================================================================

def create_crewai_merge_node(
    parallel_agent_configs: List[Dict[str, Any]],
    merge_strategy: str = "combine"
) -> Callable:
    """
    Create a merge node that aggregates results from parallel CrewAI agents.

    This is used in hybrid workflows where parallel branches need to merge
    before continuing sequentially.

    FIXED: Properly preserves original_user_input and crew_result for downstream agents.

    Args:
        parallel_agent_configs: Configs of agents that ran in parallel
        merge_strategy: How to merge ("combine", "vote", "llm_synthesis")

    Returns:
        Callable merge node function for LangGraph
    """
    def merge_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from parallel execution."""
        # === TRANSPARENCY: Setup ===
        run_id = _get_run_id(state)
        step_id = new_id("step_")
        step_name = "Merge Results"

        if _is_transparency_enabled():
            from apps.workflow.runtime.transparency import step_tracker, StepType
            from apps.workflow.runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync, publish_step_failed_sync

            input_summary = {
                "merge_strategy": merge_strategy,
                "agent_count": len(parallel_agent_configs)
            }

            step_tracker.start_step(run_id, step_id, step_name, StepType.MERGE, input_summary)
            publish_step_started_sync(run_id, step_id, step_name, StepType.MERGE, input_summary)

        try:
            logger.info(f"Merge node processing results from {len(parallel_agent_configs)} parallel agents")

            # Preserve original user input
            original_input = state.get("original_user_input", "")

            # Collect outputs from parallel agents
            parallel_outputs = []

            # First check for individual_outputs from parallel Crew execution
            individual_outputs = state.get("individual_outputs", [])
            if individual_outputs:
                parallel_outputs.extend(individual_outputs)

            # Also check parallel_output (aggregated by parallel Crew)
            parallel_output = state.get("parallel_output", "")

            # Check crew_result
            crew_result = state.get("crew_result", "")

            # Fallback: check output_schema keys from each agent config
            if not parallel_outputs:
                for agent_config in parallel_agent_configs:
                    agent_id = agent_config.get("agent_id")
                    output_schema = agent_config.get("output_schema", [])
                    for key in output_schema:
                        if key in state and state[key]:
                            parallel_outputs.append(state[key])

            # Merge based on strategy
            if merge_strategy == "combine":
                merged = "\n\n---\n\n".join(str(o) for o in parallel_outputs if o)
            elif merge_strategy == "vote":
                from collections import Counter
                counter = Counter(parallel_outputs)
                merged = counter.most_common(1)[0][0] if counter else ""
            elif merge_strategy == "prioritize":
                merged = parallel_outputs[0] if parallel_outputs else ""
            else:
                merged = "\n\n".join(str(o) for o in parallel_outputs if o)

            # Use parallel_output if no individual outputs
            if not merged and parallel_output:
                merged = parallel_output

            # Use crew_result as fallback
            if not merged and crew_result:
                merged = crew_result

            logger.info(f"Merge node aggregated {len(parallel_outputs)} outputs using {merge_strategy} strategy")

            # === TRANSPARENCY: Step completed ===
            if _is_transparency_enabled():
                output_summary = {
                    "output_preview": merged[:500] if merged else "",
                    "merge_strategy": merge_strategy,
                    "input_count": len(parallel_outputs)
                }
                step_tracker.complete_step(run_id, step_id, output_summary)
                publish_step_completed_sync(run_id, step_id, output_summary)

            # Return merged state - preserve original_user_input for downstream
            return {
                "merged_output": merged,
                "crew_result": merged,  # For downstream sequential agents
                "parallel_outputs": parallel_outputs,
                "original_user_input": original_input,  # CRITICAL: Preserve for downstream
                "messages": [{
                    "node": "merge",
                    "action": "merged_parallel_results",
                    "strategy": merge_strategy,
                    "input_count": len(parallel_outputs),
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }

        except Exception as e:
            # === TRANSPARENCY: Step failed ===
            if _is_transparency_enabled():
                step_tracker.fail_step(run_id, step_id, str(e))
                publish_step_failed_sync(run_id, step_id, str(e))

            logger.error(f"Merge node failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to merge parallel results: {e}")

    return merge_node
