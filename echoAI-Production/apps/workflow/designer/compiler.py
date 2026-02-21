"""
Workflow compiler.
Compiles workflow JSON definitions into executable LangGraph.

ARCHITECTURAL RULES (STRICTLY ENFORCED):
=========================================
1. LangGraph OWNS: Workflow topology, execution order, branching, merging, state
2. Every workflow is MATERIALIZED AS: LangGraph StateGraph
3. CrewAI is INVOKED: Inside LangGraph nodes (never controls graph)
4. CrewAI HANDLES: Agent collaboration within nodes
5. CrewAI NEVER: Controls graph traversal or state transitions
6. Workflow type: Inferred by designer, compiled here into LangGraph structure

SUPPORTED NODE TYPES:
=====================
- Agent: LLM-powered agent execution (CrewAI or direct)
- Conditional/Router: Branch routing based on conditions
- HITL: Human-in-the-loop approval checkpoints
- API: HTTP API calls (Managed connector or Ad-hoc) — SYNC, wrapped in asyncio.to_thread()
- MCP: Model Context Protocol calls (Managed or Ad-hoc) — ASYNC, awaited directly
- Code: Sandboxed Python code execution
- Self-Review: LLM-based output validation against review criteria
"""
from typing import Dict, Any, TypedDict, List, Annotated, Optional
import asyncio
import operator
import os
import json
import logging

logger = logging.getLogger(__name__)


class WorkflowCompiler:
    """
    Compiles workflow JSON to executable LangGraph.

    This compiler creates LangGraph StateGraph structures where:
    - Graph topology is defined by this compiler (NOT by agents or CrewAI)
    - Nodes can execute agents via CrewAI (but CrewAI doesn't decide flow)
    - State management is controlled by LangGraph
    """

    def __init__(self, use_crewai: bool = True):
        """
        Initialize compiler.

        Args:
            use_crewai: If True, use CrewAI for agent execution (recommended).
                       If False, use direct LLM calls (legacy mode).
        """
        self._compiled_cache = {}
        self._use_crewai = use_crewai
        self._crewai_adapter = None

        if use_crewai:
            try:
                from ..crewai_adapter import CrewAIAdapter
                self._crewai_adapter = CrewAIAdapter()
                logger.info("CrewAI adapter initialized successfully")
            except ImportError:
                logger.warning("CrewAI not available, falling back to direct LLM execution")
                self._use_crewai = False

    def _detect_start_end_nodes(
        self,
        agents_raw: list,
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect Start and End nodes in the agents list.

        Scans agent configs for type="Start" or type="End" and separates
        them from the working agents list. Used by parallel, hierarchical,
        and hybrid compile methods to wrap their internal logic.

        Args:
            agents_raw: Raw agents list from workflow (strings or dicts)
            agent_registry: Agent definitions

        Returns:
            dict with keys: start_id, start_config, end_id, end_config,
            working_agents (filtered list without Start/End)
        """
        result = {
            "start_id": None, "start_config": None,
            "end_id": None, "end_config": None,
            "working_agents": []
        }

        for agent_entry in agents_raw:
            if isinstance(agent_entry, str):
                aid = agent_entry
            elif isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
            else:
                continue

            if not aid:
                continue

            config = agent_registry.get(aid, {})
            node_type = (
                config.get("type", "")
                or config.get("metadata", {}).get("node_type", "")
            ).strip().lower()

            if node_type == "start":
                result["start_id"] = aid
                result["start_config"] = config
                logger.debug(f"Detected Start node: {aid}")
            elif node_type == "end":
                result["end_id"] = aid
                result["end_config"] = config
                logger.debug(f"Detected End node: {aid}")
            else:
                result["working_agents"].append(aid)

        return result

    def compile_to_langgraph(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> Any:
        """
        Compile workflow JSON to executable LangGraph.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            Compiled LangGraph instance (runnable)
        """
        try:
            from langgraph.graph import StateGraph, END
            from langgraph.checkpoint.memory import MemorySaver
        except ImportError:
            raise ImportError(
                "LangGraph not installed. Run: pip install langgraph langchain-core"
            )

        execution_model = workflow.get("execution_model", "sequential")

        # DEBUG: Log the execution model being used
        logger.info(f"[COMPILER] Workflow '{workflow.get('workflow_id')}' has execution_model='{execution_model}'")

        # Create state schema
        WorkflowState = self._create_state_class(workflow, agent_registry)

        # Build graph based on execution model
        # IMPORTANT: All compilation methods build LangGraph structures
        # CrewAI is ONLY used inside node functions, never for graph topology
        if execution_model == "sequential":
            return self._compile_sequential(workflow, agent_registry, WorkflowState)
        elif execution_model == "parallel":
            return self._compile_parallel(workflow, agent_registry, WorkflowState)
        elif execution_model == "hierarchical":
            return self._compile_hierarchical(workflow, agent_registry, WorkflowState)
        elif execution_model == "hybrid":
            return self._compile_hybrid(workflow, agent_registry, WorkflowState)
        else:
            logger.warning(f"Unknown execution model '{execution_model}', defaulting to sequential")
            return self._compile_sequential(workflow, agent_registry, WorkflowState)

    def _create_state_class(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]]
    ) -> type:
        """
        Create TypedDict state class for workflow.

        Args:
            workflow: Workflow definition
            agent_registry: Agent definitions

        Returns:
            TypedDict class for state
        """
        # Collect all state keys from agents
        state_keys = set()

        # Handle both embedded agent objects and ID-referenced agents
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, str):
                agent_id = agent_entry
                agent = agent_registry.get(agent_id, {})
            elif isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
                agent = agent_entry
            else:
                continue

            state_keys.update(agent.get("input_schema", []))
            state_keys.update(agent.get("output_schema", []))

        # Add workflow-level state keys
        state_keys.update(workflow.get("state_schema", {}).keys())

        # Create TypedDict dynamically
        # Use Annotated[Any, _last_writer_wins] for data fields so that
        # concurrent writes from parallel branches don't raise
        # INVALID_CONCURRENT_GRAPH_UPDATE.  Only messages uses operator.add.
        def _last_writer_wins(existing, new):
            """Reducer: last writer wins (accepts concurrent writes)."""
            return new

        _LWW = Annotated[Any, _last_writer_wins]

        fields = {key: _LWW for key in state_keys}

        # Add standard workflow fields
        fields["user_input"] = _LWW
        fields["task_description"] = _LWW
        fields["crew_result"] = _LWW

        # FIXED: Add original_user_input to preserve user query throughout workflow
        # This ensures all agents have access to the original request
        fields["original_user_input"] = _LWW

        # Parallel execution fields
        fields["parallel_output"] = _LWW
        fields["individual_outputs"] = _LWW
        fields["hierarchical_output"] = _LWW

        # Execution context fields - required for transparency tracking
        # These fields are set by executor.py and must be preserved across all nodes
        fields["run_id"] = _LWW
        fields["workflow_id"] = _LWW

        # --- Universal Integration Layer: Node output fields ---
        # last_node_output: carries the FULL output from the previous node
        # so that subsequent nodes (Agent, API, MCP, Code) receive complete context.
        # This is overwritten each step (no reducer) — only the latest output is kept.
        fields["last_node_output"] = _LWW
        # Typed result fields for specific node kinds
        fields["api_result"] = _LWW       # Latest API node response (full body + metadata)
        fields["mcp_result"] = _LWW       # Latest MCP node response
        fields["code_result"] = _LWW      # Latest Code node execution output
        fields["review_result"] = _LWW    # Latest Self-Review validation result
        # Last LLM agent output — set ONLY by Agent nodes (sequential/parallel/hierarchical).
        # API, MCP, Code, and Self-Review nodes do NOT write this key, so the End node
        # can always distinguish the last real agent output from raw API/code responses.
        fields["_last_agent_output"] = _LWW
        # Conditional routing helper (already used by _create_conditional_node)
        fields["_conditional_route"] = _LWW
        fields["_conditional_node"] = _LWW

        # Messages field accumulates across nodes
        fields["messages"] = Annotated[List[Dict[str, Any]], operator.add]

        WorkflowState = TypedDict("WorkflowState", fields)
        return WorkflowState

    def _compile_sequential(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile sequential workflow.

        ARCHITECTURE:
        - LangGraph creates linear chain of nodes: A → B → C → END
        - Each node can use CrewAI for agent execution
        - LangGraph controls the sequence (CrewAI doesn't decide "what's next")
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling sequential workflow...")

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        # Extract agent IDs (handle both embedded objects and string IDs)
        agent_ids = []
        for agent_entry in agents:
            if isinstance(agent_entry, str):
                agent_ids.append(agent_entry)
            elif isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
                if aid:
                    agent_ids.append(aid)

        # Add nodes — dispatch by type (Agent, API, MCP, Code, Self-Review, etc.)
        for agent_id in agent_ids:
            agent = agent_registry.get(agent_id, {})
            node_func = self._create_node_for_type(agent_id, agent)
            graph.add_node(agent_id, node_func)

        # Add edges based on connections (LangGraph controls sequence)
        for i, connection in enumerate(connections):
            from_agent = connection.get("from")
            to_agent = connection.get("to")

            if i == 0:
                # First connection - set entry point
                graph.set_entry_point(from_agent)

            graph.add_edge(from_agent, to_agent)

        # Set finish point
        if agent_ids:
            graph.add_edge(agent_ids[-1], END)

        # Compile with checkpointer (shared for HITL, new for non-HITL)
        checkpointer = self._get_checkpointer(workflow)
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("Sequential workflow compiled successfully")
        return compiled

    def _compile_parallel(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile parallel workflow.

        ARCHITECTURE (FIXED):
        - LangGraph creates: coordinator → parallel_crew_node → END
        - SINGLE parallel_crew_node executes ALL agents via CrewAI Crew
        - CrewAI handles true parallel agent execution INSIDE the node
        - Results are aggregated within the Crew and returned to LangGraph

        This is the correct architecture because:
        1. LangGraph's invoke() processes nodes sequentially
        2. CrewAI's Crew can execute multiple agents with true parallelism
        3. Using a single Crew node allows agents to share context
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling parallel workflow with CrewAI parallel execution...")

        graph = StateGraph(WorkflowState)

        agents = workflow.get("agents", [])

        # Detect Start/End nodes and separate from working agents
        se = self._detect_start_end_nodes(agents, agent_registry)
        start_id, start_config = se["start_id"], se["start_config"]
        end_id, end_config = se["end_id"], se["end_config"]

        # Extract agent IDs and build configs (handle both embedded and ID-referenced)
        agent_ids = []
        agent_configs = []
        for agent_entry in agents:
            if isinstance(agent_entry, str):
                aid = agent_entry
            elif isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
            else:
                continue
            if not aid:
                continue
            # Skip Start/End — they are handled separately
            if aid == start_id or aid == end_id:
                continue
            agent_ids.append(aid)
            agent_configs.append(agent_registry.get(aid, {}))

        # --- Entry point: Start node or coordinator ---
        if start_id:
            logger.info(f"Using Start node '{start_id}' as entry for parallel workflow")
            start_func = self._create_node_for_type(start_id, start_config)
            graph.add_node(start_id, start_func)
            graph.set_entry_point(start_id)
            coordinator_prev = start_id
        else:
            coordinator_prev = None

        # Preserve original user input in coordinator
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Coordinator preserves original user input and prepares for parallel execution.
            """
            from echolib.config import settings
            from echolib.utils import new_id

            logger.info(f"Coordinator preparing {len(agent_ids)} agents for parallel execution")
            # Preserve original user input for all downstream agents
            original_input = state.get("user_input") or state.get("message") or state.get("task_description") or ""

            # === TRANSPARENCY: Emit coordinator step ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync

                run_id = state.get("run_id", "unknown")
                step_id = new_id("step_")
                step_name = "Parallel Coordinator"
                input_summary = {
                    "user_input": original_input[:200] if original_input else "",
                    "agent_count": len(agent_ids)
                }

                step_tracker.start_step(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)
                publish_step_started_sync(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)

                # Complete step immediately (coordinator is just setup)
                output_summary = {"status": "prepared", "agent_count": len(agent_ids)}
                step_tracker.complete_step(run_id, step_id, output_summary)
                publish_step_completed_sync(run_id, step_id, output_summary)

            return {
                "original_user_input": original_input,
                "messages": [{
                    "node": "coordinator",
                    "action": "preparing_parallel_execution",
                    "agent_count": len(agent_ids)
                }]
            }

        graph.add_node("coordinator", coordinator)
        if coordinator_prev:
            graph.add_edge(coordinator_prev, "coordinator")
        else:
            graph.set_entry_point("coordinator")

        # Determine the final node before END (either End node or the last processing node)
        if self._use_crewai and self._crewai_adapter and agent_configs:
            # USE CREWAI FOR TRUE PARALLEL EXECUTION
            # Create SINGLE node that runs ALL parallel agents in one CrewAI Crew
            logger.info(f"Using CrewAI parallel Crew with {len(agent_configs)} agents")

            aggregation_strategy = workflow.get("aggregation_strategy", "combine")
            parallel_node_func = self._crewai_adapter.create_parallel_crew_node(
                agent_configs=agent_configs,
                aggregation_strategy=aggregation_strategy
            )

            # Single parallel execution node
            graph.add_node("parallel_execution", parallel_node_func)
            graph.add_edge("coordinator", "parallel_execution")
            last_node = "parallel_execution"

        else:
            # FALLBACK: Legacy individual node execution (not true parallel)
            logger.warning("CrewAI not available, falling back to sequential-like parallel execution")

            from ..crewai_adapter import create_crewai_merge_node

            # Add individual nodes — dispatch by type
            for agent_id in agent_ids:
                agent = agent_registry.get(agent_id, {})
                node_func = self._create_node_for_type(agent_id, agent)
                graph.add_node(agent_id, node_func)
                graph.add_edge("coordinator", agent_id)

            # Add aggregator
            def aggregator(state: Dict[str, Any]) -> Dict[str, Any]:
                """Aggregate results from parallel agents."""
                logger.info("Aggregating results from parallel execution")
                return state

            graph.add_node("aggregator", aggregator)

            for agent_id in agent_ids:
                graph.add_edge(agent_id, "aggregator")

            last_node = "aggregator"

        # --- Terminal: End node or direct END ---
        if end_id:
            logger.info(f"Using End node '{end_id}' as terminal for parallel workflow")
            end_func = self._create_node_for_type(end_id, end_config)
            graph.add_node(end_id, end_func)
            graph.add_edge(last_node, end_id)
            graph.add_edge(end_id, END)
        else:
            graph.add_edge(last_node, END)

        checkpointer = self._get_checkpointer(workflow)
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("Parallel workflow compiled successfully")
        return compiled

    def _compile_hierarchical(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile hierarchical workflow.

        ARCHITECTURE with CrewAI:
        - LangGraph creates a SINGLE node for hierarchical coordination
        - Inside that node, CrewAI Manager delegates to Workers
        - CrewAI handles the delegation logic WITHIN the node
        - LangGraph controls when/if that node executes

        ARCHITECTURE without CrewAI (legacy):
        - LangGraph creates master + sub-agent nodes with bidirectional edges
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling hierarchical workflow...")

        graph = StateGraph(WorkflowState)

        # Detect Start/End nodes from the agents array (canvas includes them)
        agents_raw = workflow.get("agents", [])
        se = self._detect_start_end_nodes(agents_raw, agent_registry)
        start_id, start_config = se["start_id"], se["start_config"]
        end_id, end_config = se["end_id"], se["end_config"]

        hierarchy = workflow.get("hierarchy", {})
        master_agent_id = hierarchy.get("master_agent")
        sub_agent_ids = hierarchy.get("delegation_order", [])
        delegation_strategy = hierarchy.get("delegation_strategy", "dynamic")

        if not master_agent_id:
            raise ValueError("Hierarchical workflow requires a master agent")

        # Get agent configurations
        master_config = agent_registry.get(master_agent_id, {})
        sub_configs = [agent_registry.get(aid, {}) for aid in sub_agent_ids if aid in agent_registry]

        # --- Entry point: Start node or direct master ---
        if start_id:
            logger.info(f"Using Start node '{start_id}' as entry for hierarchical workflow")
            start_func = self._create_node_for_type(start_id, start_config)
            graph.add_node(start_id, start_func)
            graph.set_entry_point(start_id)
            entry_prev = start_id
        else:
            entry_prev = None

        if self._use_crewai and self._crewai_adapter and sub_configs:
            # USE CREWAI FOR HIERARCHICAL DELEGATION
            # Create single LangGraph node that contains CrewAI hierarchical crew
            logger.info(f"Using CrewAI for hierarchical delegation: {len(sub_configs)} workers")

            hierarchical_node_func = self._crewai_adapter.create_hierarchical_crew_node(
                master_agent_config=master_config,
                sub_agent_configs=sub_configs,
                delegation_strategy=delegation_strategy
            )

            # LangGraph graph: [Start →] Hierarchical Node [→ End] → END
            graph.add_node("hierarchical_master", hierarchical_node_func)
            if entry_prev:
                graph.add_edge(entry_prev, "hierarchical_master")
            else:
                graph.set_entry_point("hierarchical_master")

            last_node = "hierarchical_master"

        else:
            # LEGACY MODE: Direct LLM calls with LangGraph edges
            logger.warning("CrewAI not available, using legacy hierarchical execution")

            # Add master agent node — dispatch by type
            master_func = self._create_node_for_type(master_agent_id, master_config)
            graph.add_node(master_agent_id, master_func)
            if entry_prev:
                graph.add_edge(entry_prev, master_agent_id)
            else:
                graph.set_entry_point(master_agent_id)

            # Add sub-agent nodes with bidirectional edges — dispatch by type
            for agent_id in sub_agent_ids:
                if agent_id in agent_registry:
                    agent = agent_registry.get(agent_id)
                    node_func = self._create_node_for_type(agent_id, agent)
                    graph.add_node(agent_id, node_func)
                    graph.add_edge(master_agent_id, agent_id)
                    graph.add_edge(agent_id, master_agent_id)

            last_node = master_agent_id

        # --- Terminal: End node or direct END ---
        if end_id:
            logger.info(f"Using End node '{end_id}' as terminal for hierarchical workflow")
            end_func = self._create_node_for_type(end_id, end_config)
            graph.add_node(end_id, end_func)
            graph.add_edge(last_node, end_id)
            graph.add_edge(end_id, END)
        else:
            graph.add_edge(last_node, END)

        checkpointer = self._get_checkpointer(workflow)
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("Hierarchical workflow compiled successfully")
        return compiled

    def _compile_hybrid(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type
    ) -> Any:
        """
        Compile hybrid workflow (parallel + sequential patterns).

        ARCHITECTURE (FIXED):
         coordinator → parallel_crew_node → merge → [sequential agents] → END

        The parallel section uses a SINGLE CrewAI Crew node for true parallel execution.
        Sequential sections use individual agent nodes.

        Args:
            workflow: Workflow definition with topology
            agent_registry: Agent definitions
            WorkflowState: State type

        Returns:
            Compiled LangGraph
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Compiling hybrid workflow with CrewAI parallel sections...")

        graph = StateGraph(WorkflowState)

        # Detect Start/End nodes from the agents array
        agents_raw = workflow.get("agents", [])
        se = self._detect_start_end_nodes(agents_raw, agent_registry)
        start_id, start_config = se["start_id"], se["start_config"]
        end_id, end_config = se["end_id"], se["end_config"]

        # Extract topology from workflow
        topology = workflow.get("topology", {})
        parallel_groups = topology.get("parallel_groups", [])
        sequential_chains = topology.get("sequential_chains", [])

        if not parallel_groups and not sequential_chains:
            # No topology specified - try to infer from connections
            logger.warning("No topology specified for hybrid workflow, attempting to infer")
            return self._compile_hybrid_from_connections(workflow, agent_registry, WorkflowState, graph)

        # --- Entry point: Start node or coordinator ---
        if start_id:
            logger.info(f"Using Start node '{start_id}' as entry for hybrid workflow")
            start_func = self._create_node_for_type(start_id, start_config)
            graph.add_node(start_id, start_func)
            graph.set_entry_point(start_id)
            coordinator_prev = start_id
        else:
            coordinator_prev = None

        # Create coordinator node that preserves original user input
        def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
            """Coordinator preserves user input and prepares hybrid execution."""
            from echolib.config import settings
            from echolib.utils import new_id

            original_input = state.get("user_input") or state.get("message") or state.get("task_description") or ""
            logger.info(f"Hybrid coordinator preparing execution with {len(parallel_groups)} parallel groups")

            # === TRANSPARENCY: Emit coordinator step ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync

                run_id = state.get("run_id", "unknown")
                step_id = new_id("step_")
                step_name = "Hybrid Coordinator"
                input_summary = {
                    "user_input": original_input[:200] if original_input else "",
                    "parallel_groups": len(parallel_groups),
                    "sequential_chains": len(sequential_chains)
                }

                step_tracker.start_step(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)
                publish_step_started_sync(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)

                # Complete step immediately (coordinator is just setup)
                output_summary = {"status": "prepared", "parallel_groups": len(parallel_groups)}
                step_tracker.complete_step(run_id, step_id, output_summary)
                publish_step_completed_sync(run_id, step_id, output_summary)

            return {
                "original_user_input": original_input,
                "messages": [{
                    "node": "coordinator",
                    "action": "preparing_hybrid_execution",
                    "parallel_groups": len(parallel_groups),
                    "sequential_chains": len(sequential_chains)
                }]
            }

        graph.add_node("coordinator", coordinator)
        if coordinator_prev:
            graph.add_edge(coordinator_prev, "coordinator")
        else:
            graph.set_entry_point("coordinator")

        # Collect all parallel agent configs for CrewAI parallel execution
        all_parallel_agent_configs = []
        for group in parallel_groups:
            agent_ids = group.get("agents", [])
            for agent_id in agent_ids:
                # Skip Start/End if they appear in parallel groups
                if agent_id == start_id or agent_id == end_id:
                    continue
                agent = agent_registry.get(agent_id, {})
                if agent:
                    all_parallel_agent_configs.append(agent)

        # Use CrewAI for parallel section if available and there are parallel agents
        if self._use_crewai and self._crewai_adapter and all_parallel_agent_configs:
            # CREATE SINGLE PARALLEL CREW NODE for all parallel agents
            logger.info(f"Using CrewAI parallel Crew for {len(all_parallel_agent_configs)} parallel agents")

            merge_strategy = parallel_groups[0].get("merge_strategy", "combine") if parallel_groups else "combine"
            parallel_node_func = self._crewai_adapter.create_parallel_crew_node(
                agent_configs=all_parallel_agent_configs,
                aggregation_strategy=merge_strategy
            )

            # Single parallel execution node
            graph.add_node("parallel_execution", parallel_node_func)
            graph.add_edge("coordinator", "parallel_execution")

            # Merge node processes the parallel output
            def merge_func(state: Dict[str, Any]) -> Dict[str, Any]:
                """Merge processes parallel results for sequential chain."""
                logger.info("Merge node processing parallel execution results")
                # parallel_output and crew_result are already set by parallel_execution
                return {
                    "messages": [{
                        "node": "merge",
                        "action": "parallel_results_merged",
                        "individual_output_count": len(state.get("individual_outputs", []))
                    }]
                }

            graph.add_node("merge", merge_func)
            graph.add_edge("parallel_execution", "merge")

        else:
            # FALLBACK: Legacy individual node execution (not true parallel)
            logger.warning("CrewAI not available, falling back to individual parallel nodes")

            for group in parallel_groups:
                agent_ids = group.get("agents", [])
                for agent_id in agent_ids:
                    if agent_id == start_id or agent_id == end_id:
                        continue
                    agent = agent_registry.get(agent_id, {})
                    node_func = self._create_node_for_type(agent_id, agent)
                    graph.add_node(agent_id, node_func)
                    graph.add_edge("coordinator", agent_id)

            # Create merge node
            from ..crewai_adapter import create_crewai_merge_node
            merge_strategy = parallel_groups[0].get("merge_strategy", "combine") if parallel_groups else "combine"
            merge_func = create_crewai_merge_node(all_parallel_agent_configs, merge_strategy)
            graph.add_node("merge", merge_func)

            # Connect parallel agents to merge
            for group in parallel_groups:
                agent_ids = group.get("agents", [])
                for agent_id in agent_ids:
                    if agent_id != start_id and agent_id != end_id:
                        graph.add_edge(agent_id, "merge")

        # Add sequential chain after merge (LangGraph edges)
        prev_node = "merge"
        for chain in sequential_chains:
            agent_ids = chain.get("agents", [])
            for agent_id in agent_ids:
                # Skip Start/End in sequential chains
                if agent_id == start_id or agent_id == end_id:
                    continue
                agent = agent_registry.get(agent_id, {})
                node_func = self._create_node_for_type(agent_id, agent)
                graph.add_node(agent_id, node_func)
                # LangGraph creates sequential edge
                graph.add_edge(prev_node, agent_id)
                prev_node = agent_id

        # --- Terminal: End node or direct END ---
        if end_id:
            logger.info(f"Using End node '{end_id}' as terminal for hybrid workflow")
            end_func = self._create_node_for_type(end_id, end_config)
            graph.add_node(end_id, end_func)
            graph.add_edge(prev_node, end_id)
            graph.add_edge(end_id, END)
        else:
            graph.add_edge(prev_node, END)

        checkpointer = self._get_checkpointer(workflow)
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("Hybrid workflow compiled successfully")
        return compiled

    def _compile_hybrid_from_connections(
        self,
        workflow: Dict[str, Any],
        agent_registry: Dict[str, Dict[str, Any]],
        WorkflowState: type,
        graph: Any
    ) -> Any:
        """
        Build hybrid workflow by properly traversing the entire connection graph.

        This method handles ANY workflow structure with:
        - Sequential sections (linear chains)
        - Parallel sections (fan-out from single node to multiple targets)
        - Merge points (fan-in from multiple sources to single target)
        - Multiple parallel/merge sections in the same workflow
        - HITL nodes in any position
        - Conditional/Router branching (routes to ONE target, not parallel)

        Algorithm:
        1. Build adjacency lists from connections (outgoing and incoming edges)
        2. Identify node types: conditional nodes, parallel sources, merge targets, entry points, terminals
        3. Use BFS traversal starting from entry point
        4. Handle CONDITIONAL nodes using LangGraph add_conditional_edges (routes to ONE branch)
        5. Handle parallel sources by creating CrewAI parallel Crew nodes (executes ALL branches)
        6. Handle merge targets by connecting parallel outputs
        7. Handle sequential nodes individually
        8. Connect terminal nodes to LangGraph END

        IMPORTANT:
        - Conditional/Router nodes are NOT parallel sources - they route to ONE target based on a condition
        - This function handles Start/End pseudo-nodes from canvas and properly filters them
        """
        from langgraph.graph import END
        from langgraph.checkpoint.memory import MemorySaver
        from collections import deque

        logger.info("Building hybrid workflow by traversing connection graph...")

        agents_raw = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        if not agents_raw:
            raise ValueError("No agents defined for hybrid workflow")

        # Extract agent IDs (handle both embedded objects and string IDs)
        agents = []
        for agent_entry in agents_raw:
            if isinstance(agent_entry, str):
                agents.append(agent_entry)
            elif isinstance(agent_entry, dict):
                aid = agent_entry.get("agent_id")
                if aid:
                    agents.append(aid)

        # =====================================================================
        # STEP 1: Normalize connections
        # =====================================================================
        # NOTE: Start/End nodes are now REAL nodes in the agents list (handled
        # by _create_node_for_type as passthrough functions). We no longer
        # filter them out as pseudo-nodes. Only filter connections that
        # reference truly unknown node IDs.
        normalized_connections = []

        for conn in connections:
            from_node = conn.get("from", "")
            to_node = conn.get("to", "")

            # Convert to string for comparison (handle integer IDs from canvas)
            from_node_str = str(from_node) if from_node else ""
            to_node_str = str(to_node) if to_node else ""

            # Only include connections between actual agents (Start/End are real agents now)
            if from_node_str in agents and to_node_str in agents:
                normalized_connections.append({
                    "from": from_node_str,
                    "to": to_node_str
                })
            else:
                logger.debug(f"Skipping connection {from_node_str} -> {to_node_str} (unknown agents)")

        # Use normalized connections for analysis
        connections = normalized_connections
        logger.info(f"Using {len(connections)} agent-to-agent connections")

        # =====================================================================
        # STEP 2: Build adjacency lists
        # =====================================================================
        outgoing = {agent: [] for agent in agents}  # node -> [targets]
        incoming = {agent: [] for agent in agents}  # node -> [sources]

        for conn in connections:
            from_node = conn.get("from")
            to_node = conn.get("to")
            if from_node in outgoing and to_node in incoming:
                outgoing[from_node].append(to_node)
                incoming[to_node].append(from_node)

        # =====================================================================
        # STEP 3: Identify node types
        # =====================================================================
        # Build node_types dictionary for filtering
        node_types = {}
        for agent_id in agents:
            agent_config = agent_registry.get(agent_id, {})
            # Check both metadata.node_type and top-level type field
            node_type = agent_config.get("metadata", {}).get("node_type", "")
            if not node_type:
                node_type = agent_config.get("type", "")
            node_types[agent_id] = node_type

        # Conditional/Router nodes: nodes with multiple outgoing edges that should BRANCH (not parallel)
        # These nodes route to ONE target based on a condition, not all targets simultaneously
        conditional_nodes = {
            a for a in agents
            if len(outgoing[a]) > 1
            and node_types.get(a) in ("Conditional", "Router", "conditional", "router")
        }

        # Parallel sources: nodes with multiple outgoing edges EXCLUDING Conditional/Router
        # These nodes truly fan out to execute multiple branches in parallel
        parallel_sources = {
            a for a in agents
            if len(outgoing[a]) > 1
            and a not in conditional_nodes
        }

        # Merge targets: nodes with multiple incoming edges
        merge_targets = {a for a in agents if len(incoming[a]) > 1}

        logger.info(f"  Conditional nodes ({len(conditional_nodes)}): {conditional_nodes}")

        # Entry point: node with no incoming edges
        # (Start node naturally has no incoming edges since it's the first node)
        entry_candidates = [a for a in agents if not incoming[a]]
        entry_point = entry_candidates[0] if entry_candidates else agents[0]

        # Detect if the entry point is a Start node (passthrough, not an agent)
        entry_config = agent_registry.get(entry_point, {})
        entry_type = (
            entry_config.get("type", "")
            or entry_config.get("metadata", {}).get("node_type", "")
        ).strip().lower()
        entry_is_start_node = entry_type == "start"

        # Terminal nodes: nodes with no outgoing edges
        terminal_nodes = {a for a in agents if not outgoing[a]}

        logger.info(f"Graph analysis complete:")
        logger.info(f"  Entry point: {entry_point}")
        logger.info(f"  Parallel sources ({len(parallel_sources)}): {parallel_sources}")
        logger.info(f"  Merge targets ({len(merge_targets)}): {merge_targets}")
        logger.info(f"  Terminal nodes ({len(terminal_nodes)}): {terminal_nodes}")

        # =====================================================================
        # STEP 4: Create entry point node
        # =====================================================================
        # If the entry point is a Start node, use it directly as the LangGraph
        # entry (the _create_node_for_type dispatcher handles Start passthrough).
        # Otherwise, inject a coordinator node for backward compatibility.
        if entry_is_start_node:
            logger.info(f"Using Start node '{entry_point}' as direct entry point (no coordinator)")
            start_func = self._create_node_for_type(entry_point, entry_config)
            graph.add_node(entry_point, start_func)
            graph.set_entry_point(entry_point)
            bfs_entry_prev = entry_point  # BFS starts AFTER the Start node
            nodes_created_init = {entry_point}
            visited_init = {entry_point}  # Start is already processed
        else:
            # Legacy: no Start node — inject coordinator
            def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
                """Coordinator preserves user input and prepares for hybrid execution."""
                from echolib.config import settings
                from echolib.utils import new_id

                original_input = (
                    state.get("user_input")
                    or state.get("message")
                    or state.get("task_description")
                    or ""
                )
                logger.info(f"Hybrid coordinator starting execution with {len(agents)} agents")

                # === TRANSPARENCY: Emit coordinator step ===
                if settings.transparency_enabled:
                    from ..runtime.transparency import step_tracker, StepType
                    from ..runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync

                    run_id = state.get("run_id", "unknown")
                    step_id = new_id("step_")
                    step_name = "Connection-Based Coordinator"
                    input_summary = {
                        "user_input": original_input[:200] if original_input else "",
                        "total_agents": len(agents),
                        "parallel_sections": len(parallel_sources),
                        "merge_points": len(merge_targets)
                    }

                    step_tracker.start_step(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)
                    publish_step_started_sync(run_id, step_id, step_name, StepType.COORDINATOR, input_summary)

                    # Complete step immediately (coordinator is just setup)
                    output_summary = {"status": "prepared", "total_agents": len(agents)}
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                return {
                    "original_user_input": original_input,
                    "messages": [{
                        "node": "coordinator",
                        "action": "hybrid_execution_start",
                        "total_agents": len(agents),
                        "parallel_sections": len(parallel_sources),
                        "merge_points": len(merge_targets)
                    }]
                }

            graph.add_node("coordinator", coordinator)
            graph.set_entry_point("coordinator")
            bfs_entry_prev = "coordinator"
            nodes_created_init = set()
            visited_init = set()

        # =====================================================================
        # STEP 5: BFS traversal to build the full graph
        # =====================================================================
        visited = visited_init  # Agents that have been processed
        nodes_created = nodes_created_init  # LangGraph nodes that have been created
        parallel_crew_counter = 0  # Counter for unique parallel crew names
        conditional_targets = set()  # Track nodes reached via conditional edges

        # Track which parallel crew a merge target should connect from
        merge_target_to_parallel_crew = {}

        # Queue: (agent_id, previous_langgraph_node)
        # If Start node was used, start BFS from Start's outgoing edges
        if entry_is_start_node:
            queue = deque(
                [(next_node, entry_point) for next_node in outgoing.get(entry_point, [])]
            )
        else:
            queue = deque([(entry_point, "coordinator")])

        while queue:
            current, prev_lg_node = queue.popleft()

            if current is None or current in visited:
                continue

            visited.add(current)

            # Skip if this node is a target of a parallel section (handled by Crew)
            # But only if it's NOT also a merge target (merge targets need individual nodes)
            # IMPORTANT: Do NOT skip nodes that are targets of conditional nodes
            is_parallel_target = any(
                current in outgoing.get(ps, [])
                for ps in parallel_sources
            )

            # Check if this node is a target of a conditional node (should NOT be skipped)
            is_conditional_target = any(
                current in outgoing.get(cn, [])
                for cn in conditional_nodes
            )

            if is_parallel_target and current not in merge_targets and current not in parallel_sources and not is_conditional_target:
                # This node is handled by a parallel Crew, skip individual creation
                logger.debug(f"Skipping {current} - handled by parallel Crew")
                continue

            # Get agent config
            agent_config = agent_registry.get(current, {})

            # Check if this is a HITL node
            node_type = agent_config.get("metadata", {}).get("node_type", "")
            is_hitl = node_type == "HITL" or agent_config.get("type") == "HITL"

            # =================================================================
            # PRE-STEP: If this node is a merge target, record all its
            # incoming source nodes so we can wire edges AFTER the node is
            # created (regardless of which CASE below handles creation).
            # This is essential when a node is BOTH a merge target AND
            # a parallel source — the elif chain would skip CASE B.
            # =================================================================
            _merge_sources_for_current: list = []
            if current in merge_targets and current not in merge_target_to_parallel_crew:
                for source_agent in agents:
                    if current in outgoing.get(source_agent, []):
                        if source_agent in conditional_nodes:
                            continue  # conditional_edges handle this
                        if source_agent in merge_target_to_parallel_crew:
                            _merge_sources_for_current.append(
                                merge_target_to_parallel_crew[source_agent]
                            )
                        elif source_agent in nodes_created:
                            _merge_sources_for_current.append(source_agent)

            # =====================================================================
            # CASE A0: Current node is a CONDITIONAL/ROUTER (branches to ONE target)
            # =====================================================================
            # IMPORTANT: Conditional nodes must NOT execute branches in parallel.
            # They evaluate a condition and route to exactly ONE branch.
            if current in conditional_nodes:
                logger.info(f"Processing conditional node: {current}")

                # Create the conditional node itself
                if current not in nodes_created:
                    conditional_func = self._create_conditional_node(current, agent_config)
                    graph.add_node(current, conditional_func)
                    nodes_created.add(current)
                    graph.add_edge(prev_lg_node, current)
                    logger.info(f"Created conditional node: {current}")

                # Get branch targets from config
                branch_targets = outgoing[current]
                branches_config = agent_config.get("config", {}).get("branches", [])

                # Resolve stale numeric targetNodeIds to actual agent IDs.
                # Workflows saved before the node_mapper fix may still contain
                # frontend canvas IDs (e.g. 1008, 3010) instead of agent IDs.
                # Canvas IDs = base_offset + index_in_agents_list.
                # The base_offset varies per workflow (1000, 2000, 3000, …).
                # Strategy: infer base_offset dynamically, then map all branches.
                has_unresolved = any(
                    b.get("targetNodeId") is not None
                    and str(b.get("targetNodeId")) not in branch_targets
                    for b in branches_config
                )
                if has_unresolved:
                    # Build canvas_id → agent_id by detecting the base offset.
                    # For each branch's numeric targetNodeId, try every possible
                    # base = tid - i (for i in range(len(agents))). A base is valid
                    # when the resulting agent_id is an actual connection target.
                    canvas_to_agent: Dict[str, str] = {}
                    base_offset = None
                    for branch in branches_config:
                        tid = branch.get("targetNodeId")
                        if tid is None:
                            continue
                        try:
                            tid_int = int(tid)
                        except (ValueError, TypeError):
                            continue
                        if str(tid) in branch_targets:
                            continue  # already resolved
                        # Try every possible index to find consistent base
                        for i, aid in enumerate(agents):
                            candidate_base = tid_int - i
                            if candidate_base < 0:
                                continue
                            if aid in branch_targets:
                                # Verify this base works: check other branches
                                base_offset = candidate_base
                                break
                        if base_offset is not None:
                            break

                    if base_offset is not None:
                        for branch in branches_config:
                            tid = branch.get("targetNodeId")
                            if tid is None or str(tid) in branch_targets:
                                continue
                            try:
                                idx = int(tid) - base_offset
                            except (ValueError, TypeError):
                                continue
                            if 0 <= idx < len(agents):
                                resolved_id = agents[idx]
                                if resolved_id in branch_targets:
                                    logger.info(
                                        f"Resolved branch targetNodeId {tid} -> {resolved_id}"
                                    )
                                    branch["targetNodeId"] = resolved_id

                # Create routing function for LangGraph conditional edges
                routing_func = self._create_conditional_routing_function(
                    current, agent_config, branch_targets, branches_config
                )

                # Build path_map: maps routing function return values to target nodes
                # The routing function returns the target node ID directly
                path_map = {target: target for target in branch_targets}

                # Add conditional edges using LangGraph's native routing
                graph.add_conditional_edges(
                    current,
                    routing_func,
                    path_map
                )
                logger.info(f"Added conditional edges from {current} to targets: {branch_targets}")

                # Mark all branch targets as conditional targets (they already have conditional edges)
                for target in branch_targets:
                    conditional_targets.add(target)

                # Queue all branch targets for individual processing
                # Each branch is processed as a separate sequential chain
                for target in branch_targets:
                    if target not in visited:
                        queue.append((target, current))

            # =====================================================================
            # CASE A: Current node is a PARALLEL SOURCE (fans out to multiple targets)
            # =====================================================================
            elif current in parallel_sources:
                logger.info(f"Processing parallel source: {current}")

                # First, create the parallel source node itself (if it's not just routing)
                # Check if this is a pure routing node (like Conditional) or an agent
                agent_role = agent_config.get("role", "")
                is_pure_router = agent_config.get("type") in ("Conditional", "Router")

                if not is_pure_router and current not in nodes_created:
                    # Create the source node — dispatch by type
                    source_func = self._create_node_for_type(current, agent_config)
                    graph.add_node(current, source_func)
                    nodes_created.add(current)
                    # Wire incoming edges (merge-aware)
                    if _merge_sources_for_current:
                        for src in _merge_sources_for_current:
                            graph.add_edge(src, current)
                            logger.info(f"Connected {src} -> {current} (merge+parallel)")
                    elif current not in conditional_targets:
                        graph.add_edge(prev_lg_node, current)
                    prev_lg_node = current
                    logger.info(f"Created node for parallel source: {current} (type={agent_config.get('type', 'Agent')})")
                elif is_pure_router and current not in nodes_created:
                    # For pure routers, still create the node
                    router_func = self._create_node_for_type(current, agent_config)
                    graph.add_node(current, router_func)
                    nodes_created.add(current)
                    if _merge_sources_for_current:
                        for src in _merge_sources_for_current:
                            graph.add_edge(src, current)
                            logger.info(f"Connected {src} -> {current} (merge+router)")
                    elif current not in conditional_targets:
                        graph.add_edge(prev_lg_node, current)
                    prev_lg_node = current
                    logger.info(f"Created router node for: {current}")

                # Get parallel targets
                parallel_targets = outgoing[current]
                parallel_target_configs = [
                    agent_registry.get(t, {})
                    for t in parallel_targets
                    if t in agent_registry
                ]

                # Find the merge target for this parallel group
                merge_target_for_group = None
                for target in parallel_targets:
                    for next_node in outgoing.get(target, []):
                        if next_node in merge_targets:
                            merge_target_for_group = next_node
                            break
                    if merge_target_for_group:
                        break

                # Create parallel Crew node for the targets
                if self._use_crewai and self._crewai_adapter and parallel_target_configs:
                    parallel_crew_counter += 1
                    parallel_node_name = f"parallel_crew_{parallel_crew_counter}"

                    parallel_func = self._crewai_adapter.create_parallel_crew_node(
                        agent_configs=parallel_target_configs,
                        aggregation_strategy="combine"
                    )
                    graph.add_node(parallel_node_name, parallel_func)
                    nodes_created.add(parallel_node_name)
                    graph.add_edge(prev_lg_node, parallel_node_name)

                    logger.info(f"Created parallel Crew '{parallel_node_name}' with {len(parallel_targets)} agents: {parallel_targets}")

                    # Mark parallel targets as visited (they're in the Crew)
                    for target in parallel_targets:
                        visited.add(target)

                    # If there's a merge target, record the connection
                    if merge_target_for_group:
                        merge_target_to_parallel_crew[merge_target_for_group] = parallel_node_name
                        # Queue the merge target for processing
                        queue.append((merge_target_for_group, parallel_node_name))
                    else:
                        # No merge target - parallel targets might be terminal
                        # Check if any parallel target has outgoing connections
                        has_continuation = False
                        for target in parallel_targets:
                            for next_node in outgoing.get(target, []):
                                if next_node not in visited:
                                    queue.append((next_node, parallel_node_name))
                                    has_continuation = True

                        if not has_continuation:
                            # Parallel targets are terminal
                            graph.add_edge(parallel_node_name, END)
                            logger.info(f"Parallel crew {parallel_node_name} connects to END")
                else:
                    # Fallback: create individual nodes for each target
                    logger.warning("CrewAI not available, creating individual nodes for parallel targets")
                    for target in parallel_targets:
                        if target not in visited:
                            queue.append((target, prev_lg_node))

            # =====================================================================
            # CASE B: Current node is a MERGE TARGET (receives from multiple sources)
            # =====================================================================
            elif current in merge_targets:
                logger.info(f"Processing merge target: {current}")

                # Create the merge target node — dispatch by type
                if current not in nodes_created:
                    merge_func = self._create_node_for_type(current, agent_config)
                    graph.add_node(current, merge_func)
                    nodes_created.add(current)

                    # Wire incoming edges using pre-computed merge sources
                    if current in merge_target_to_parallel_crew:
                        parallel_crew_name = merge_target_to_parallel_crew[current]
                        graph.add_edge(parallel_crew_name, current)
                        logger.info(f"Connected {parallel_crew_name} -> {current}")
                    elif _merge_sources_for_current:
                        for src in _merge_sources_for_current:
                            graph.add_edge(src, current)
                            logger.info(f"Connected {src} -> {current} (merge)")
                    elif current not in conditional_targets:
                        # Only add fallback edge if this node is NOT already
                        # wired via conditional_edges.  Adding a regular edge
                        # to a conditional target would make it ALWAYS execute,
                        # bypassing the conditional routing entirely.
                        graph.add_edge(prev_lg_node, current)
                        logger.info(f"Fallback: connected {prev_lg_node} -> {current}")

                # Queue outgoing nodes
                for next_node in outgoing.get(current, []):
                    if next_node not in visited:
                        queue.append((next_node, current))

                # If terminal, connect to END
                if current in terminal_nodes:
                    graph.add_edge(current, END)
                    logger.info(f"Terminal merge target {current} connects to END")

            # =====================================================================
            # CASE C: Regular sequential node
            # =====================================================================
            else:
                logger.info(f"Processing sequential node: {current}")

                if current not in nodes_created:
                    # Create node — dispatch by type (Agent, API, MCP, Code, etc.)
                    node_func = self._create_node_for_type(current, agent_config)
                    graph.add_node(current, node_func)
                    nodes_created.add(current)
                    # Wire incoming edges (merge-aware)
                    if _merge_sources_for_current:
                        for src in _merge_sources_for_current:
                            graph.add_edge(src, current)
                            logger.info(f"Connected {src} -> {current} (merge+seq)")
                    elif current not in conditional_targets:
                        # Only add edge if NOT a conditional target
                        # (conditional targets already have edges via add_conditional_edges)
                        graph.add_edge(prev_lg_node, current)
                    logger.info(f"Created sequential node: {current}")

                # Queue outgoing nodes
                for next_node in outgoing.get(current, []):
                    if next_node not in visited:
                        queue.append((next_node, current))

                # If terminal, connect to END
                if current in terminal_nodes:
                    graph.add_edge(current, END)
                    logger.info(f"Terminal node {current} connects to END")

        # =====================================================================
        # STEP 6: Verify graph completeness
        # =====================================================================
        unvisited = set(agents) - visited
        if unvisited:
            logger.warning(f"Some agents were not visited during traversal: {unvisited}")
            # These might be disconnected nodes - add them with edge to END
            for agent_id in unvisited:
                if agent_id not in nodes_created:
                    agent_config = agent_registry.get(agent_id, {})
                    node_func = self._create_node_for_type(agent_id, agent_config)
                    graph.add_node(agent_id, node_func)
                    nodes_created.add(agent_id)
                    graph.add_edge("coordinator", agent_id)
                    graph.add_edge(agent_id, END)
                    logger.info(f"Added disconnected node {agent_id} with direct path to END")

        logger.info(f"Hybrid workflow graph complete: {len(nodes_created)} nodes created")

        # =====================================================================
        # STEP 7: Compile with checkpointer (shared for HITL workflows)
        # =====================================================================
        checkpointer = self._get_checkpointer(workflow)
        return graph.compile(checkpointer=checkpointer)

    def _create_conditional_node(
        self,
        node_id: str,
        agent_config: Dict[str, Any]
    ):
        """
        Create a conditional/router node function.

        This node evaluates conditions and stores the routing decision in state.
        The actual routing is handled by LangGraph's add_conditional_edges.

        Args:
            node_id: Node identifier
            agent_config: Node configuration with branches

        Returns:
            Callable node function that evaluates conditions
        """
        node_name = agent_config.get("name", node_id)
        branches_config = agent_config.get("config", {}).get("branches", [])

        def conditional_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Conditional node that evaluates branch conditions.

            The routing decision is stored in state for the routing function
            to use when LangGraph invokes add_conditional_edges.
            """
            # Log non-internal state keys for debugging conditional evaluation
            user_state_keys = {
                k: (type(v).__name__, repr(v)[:100])
                for k, v in state.items()
                if not k.startswith("_") and k != "messages"
            }
            logger.info(
                f"Conditional node '{node_name}' evaluating branches. "
                f"State vars: {{{', '.join(f'{k}({t})={val}' for k, (t, val) in user_state_keys.items())}}}"
            )

            # Evaluate each branch condition against current state
            selected_branch = None
            selected_target = None

            for branch in branches_config:
                branch_type = branch.get("type", "")
                condition = branch.get("condition", "")
                target_node_id = branch.get("targetNodeId")

                if branch_type == "else":
                    # Else branch is the fallback - save it but continue checking
                    if selected_branch is None:
                        selected_branch = branch
                        selected_target = str(target_node_id) if target_node_id else None
                    continue

                if branch_type in ("if", "elif") and condition:
                    # For elif: only evaluate if no earlier branch has already matched
                    if branch_type == "elif" and selected_branch is not None:
                        continue
                    # Evaluate the condition against state
                    try:
                        # Create evaluation context from state
                        eval_context = dict(state)
                        # Safe evaluation of simple conditions
                        result = self._evaluate_condition(condition, eval_context)
                        if result:
                            selected_branch = branch
                            selected_target = str(target_node_id) if target_node_id else None
                            logger.info(f"Condition '{condition}' ({branch_type}) evaluated to True, routing to {selected_target}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to evaluate condition '{condition}': {e}")
                        continue

            # If no condition matched and we have an else branch, use it
            if selected_target is None and branches_config:
                # Find the else branch
                for branch in branches_config:
                    if branch.get("type") == "else":
                        selected_target = str(branch.get("targetNodeId"))
                        logger.info(f"No conditions matched, using else branch to {selected_target}")
                        break

            # Store routing decision in state for the routing function
            return {
                "_conditional_route": selected_target,
                "_conditional_node": node_id,
                "messages": [{
                    "node": node_id,
                    "type": "conditional",
                    "name": node_name,
                    "selected_route": selected_target,
                    "action": "branch_evaluated"
                }]
            }

        return conditional_node

    def _create_conditional_routing_function(
        self,
        node_id: str,
        agent_config: Dict[str, Any],
        branch_targets: List[str],
        branches_config: List[Dict[str, Any]]
    ):
        """
        Create routing function for LangGraph conditional edges.

        This function is called by LangGraph to determine which branch to take.
        It reads the routing decision from state (set by _create_conditional_node).

        Args:
            node_id: Conditional node identifier
            agent_config: Node configuration
            branch_targets: List of possible target node IDs
            branches_config: Branch configurations with conditions

        Returns:
            Callable routing function that returns the target node ID
        """
        # Determine default target (else branch or last target)
        default_target = None
        for branch in branches_config:
            if branch.get("type") == "else":
                target = branch.get("targetNodeId")
                default_target = str(target) if target else None
                break

        if default_target is None and branch_targets:
            default_target = branch_targets[-1]

        def routing_function(state: Dict[str, Any]) -> str:
            """
            Routing function that returns the target node ID.

            This reads the decision made by the conditional node and returns
            the appropriate target for LangGraph to route to.
            """
            # Get the routing decision from state
            selected_route = state.get("_conditional_route")

            if selected_route and selected_route in branch_targets:
                logger.info(f"Routing from {node_id} to {selected_route}")
                return selected_route

            # Fallback: evaluate conditions directly if not in state
            # This handles cases where the conditional node result isn't in state
            _matched = False
            for branch in branches_config:
                branch_type = branch.get("type", "")
                condition = branch.get("condition", "")
                target_node_id = branch.get("targetNodeId")

                if branch_type in ("if", "elif") and condition:
                    # elif only fires if no earlier branch matched
                    if branch_type == "elif" and _matched:
                        continue
                    try:
                        eval_context = dict(state)
                        result = self._evaluate_condition(condition, eval_context)
                        if result:
                            _matched = True
                            target = str(target_node_id) if target_node_id else None
                            if target and target in branch_targets:
                                logger.info(f"Direct condition evaluation ({branch_type}): routing to {target}")
                                return target
                    except Exception as e:
                        logger.warning(f"Routing function condition eval failed: {e}")
                        continue

            # Use default target
            logger.info(f"Using default route from {node_id} to {default_target}")
            return default_target

        return routing_function

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a condition string against a context.

        Supports both Python and JavaScript-style conditions:
        - Python:     "risk_score > 70 and status == 'approved'"
        - JavaScript: "extractionConfidence >= 0.9 && missingFields.length === 0"
        - Mixed:      "total_days <= 3 AND coverage_risk !== 'high'"

        Handles:
        - && / ||           → and / or
        - === / !==         → == / !=
        - .length           → len()
        - !variable         → not variable
        - AND / OR          → and / or (case-insensitive)
        - Numeric coercion  → string "42" becomes int 42

        Args:
            condition: Condition string to evaluate
            context: Dictionary of variable values

        Returns:
            Boolean result of condition evaluation
        """
        import re as _re

        if not condition or not condition.strip():
            return False

        normalized = condition.strip()

        # ─── JavaScript → Python syntax translation ───

        # Strict equality / inequality BEFORE loose equality
        normalized = normalized.replace("!==", "!=").replace("===", "==")

        # Logical operators
        normalized = normalized.replace("&&", " and ").replace("||", " or ")

        # Case-insensitive AND / OR (word boundaries)
        normalized = _re.sub(r'\bAND\b', 'and', normalized)
        normalized = _re.sub(r'\bOR\b', 'or', normalized)
        normalized = _re.sub(r'\bNOT\b', 'not', normalized)

        # .length → len() — e.g. "missingFields.length" → "len(missingFields)"
        normalized = _re.sub(
            r'(\w+)\.length\b',
            r'len(\1)',
            normalized
        )

        # JavaScript negation: "!varName" → "not varName"
        # Only match "!" that is NOT part of "!=" and is followed by a word char
        normalized = _re.sub(r'(?<!=)!(\w)', r'not \1', normalized)

        logger.info(
            f"[CONDITION EVAL] Original: '{condition}' → Normalized: '{normalized}'"
        )

        # ─── Extract variable names referenced in the condition ───
        # This helps debug which variables are present/missing in state
        referenced_vars = set(_re.findall(r'\b([a-zA-Z_]\w*)\b', normalized))
        # Remove Python keywords and builtins from the set
        _ignore = {
            'and', 'or', 'not', 'True', 'False', 'None', 'true', 'false',
            'null', 'len', 'str', 'int', 'float', 'bool', 'abs', 'min',
            'max', 'isinstance', 'list', 'dict', 'in', 'is', 'if', 'else'
        }
        referenced_vars -= _ignore

        # ─── Build safe evaluation environment ───
        safe_globals = {"__builtins__": {}}
        safe_locals = {}

        for k, v in context.items():
            # Skip internal keys for logging clarity
            if k.startswith("_"):
                safe_locals[k] = v
                continue
            # Coerce string values to their natural Python types
            if isinstance(v, str):
                stripped = v.strip()
                # Try JSON parse first (handles arrays, objects, booleans, null)
                if stripped and stripped[0] in ('[', '{'):
                    try:
                        import json as _json
                        v = _json.loads(stripped)
                    except (ValueError, TypeError):
                        pass
                elif stripped.lower() in ('true', 'yes'):
                    v = True
                elif stripped.lower() in ('false', 'no'):
                    v = False
                elif stripped.lower() in ('null', 'none'):
                    v = None
                else:
                    # Try numeric coercion
                    try:
                        v = int(stripped)
                    except ValueError:
                        try:
                            v = float(stripped)
                        except ValueError:
                            pass
            safe_locals[k] = v

        # Log which referenced variables are present/missing in state
        present_vars = {}
        missing_vars = []
        for var in referenced_vars:
            if var in safe_locals:
                present_vars[var] = safe_locals[var]
            else:
                missing_vars.append(var)

        if present_vars:
            logger.info(
                f"[CONDITION EVAL] Variables found in state: "
                f"{{{', '.join(f'{k}={v!r}' for k, v in present_vars.items())}}}"
            )
        if missing_vars:
            logger.warning(
                f"[CONDITION EVAL] Variables NOT in state: {missing_vars} — "
                f"condition will likely evaluate to False. "
                f"Ensure the upstream agent sets these variables."
            )

        # Common boolean / null literals
        safe_locals["True"] = True
        safe_locals["False"] = False
        safe_locals["true"] = True
        safe_locals["false"] = False
        safe_locals["None"] = None
        safe_locals["null"] = None

        # Safe builtins needed for translated expressions
        safe_locals["len"] = len
        safe_locals["str"] = str
        safe_locals["int"] = int
        safe_locals["float"] = float
        safe_locals["bool"] = bool
        safe_locals["abs"] = abs
        safe_locals["min"] = min
        safe_locals["max"] = max
        safe_locals["isinstance"] = isinstance
        safe_locals["list"] = list
        safe_locals["dict"] = dict

        try:
            result = eval(normalized, safe_globals, safe_locals)
            bool_result = bool(result)
            logger.info(
                f"[CONDITION EVAL] Result: '{normalized}' → {bool_result}"
            )
            return bool_result
        except NameError as e:
            # Variable not found in context — treat as False
            logger.warning(
                f"[CONDITION EVAL] NameError for '{condition}': {e}. "
                f"Variable is missing from workflow state."
            )
            return False
        except Exception as e:
            logger.warning(
                f"[CONDITION EVAL] Error for '{condition}' "
                f"(normalized: '{normalized}'): {e}"
            )
            return False

    def _create_agent_node(
        self,
        agent_id: str,
        agent_config: Dict[str, Any]
    ):
        """
        Create agent node function with CrewAI or direct LLM execution and HITL support.

        ARCHITECTURE:
        - This creates a FUNCTION that LangGraph will call
        - Inside this function, we can use CrewAI for agent execution
        - CrewAI is invoked INSIDE the function, not for graph control

        Args:
            agent_id: Agent identifier
            agent_config: Agent configuration

        Returns:
            Callable node function compatible with LangGraph
        """
        # Check if this is a HITL node
        node_type = agent_config.get("metadata", {}).get("node_type")
        is_hitl_node = (node_type == "HITL")

        # If CrewAI is enabled, use CrewAI adapter for agent execution
        if self._use_crewai and self._crewai_adapter and not is_hitl_node:
            logger.info(f"Creating CrewAI-powered node for agent: {agent_id}")
            return self._crewai_adapter.create_sequential_agent_node(agent_config)

        # Otherwise, use legacy direct LLM execution
        logger.info(f"Creating direct LLM node for agent: {agent_id}")

        def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Agent node execution function with real LLM calls and HITL support.

            Args:
                state: Current workflow state

            Returns:
                Updated state
            """
            from ..runtime.hitl import HITLManager
            from echolib.config import settings
            from echolib.utils import new_id
            import asyncio

            # Get agent configuration
            agent_name = agent_config.get("name", agent_id)
            agent_role = agent_config.get("role", "Processing")
            agent_description = agent_config.get("description", "")
            input_schema = agent_config.get("input_schema", [])
            output_schema = agent_config.get("output_schema", [])
            llm_config = agent_config.get("llm", {})

            # Extract inputs from state
            inputs = {key: state.get(key) for key in input_schema if key in state}

            # === TRANSPARENCY: Setup ===
            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import publish_step_started_sync, publish_step_completed_sync, publish_step_failed_sync

                user_input = (
                    state.get("original_user_input")
                    or state.get("user_input")
                    or state.get("message")
                    or ""
                )
                input_summary = {
                    "user_input": user_input[:200] if user_input else "",
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "is_hitl": is_hitl_node
                }

                step_tracker.start_step(run_id, step_id, agent_name, StepType.LLM_CALL if not is_hitl_node else StepType.AGENT, input_summary)
                publish_step_started_sync(run_id, step_id, agent_name, StepType.LLM_CALL if not is_hitl_node else StepType.AGENT, input_summary)

            try:
                # If this is a HITL node, request approval BEFORE executing
                if is_hitl_node:
                    hitl = HITLManager()

                    # Get workflow and run IDs from state
                    workflow_id = state.get("workflow_id", "unknown")

                    # Request approval
                    context = {
                        "agent_output": {"pending": "awaiting approval before execution"},
                        "state_snapshot": dict(state),
                        "inputs": inputs
                    }

                    interrupt_info = hitl.request_approval(
                        run_id=run_id,
                        workflow_id=workflow_id,
                        blocked_at=agent_id,
                        context=context
                    )

                    # Check HITL status - execution pauses here
                    status = hitl.get_status(run_id)

                    if status.get("state") == "rejected":
                        # Execution was rejected
                        raise RuntimeError(f"Workflow rejected at HITL checkpoint: {agent_id}")

                    elif status.get("state") == "modified":
                        # Agent was modified - reload configuration
                        # In production, reload agent config here
                        pass

                # Build prompt for LLM (only if not a pure HITL node)
                if not is_hitl_node or llm_config:
                    prompt = f"""You are {agent_name}, a specialized agent with the following role:
{agent_role}

{agent_description}

Your task is to process the following inputs and generate outputs:

Inputs:
{inputs}

Please provide your response in a clear, structured format. Focus on your specific role and responsibilities."""

                    # Execute real LLM call
                    try:
                        llm_response = self._execute_llm_call(llm_config, prompt)

                        # Create outputs based on LLM response
                        outputs = {}
                        for key in output_schema:
                            outputs[key] = llm_response

                    except Exception as e:
                        # Fallback if LLM call fails
                        outputs = {
                            key: f"Error in {agent_name}: {str(e)}" for key in output_schema
                        }
                else:
                    # HITL node - just pass through
                    outputs = {key: state.get(key) for key in output_schema if key in state}

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_text = next(iter(outputs.values()), "") if outputs else ""
                    output_summary = {
                        "output_preview": output_text[:500] if output_text else "",
                        "agent_name": agent_name,
                        "agent_role": agent_role
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                # Return only new data - LangGraph merges via state schema
                # messages uses operator.add, so return ONLY the new message
                return {
                    **outputs,
                    "crew_result": next(iter(outputs.values()), "") if outputs else "",
                    "messages": [{
                        "agent": agent_id,
                        "role": agent_role,
                        "inputs": inputs,
                        "outputs": outputs,
                        "hitl_checkpoint": is_hitl_node
                    }]
                }

            except Exception as e:
                # === TRANSPARENCY: Step failed ===
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))
                raise

        return agent_node

    # =====================================================================
    # UNIVERSAL INTEGRATION LAYER — New Node Types
    # =====================================================================

    def _create_api_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create an API node function for LangGraph.

        Supports two modes:
        - Managed Mode: Uses a pre-registered connector via connector_id.
          The connector's secure config is loaded from storage at runtime.
        - Ad-hoc Mode: Uses inline config (url, method, auth, headers, body)
          to create an ephemeral HTTPConnector for a single execution.

        IMPORTANT: APIConnector.invoke() and HTTPConnector.execute() are both
        SYNCHRONOUS. We wrap them with asyncio.to_thread() to avoid blocking
        the async event loop.

        The FULL response data is stored in state['last_node_output'] and
        state['api_result'] so the next node receives complete context.

        Args:
            node_id: Unique node identifier in the workflow graph.
            node_config: Node configuration dict. May contain:
                - connector_id (str): For Managed Mode
                - url/base_url (str): For Ad-hoc Mode
                - method (str): HTTP method (default GET)
                - headers (dict): Custom headers
                - query_params (dict): URL query parameters
                - body (dict): Request body
                - authentication/auth_config (dict): Auth configuration
                - payload (dict): For Managed Mode invocation payload

        Returns:
            Async callable node function compatible with LangGraph StateGraph.
        """
        config = node_config.get("config", node_config)
        # Fallback: resolve config from metadata for pre-fix workflows
        if not config.get("url") and not config.get("base_url") and not config.get("connector_id"):
            config = node_config.get("metadata", {}).get("api_config", config)
        node_name = node_config.get("name", node_id)

        def _interpolate_template(template: str, state: Dict[str, Any]) -> str:
            """Replace {variable} placeholders in a string with values from state."""
            import re
            def _replacer(match):
                key = match.group(1)
                val = state.get(key)
                if val is not None:
                    return str(val)
                return match.group(0)  # leave unreplaced if not in state
            return re.sub(r"\{(\w+)\}", _replacer, template)

        def _interpolate_dict(d: Any, state: Dict[str, Any]) -> Any:
            """Recursively interpolate {variable} placeholders in dicts/lists/strings."""
            if isinstance(d, str):
                return _interpolate_template(d, state)
            if isinstance(d, dict):
                return {k: _interpolate_dict(v, state) for k, v in d.items()}
            if isinstance(d, list):
                return [_interpolate_dict(item, state) for item in d]
            return d

        def api_node(state: Dict[str, Any]) -> Dict[str, Any]:
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # ─── Interpolate template variables from state ───
            # URLs like "https://api.example.com/v1/users/{employee_id}/balance"
            # and body/headers/query_params containing {variable} placeholders
            # are resolved against current workflow state before the HTTP call.
            live_config = _interpolate_dict(config, state)

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync,
                    publish_step_failed_sync
                )
                input_summary = {
                    "node_name": node_name,
                    "node_type": "API",
                    "mode": "managed" if live_config.get("connector_id") else "ad-hoc",
                    "method": live_config.get("method", "GET"),
                    "url": live_config.get("url") or live_config.get("base_url", ""),
                }
                step_tracker.start_step(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)
                publish_step_started_sync(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)

            try:
                connector_id = live_config.get("connector_id")

                if connector_id:
                    # ─── MANAGED MODE ───
                    # Load the registered connector and invoke it.
                    # APIConnector.invoke() is SYNC → run in thread.
                    from echolib.services import ConnectorManager
                    api_manager = ConnectorManager().get_manager("api")

                    payload = live_config.get("payload")
                    # APIConnector.invoke() is SYNC — call directly
                    result = api_manager.invoke(connector_id, payload)
                else:
                    # ─── AD-HOC MODE ───
                    # Build an ephemeral HTTPConnector from inline config.
                    from echolib.Get_connector.Get_API.connectors.factory import ConnectorFactory
                    from echolib.Get_connector.Get_API.models.config import ConnectorConfig

                    auth_config = (
                        live_config.get("auth_config")
                        or live_config.get("authentication")
                        or live_config.get("auth")
                        or {"type": "none"}
                    )
                    # If authentication is a simple string (e.g. "bearer"), normalize it
                    if isinstance(auth_config, str):
                        auth_config = {"type": auth_config}

                    # Separate base_url from path for proper URL construction.
                    # The user may configure a full URL like
                    #   "https://api.hris.com/v1/employees/123/pto-balance"
                    # ConnectorConfig expects base_url (scheme+host) and execute()
                    # takes the endpoint path separately.
                    raw_url = live_config.get("base_url") or live_config.get("url", "")
                    endpoint = live_config.get("endpoint", "")
                    if raw_url and not endpoint:
                        # Split full URL into base (scheme+host) and path
                        from urllib.parse import urlparse
                        parsed = urlparse(raw_url)
                        base_url = f"{parsed.scheme}://{parsed.netloc}"
                        endpoint = parsed.path or "/"
                        if parsed.query:
                            endpoint += f"?{parsed.query}"
                    else:
                        base_url = raw_url

                    temp_config = ConnectorConfig(
                        name=f"adhoc_{node_id}",
                        base_url=base_url,
                        auth=auth_config,
                        default_headers=live_config.get("headers"),
                    )
                    connector = ConnectorFactory.create(temp_config)

                    # HTTPConnector.execute() is SYNC — call directly
                    raw_response = connector.execute(
                        method=live_config.get("method", "GET"),
                        endpoint=endpoint,
                        headers=live_config.get("headers"),
                        query_params=live_config.get("query_params"),
                        body=live_config.get("body"),
                    )

                    # Normalize ExecuteResponse to a plain dict for state storage
                    result = {
                        "success": raw_response.success,
                        "status_code": raw_response.status_code,
                        "data": raw_response.body,
                        "headers": dict(raw_response.headers) if raw_response.headers else {},
                        "error": raw_response.error,
                        "elapsed_seconds": raw_response.elapsed_seconds,
                    }

                # === Build full context for the next node ===
                api_data = result.get("data") if isinstance(result, dict) else result

                # ─── Map API response fields into workflow state ───
                # If the API returns a JSON dict, promote its keys into state so
                # downstream conditional nodes and agents can reference them
                # (e.g. pto_balance, approval_status) without manual wiring.
                state_updates: Dict[str, Any] = {}
                if isinstance(api_data, dict):
                    state_updates.update(api_data)
                elif isinstance(api_data, str):
                    try:
                        parsed_data = json.loads(api_data)
                        if isinstance(parsed_data, dict):
                            state_updates.update(parsed_data)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_summary = {
                        "node_name": node_name,
                        "success": result.get("success") if isinstance(result, dict) else True,
                        "status_code": result.get("status_code") if isinstance(result, dict) else None,
                        "data_preview": str(api_data)[:500] if api_data else "",
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                state_updates.update({
                    "last_node_output": api_data,
                    "api_result": result,
                    "crew_result": json.dumps(api_data) if not isinstance(api_data, str) else api_data,
                    "messages": [{
                        "node": node_id,
                        "type": "api",
                        "name": node_name,
                        "mode": "managed" if connector_id else "ad-hoc",
                        "success": result.get("success") if isinstance(result, dict) else True,
                        "status_code": result.get("status_code") if isinstance(result, dict) else None,
                        "data": api_data,
                    }],
                })
                return state_updates

            except Exception as e:
                logger.error(f"API node '{node_name}' failed: {e}")
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                error_result = {"success": False, "error": str(e), "node": node_id}
                return {
                    "last_node_output": error_result,
                    "api_result": error_result,
                    "crew_result": f"API Error in {node_name}: {e}",
                    "messages": [{
                        "node": node_id,
                        "type": "api",
                        "name": node_name,
                        "error": str(e),
                    }],
                }

        return api_node

    def _create_mcp_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create an MCP (Model Context Protocol) node function for LangGraph.

        Supports two modes:
        - Managed Mode: Uses a pre-registered MCP connector via connector_id.
        - Ad-hoc Mode: Uses inline config to create an ephemeral HTTPMCPConnector.

        IMPORTANT: MCPConnector.invoke_async() is ASYNC — await it directly.
        No thread wrapping needed.

        The FULL response data is stored in state['last_node_output'] and
        state['mcp_result'] so the next node receives complete context.

        Args:
            node_id: Unique node identifier.
            node_config: Node configuration dict. May contain:
                - connector_id (str): For Managed Mode
                - endpoint_url (str): For Ad-hoc Mode
                - method (str): HTTP method (default POST)
                - name, description, auth_config, input_schema, output_schema
                - headers, query_params, payload

        Returns:
            Async callable node function compatible with LangGraph StateGraph.
        """
        config = node_config.get("config", node_config)
        # Fallback: resolve config from metadata for pre-fix workflows
        if not config.get("endpoint_url") and not config.get("connector_id") and not config.get("serverName"):
            config = node_config.get("metadata", {}).get("mcp_config", config)
        node_name = node_config.get("name", node_id)

        def _interpolate_template(template: str, state: Dict[str, Any]) -> str:
            """Replace {variable} placeholders in a string with values from state."""
            import re
            def _replacer(match):
                key = match.group(1)
                val = state.get(key)
                if val is not None:
                    return str(val)
                return match.group(0)
            return re.sub(r"\{(\w+)\}", _replacer, template)

        def _interpolate_dict(d: Any, state: Dict[str, Any]) -> Any:
            """Recursively interpolate {variable} placeholders in dicts/lists/strings."""
            if isinstance(d, str):
                return _interpolate_template(d, state)
            if isinstance(d, dict):
                return {k: _interpolate_dict(v, state) for k, v in d.items()}
            if isinstance(d, list):
                return [_interpolate_dict(item, state) for item in d]
            return d

        def mcp_node(state: Dict[str, Any]) -> Dict[str, Any]:
            import asyncio
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # ─── Interpolate template variables from state ───
            live_config = _interpolate_dict(config, state)

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync,
                    publish_step_failed_sync
                )
                input_summary = {
                    "node_name": node_name,
                    "node_type": "MCP",
                    "mode": "managed" if live_config.get("connector_id") else "ad-hoc",
                }
                step_tracker.start_step(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)
                publish_step_started_sync(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)

            try:
                connector_id = live_config.get("connector_id")

                if connector_id:
                    # ─── MANAGED MODE ───
                    # MCPConnector.invoke_async() is ASYNC — run in sync context
                    from echolib.services import ConnectorManager
                    mcp_manager = ConnectorManager().get_manager("mcp")
                    # MCPConnector.invoke_async() is ASYNC — run in new event loop
                    result = asyncio.run(
                        mcp_manager.invoke_async(
                            connector_id, payload=live_config.get("payload")
                        )
                    )
                else:
                    # ─── AD-HOC MODE ───
                    # Detect transport: STDIO (Azure MCP / command-based) vs HTTP
                    raw_command = live_config.get("command", "")
                    transport_type = live_config.get("transport_type", "http")

                    if raw_command or transport_type == "stdio":
                        # ─── STDIO MODE (Azure MCP Server via mcp Python library) ───
                        import os as _os
                        import sys as _sys
                        from mcp import ClientSession, StdioServerParameters
                        from mcp.client.stdio import stdio_client

                        # Windows requires ProactorEventLoop for subprocess STDIO
                        if _sys.platform == "win32":
                            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

                        # Parse "npx -y @azure/mcp@latest server start" → command + args list
                        parts = raw_command.strip().split() if raw_command.strip() else ["npx", "-y", "@azure/mcp@latest", "server", "start"]
                        cmd = parts[0]
                        cmd_args = parts[1:] if len(parts) > 1 else []

                        tool_name = live_config.get("toolName") or live_config.get("tool_name", "")
                        tool_arguments = live_config.get("arguments") or live_config.get("payload") or {}

                        server_params = StdioServerParameters(
                            command=cmd,
                            args=cmd_args,
                            env=_os.environ.copy()
                        )

                        async def _run_stdio_mcp():
                            async with stdio_client(server_params) as (read, write):
                                async with ClientSession(read, write) as session:
                                    await session.initialize()

                                    # Always discover available tools first
                                    tools_response = await session.list_tools()
                                    available_tools = [t.name for t in tools_response.tools] if tools_response.tools else []

                                    _tool_name = tool_name
                                    if not _tool_name:
                                        # Auto-select first available tool
                                        if available_tools:
                                            _tool_name = available_tools[0]
                                        else:
                                            return {
                                                "success": False,
                                                "output": "MCP server returned no tools. Check az login and server startup.",
                                                "tool_name": "",
                                                "available_tools": [],
                                                "metadata": {"transport": "stdio", "command": raw_command}
                                            }
                                    elif _tool_name not in available_tools:
                                        # Specified tool not found — report available tools
                                        tools_list = "\n".join(f"  - {t}" for t in available_tools) if available_tools else "  (none)"
                                        return {
                                            "success": False,
                                            "output": (
                                                f"Tool '{_tool_name}' not found in MCP server.\n"
                                                f"Available tools:\n{tools_list}"
                                            ),
                                            "tool_name": _tool_name,
                                            "available_tools": available_tools,
                                            "metadata": {"transport": "stdio", "command": raw_command}
                                        }

                                    call_result = await session.call_tool(_tool_name, tool_arguments)

                                    # Extract text content from MCP result
                                    output_text = ""
                                    if call_result.content:
                                        output_text = "\n".join(
                                            item.text for item in call_result.content
                                            if hasattr(item, "text") and item.text
                                        )
                                    is_error = getattr(call_result, "isError", False)
                                    return {
                                        "success": not is_error,
                                        "output": output_text,
                                        "tool_name": _tool_name,
                                        "available_tools": available_tools,
                                        "metadata": {"transport": "stdio", "command": raw_command}
                                    }

                        try:
                            result = asyncio.run(_run_stdio_mcp())
                        except BaseException as _eg:
                            # Unwrap anyio ExceptionGroup to surface the real error
                            subs = getattr(_eg, "exceptions", None)
                            if subs:
                                real_msgs = "; ".join(
                                    f"{type(e).__name__}: {e}" for e in subs
                                )
                                logger.error(
                                    f"MCP STDIO sub-exceptions: {real_msgs}",
                                    exc_info=_eg
                                )
                                raise RuntimeError(
                                    f"MCP STDIO failed — {real_msgs}"
                                ) from _eg
                            raise

                    else:
                        # ─── HTTP MODE (existing path — unchanged) ───
                        from echolib.Get_connector.Get_MCP.http_script import HTTPMCPConnector

                        adhoc_config = {
                            "name": live_config.get("name", f"adhoc_mcp_{node_id}"),
                            "description": live_config.get("description", f"Ad-hoc MCP call from node {node_id}"),
                            "auth_config": live_config.get("auth_config") or live_config.get("authentication") or {"type": "none"},
                            "input_schema": live_config.get("input_schema", {}),
                            "output_schema": live_config.get("output_schema", {}),
                            "endpoint_url": live_config.get("endpoint_url") or live_config.get("url", ""),
                            "method": live_config.get("method", "POST"),
                            "headers": live_config.get("headers"),
                            "query_params": live_config.get("query_params"),
                            "timeout": live_config.get("timeout", 30),
                        }
                        connector = HTTPMCPConnector(**adhoc_config)

                        # HTTPMCPConnector.test() is ASYNC — run in new event loop
                        payload = live_config.get("payload", {})
                        result = asyncio.run(connector.test(payload=payload))

                # === Build full context for the next node ===
                mcp_data = result.get("output") if isinstance(result, dict) else result

                # ─── Map MCP response fields into workflow state ───
                state_updates: Dict[str, Any] = {}
                if isinstance(mcp_data, dict):
                    state_updates.update(mcp_data)
                elif isinstance(mcp_data, str):
                    try:
                        parsed_data = json.loads(mcp_data)
                        if isinstance(parsed_data, dict):
                            state_updates.update(parsed_data)
                    except (json.JSONDecodeError, TypeError):
                        pass

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_summary = {
                        "node_name": node_name,
                        "success": result.get("success") if isinstance(result, dict) else True,
                        "data_preview": str(mcp_data)[:500] if mcp_data else "",
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                state_updates.update({
                    "last_node_output": mcp_data,
                    "mcp_result": result,
                    "crew_result": json.dumps(mcp_data) if not isinstance(mcp_data, str) else mcp_data,
                    "messages": [{
                        "node": node_id,
                        "type": "mcp",
                        "name": node_name,
                        "mode": "managed" if connector_id else "ad-hoc",
                        "success": result.get("success") if isinstance(result, dict) else True,
                        "data": mcp_data,
                    }],
                })
                return state_updates

            except Exception as e:
                logger.error(f"MCP node '{node_name}' failed: {e}")
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                error_result = {"success": False, "error": str(e), "node": node_id}
                return {
                    "last_node_output": error_result,
                    "mcp_result": error_result,
                    "crew_result": f"MCP Error in {node_name}: {e}",
                    "messages": [{
                        "node": node_id,
                        "type": "mcp",
                        "name": node_name,
                        "error": str(e),
                    }],
                }

        return mcp_node

    def _create_code_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create a Code execution node for LangGraph.

        Executes user-provided Python code in a sandboxed environment.
        The code has access to:
        - All current state variables (read-only injection)
        - A captured `result` variable for output
        - Standard library modules only (no arbitrary imports unless in packages list)

        stdout output and the `result` variable are captured and passed to the
        next node via state['last_node_output'] and state['code_result'].

        Args:
            node_id: Unique node identifier.
            node_config: Node configuration dict. May contain:
                - code (str): Python source code to execute
                - language (str): Programming language (currently only "python")
                - packages (list): Optional list of allowed packages

        Returns:
            Async callable node function compatible with LangGraph StateGraph.
        """
        config = node_config.get("config", node_config)
        # Fallback: resolve config from metadata for pre-fix workflows
        if not config.get("code"):
            config = node_config.get("metadata", {}).get("code_config", config)
        node_name = node_config.get("name", node_id)

        def code_node(state: Dict[str, Any]) -> Dict[str, Any]:
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync,
                    publish_step_failed_sync
                )
                input_summary = {
                    "node_name": node_name,
                    "node_type": "Code",
                    "language": config.get("language", "python"),
                }
                step_tracker.start_step(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)
                publish_step_started_sync(run_id, step_id, node_name, StepType.TOOL_CALL, input_summary)

            try:
                code = config.get("code", "")
                language = config.get("language", "python")

                from ..runtime.code_executor import CodeExecutor
                code_result = CodeExecutor.execute(code, language, state)

                code_output = code_result["output"]
                stdout_output = code_result.get("stdout", "")

                if not code_result["success"]:
                    raise RuntimeError(code_result.get("error", "Code execution failed"))

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_summary = {
                        "node_name": node_name,
                        "success": True,
                        "stdout_preview": stdout_output[:500] if stdout_output else "",
                        "result_preview": str(code_output)[:500] if code_output else "",
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                return {
                    "last_node_output": code_output,
                    "code_result": code_result,
                    "crew_result": str(code_output) if code_output else stdout_output,
                    "messages": [{
                        "node": node_id,
                        "type": "code",
                        "name": node_name,
                        "success": True,
                        "output": code_output,
                        "stdout": stdout_output,
                    }],
                }

            except Exception as e:
                logger.error(f"Code node '{node_name}' failed: {e}")
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                error_result = {"success": False, "error": str(e), "node": node_id}
                return {
                    "last_node_output": error_result,
                    "code_result": error_result,
                    "crew_result": f"Code Error in {node_name}: {e}",
                    "messages": [{
                        "node": node_id,
                        "type": "code",
                        "name": node_name,
                        "error": str(e),
                    }],
                }

        return code_node

    def _create_self_review_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create a Self-Review node for LangGraph.

        Uses an LLM to validate the output from the previous node against
        a review prompt and minimum confidence threshold. The LLM acts as
        a quality gate, assessing whether the prior output meets criteria.

        The review result (pass/fail, confidence, feedback) is stored in
        state['last_node_output'] and state['review_result'].

        Args:
            node_id: Unique node identifier.
            node_config: Node configuration dict. May contain:
                - reviewPrompt (str): Criteria for the LLM to evaluate against
                - minConfidence (float): Minimum confidence score (0.0–1.0)
                - model (dict): Optional LLM model override

        Returns:
            Async callable node function compatible with LangGraph StateGraph.
        """
        config = node_config.get("config", node_config)
        node_name = node_config.get("name", node_id)

        async def self_review_node(state: Dict[str, Any]) -> Dict[str, Any]:
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync,
                    publish_step_failed_sync
                )
                input_summary = {
                    "node_name": node_name,
                    "node_type": "Self-Review",
                    "min_confidence": config.get("minConfidence", 0.8),
                }
                step_tracker.start_step(run_id, step_id, node_name, StepType.LLM_CALL, input_summary)
                publish_step_started_sync(run_id, step_id, node_name, StepType.LLM_CALL, input_summary)

            try:
                review_prompt = config.get("reviewPrompt", "Review the output for quality and completeness.")
                min_confidence = config.get("minConfidence", 0.8)
                llm_config = config.get("model", {})

                # Gather the output from the previous node to review
                previous_output = (
                    state.get("last_node_output")
                    or state.get("crew_result")
                    or ""
                )
                previous_output_str = (
                    json.dumps(previous_output, default=str)
                    if not isinstance(previous_output, str)
                    else previous_output
                )

                # Build the LLM prompt for self-review
                llm_prompt = f"""You are a quality assurance reviewer. Evaluate the following output against the review criteria.

REVIEW CRITERIA:
{review_prompt}

OUTPUT TO REVIEW:
{previous_output_str[:4000]}

Respond in the following JSON format ONLY (no other text):
{{
  "passed": true or false,
  "confidence": 0.0 to 1.0,
  "feedback": "Brief explanation of your assessment",
  "issues": ["list of specific issues found, if any"]
}}"""

                # Execute the LLM call in a thread (sync LLM calls)
                llm_response = await asyncio.to_thread(
                    self._execute_llm_call, llm_config, llm_prompt
                )

                # Parse the LLM response
                try:
                    # Try to extract JSON from the response
                    review_data = json.loads(llm_response)
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as a textual review
                    review_data = {
                        "passed": True,
                        "confidence": 0.7,
                        "feedback": llm_response,
                        "issues": [],
                    }

                # Apply the confidence threshold
                confidence = review_data.get("confidence", 0.0)
                passed = review_data.get("passed", False) and confidence >= min_confidence

                review_result = {
                    "success": True,
                    "passed": passed,
                    "confidence": confidence,
                    "min_confidence": min_confidence,
                    "feedback": review_data.get("feedback", ""),
                    "issues": review_data.get("issues", []),
                }

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_summary = {
                        "node_name": node_name,
                        "passed": passed,
                        "confidence": confidence,
                        "feedback_preview": review_data.get("feedback", "")[:300],
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                return {
                    "last_node_output": review_result,
                    "review_result": review_result,
                    "crew_result": review_data.get("feedback", ""),
                    "messages": [{
                        "node": node_id,
                        "type": "self_review",
                        "name": node_name,
                        "passed": passed,
                        "confidence": confidence,
                        "feedback": review_data.get("feedback", ""),
                        "issues": review_data.get("issues", []),
                    }],
                }

            except Exception as e:
                logger.error(f"Self-Review node '{node_name}' failed: {e}")
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))

                error_result = {"success": False, "error": str(e), "node": node_id}
                return {
                    "last_node_output": error_result,
                    "review_result": error_result,
                    "crew_result": f"Self-Review Error in {node_name}: {e}",
                    "messages": [{
                        "node": node_id,
                        "type": "self_review",
                        "name": node_name,
                        "error": str(e),
                    }],
                }

        return self_review_node

    # =====================================================================
    # Node Type Dispatcher
    # =====================================================================

    def _create_start_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create a Start node function for LangGraph.

        The Start node is a PASSTHROUGH — it does NOT call an LLM.
        It reads the user's input from state, maps it to the defined
        input variables (from the canvas inputVariables config), and
        forwards everything to the next node.

        Input variables (e.g., "query") are already initialized by the
        executor, but this node explicitly sets them in state and
        populates last_node_output so downstream nodes receive context.

        Args:
            node_id: Node identifier.
            node_config: Agent config dict containing input_schema from
                         the canvas Start node's inputVariables.

        Returns:
            Callable node function compatible with LangGraph.
        """
        agent_name = node_config.get("name", "Start")
        input_schema = node_config.get("input_schema", [])

        def start_node(state: Dict[str, Any]) -> Dict[str, Any]:
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync
                )
                input_summary = {
                    "node_name": agent_name,
                    "node_type": "Start",
                    "input_variables": input_schema,
                }
                step_tracker.start_step(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )
                publish_step_started_sync(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )

            # === Core Logic: Passthrough with variable injection ===
            user_input = (
                state.get("original_user_input")
                or state.get("user_input")
                or state.get("task_description")
                or ""
            )

            # Map user_input to each defined input variable
            # e.g., input_schema=["query"] → {"query": user_input}
            outputs = {}
            for var_name in input_schema:
                # Use the value already in state if set (executor may have mapped it),
                # otherwise default to user_input
                outputs[var_name] = state.get(var_name) or user_input

            logger.info(
                f"Start node '{agent_name}' passed through. "
                f"Input variables: {list(outputs.keys())}, "
                f"user_input length: {len(user_input)}"
            )

            # === TRANSPARENCY: Step completed ===
            if settings.transparency_enabled:
                output_summary = {
                    "node_name": agent_name,
                    "node_type": "Start",
                    "variables_set": list(outputs.keys()),
                    "output_preview": user_input[:200] if user_input else "",
                }
                step_tracker.complete_step(run_id, step_id, output_summary)
                publish_step_completed_sync(run_id, step_id, output_summary)

            return {
                **outputs,
                "last_node_output": user_input,
                "crew_result": user_input,
                "messages": [{
                    "agent": node_id,
                    "role": "Start",
                    "node_type": "Start",
                    "inputs": {"user_input": user_input},
                    "outputs": outputs,
                }]
            }

        return start_node

    def _create_end_node(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Create an End node function for LangGraph.

        The End node is a PASSTHROUGH — it does NOT call an LLM.
        It collects the final output from the previous node(s) and maps
        it to the defined output variables (from the canvas outputVariables
        config).

        It reads from crew_result / last_node_output (the accumulated result
        from the last processing node) and sets the output variables.

        Args:
            node_id: Node identifier.
            node_config: Agent config dict containing output_schema from
                         the canvas End node's outputVariables.

        Returns:
            Callable node function compatible with LangGraph.
        """
        agent_name = node_config.get("name", "End")
        output_schema = node_config.get("output_schema", [])

        def end_node(state: Dict[str, Any]) -> Dict[str, Any]:
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            step_id = new_id("step_")

            # === TRANSPARENCY: Step started ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync, publish_step_completed_sync
                )
                input_summary = {
                    "node_name": agent_name,
                    "node_type": "End",
                    "output_variables": output_schema,
                }
                step_tracker.start_step(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )
                publish_step_started_sync(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )

            # === Core Logic: Collect final output ===
            # Prefer _last_agent_output (set only by LLM agent nodes, never by API/MCP/Code).
            # This prevents raw HTTP responses or code output from polluting output variables.
            # Fall back to crew_result only if no agent has run yet.
            final_output = (
                state.get("_last_agent_output")
                or state.get("crew_result")
                or state.get("last_node_output")
                or state.get("review_result")
                or state.get("code_result")
                or state.get("api_result")
                or state.get("mcp_result")
                or ""
            )

            # If final_output is a dict, convert to string for variable mapping
            if isinstance(final_output, dict):
                import json
                final_output_str = json.dumps(final_output, indent=2)
            else:
                final_output_str = str(final_output) if final_output else ""

            # Try intelligent extraction of output variables from the agent output text.
            # This extracts structured values (e.g. risk_score: 65.5) before falling back
            # to the full text, so each output variable gets a meaningful value.
            extracted_vars: Dict[str, Any] = {}
            if output_schema and final_output_str:
                try:
                    from ..crewai_adapter import CrewAIAdapter
                    extracted_vars = CrewAIAdapter._extract_state_variables(
                        final_output_str, state
                    )
                except Exception as _ext_err:
                    logger.warning(f"End node variable extraction failed: {_ext_err}")

            # Map to defined output variables
            # Priority: (1) intelligently extracted value, (2) value explicitly set in state
            # by a prior node, (3) full agent output text as fallback.
            outputs = {}
            for var_name in output_schema:
                if var_name in extracted_vars:
                    outputs[var_name] = extracted_vars[var_name]
                else:
                    existing = state.get(var_name)
                    if existing not in (None, ""):
                        outputs[var_name] = existing
                    else:
                        outputs[var_name] = final_output_str

            logger.info(
                f"End node '{agent_name}' collected output. "
                f"Output variables: {list(outputs.keys())}, "
                f"output length: {len(final_output_str)}"
            )

            # === TRANSPARENCY: Step completed ===
            if settings.transparency_enabled:
                output_summary = {
                    "node_name": agent_name,
                    "node_type": "End",
                    "variables_collected": list(outputs.keys()),
                    "output_preview": final_output_str[:500] if final_output_str else "",
                }
                step_tracker.complete_step(run_id, step_id, output_summary)
                publish_step_completed_sync(run_id, step_id, output_summary)

            return {
                **outputs,
                "crew_result": final_output_str,
                "last_node_output": final_output_str,
                "messages": [{
                    "agent": node_id,
                    "role": "End",
                    "node_type": "End",
                    "inputs": {"final_output": final_output_str[:200]},
                    "outputs": outputs,
                }]
            }

        return end_node

    # ========================================================================
    # HITL NODE SUPPORT (Additive — does NOT modify existing methods)
    # ========================================================================

    # Shared MemorySaver instances for HITL workflows.
    # Key = workflow_id, Value = MemorySaver instance.
    # This ensures the same checkpointer survives across compile calls
    # (initial execution + resume), so LangGraph can restore state.
    _hitl_checkpointers: Dict[str, Any] = {}

    def _get_checkpointer(self, workflow: Dict[str, Any]):
        """
        Return the appropriate checkpointer for this workflow.

        - Workflows with HITL nodes -> PostgresSaver (persistent) or shared MemorySaver
        - All other workflows -> new MemorySaver() (existing behavior, unchanged)

        For HITL workflows, the checkpointer MUST be the same instance across
        the initial execution and resume calls. Otherwise the checkpoint
        (graph state at the interrupt point) is lost and resume re-executes
        from scratch.

        Args:
            workflow: Workflow definition dict.

        Returns:
            A LangGraph checkpointer instance.
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

        workflow_id = workflow.get("workflow_id", "unknown")

        # HITL workflow -> try PostgresSaver first (persistent across restarts)
        # Cache the PostgresSaver instance so the same connection is reused
        try:
            if "_postgres_checkpointer" not in self._hitl_checkpointers:
                from apps.workflow.runtime.checkpointer import get_postgres_checkpointer
                self._hitl_checkpointers["_postgres_checkpointer"] = get_postgres_checkpointer()
                logger.info("Using PostgresSaver checkpointer for HITL-enabled workflow")
            else:
                logger.info("Reusing cached PostgresSaver checkpointer")
            return self._hitl_checkpointers["_postgres_checkpointer"]
        except Exception as e:
            logger.warning(
                f"PostgresSaver unavailable ({e}), using shared MemorySaver "
                f"for workflow '{workflow_id}'."
            )
            # Shared MemorySaver — same instance for this workflow_id
            if workflow_id not in self._hitl_checkpointers:
                self._hitl_checkpointers[workflow_id] = MemorySaver()
                logger.info(f"Created shared MemorySaver for HITL workflow '{workflow_id}'")
            else:
                logger.info(f"Reusing shared MemorySaver for HITL workflow '{workflow_id}'")
            return self._hitl_checkpointers[workflow_id]

    @staticmethod
    def _extract_preceding_output(state: dict) -> str:
        """Extract the most recent agent output from state for HITL review.

        Checks the keys that crewai_adapter uses to store results, in priority
        order. Falls back to the last message content if none of the dedicated
        keys are populated.
        """
        for key in (
            "crew_result",
            "result",
            "hierarchical_output",
            "parallel_output",
            "merged_output",
        ):
            val = state.get(key)
            if val and isinstance(val, str) and val.strip():
                return val
        # Fallback: last message content
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                return last.get("outputs", {}).get("result", str(last))
        return ""

    @staticmethod
    def _get_preceding_node_name(state: dict) -> str:
        """Get the name of the most recent agent that produced output."""
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                return last.get("agent", last.get("node", ""))
        return ""

    def _create_hitl_node(
        self,
        node_id: str,
        node_config: Dict[str, Any],
    ):
        """
        Create a dedicated HITL node function that uses LangGraph interrupt().

        This is the correct HITL implementation that actually pauses the graph
        using LangGraph's native interrupt mechanism. The interrupt() call:
        1. Saves graph state via the checkpointer (PostgresSaver for HITL)
        2. Returns immediately to the caller with the interrupt payload
        3. On resume via Command(resume=value), the interrupt() returns the value

        Unlike the existing _create_agent_node HITL logic (which does not actually
        pause), this method uses the official LangGraph interrupt API.

        Args:
            node_id: Node identifier.
            node_config: Full node configuration dict.

        Returns:
            Callable node function compatible with LangGraph.
        """
        # Extract HITL configuration from node metadata
        metadata = node_config.get("metadata", {})
        hitl_config = metadata.get("hitl_config", {})
        config_section = node_config.get("config", {})

        # Merge config sources (config section may also contain hitl fields)
        merged_hitl_config = {
            "title": hitl_config.get("title") or config_section.get("title", ""),
            "message": hitl_config.get("message") or config_section.get("message", ""),
            "priority": hitl_config.get("priority") or config_section.get("priority", "medium"),
        }
        # Preserve allowed_decisions / allowEdit / allowDefer for derive_allowed_decisions
        for key in ("allowed_decisions", "allowEdit", "allowDefer"):
            if key in hitl_config:
                merged_hitl_config[key] = hitl_config[key]
            elif key in config_section:
                merged_hitl_config[key] = config_section[key]

        agent_name = node_config.get("name", node_id)

        def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            HITL node execution function.

            Calls HITLManager.request_approval() to record the HITL checkpoint,
            then calls interrupt() to pause the LangGraph execution.

            When resumed via Command(resume=value), the interrupt() returns
            the human's decision, which this node processes accordingly.

            Args:
                state: Current workflow state.

            Returns:
                Updated state with HITL decision results.
            """
            from langgraph.types import interrupt
            from langgraph.errors import GraphInterrupt
            from ..runtime.hitl import HITLManager, derive_allowed_decisions
            from echolib.config import settings
            from echolib.utils import new_id

            run_id = state.get("run_id", "unknown")
            workflow_id = state.get("workflow_id", "unknown")
            step_id = new_id("step_")

            # === TRANSPARENCY: Setup ===
            if settings.transparency_enabled:
                from ..runtime.transparency import step_tracker, StepType
                from ..runtime.event_publisher import (
                    publish_step_started_sync,
                    publish_step_completed_sync,
                    publish_step_failed_sync,
                )

                input_summary = {
                    "user_input": state.get("original_user_input", "")[:200],
                    "agent_name": agent_name,
                    "agent_role": "HITL Review",
                    "is_hitl": True,
                }
                step_tracker.start_step(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )
                publish_step_started_sync(
                    run_id, step_id, agent_name, StepType.AGENT, input_summary
                )

            # Extract preceding agent output BEFORE try so it is available
            # in both the try block and the except GraphInterrupt handler.
            preceding_output = WorkflowCompiler._extract_preceding_output(state)
            preceding_node_name = WorkflowCompiler._get_preceding_node_name(state)

            try:
                # Resolve allowed decisions from config
                allowed = derive_allowed_decisions(merged_hitl_config)

                # Build interrupt payload with full HITL context
                interrupt_payload = {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "node_id": node_id,
                    "title": merged_hitl_config.get("title", ""),
                    "message": merged_hitl_config.get("message", ""),
                    "priority": merged_hitl_config.get("priority", "medium"),
                    "allowed_decisions": allowed,
                    "preceding_output": preceding_output,
                    "preceding_node_name": preceding_node_name,
                    "state_snapshot": {
                        k: v for k, v in state.items()
                        if not k.startswith("_") and k != "messages"
                    },
                }

                # Record HITL request in HITLManager for audit trail
                # Use the DI container singleton if available, otherwise create instance
                try:
                    from echolib.di import container
                    hitl = container.resolve('workflow.hitl')
                except Exception:
                    hitl = HITLManager()

                hitl.request_approval(
                    run_id=run_id,
                    workflow_id=workflow_id,
                    blocked_at=node_id,
                    context={
                        "agent_output": preceding_output,
                        "preceding_node_name": preceding_node_name,
                        "state_snapshot": dict(state),
                        "hitl_config": merged_hitl_config,
                    },
                )

                # === INTERRUPT: Pause graph execution ===
                # This call saves state via PostgresSaver and returns to caller.
                # When resumed, human_decision receives the Command(resume=value).
                human_decision = interrupt(interrupt_payload)

                # === RESUMED: Process human decision ===
                action = "approve"
                decision_payload = {}

                if isinstance(human_decision, dict):
                    action = human_decision.get("action", "approve")
                    decision_payload = human_decision.get("payload", {})

                # Record the decision outcome
                if action == "approve":
                    hitl.approve(run_id=run_id, actor="human_reviewer")
                elif action == "reject":
                    hitl.reject(run_id=run_id, actor="human_reviewer")
                    raise RuntimeError(
                        f"Workflow rejected at HITL checkpoint: {node_id}"
                    )
                elif action == "edit":
                    hitl.modify(
                        run_id=run_id,
                        actor="human_reviewer",
                        changes=decision_payload,
                    )

                # === TRANSPARENCY: Step completed ===
                if settings.transparency_enabled:
                    output_summary = {
                        "output_preview": f"HITL {action}: {node_id}",
                        "agent_name": agent_name,
                        "agent_role": "HITL Review",
                        "hitl_action": action,
                    }
                    step_tracker.complete_step(run_id, step_id, output_summary)
                    publish_step_completed_sync(run_id, step_id, output_summary)

                # Return state updates
                output_schema = node_config.get("output_schema", [])
                outputs = {key: state.get(key) for key in output_schema if key in state}

                # Merge edit changes into outputs so downstream agents see them
                if action == "edit" and decision_payload:
                    # Frontend sends: {"changes": {"edit_feedback": "...", ...}}
                    # Unwrap the inner changes dict; fall back to payload itself
                    # for clients that send flat payloads.
                    changes = decision_payload.get("changes", decision_payload)

                    # Separate textual feedback from explicit field overwrites
                    edit_feedback = (
                        changes.get("edit_feedback")
                        or changes.get("prompt")
                    )

                    # Apply explicit field-level changes
                    # (e.g. {"analysis": "replacement text"})
                    for key, value in changes.items():
                        if key in ("edit_feedback", "prompt"):
                            continue  # meta-keys, not state fields
                        if key in output_schema or key in state:
                            outputs[key] = value

                    # Inject textual feedback so downstream agents see it
                    if edit_feedback:
                        feedback_prefix = (
                            "[HUMAN REVIEWER FEEDBACK — you MUST incorporate "
                            f"this feedback into your response]: {edit_feedback}"
                        )

                        # Prepend to output_schema fields that were NOT
                        # explicitly overwritten by the reviewer
                        for key in output_schema:
                            if key not in changes:
                                existing = outputs.get(key) or state.get(key, "")
                                if isinstance(existing, str) and existing:
                                    outputs[key] = f"{feedback_prefix}\n\n{existing}"

                        # Update all context keys that downstream agents
                        # read from across workflow types:
                        #   sequential  → crew_result  (crewai_adapter:857)
                        #   hierarchical→ crew_result  (crewai_adapter:396)
                        #   parallel    → crew_result  (crewai_adapter:617)
                        #   hybrid merge→ crew_result, parallel_output,
                        #                 merged_output (crewai_adapter:1220-1270)
                        for ctx_key in (
                            "crew_result",
                            "parallel_output",
                            "merged_output",
                            "hierarchical_output",
                        ):
                            ctx_existing = state.get(ctx_key, "")
                            if isinstance(ctx_existing, str) and ctx_existing:
                                outputs[ctx_key] = (
                                    f"{feedback_prefix}\n\n{ctx_existing}"
                                )

                return {
                    **outputs,
                    "messages": [{
                        "agent": node_id,
                        "role": "HITL Review",
                        "inputs": {"hitl_config": merged_hitl_config},
                        "outputs": {"action": action, "payload": decision_payload},
                        "hitl_checkpoint": True,
                        "hitl_action": action,
                    }],
                }

            except RuntimeError:
                # Re-raise rejection errors
                raise
            except GraphInterrupt:
                # === TRANSPARENCY: HITL node paused (NOT a failure) ===
                # interrupt() raises GraphInterrupt to halt the graph.
                # Emit a completed event with paused metadata so the
                # frontend shows "waiting for review" instead of "failed".
                if settings.transparency_enabled:
                    paused_summary = {
                        "output_preview": f"⏸️ Waiting for human review: {merged_hitl_config.get('title', node_id)}",
                        "agent_name": agent_name,
                        "agent_role": "HITL Review",
                        "hitl_paused": True,
                        "hitl_config": {
                            "title": merged_hitl_config.get("title", ""),
                            "priority": merged_hitl_config.get("priority", "medium"),
                        },
                        "preceding_output": preceding_output,
                        "preceding_node_name": preceding_node_name,
                    }
                    step_tracker.complete_step(run_id, step_id, paused_summary)
                    publish_step_completed_sync(run_id, step_id, paused_summary)
                raise  # Re-raise so LangGraph handles the interrupt
            except Exception as e:
                # === TRANSPARENCY: Step failed ===
                if settings.transparency_enabled:
                    step_tracker.fail_step(run_id, step_id, str(e))
                    publish_step_failed_sync(run_id, step_id, str(e))
                raise

        return hitl_node

    def _create_node_for_type(
        self,
        node_id: str,
        node_config: Dict[str, Any]
    ):
        """
        Dispatch to the correct node creation method based on node type.

        This is the central router that the compile methods call instead of
        directly calling _create_agent_node. It inspects the node's type
        field and delegates accordingly.

        Supported types:
            - "Agent" (default) → _create_agent_node
            - "API"             → _create_api_node
            - "MCP"             → _create_mcp_node
            - "Code"            → _create_code_node
            - "Self-Review"     → _create_self_review_node
            - "Conditional"/"Router" → _create_conditional_node (handled separately in BFS)
            - "HITL"            → _create_agent_node (has built-in HITL logic)
            - "Start"           → _create_start_node (passthrough, maps inputVariables)
            - "End"             → _create_end_node (passthrough, collects outputVariables)

        Args:
            node_id: Node identifier.
            node_config: Full node configuration dict.

        Returns:
            Callable node function compatible with LangGraph.
        """
        # Determine node type from config — check multiple locations
        node_type = (
            node_config.get("type")
            or node_config.get("metadata", {}).get("node_type")
            or node_config.get("config", {}).get("type")
            or "Agent"
        )

        node_type_normalized = node_type.strip().lower()

        if node_type_normalized == "api":
            logger.info(f"Creating API node: {node_id} ('{node_config.get('name', node_id)}')")
            return self._create_api_node(node_id, node_config)

        elif node_type_normalized == "mcp":
            logger.info(f"Creating MCP node: {node_id} ('{node_config.get('name', node_id)}')")
            return self._create_mcp_node(node_id, node_config)

        elif node_type_normalized == "code":
            logger.info(f"Creating Code node: {node_id} ('{node_config.get('name', node_id)}')")
            return self._create_code_node(node_id, node_config)

        elif node_type_normalized in ("self-review", "self_review", "selfreview"):
            logger.info(f"Creating Self-Review node: {node_id} ('{node_config.get('name', node_id)}')")
            return self._create_self_review_node(node_id, node_config)

        elif node_type_normalized in ("conditional", "router"):
            # Conditional nodes are handled separately in the BFS traversal,
            # but this fallback exists for safety.
            logger.info(f"Creating Conditional node: {node_id}")
            return self._create_conditional_node(node_id, node_config)

        elif node_type_normalized == "start":
            logger.info(f"Creating Start node: {node_id} (passthrough, no LLM)")
            return self._create_start_node(node_id, node_config)

        elif node_type_normalized == "end":
            logger.info(f"Creating End node: {node_id} (passthrough, no LLM)")
            return self._create_end_node(node_id, node_config)

        elif node_type_normalized == "hitl":
            # Dedicated HITL node with LangGraph interrupt() support.
            # Uses _create_hitl_node which properly pauses graph execution.
            logger.info(f"Creating HITL node: {node_id} (interrupt-based pause)")
            return self._create_hitl_node(node_id, node_config)

        else:
            # Default: Agent node (also handles HITL via internal check)
            logger.info(f"Creating Agent node: {node_id} (type='{node_type}')")
            return self._create_agent_node(node_id, node_config)

    def _execute_llm_call(self, llm_config: Dict[str, Any], prompt: str) -> str:
        """
        Execute actual LLM call based on provider.

        Args:
            llm_config: LLM configuration (provider, model, temperature)
            prompt: Prompt to send to LLM

        Returns:
            LLM response text
        """
        provider = llm_config.get("provider", os.getenv("LLM_PROVIDER", "openrouter"))
        model = llm_config.get(
            "model",
            os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
        )
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 1000)

        if provider == "openrouter":
            return self._call_openrouter(model, prompt, temperature, max_tokens)
        elif provider == "openai":
            return self._call_openai(model, prompt, temperature, max_tokens)
        elif provider == "anthropic":
            return self._call_anthropic(model, prompt, temperature, max_tokens)
        elif provider == "azure":
            return self._call_azure(model, prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _call_openai(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenAI API or Ollama using ChatOpenAI."""
        try:
            from langchain_openai import ChatOpenAI

            # Check if using Ollama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://10.188.100.130:8002/v1")
            use_ollama = os.getenv("USE_OLLAMA", "true").lower() == "true"

            if use_ollama:
                # Use Ollama endpoint
                llm = ChatOpenAI(
                    base_url=ollama_url,
                    api_key="ollama",
                    model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Use OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            response = llm.invoke(prompt)
            return response.content

        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

    def _call_anthropic(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API."""
        try:
            import anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")

    def _call_azure(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call Azure OpenAI API."""
        try:
            from openai import AzureOpenAI

            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

            if not api_key or not endpoint:
                raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set")

            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {e}")

    def _call_openrouter(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call OpenRouter via ChatOpenAI wrapper."""
        try:
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set")

            llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )

            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {e}")

    def _determine_graph_type(self, execution_model: str) -> str:
        """
        Determine LangGraph graph type from execution model.

        Args:
            execution_model: Execution model

        Returns:
            LangGraph graph type
        """
        mapping = {
            "sequential": "StateGraph",
            "parallel": "StateGraph",
            "hierarchical": "StateGraph",
            "hybrid": "StateGraph"
        }
        return mapping.get(execution_model, "StateGraph")
