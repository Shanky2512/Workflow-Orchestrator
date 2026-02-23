"""
Workflow executor.
Executes workflows using LangGraph runtime.
"""
import re
import logging
from typing import Dict, Any, Optional
from echolib.utils import new_id
from echolib.config import settings
from apps.workflow.visualization.node_mapper import normalize_agent_config

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """
    Workflow execution service.
    Compiles and runs workflows via LangGraph.
    """

    def __init__(self, storage, compiler, agent_registry, guards=None):
        """
        Initialize executor.

        Args:
            storage: WorkflowStorage instance
            compiler: WorkflowCompiler instance
            agent_registry: AgentRegistry instance
            guards: RuntimeGuards instance (optional)
        """
        self.storage = storage
        self.compiler = compiler
        self.agent_registry = agent_registry
        self.guards = guards

    def _extract_workflow_variables(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract input variables from workflow state_schema.

        The state_schema contains variable definitions from Start/End nodes.
        This method extracts input variables and their types.

        Args:
            workflow: Workflow definition

        Returns:
            Dict of variable names to their default values based on type
        """
        state_schema = workflow.get("state_schema", {})
        variables = {}

        for var_name, var_type in state_schema.items():
            # Set default value based on type
            if var_type == "number":
                variables[var_name] = 0
            elif var_type == "boolean":
                variables[var_name] = False
            elif var_type == "array":
                variables[var_name] = []
            elif var_type == "object":
                variables[var_name] = {}
            else:
                # Default to empty string for string and unknown types
                variables[var_name] = ""

        return variables

    def _substitute_variables(
        self,
        text: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Substitute variable placeholders in text with actual values.

        Supports formats: {variable_name}, {{variable_name}}, ${variable_name}

        Args:
            text: Text containing variable placeholders
            variables: Dict of variable names to values

        Returns:
            Text with variables substituted
        """
        if not text or not variables:
            return text

        result = text

        # Substitute {variable_name} format
        for var_name, var_value in variables.items():
            # Handle single brace format: {var_name}
            result = result.replace(f"{{{var_name}}}", str(var_value))
            # Handle double brace format: {{var_name}}
            result = result.replace(f"{{{{{var_name}}}}}", str(var_value))
            # Handle shell-style format: ${var_name}
            result = result.replace(f"${{{var_name}}}", str(var_value))

        return result

    def execute_workflow(
        self,
        workflow_id: str,
        execution_mode: str,
        version: Optional[str] = None,
        input_payload: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow identifier
            execution_mode: "test" or "final"
            version: Version (required for final mode)
            input_payload: Input data for workflow

        Returns:
            Execution result

        Raises:
            RuntimeError: If workflow not in correct state
        """
        if input_payload is None:
            input_payload = {}

        # Load workflow based on mode
        if execution_mode == "draft":
            # Try draft first, fall back to temp for backwards compatibility
            try:
                workflow = self.storage.load_workflow(
                    workflow_id=workflow_id,
                    state="draft"
                )
            except FileNotFoundError:
                # Fallback: try temp folder (for workflows saved before draft-chat feature)
                print(f"Draft not found for {workflow_id}, trying temp folder...")
                workflow = self.storage.load_workflow(
                    workflow_id=workflow_id,
                    state="temp"
                )
            # Draft workflows can be in draft, validated, or testing status (for temp fallback)
            if workflow.get("status") not in ("draft", "validated", "testing", "final"):
                raise RuntimeError("Workflow not in usable state")

        elif execution_mode == "test":
            workflow = self.storage.load_workflow(
                workflow_id=workflow_id,
                state="temp"
            )
            if workflow.get("status") != "testing":
                raise RuntimeError("Workflow not in testing state")

        elif execution_mode == "final":
            if not version:
                # Auto-resolve latest version when none provided
                versions = self.storage.list_versions(workflow_id)
                if not versions:
                    raise ValueError(
                        f"No final versions found for workflow '{workflow_id}'"
                    )
                version = versions[-1]
                print(f"[EXECUTOR] Auto-resolved latest version: {version}")
            workflow = self.storage.load_workflow(
                workflow_id=workflow_id,
                state="final",
                version=version
            )
            if workflow.get("status") != "final":
                raise RuntimeError("Workflow not finalized")

        else:
            raise ValueError(f"Invalid execution mode: {execution_mode}")

        # Apply guards if available
        if self.guards:
            self.guards.check_before_execution(workflow)

        # DEBUG: Log workflow execution_model after loading
        print(f"[EXECUTOR] Loaded workflow '{workflow_id}' with execution_model='{workflow.get('execution_model')}'")

        # Extract workflow variables from state_schema
        workflow_variables = self._extract_workflow_variables(workflow)

        # Load agent definitions - handle both embedded and ID-referenced formats
        agent_defs = {}
        for agent_entry in workflow.get("agents", []):
            # Handle both formats:
            # 1. New format: agent_entry is a full agent dict
            # 2. Old format: agent_entry is a string agent_id
            if isinstance(agent_entry, dict):
                # New format: agent is embedded in workflow
                agent_id = agent_entry.get("agent_id")
                if agent_id:
                    agent_defs[agent_id] = agent_entry
            elif isinstance(agent_entry, str):
                # Old format: agent_entry is an ID, load from registry
                agent_id = agent_entry
                try:
                    agent = self.agent_registry.get_agent(agent_id)
                    if agent is None:
                        # Agent not in cache - try reloading from disk
                        print(f"Warning: Agent '{agent_id}' not in cache, reloading registry...")
                        self.agent_registry._load_all()  # Reload all agents from disk
                        agent = self.agent_registry.get_agent(agent_id)
                        if agent is None:
                            raise RuntimeError(f"Agent '{agent_id}' not found in registry after reload")
                    agent_defs[agent_id] = agent
                except FileNotFoundError:
                    raise RuntimeError(f"Agent '{agent_id}' not found in registry")

        # Normalize agent configs — ensures top-level 'config' key exists
        # for API/Code/MCP nodes, handling pre-fix workflows
        for agent_id in agent_defs:
            normalize_agent_config(agent_defs[agent_id])

        # Compile workflow to LangGraph
        compiled_graph = self.compiler.compile_to_langgraph(workflow, agent_defs)

        # Execute workflow with LangGraph
        # Use run_id from input_payload if provided (for transparency tracking),
        # otherwise generate a new one
        if input_payload is None:
            input_payload = {}
        run_id = input_payload.get("run_id") or new_id("run_")

        try:
            # Prepare initial state with workflow and run IDs

            # Normalize user input: ensure "user_input" key is set
            # Accept common key names from frontend
            # FIX: Handle structured payloads (code, language, etc.) that don't use standard keys
            user_input = (
                input_payload.get("user_input")
                or input_payload.get("message")
                or input_payload.get("question")
                or input_payload.get("input")
                or input_payload.get("task_description")
                or input_payload.get("prompt")
            )

            # FIX: If no standard key found but payload has data, serialize entire payload
            # This handles structured inputs like {"code": "...", "language": "python"}
            if not user_input and input_payload:
                import json
                # Check if there are any meaningful keys in the payload (not just metadata)
                meaningful_keys = {k for k in input_payload.keys()
                                   if k not in ("workflow_id", "run_id", "mode", "version", "context")}
                if meaningful_keys:
                    # Serialize the structured input as JSON string for agents to parse
                    user_input = json.dumps(input_payload, indent=2)
                    print(f"[Executor] Serialized structured payload as user_input: {user_input[:200]}...")

            # Final fallback to empty string
            if not user_input:
                user_input = ""

            # FIXED: Set initial state with original_user_input preserved
            # This ensures all agents throughout the workflow have access
            # to the original user request, not just the first agent

            # Merge workflow variables with input payload
            # Input payload values override default variable values
            merged_variables = {**workflow_variables}
            for key, value in input_payload.items():
                if key in merged_variables or key not in ["workflow_id", "run_id", "mode", "version"]:
                    merged_variables[key] = value

            # FIX Bug 4: Map user message to workflow state variables
            # When user sends a message (e.g., "2 + 2 - 5"), it should populate
            # the first workflow state variable (typically 'query') so agents
            # with input_schema=["query"] receive the user's input correctly.
            if user_input and workflow_variables:
                # Get workflow state variable names in order
                state_var_names = list(workflow_variables.keys())
                if state_var_names:
                    # Map user_input to first state variable (usually 'query')
                    first_var = state_var_names[0]
                    # Only set if the variable is still at its default (empty) value
                    if not merged_variables.get(first_var):
                        merged_variables[first_var] = user_input
                        print(f"[Executor] Mapped user_input to workflow variable '{first_var}'")

            # Substitute variables in user_input if it contains placeholders
            user_input = self._substitute_variables(user_input, merged_variables)

            initial_state = {
                **merged_variables,  # Include all workflow variables
                **input_payload,
                "user_input": user_input,
                "original_user_input": user_input,  # CRITICAL: Preserve original
                "task_description": user_input,     # Alias for compatibility
                "messages": [],
                "workflow_id": workflow_id,
                "run_id": run_id,
                "_workflow_variables": workflow_variables,  # Track defined variables
                "_state_schema": workflow.get("state_schema", {})  # Track variable types
            }

            # Audit log user input (prompt injection detection)
            try:
                from echolib.observability import audit_log_input
                audit_log_input(
                    run_id=run_id,
                    user_input=user_input,
                    user_id=str(input_payload.get("user_id", "system")),
                    workflow_id=workflow_id,
                )
            except Exception as _audit_err:
                logger.debug(f"Audit log skipped: {_audit_err}")

            # Run the compiled graph
            config = {"configurable": {"thread_id": run_id}}

            # --- Langfuse Tracing ---
            try:
                if settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
                    from langfuse.langchain import CallbackHandler
                    langfuse_handler = CallbackHandler()
                    config["callbacks"] = [langfuse_handler]
                    config["metadata"] = {
                        "langfuse_session_id": str(run_id),
                        "langfuse_user_id": str(input_payload.get("user_id", "system")),
                        "langfuse_tags": [
                            f"workflow:{workflow_id}",
                            f"mode:{execution_mode}",
                        ],
                    }
            except Exception as _lf_err:
                logger.debug(f"Langfuse handler skipped: {_lf_err}")

            final_state = compiled_graph.invoke(initial_state, config)

            result = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_mode": execution_mode,
                "output": final_state,
                "messages": final_state.get("messages", [])
            }

        except Exception as e:
            import traceback
            logger.error(f"[EXECUTOR] Workflow execution FAILED for {workflow_id}: {e}")
            logger.error(f"[EXECUTOR] Traceback:\n{traceback.format_exc()}")
            result = {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "execution_mode": execution_mode,
                "error": str(e),
                "output": {}
            }

        return result

    def load_for_execution(
        self,
        workflow_id: str,
        mode: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load workflow for execution.

        Args:
            workflow_id: Workflow identifier
            mode: "draft", "test", or "final"
            version: Version (for final mode)

        Returns:
            Workflow definition
        """
        if mode == "draft":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="draft"
            )
        elif mode == "test":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="temp"
            )
        elif mode == "final":
            return self.storage.load_workflow(
                workflow_id=workflow_id,
                state="final",
                version=version
            )
        else:
            raise ValueError("Invalid execution mode")

    # ========================================================================
    # HITL-AWARE EXECUTION (Additive — does NOT modify existing methods)
    # ========================================================================

    def _load_workflow(
        self,
        workflow_id: str,
        mode: str,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load workflow definition by mode.

        Extracted helper that mirrors the loading logic in execute_workflow
        without modifying the original method.

        Args:
            workflow_id: Workflow identifier.
            mode: "draft", "test", or "final".
            version: Version string (for final mode).

        Returns:
            Workflow definition dict.

        Raises:
            ValueError: If mode is invalid or no versions found.
            FileNotFoundError: If workflow not found.
        """
        if mode == "draft":
            try:
                return self.storage.load_workflow(
                    workflow_id=workflow_id, state="draft"
                )
            except FileNotFoundError:
                return self.storage.load_workflow(
                    workflow_id=workflow_id, state="temp"
                )
        elif mode == "test":
            return self.storage.load_workflow(
                workflow_id=workflow_id, state="temp"
            )
        elif mode == "final":
            if not version:
                versions = self.storage.list_versions(workflow_id)
                if not versions:
                    raise ValueError(
                        f"No final versions found for workflow '{workflow_id}'"
                    )
                version = versions[-1]
            return self.storage.load_workflow(
                workflow_id=workflow_id, state="final", version=version
            )
        raise ValueError(f"Invalid execution mode: {mode}")

    def _load_agent_defs(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load agent definitions from workflow.

        Extracted helper that mirrors the agent loading logic in execute_workflow
        without modifying the original method.

        Args:
            workflow: Workflow definition dict.

        Returns:
            Dict mapping agent_id to agent definition.
        """
        agent_defs = {}
        for agent_entry in workflow.get("agents", []):
            if isinstance(agent_entry, dict):
                agent_id = agent_entry.get("agent_id")
                if agent_id:
                    agent_defs[agent_id] = agent_entry
            elif isinstance(agent_entry, str):
                agent_id = agent_entry
                agent = self.agent_registry.get_agent(agent_id)
                if agent is None:
                    self.agent_registry._load_all()
                    agent = self.agent_registry.get_agent(agent_id)
                if agent:
                    agent_defs[agent_id] = agent

        for agent_id in agent_defs:
            normalize_agent_config(agent_defs[agent_id])

        return agent_defs

    def execute_workflow_hitl(
        self,
        workflow_id: str,
        execution_mode: str,
        version: Optional[str] = None,
        input_payload: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow with HITL interrupt support.

        Same as execute_workflow but:
        1. Detects __interrupt__ in invoke() return value
        2. Returns status="interrupted" with interrupt payload
        3. Uses the HITL-aware checkpointer (PostgresSaver for HITL workflows)

        This method is used ONLY for workflows that contain HITL nodes.
        Non-HITL workflows continue using execute_workflow() unchanged.

        Args:
            workflow_id: Workflow identifier.
            execution_mode: "test" or "final".
            version: Version (required for final mode).
            input_payload: Input data for workflow.

        Returns:
            Execution result dict. If workflow pauses at HITL:
            {
                "run_id": "...",
                "status": "interrupted",
                "interrupt": {...},  # HITL context for frontend
                ...
            }
        """
        if input_payload is None:
            input_payload = {}

        # Load workflow
        workflow = self._load_workflow(workflow_id, execution_mode, version)

        # Apply guards if available
        if self.guards:
            self.guards.check_before_execution(workflow)

        logger.info(
            f"[EXECUTOR-HITL] Executing workflow '{workflow_id}' "
            f"(mode={execution_mode})"
        )

        # Extract workflow variables
        workflow_variables = self._extract_workflow_variables(workflow)

        # Load agent definitions
        agent_defs = self._load_agent_defs(workflow)

        # Compile workflow to LangGraph
        compiled_graph = self.compiler.compile_to_langgraph(workflow, agent_defs)

        # Prepare run_id
        run_id = input_payload.get("run_id") or new_id("run_")

        try:
            # Normalize user input (same logic as execute_workflow)
            import json as _json

            user_input = (
                input_payload.get("user_input")
                or input_payload.get("message")
                or input_payload.get("question")
                or input_payload.get("input")
                or input_payload.get("task_description")
                or input_payload.get("prompt")
            )

            if not user_input and input_payload:
                meaningful_keys = {
                    k for k in input_payload.keys()
                    if k not in ("workflow_id", "run_id", "mode", "version", "context")
                }
                if meaningful_keys:
                    user_input = _json.dumps(input_payload, indent=2)

            if not user_input:
                user_input = ""

            # Merge variables
            merged_variables = {**workflow_variables}
            for key, value in input_payload.items():
                if key in merged_variables or key not in [
                    "workflow_id", "run_id", "mode", "version"
                ]:
                    merged_variables[key] = value

            if user_input and workflow_variables:
                state_var_names = list(workflow_variables.keys())
                if state_var_names:
                    first_var = state_var_names[0]
                    if not merged_variables.get(first_var):
                        merged_variables[first_var] = user_input

            user_input = self._substitute_variables(user_input, merged_variables)

            initial_state = {
                **merged_variables,
                **input_payload,
                "user_input": user_input,
                "original_user_input": user_input,
                "task_description": user_input,
                "messages": [],
                "workflow_id": workflow_id,
                "run_id": run_id,
                "_workflow_variables": workflow_variables,
                "_state_schema": workflow.get("state_schema", {}),
            }

            # Audit log user input (prompt injection detection)
            try:
                from echolib.observability import audit_log_input
                audit_log_input(
                    run_id=run_id,
                    user_input=user_input,
                    user_id=str(input_payload.get("user_id", "system")),
                    workflow_id=workflow_id,
                )
            except Exception as _audit_err:
                logger.debug(f"Audit log skipped: {_audit_err}")

            # Run the compiled graph
            config = {"configurable": {"thread_id": run_id}}

            # --- Langfuse Tracing ---
            try:
                if settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
                    from langfuse.langchain import CallbackHandler
                    langfuse_handler = CallbackHandler()
                    config["callbacks"] = [langfuse_handler]
                    config["metadata"] = {
                        "langfuse_session_id": str(run_id),
                        "langfuse_user_id": str(input_payload.get("user_id", "system")),
                        "langfuse_tags": [
                            f"workflow:{workflow_id}",
                            f"mode:{execution_mode}",
                        ],
                    }
            except Exception as _lf_err:
                logger.debug(f"Langfuse handler skipped: {_lf_err}")

            result = compiled_graph.invoke(initial_state, config)

            # CHECK: Did the graph pause at an interrupt?
            if "__interrupt__" in result:
                interrupt_data = result["__interrupt__"]
                # Extract the interrupt payload (HITL context)
                # interrupt_data is a list/tuple of Interrupt objects
                interrupt_payload = None
                if interrupt_data and len(interrupt_data) > 0:
                    first_interrupt = interrupt_data[0]
                    # Interrupt objects have a .value attribute
                    if hasattr(first_interrupt, 'value'):
                        interrupt_payload = first_interrupt.value
                    else:
                        interrupt_payload = first_interrupt

                logger.info(
                    f"[EXECUTOR-HITL] Workflow '{workflow_id}' interrupted "
                    f"at HITL node (run_id={run_id})"
                )

                return {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "status": "interrupted",
                    "execution_mode": execution_mode,
                    "interrupt": interrupt_payload,
                    "output": result,
                    "messages": result.get("messages", []),
                }

            # Normal completion (no interrupt)
            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_mode": execution_mode,
                "output": result,
                "messages": result.get("messages", []),
            }

        except Exception as e:
            import traceback
            logger.error(
                f"[EXECUTOR-HITL] Workflow execution FAILED for "
                f"{workflow_id}: {e}"
            )
            logger.error(f"[EXECUTOR-HITL] Traceback:\n{traceback.format_exc()}")
            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "execution_mode": execution_mode,
                "error": str(e),
                "output": {},
            }

    def resume_workflow(
        self,
        workflow_id: str,
        run_id: str,
        action: str,
        payload: Dict[str, Any] = None,
        execution_mode: str = "draft",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resume a workflow that was paused at a HITL interrupt.

        Uses LangGraph's Command(resume=...) to continue execution
        from the exact point where interrupt() was called.

        The resume value becomes the return value of interrupt() inside
        the HITL node function, so the node can read the human's decision.

        Args:
            workflow_id: Workflow identifier.
            run_id: The original run_id (used as thread_id for checkpoint lookup).
            action: Human decision ("approve", "reject", "edit", "defer").
            payload: Additional data (edit content, rationale, etc.).
            execution_mode: "draft", "test", or "final".
            version: Version (for final mode).

        Returns:
            Execution result dict (may be another interrupt or final completion).
        """
        from langgraph.types import Command

        logger.info(
            f"[EXECUTOR-HITL] Resuming workflow '{workflow_id}' "
            f"(run_id={run_id}, action={action})"
        )

        # Re-compile the graph (needed to get the same graph structure).
        # The checkpointer will restore state from the DB using thread_id=run_id.
        workflow = self._load_workflow(workflow_id, execution_mode, version)
        agent_defs = self._load_agent_defs(workflow)
        compiled_graph = self.compiler.compile_to_langgraph(workflow, agent_defs)

        # Build resume command
        resume_value = {
            "action": action,
            "payload": payload or {},
        }

        config = {"configurable": {"thread_id": run_id}}

        # --- Langfuse Tracing ---
        try:
            if settings.LANGFUSE_TRACING_ENABLED and settings.LANGFUSE_PUBLIC_KEY:
                from langfuse.langchain import CallbackHandler
                langfuse_handler = CallbackHandler()
                config["callbacks"] = [langfuse_handler]
                config["metadata"] = {
                    "langfuse_session_id": str(run_id),
                    "langfuse_user_id": "system",
                    "langfuse_tags": [
                        f"workflow:{workflow_id}",
                        f"mode:{execution_mode}",
                        "resume",
                    ],
                }
        except Exception as _lf_err:
            logger.debug(f"Langfuse handler skipped on resume: {_lf_err}")

        try:
            # Resume from interrupt -- Command(resume=...) becomes the return
            # value of the interrupt() call inside the HITL node
            result = compiled_graph.invoke(Command(resume=resume_value), config)

            # Check if we hit ANOTHER interrupt (chained HITL nodes)
            if "__interrupt__" in result:
                interrupt_data = result["__interrupt__"]
                interrupt_payload = None
                if interrupt_data and len(interrupt_data) > 0:
                    first_interrupt = interrupt_data[0]
                    if hasattr(first_interrupt, 'value'):
                        interrupt_payload = first_interrupt.value
                    else:
                        interrupt_payload = first_interrupt

                logger.info(
                    f"[EXECUTOR-HITL] Resumed workflow hit another interrupt "
                    f"(run_id={run_id})"
                )

                return {
                    "run_id": run_id,
                    "workflow_id": workflow_id,
                    "status": "interrupted",
                    "interrupt": interrupt_payload,
                    "output": result,
                    "messages": result.get("messages", []),
                }

            logger.info(
                f"[EXECUTOR-HITL] Workflow resumed and completed "
                f"(run_id={run_id})"
            )

            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "completed",
                "output": result,
                "messages": result.get("messages", []),
            }

        except Exception as e:
            logger.error(
                f"[EXECUTOR-HITL] Resume failed for run '{run_id}': {e}",
                exc_info=True,
            )
            return {
                "run_id": run_id,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "output": {},
            }
