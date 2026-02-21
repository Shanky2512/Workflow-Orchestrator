"""
Agent factory.
Creates runtime agent instances from definitions.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class AgentRuntime:
    """
    Runtime wrapper around an agent definition that exposes a .chat() method.

    Stores the original agent definition dict and makes its keys accessible
    as attributes so existing code that reads agent_id, name, role, etc.
    continues to work unchanged.
    """

    def __init__(
        self,
        agent_def: Dict[str, Any],
        llm_client: Any = None,
        tools: List[Any] = None,
    ):
        self._agent_def = agent_def
        self._llm_client = llm_client
        self._tools = tools or []

        # Expose dict keys as direct attributes
        for key, value in agent_def.items():
            setattr(self, key, value)

        # Ensure common fields exist even if missing from agent_def
        if not hasattr(self, "agent_id"):
            self.agent_id = agent_def.get("agent_id") or agent_def.get("id")
        if not hasattr(self, "runtime_ready"):
            self.runtime_ready = True

    # ---- dict-like access for backward compatibility ----

    def get(self, key: str, default=None):
        return self._agent_def.get(key, default)

    def __getitem__(self, key: str):
        return self._agent_def[key]

    def __contains__(self, key: str):
        return key in self._agent_def

    def keys(self):
        return self._agent_def.keys()

    def values(self):
        return self._agent_def.values()

    def items(self):
        return self._agent_def.items()

    # ---- core chat method ----

    def chat(
        self,
        message: str,
        history: list = None,
        context: dict = None,
        tools: list = None,
        metadata: dict = None,
    ) -> Dict[str, str]:
        """
        Send a message to the agent and return the LLM response.

        Builds a conversation from the agent's system prompt, any prior
        history, and the current user message, then calls the LLM via
        langchain_openai.ChatOpenAI using config from llm_provider.json.

        Returns:
            Dict with keys "content" (str) and "role" ("assistant").
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        # --- Resolve LLM connection details from llm_provider.json ---
        provider_file = Path(__file__).resolve().parent.parent.parent.parent / "llm_provider.json"
        base_url = "http://10.188.100.131:8004/v1"
        model_name = "mistral-nemo:12b-instruct-2407-fp16"
        api_key = "ollama"

        if provider_file.exists():
            with open(provider_file, "r") as f:
                data = json.load(f)
            models = data.get("models", [])
            ollama_model = next(
                (m for m in models if m.get("provider") == "ollama"), None
            )
            if ollama_model:
                base_url = ollama_model.get("base_url", base_url)
                model_name = ollama_model.get("model_name", model_name)
                api_key = ollama_model.get("api_key", api_key)

        # Environment variable override
        base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        # --- Build system prompt from agent definition ---
        system_prompt = (
            self._agent_def.get("prompt")
            or self._agent_def.get("system_prompt")
            or ""
        )
        if not system_prompt:
            role = self._agent_def.get("role") or ""
            description = self._agent_def.get("description") or ""
            if role or description:
                system_prompt = f"You are {role}. {description}".strip()
            else:
                system_prompt = "You are a helpful AI assistant."

        # --- Build LangChain messages ---
        messages = [SystemMessage(content=system_prompt)]

        # Append prior conversation history
        for turn in (history or []):
            role_val = turn.get("role", "user") if isinstance(turn, dict) else "user"
            content_val = turn.get("content", str(turn)) if isinstance(turn, dict) else str(turn)
            if role_val == "assistant":
                messages.append(AIMessage(content=content_val))
            else:
                messages.append(HumanMessage(content=content_val))

        # Append the current user message
        messages.append(HumanMessage(content=message))

        # --- Determine temperature from agent LLM config ---
        llm_config = self._agent_def.get("llm") or {}
        temperature = 0.2
        if isinstance(llm_config, dict):
            temperature = llm_config.get("temperature", temperature)

        # --- Invoke the LLM ---
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
        )

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        return {"content": response_text, "role": "assistant"}


class AgentFactory:
    """
    Agent factory service.
    Instantiates agents with LLM and tool bindings.
    """

    def __init__(self, tool_registry=None):
        """
        Initialize factory.

        Args:
            tool_registry: Tool registry for binding tools to agents
        """
        self.tool_registry = tool_registry or {}
        self._llm_clients = {}

    def create_agent(
        self,
        agent_def: Dict[str, Any],
        bind_tools: bool = True
    ) -> AgentRuntime:
        """
        Create runtime agent instance from definition.

        Args:
            agent_def: Agent definition
            bind_tools: Whether to bind tools to agent

        Returns:
            AgentRuntime instance with .chat() method
        """
        agent_id = agent_def.get("agent_id") or agent_def.get("id")
        llm_config = agent_def.get("llm", {})

        # Create LLM client config
        llm = self._create_llm_client(llm_config)

        # Bind tools if requested
        tools = []
        if bind_tools:
            tool_ids = agent_def.get("tools", [])
            tools = self._bind_tools(tool_ids)

        # Build a merged definition dict that contains everything the
        # original agent_def has plus runtime fields, so AgentRuntime
        # exposes all original keys as attributes.
        runtime_def = dict(agent_def)
        runtime_def["agent_id"] = agent_id
        runtime_def["llm"] = llm
        runtime_def["tools"] = tools
        runtime_def["constraints"] = agent_def.get("constraints", {})
        runtime_def["runtime_ready"] = True

        return AgentRuntime(agent_def=runtime_def, llm_client=llm, tools=tools)

    def create_agents_for_workflow(
        self,
        agent_definitions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create multiple agents for a workflow.

        Args:
            agent_definitions: Dict of agent_id -> agent_def

        Returns:
            Dict of agent_id -> agent_instance
        """
        instances = {}
        for agent_id, agent_def in agent_definitions.items():
            instances[agent_id] = self.create_agent(agent_def)
        return instances

    def _create_llm_client(self, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create LLM client from config.

        Args:
            llm_config: LLM configuration

        Returns:
            LLM client instance (placeholder)

        TODO: Implement actual LLM client creation
        """
        provider = llm_config.get("provider", os.getenv("LLM_PROVIDER", "openrouter"))
        model = llm_config.get(
            "model",
            os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
        )
        temperature = llm_config.get("temperature", 0.2)

        # Cache clients
        cache_key = f"{provider}:{model}"
        if cache_key in self._llm_clients:
            return self._llm_clients[cache_key]

        # Placeholder LLM client
        llm_client = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "ready": True
        }

        self._llm_clients[cache_key] = llm_client
        return llm_client

    def _bind_tools(self, tool_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Bind tools to agent.

        Args:
            tool_ids: List of tool identifiers

        Returns:
            List of bound tool instances

        TODO: Implement actual MCP tool binding
        """
        tools = []
        for tool_id in tool_ids:
            tool_def = self.tool_registry.get(tool_id)
            if tool_def:
                # Support both dict and ToolDef (Pydantic) objects
                name = (
                    tool_def.name if hasattr(tool_def, 'name') else
                    (tool_def.get("name") if isinstance(tool_def, dict) else str(tool_def))
                )
                tools.append({
                    "tool_id": tool_id,
                    "name": name,
                    "bound": True
                })
        return tools

    def validate_agent_config(self, agent_def: Dict[str, Any]) -> bool:
        """
        Validate agent configuration before creation.

        Args:
            agent_def: Agent definition

        Returns:
            True if valid

        TODO: Add comprehensive validation
        """
        # Check required fields
        if not agent_def.get("agent_id") and not agent_def.get("id"):
            return False

        llm_config = agent_def.get("llm")
        if not llm_config or not llm_config.get("provider") or not llm_config.get("model"):
            return False

        return True

    def get_llm_client(self, provider: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM client.

        Args:
            provider: LLM provider
            model: Model name

        Returns:
            LLM client or None
        """
        cache_key = f"{provider}:{model}"
        return self._llm_clients.get(cache_key)
