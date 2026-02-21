import json
import os
import re
from pathlib import Path
from typing import Callable, List, Dict, Optional

from echolib.interfaces import ICredentialStore, ILogger
from echolib.types import Agent, AgentTemplate, PromptRequest, ValidationResult, LLMConfig
from echolib.utils import new_id
from echolib.llm_factory import LLMProvider
from echolib.services import LLMService, LangGraphBuilder, TemplateRepository, ToolService


class AgentService:
    def __init__(
        self,
        tpl_repo: TemplateRepository,
        graph_builder: LangGraphBuilder,
        tool_svc: ToolService,
        llm: Optional[LLMProvider] = None,
        llm_svc: Optional[LLMService] = None
    ):
        self.tpl_repo = tpl_repo
        self.graph_builder = graph_builder
        self.tool_svc = tool_svc
        self.llm = llm
        self.llm_svc = llm_svc
        # Default to the agent card template so callers don't need to supply one.
        self.template: AgentTemplate = self.tpl_repo.getAgentTemplate("agentCard")
        self.agents: dict[str, Agent] = {}
    
    def _get_available_tools(self) -> List[Dict]:
        """Return the tool catalog from ToolService in a tolerant way."""
        # Try common method names to fetch tools; adapt if your ToolService differs.
        for attr in ("list", "list_tools", "get_tools", "all"):
            if hasattr(self.tool_svc, attr):
                tools = getattr(self.tool_svc, attr)()
                # Normalize: expect list of dicts with at least an 'id' field.
                return tools or []
        return []

    def _validate_tools(self, tool_ids: List[str]) -> None:
        """Validate user-passed tool IDs against actual service tools."""
        catalog = self._get_available_tools()
        available = {t.get("id") for t in catalog if isinstance(t, dict) and t.get("id")}
        invalid = [t for t in (tool_ids or []) if t not in available]
        if invalid:
            raise ValueError(
                f"Invalid tools passed: {invalid}. Valid tools: {sorted([t for t in available if t])}"
            )
    def _extract_model_from_prompt(self, prompt: str, available_model_ids: List[str]) -> Optional[str]:
        """If user named a model explicitly in the prompt, try to detect it."""
        p = (prompt or "").lower()
        for mid in available_model_ids:
            if not mid:
                continue
            if mid.lower() in p:
                return mid
        return None

    def _ensure_llm_in_spec(self, spec: Dict) -> Dict:
        """
        Ensure spec has an 'llm' block. If missing, pick a default Ollama model.
        """
        if isinstance(spec.get("llm"), dict) and spec["llm"].get("model"):
            return spec  # already provided by user or LLM

        # Ask LLMService for available models
        model_id: Optional[str] = None
        if self.llm_svc:
            models = self.llm_svc.list_models()
            model_list = models if isinstance(models, list) else models.get("data", models.get("models", []))
            if not isinstance(model_list, list):
                model_list = []
            ids: List[str] = [
                str(m["id"]) if isinstance(m, dict) else str(m)
                for m in model_list
                if (isinstance(m, dict) and m.get("id")) or (isinstance(m, str) and m)
            ]
            # Try to detect if user named a specific model in the prompt
            model_id = self._extract_model_from_prompt(spec.get("purpose", "") + " " + spec.get("agent_name", ""), ids)
            # If still none, pick default from service (prefers "gpt-oss")
            if not model_id:
                pick_default = getattr(self.llm_svc, "pick_default_model", None)
                if callable(pick_default):
                    chosen = pick_default(preferred_prefix="gpt-oss")
                    model_id = chosen if isinstance(chosen, str) else model_id

        # Ultimate fallback: try llm_provider.json Ollama model, else default
        if not model_id:
            try:
                provider_file = Path(__file__).resolve().parent.parent.parent.parent / "llm_provider.json"
                if provider_file.exists():
                    with open(provider_file, "r") as f:
                        data = json.load(f)
                    ollama = next((m for m in data.get("models", []) if m.get("provider") == "ollama"), None)
                    if ollama:
                        model_id = ollama.get("model_name", "mistral-nemo:12b-instruct-2407-fp16")
            except Exception:
                pass
        model_id = model_id or "mistral-nemo:12b-instruct-2407-fp16"

        # Fill in a minimal llm config for Ollama
        spec["llm"] = {
            "provider": "ollama",
            "model": model_id,
            "temperature": 0.2,
            "max_tokens": 2048
        }
        return spec

    def _call_llm_for_agent_spec(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Call Ollama (or configured LLM) to generate agent spec from prompts.
        Uses LangChain ChatOpenAI with Ollama base_url from llm_provider.json or env.
        Returns the raw text response, or None on failure.
        """
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

            # Load Ollama config from llm_provider.json
            provider_file = Path(__file__).resolve().parent.parent.parent.parent / "llm_provider.json"
            base_url = "http://10.188.100.131:8004/v1"
            model_name = "mistral-nemo:12b-instruct-2407-fp16"

            if provider_file.exists():
                with open(provider_file, "r") as f:
                    data = json.load(f)
                models = data.get("models", [])
                ollama_model = next((m for m in models if m.get("provider") == "ollama"), None)
                if ollama_model:
                    base_url = ollama_model.get("base_url", base_url)
                    model_name = ollama_model.get("model_name", model_name)
                    if model_id and model_id in [m.get("id") for m in models]:
                        cfg = next((m for m in models if m.get("id") == model_id), ollama_model)
                        model_name = cfg.get("model_name", model_name)
                        if cfg.get("provider") == "ollama":
                            base_url = cfg.get("base_url", base_url)

            # Env overrides
            base_url = os.getenv("OLLAMA_BASE_URL", base_url)
            if not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"

            llm = ChatOpenAI(
                base_url=base_url,
                api_key="ollama",
                model=model_name,
                temperature=0.2,
                max_tokens=2048,
            )

            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.invoke(full_prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip() if text else None
        except Exception:
            return None

    # ---------- Prompt → Agent-Card inference ----------
    def _infer_agent_card_from_prompt(self, prompt: str) -> Dict:
        """
        Use an LLM (Ollama) to infer an agent-card-like JSON from the user prompt.
        Fields: agent_name, purpose, goals, tools, input_schema, output_schema, llm (optional).
        Fallback to a lightweight heuristic if LLM call fails.
        """
        prompt = (prompt or "").strip()
        catalog = self._get_available_tools()
        tool_ids = [t.get("id") for t in catalog if isinstance(t, dict) and t.get("id")]

        # --- Preferred path: call Ollama via _call_llm_for_agent_spec ---
        sys_msg = (
            "You are an assistant that converts a natural-language request into a JSON agent spec.\n"
            "Return ONLY a valid JSON object (no markdown, no code blocks) with these keys:\n"
            "- agent_name: string\n"
            "- purpose: string (what the agent does)\n"
            "- goals: array of strings\n"
            "- tools: array of tool ids (use empty array [] if none needed)\n"
            "- input_schema: array with one JSON Schema object, e.g. [{\"type\":\"object\",\"properties\":{\"input\":{\"type\":\"string\"}},\"required\":[\"input\"]}]\n"
            "- output_schema: array with one JSON Schema object, e.g. [{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"string\"}}}]\n"
            "- reasoning_working_style: string (optional)\n"
            "- error_handling_patterns: string (optional)\n"
            "- example_workflows: array (optional)\n"
            "- llm: object with provider, model, temperature, max_tokens (optional)\n"
            "If no tool is needed, return empty array for tools. Prefer concise, production-ready schemas."
        )
        if tool_ids:
            sys_msg += f"\nAvailable tool ids: {', '.join(str(t) for t in tool_ids if t)}. Use only these if tools are needed."
        usr_msg = f"User prompt:\n{prompt}\n\nProduce the JSON object only."

        text = self._call_llm_for_agent_spec(sys_msg, usr_msg)
        if text:
            try:
                # If the LLM wrapped JSON in a code block, strip it
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
                json_text = m.group(1).strip() if m else text.strip()
                spec = json.loads(json_text)
                if isinstance(spec, dict):
                    return spec
            except (json.JSONDecodeError, ValueError):
                pass

        # --- Fallback: simple heuristic (no external calls) ---
        lower = prompt.lower()
        guessed_name = (prompt[:40] + "...") if len(prompt) > 40 else (prompt or "New Agent")
        guessed_name = guessed_name.replace("\n", " ").strip() or "New Agent"

        desired = []
        if any(k in lower for k in ("search", "web", "crawl", "browse")):
            desired.append("web_search")
        if any(k in lower for k in ("sentiment", "tone", "emotion")):
            desired.append("sentiment_analysis")
        if any(k in lower for k in ("sql", "database", "query db", "postgres", "mysql")):
            desired.append("database_query")
        if any(k in lower for k in ("summaris", "summariz", "digest")):
            desired.append("document_summarize")

        tools = [t for t in desired if t in (tool_ids or [])]

        return {
            "agent_name": guessed_name,
            "purpose": prompt or "Auto-generated agent",
            "goals": [],
            "tools": tools,  # validated again later
            "input_assumptions": "",
            "output_definitions": "",
            "reasoning_working_style": "",
            "error_handling_patterns": "",
            "example_workflows": [],
            # input/output schemas: provide generic pass-through when we cannot infer
            "input_schema": [{
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"]
            }],
            "output_schema": [{
                "type": "object",
                "properties": {"answer": {"type": "string"}}
            }],
            # no llm by default in heuristic; your template or later logic may supply it
        }


    def createFromPrompt(self, prompt: str, template=None) -> Agent:
        """
        1) Infer an agent JSON from the user prompt (LLM preferred, heuristic fallback)
        2) Validate tools against ToolService
        3) Delegate creation to createFromCanvasCard (which merges with template)
        """
        template = template or self.template
        if template is None:
            raise ValueError("Agent template is not configured")

        # 1) Infer spec from prompt (LLM or heuristic)
        spec = self._infer_agent_card_from_prompt(prompt)

        # 2) Ensure spec has llm config (Ollama default if missing)
        spec = self._ensure_llm_in_spec(spec)

        # 3) Tools: prefer explicit "tools" array if present; validate them
        tools = []
        if isinstance(spec.get("tools"), list):
            tools = [t for t in spec["tools"] if isinstance(t, str)]
            if tools:
                self._validate_tools(tools)

        # 4) Assemble the card JSON for createFromCanvasCard (includes schemas + llm)
        cardJSON: Dict = {
            "agent_name": spec.get("agent_name") or "unnamed-agent",
            "purpose": spec.get("purpose") or "",
            "goals": spec.get("goals") or [],
            "input_assumptions": spec.get("input_assumptions") or "",
            "output_definitions": spec.get("output_definitions") or "",
            "tools": tools,  # explicit tools preferred by your service
            "tools_apis_used": spec.get("tools_apis_used") or tools,  # optional echo
            "reasoning_working_style": spec.get("reasoning_working_style") or "",
            "error_handling_patterns": spec.get("error_handling_patterns") or "",
            "example_workflows": spec.get("example_workflows") or [],
            "capabilities": spec.get("capabilities") or [],  # optional; your service also derives these
            "input_schema": spec.get("input_schema") or [],
            "output_schema": spec.get("output_schema") or [],
            "llm": spec.get("llm"),  # always set after _ensure_llm_in_spec
        }

        # 5) Create the agent via createFromCanvasCard
        return self.createFromCanvasCard(cardJSON, template)

    def createFromCanvasCard(self, cardJSON: dict, template=None) -> Agent:
        """
        Create an LLM-capable Agent from a canvas card JSON and an AgentTemplate.

        This mirrors the implementation in `backend/echolib/services.py::AgentService`.
        """
        template = template or self.template
        if template is None:
            raise ValueError("Agent template is not configured")

        def _deep_update(target: Dict, src: Dict) -> Dict:
            """Recursively merge src into target (src wins)."""
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(target.get(k), dict):
                    _deep_update(target[k], v)
                else:
                    target[k] = v
            return target

        # 1) Load base template JSON and merge card JSON over it
        base_tpl = template.to_json()
        merged: Dict = _deep_update(base_tpl.copy(), cardJSON or {})

        # 2) Derive core Agent fields
        name = (
            merged.get("name")
            or merged.get("agent_name")
            or cardJSON.get("name")
            or "unnamed-agent"
        )
        description = merged.get("description") or merged.get("purpose")

        # 3) Derive tools
        # 3a) Primary: explicit tools list on the card, e.g. "tools": ["toolA", "toolB"]
        tools: List[str] = []
        explicit_tools = cardJSON.get("tools")
        if isinstance(explicit_tools, list):
            tools = [t for t in explicit_tools if isinstance(t, str)]

        # 3b) Fallback: derive from capabilities or template tools_apis_used
        capabilities = merged.get("capabilities") or []
        if not tools and isinstance(capabilities, list):
            for cap in capabilities:
                if isinstance(cap, dict):
                    tool_id = cap.get("id")
                    if tool_id:
                        tools.append(tool_id)
        if not tools and isinstance(merged.get("tools_apis_used"), list):
            tools = list(merged["tools_apis_used"])

        # 4) Derive input/output schema lists from capabilities or cardJSON
        input_schema: List[dict] = []
        output_schema: List[dict] = []
        for cap in capabilities:
            if isinstance(cap, dict):
                if "input_schema" in cap:
                    input_schema.append(cap["input_schema"])
                if "output_schema" in cap:
                    output_schema.append(cap["output_schema"])
        if not input_schema and isinstance(merged.get("input_schema"), list):
            input_schema = [s for s in merged["input_schema"] if isinstance(s, dict)]
        if not output_schema and isinstance(merged.get("output_schema"), list):
            output_schema = [s for s in merged["output_schema"] if isinstance(s, dict)]

        # 5) LLM configuration from merged JSON (set by createFromPrompt or card)
        llm_cfg = None
        llm_data = merged.get("llm")
        if isinstance(llm_data, dict):
            llm_cfg = LLMConfig(**llm_data)

        # 6) Construct a fully populated, LLM-capable Agent
        a = Agent(
            id=new_id("agt_"),
            name=name,
            description=description,
            llm=llm_cfg,
            tools=tools or None,
            input_schema=input_schema or None,
            output_schema=output_schema or None,
            metadata=merged,
        )

        self.agents[a.id] = a
        return a

    def validateA2A(self, agent: Agent) -> ValidationResult:
        return ValidationResult(ok=True)

    def listAgents(self) -> List[Agent]:
        return list(self.agents.values())