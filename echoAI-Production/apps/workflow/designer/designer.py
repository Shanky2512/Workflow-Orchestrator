"""
LLM-based workflow designer.
Analyzes user prompts and generates workflow + agent definitions using real LLM.

LLM Provider Configuration:
---------------------------
This module supports multiple LLM providers. Configure via .env file:
- OPTION 1: Ollama (On-Premise) - Set USE_OLLAMA=true
- OPTION 2: OpenRouter (Current) - Set USE_OPENROUTER=true
- OPTION 3: Azure OpenAI - Set USE_AZURE=true
- OPTION 4: OpenAI Direct - Set USE_OPENAI=true

See .env file for detailed configuration options.
"""
import json
import os
from typing import Dict, Any, Tuple, List, Optional, Set
from echolib.utils import new_id
from datetime import datetime


# Cache for loaded tools from registry
_TOOLS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None
_TOOLS_CACHE_TIME: Optional[float] = None
_TOOLS_CACHE_TTL: float = 300.0  # 5 minutes cache TTL


def _get_tools_storage_path() -> str:
    """Get the path to the tools storage directory."""
    # Path relative to this file: designer.py -> designer/ -> workflow/ -> apps/ -> storage/tools/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "storage", "tools")


def _load_available_tools(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load available tools from tool_index.json and individual tool files.

    Returns a dict with tool_id -> {name, description, tags, keywords} for each active tool.
    Results are cached to avoid repeated file I/O.

    Args:
        force_reload: If True, bypass cache and reload from disk

    Returns:
        Dict mapping tool_id to tool metadata including name, description, and tags
    """
    global _TOOLS_CACHE, _TOOLS_CACHE_TIME

    import time
    current_time = time.time()

    # Return cached data if valid
    if not force_reload and _TOOLS_CACHE is not None and _TOOLS_CACHE_TIME is not None:
        if current_time - _TOOLS_CACHE_TIME < _TOOLS_CACHE_TTL:
            return _TOOLS_CACHE

    tools_path = _get_tools_storage_path()
    tool_index_path = os.path.join(tools_path, "tool_index.json")

    tools_data: Dict[str, Dict[str, Any]] = {}

    try:
        # Load tool index
        if os.path.exists(tool_index_path):
            with open(tool_index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            tools_dict = index_data.get("tools", {})

            for tool_id, tool_info in tools_dict.items():
                # Only include active tools
                if tool_info.get("status") != "active":
                    continue

                # Start with basic info from index
                tool_entry = {
                    "name": tool_info.get("name", tool_id),
                    "tags": tool_info.get("tags", []),
                    "description": "",  # Will be loaded from individual file
                    "keywords": []  # Derived from tags for matching
                }

                # Try to load full description from individual tool file
                tool_file_path = os.path.join(tools_path, f"{tool_id}.json")
                if os.path.exists(tool_file_path):
                    try:
                        with open(tool_file_path, 'r', encoding='utf-8') as f:
                            tool_detail = json.load(f)
                        tool_entry["description"] = tool_detail.get("description", "")
                        # Merge tags from individual file if present
                        if "tags" in tool_detail:
                            tool_entry["tags"] = list(set(tool_entry["tags"] + tool_detail.get("tags", [])))
                    except (json.JSONDecodeError, IOError):
                        pass

                # Build keywords from tags for matching
                tool_entry["keywords"] = [tag.lower().replace("_", " ") for tag in tool_entry["tags"]]

                tools_data[tool_id] = tool_entry

        # Update cache
        _TOOLS_CACHE = tools_data
        _TOOLS_CACHE_TIME = current_time

    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load tools from registry: {e}")
        # Return cached data if available, otherwise empty dict
        if _TOOLS_CACHE is not None:
            return _TOOLS_CACHE

    return tools_data


def _get_tool_selection_rules() -> Dict[str, List[str]]:
    """
    Get tool selection rules by building keyword lists from tool tags.

    This provides backward compatibility with TOOL_SELECTION_RULES while
    loading data dynamically from the registry.

    Returns:
        Dict mapping tool_id to list of keywords for matching
    """
    tools = _load_available_tools()

    # Build keyword rules from tags
    # Expand tags into related keywords for better matching
    rules: Dict[str, List[str]] = {}

    # Keyword expansion mappings for common tags
    tag_expansions = {
        "math": ["calculate", "calculation", "calculator", "math", "mathematics", "compute", "computation"],
        "calculation": ["calculate", "calculation", "compute", "formula"],
        "arithmetic": ["add", "subtract", "multiply", "divide", "sum", "difference", "product"],
        "statistics": ["average", "mean", "median", "variance", "stddev", "statistics", "statistical"],
        "linear_algebra": ["matrix", "vector", "dot product", "linear algebra"],
        "code": ["code", "coding", "program", "programming", "script", "scripting", "develop", "developer"],
        "generation": ["generate", "generating", "create", "creating", "build", "building"],
        "crewai": ["crewai", "crew", "agent framework"],
        "langgraph": ["langgraph", "langchain", "graph", "workflow"],
        "workflow": ["workflow", "orchestration", "pipeline", "automation"],
        "agent": ["agent", "agents", "ai agent", "autonomous"],
        "automation": ["automation", "automate", "automated", "execute", "execution"],
        "review": ["review", "reviewer", "reviewing", "code review", "peer review", "check", "checking"],
        "analysis": ["analyze", "analysis", "inspect", "inspection", "examine", "audit"],
        "security": ["security", "secure", "vulnerability", "exploit", "injection"],
        "quality": ["quality", "clean code", "best practices", "standards"],
        "best_practices": ["best practices", "standards", "convention", "lint", "linting"],
        "refactoring": ["refactor", "refactoring", "improve", "improvement", "optimize", "optimization"],
        "file": ["file", "files", "document", "documents", "doc", "docs"],
        "reader": ["read", "reading", "parse", "parsing", "extract", "extraction", "load", "loading"],
        "pdf": ["pdf", "pdfs", "document"],
        "json": ["json", "data", "config", "configuration"],
        "xml": ["xml", "markup", "data"],
        "csv": ["csv", "spreadsheet", "table", "data", "excel"],
        "parser": ["parse", "parsing", "parser", "extract", "extraction"],
        "document": ["document", "documents", "report", "invoice", "contract", "content"],
        "rag": ["rag", "retrieval", "retrieval augmented", "embeddings"],
        "embeddings": ["embedding", "embeddings", "vector", "semantic"],
        "search": ["search", "searching", "find", "lookup", "discover", "query"],
        "web": ["web", "internet", "online", "browse", "browsing"],
        "internet": ["internet", "online", "web", "network"],
        "research": ["research", "investigate", "explore", "analyze", "information"],
        "bing": ["bing", "search engine", "web search"]
    }

    for tool_id, tool_info in tools.items():
        keywords = set()

        # Add tool name words
        name_words = tool_info["name"].lower().split()
        keywords.update(name_words)

        # Add description words (key terms only)
        if tool_info["description"]:
            desc_words = tool_info["description"].lower().split()
            # Add significant words from description
            for word in desc_words:
                word = word.strip(".,!?;:()[]{}\"'")
                if len(word) > 3:  # Skip short words
                    keywords.add(word)

        # Expand tags into keywords
        for tag in tool_info["tags"]:
            tag_lower = tag.lower().replace("_", " ")
            keywords.add(tag_lower)

            # Add expanded keywords for known tags
            if tag.lower() in tag_expansions:
                keywords.update(tag_expansions[tag.lower()])

        rules[tool_id] = list(keywords)

    return rules


def _format_tools_for_llm_prompt() -> str:
    """
    Format available tools with descriptions for LLM prompts.

    Returns a formatted string listing all tools with their IDs, names, and descriptions.
    This format is designed to be clear to the LLM about which exact tool IDs to use.
    """
    tools = _load_available_tools()

    if not tools:
        return "(No tools available)"

    lines = ["## AVAILABLE TOOLS (You MUST ONLY use these exact tool IDs - NEVER invent new ones)"]
    lines.append("")

    for tool_id, tool_info in sorted(tools.items()):
        name = tool_info.get("name", tool_id)
        description = tool_info.get("description", "No description available")
        # Truncate long descriptions
        if len(description) > 150:
            description = description[:147] + "..."
        lines.append(f"- {tool_id}: {name} - {description}")

    lines.append("")
    lines.append("CRITICAL: You MUST select tools ONLY from the above list using exact tool_id values. Do NOT invent tool names like 'tool_code_executor' or 'tool_data_analyzer'. If no tool matches, use an empty tools list.")

    return "\n".join(lines)


def _get_valid_tool_ids() -> Set[str]:
    """
    Get a set of valid tool IDs from the registry.

    Returns:
        Set of valid tool_id strings
    """
    tools = _load_available_tools()
    return set(tools.keys())


# Fallback tool selection rules for when registry is unavailable
# This provides backward compatibility
FALLBACK_TOOL_SELECTION_RULES = {
    "tool_web_search": [
        "research", "analyze", "analysis", "search", "web", "explore", "investigate",
        "find", "lookup", "browse", "internet", "online", "query", "discover",
        "information", "info", "data", "facts", "knowledge", "learn", "learning",
        "news", "trends", "trending", "latest", "current", "today", "recent", "update",
        "financial", "finance", "stock", "market", "trading", "invest", "investment",
        "travel", "trip", "vacation", "booking", "flight", "hotel", "destination"
    ],
    "tool_file_reader": [
        "file", "files", "document", "documents", "doc", "docs",
        "pdf", "pdfs", "word", "docx", "txt", "text",
        "csv", "excel", "xlsx", "spreadsheet", "json", "xml", "yaml",
        "read", "reading", "parse", "parsing", "extract", "extraction",
        "load", "loading", "import", "importing", "open", "opening",
        "content", "contents", "report", "invoice", "contract"
    ],
    "tool_code_generator": [
        "code", "coding", "program", "programming", "script", "scripting",
        "python", "javascript", "typescript", "java", "develop", "developer",
        "build", "building", "create", "creating", "generate", "generating",
        "write", "writing", "implement", "implementation", "execute", "execution",
        "software", "application", "app", "api", "backend", "frontend",
        "function", "method", "class", "algorithm", "module", "library"
    ],
    "tool_code_reviewer": [
        "review", "reviewer", "reviewing", "code review", "peer review",
        "check", "checking", "inspect", "inspection", "examine", "audit",
        "quality", "code quality", "clean code", "best practices", "standards",
        "bug", "bugs", "issue", "issues", "error", "errors", "problem",
        "security", "secure", "vulnerability", "exploit",
        "improve", "improvement", "optimize", "optimization", "refactor"
    ],
    "tool_calculator": [
        "calculate", "calculation", "calculator", "math", "mathematics",
        "compute", "computation", "add", "subtract", "multiply", "divide",
        "average", "mean", "median", "statistics", "percentage", "ratio",
        "financial", "budget", "interest", "loan", "tax", "profit", "margin",
        "convert", "conversion", "unit", "measurement", "currency"
    ]
}


def _get_effective_tool_selection_rules() -> Dict[str, List[str]]:
    """
    Get the effective tool selection rules.

    Tries to load from registry first, falls back to hardcoded rules if unavailable.
    """
    try:
        rules = _get_tool_selection_rules()
        if rules:
            return rules
    except Exception:
        pass

    return FALLBACK_TOOL_SELECTION_RULES


class WorkflowDesigner:
    """
    Workflow designer service.
    Uses LLM to generate workflows from natural language prompts.
    """

    def __init__(self, llm_service=None, api_key: Optional[str] = None, agent_registry=None):
        """
        Initialize designer.

        Args:
            llm_service: LLM service for prompt analysis
            api_key: OpenAI API key (optional, reads from env if not provided)
            agent_registry: Agent registry for saving agents
        """
        self.llm_service = llm_service
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._openai_client = None
        self.agent_registry = agent_registry

    def _get_openai_client(self):
        """
        Get LLM client using centralized LLM Manager.

        All LLM configuration is now in llm_manager.py
        To change provider/model, edit llm_manager.py
        """
        if self._openai_client is None:
            try:
                from llm_manager import LLMManager

                # Get LLM from centralized manager
                # Uses default configuration from llm_manager.py
                # Temperature and max_tokens can be overridden if needed
                self._openai_client = LLMManager.get_llm(
                    temperature=0.3,
                    max_tokens=4000
                )

            except Exception as e:
                raise RuntimeError(f"Failed to get LLM from LLMManager: {e}")

        return self._openai_client

    def _select_tools_for_agent(
        self,
        agent_name: str,
        agent_role: str,
        agent_goal: str,
        agent_description: str,
        user_prompt: str
    ) -> List[str]:
        """
        Auto-select appropriate tools based on agent purpose and keywords.

        Uses keyword matching to determine which tools are relevant for the agent.
        Tools are loaded dynamically from the registry.
        Maximum of 2 tools will be selected to avoid over-tooling.

        Args:
            agent_name: Name of the agent
            agent_role: Role/title of the agent
            agent_goal: Goal of the agent
            agent_description: Description of what the agent does
            user_prompt: The original user prompt for context

        Returns:
            List of tool IDs (max 2), empty list if no clear match
        """
        # Build keyword set from all agent context
        text_to_analyze = " ".join([
            agent_name.lower(),
            agent_role.lower(),
            agent_goal.lower(),
            agent_description.lower(),
            user_prompt.lower()
        ])

        # Extract words for matching
        keywords: Set[str] = set()
        words = [w.strip(".,!?;:()[]{}\"'") for w in text_to_analyze.split()]
        keywords.update(w for w in words if len(w) > 2)

        if not keywords:
            return []

        # Get tool selection rules (dynamically loaded or fallback)
        tool_selection_rules = _get_effective_tool_selection_rules()

        # Score each tool based on keyword matches
        tool_scores: Dict[str, int] = {}

        for tool_id, tool_keywords in tool_selection_rules.items():
            score = 0
            for keyword in tool_keywords:
                # Check if keyword appears in the combined text
                if keyword in text_to_analyze:
                    score += 2  # Higher score for direct match
                elif keyword in keywords:
                    score += 1

            if score > 0:
                tool_scores[tool_id] = score

        if not tool_scores:
            return []

        # Sort by score and take top 2
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool_id for tool_id, score in sorted_tools[:2]]

        return selected_tools

    def design_from_prompt(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Design workflow and agents from user prompt using real LLM.

        Args:
            user_prompt: Natural language description of desired workflow
            default_llm: Default LLM configuration for agents

        Returns:
            Tuple of (workflow_definition, agent_definitions)
        """
        if default_llm is None:
            # DEFAULT LLM for agents - delegates to LLMManager
            # Agents will use defaults from llm_manager.py unless overridden
            default_llm = {
                # Leave empty to use LLMManager defaults
                # Or specify: "provider": "openai", "model": "gpt-4", etc.
                "temperature": 0.3
            }

        # Always try LLM first (OpenRouter is available)
        try:
            return self._design_with_llm(user_prompt, default_llm)
        except Exception as e:
            print(f"LLM design failed, falling back to heuristics: {e}")
            return self._design_with_heuristics(user_prompt, default_llm)

    def _design_with_llm(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Design workflow using real LLM analysis."""

        # Enhanced system prompt with better pattern detection
        system_prompt = """You are an expert workflow architect. Analyze the user's request and design an optimal multi-agent workflow.

## WORKFLOW TYPES AND WHEN TO USE THEM

1. **Sequential**: Linear chain where tasks must happen in specific order
   - Example: "Generate code, then review it, then deploy"
   - Pattern: Output of A feeds into B feeds into C
   - Use when: Dependencies exist between stages

2. **Parallel**: Independent tasks that can run simultaneously
   - Example: "Analyze code for bugs, security issues, and performance problems"
   - Pattern: Multiple agents process same/different inputs concurrently
   - Use when: Tasks are independent and can benefit from concurrency

3. **Hierarchical**: Manager coordinates specialist workers
   - Example: "Project manager assigns tasks to frontend, backend, and DevOps teams"
   - Pattern: One manager delegates to multiple workers
   - Use when: Central coordination and delegation is needed

4. **Hybrid**: Mixed patterns combining parallel and sequential
   - Example: "Three agents analyze different aspects in parallel, then synthesizer combines results, then reviewer validates"
   - Pattern: Parallel stages → merge → sequential stages
   - Use when: Some stages benefit from parallelism, others need sequential processing

## DECISION TREE

Ask yourself these questions in order:
1. Is there ONE agent coordinating/managing others? → **Hierarchical**
2. Are there distinct stages where some run in parallel, then merge into sequential? → **Hybrid**
3. Can ALL tasks run simultaneously with no dependencies? → **Parallel**
4. Must tasks happen in a specific order? → **Sequential**

## RESPONSE FORMAT

Return JSON with this structure:

For Sequential/Parallel/Hierarchical:
{
  "execution_model": "sequential|parallel|hierarchical",
  "workflow_name": "Brief name",
  "reasoning": "1-2 sentences why you chose this model",
  "agents": [
    {
      "name": "Agent name",
      "role": "Clear role",
      "goal": "What this agent aims to achieve",
      "description": "What this agent does",
      "input_schema": ["input_keys"],
      "output_schema": ["output_keys"]
    }
  ]
}

For Hybrid workflows, ALSO include topology:
{
  "execution_model": "hybrid",
  "workflow_name": "Brief name",
  "reasoning": "Why hybrid is needed",
  "agents": [...],
  "topology": {
    "parallel_groups": [
      {
        "agents": [0, 1, 2],  // indices into agents array
        "merge_strategy": "combine"  // "combine", "vote", or "prioritize"
      }
    ],
    "sequential_chains": [
      {
        "agents": [3, 4]  // indices into agents array (after merge)
      }
    ]
  }
}

For Hierarchical workflows, ALSO include hierarchy:
{
  "execution_model": "hierarchical",
  "hierarchy": {
    "master_agent_index": 0,  // index in agents array
    "sub_agent_indices": [1, 2, 3],  // worker indices
    "delegation_strategy": "dynamic"  // "dynamic", "all", or "sequential"
  }
}

## EXAMPLES

User: "Generate Python code for an API endpoint"
→ Sequential (design → implement → test)

User: "Check code for security, performance, and maintainability issues"
→ Parallel (3 independent analyses)

User: "A tech lead coordinates frontend, backend, and DevOps work"
→ Hierarchical (1 manager + 3 specialists)

User: "Extract data from 3 sources in parallel, then transform, then load to database"
→ Hybrid (3 parallel extractors → transformer → loader)

## RULES

1. Design 2-5 agents (more complex tasks may need more)
2. Each agent needs clear role, goal, and I/O schema
3. For sequential: ensure output keys match next agent's input keys
4. For parallel: agents should have similar input but different focus areas
5. For hierarchical: manager's goal should mention coordination/delegation
6. For hybrid: be explicit about which agents run in parallel vs sequential
7. Always include "reasoning" to explain your choice
8. Be practical and concise

Now analyze the user's request:"""

        llm = self._get_openai_client()

        # Combine system and user prompts for ChatOpenAI
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse LLM response
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        llm_output = json.loads(content)

        # Build workflow from LLM design
        return self._build_workflow_from_llm_response(
            llm_output, user_prompt, default_llm
        )

    def _build_workflow_from_llm_response(
        self,
        llm_output: Dict[str, Any],
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Build workflow structure from LLM response."""

        workflow_id = new_id("wf_")
        timestamp = datetime.utcnow().isoformat()

        execution_model = llm_output.get("execution_model", "sequential")
        workflow_name = llm_output.get("workflow_name", "Workflow from prompt")

        # Create agents from LLM design
        agents = []
        for agent_spec in llm_output.get("agents", []):
            agent_id = new_id("agt_")
            agent_name = agent_spec.get("name", "Agent")
            agent_role = agent_spec.get("role", "Processing")
            agent_goal = agent_spec.get("goal") or f"Complete task: {agent_name}"
            agent_description = agent_spec.get("description", "")

            # Auto-select tools based on agent purpose and user prompt
            selected_tools = self._select_tools_for_agent(
                agent_name=agent_name,
                agent_role=agent_role,
                agent_goal=agent_goal,
                agent_description=agent_description,
                user_prompt=user_prompt
            )

            agents.append({
                "agent_id": agent_id,
                "name": agent_name,
                "role": agent_role,
                "goal": agent_goal,
                "description": agent_description,
                "llm": default_llm.copy(),
                "tools": selected_tools,  # Auto-selected tools based on agent purpose
                "input_schema": agent_spec.get("input_schema", ["input"]),
                "output_schema": agent_spec.get("output_schema", ["output"]),
                "constraints": {
                    "max_steps": 10,
                    "timeout_seconds": 60
                },
                "permissions": {
                    "can_call_agents": execution_model == "hierarchical" and len(agents) == 0
                },
                "metadata": {
                    "created_at": timestamp
                }
            })

        # Generate connections (for non-hybrid workflows)
        connections = self._generate_connections(agents, execution_model)

        # Build topology for hybrid workflows
        topology = None
        if execution_model == "hybrid":
            llm_topology = llm_output.get("topology", {})
            parallel_groups_indices = llm_topology.get("parallel_groups", [])
            sequential_chains_indices = llm_topology.get("sequential_chains", [])

            # Convert agent indices to agent IDs
            topology = {
                "parallel_groups": [],
                "sequential_chains": []
            }

            for group in parallel_groups_indices:
                agent_indices = group.get("agents", [])
                topology["parallel_groups"].append({
                    "agents": [agents[i]["agent_id"] for i in agent_indices if i < len(agents)],
                    "merge_strategy": group.get("merge_strategy", "combine")
                })

            for chain in sequential_chains_indices:
                agent_indices = chain.get("agents", [])
                topology["sequential_chains"].append({
                    "agents": [agents[i]["agent_id"] for i in agent_indices if i < len(agents)]
                })

        # Build hierarchy for hierarchical workflows
        hierarchy = None
        if execution_model == "hierarchical":
            llm_hierarchy = llm_output.get("hierarchy", {})
            master_index = llm_hierarchy.get("master_agent_index", 0)
            sub_indices = llm_hierarchy.get("sub_agent_indices", list(range(1, len(agents))))
            delegation_strategy = llm_hierarchy.get("delegation_strategy", "dynamic")

            if len(agents) > 0:
                hierarchy = {
                    "master_agent": agents[master_index]["agent_id"] if master_index < len(agents) else agents[0]["agent_id"],
                    "delegation_order": [agents[i]["agent_id"] for i in sub_indices if i < len(agents)],
                    "delegation_strategy": delegation_strategy
                }

        # Build workflow
        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_name,
            "description": user_prompt[:200],
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": [agent["agent_id"] for agent in agents],
            "connections": connections,
            "hierarchy": hierarchy,
            "topology": topology,  # For hybrid workflows
            "state_schema": {},
            "human_in_loop": {
                "enabled": False,
                "review_points": []
            },
            "metadata": {
                "created_by": "designer_llm",
                "created_at": timestamp,
                "reasoning": llm_output.get("reasoning", ""),  # LLM's reasoning for workflow type
                "tags": ["auto-generated", "llm-designed"]
            }
        }

        # Convert agents list to dict
        agent_dict = {agent["agent_id"]: agent for agent in agents}

        # Save agents to registry if available
        if self.agent_registry:
            for agent in agents:
                try:
                    self.agent_registry.register_agent(agent)
                except Exception as e:
                    print(f"Warning: Failed to register agent {agent.get('agent_id')}: {e}")

        return workflow, agent_dict

    def _design_with_heuristics(
        self,
        user_prompt: str,
        default_llm: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Fallback: Design workflow using simple heuristics."""

        workflow_id = new_id("wf_")
        timestamp = datetime.utcnow().isoformat()

        # Determine execution model
        execution_model = self._infer_execution_model(user_prompt)

        # Generate agents
        agents = self._generate_agents_heuristic(user_prompt, default_llm)

        # Generate connections
        connections = self._generate_connections(agents, execution_model)

        # Build hierarchy if needed
        hierarchy = None
        if execution_model == "hierarchical" and len(agents) > 0:
            hierarchy = {
                "master_agent": agents[0]["agent_id"],
                "delegation_order": [a["agent_id"] for a in agents[1:]]
            }

        # Build workflow
        workflow = {
            "workflow_id": workflow_id,
            "name": "Workflow from prompt",
            "description": user_prompt[:200],
            "status": "draft",
            "version": "0.1",
            "execution_model": execution_model,
            "agents": [agent["agent_id"] for agent in agents],
            "connections": connections,
            "hierarchy": hierarchy,
            "state_schema": {},
            "human_in_loop": {
                "enabled": False,
                "review_points": []
            },
            "metadata": {
                "created_by": "designer_heuristic",
                "created_at": timestamp,
                "tags": ["auto-generated"]
            }
        }

        agent_dict = {agent["agent_id"]: agent for agent in agents}

        # Save agents to registry if available
        if self.agent_registry:
            for agent in agents:
                try:
                    self.agent_registry.register_agent(agent)
                except Exception as e:
                    print(f"Warning: Failed to register agent {agent.get('agent_id')}: {e}")

        return workflow, agent_dict

    def _infer_execution_model(self, prompt: str) -> str:
        """Infer execution model from prompt using keywords."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["coordinate", "orchestrate", "manage", "master"]):
            return "hierarchical"
        elif any(word in prompt_lower for word in ["parallel", "simultaneously", "at once", "concurrent"]):
            return "parallel"
        elif any(word in prompt_lower for word in ["step", "then", "after", "sequence", "pipeline"]):
            return "sequential"
        else:
            return "sequential"

    def _generate_agents_heuristic(
        self,
        prompt: str,
        default_llm: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic agents using heuristics."""
        timestamp = datetime.utcnow().isoformat()

        # Auto-select tools for analyzer agent
        analyzer_tools = self._select_tools_for_agent(
            agent_name="Analyzer",
            agent_role="Data Analysis",
            agent_goal="Analyze input data thoroughly",
            agent_description="Analyzes input data",
            user_prompt=prompt
        )

        # Auto-select tools for synthesizer agent
        synthesizer_tools = self._select_tools_for_agent(
            agent_name="Synthesizer",
            agent_role="Result Synthesis",
            agent_goal="Synthesize analysis into final output",
            agent_description="Synthesizes analysis into final output",
            user_prompt=prompt
        )

        agents = [
            {
                "agent_id": new_id("agt_"),
                "name": "Analyzer",
                "role": "Data Analysis",
                "goal": "Analyze input data thoroughly",
                "description": "Analyzes input data",
                "llm": default_llm.copy(),
                "tools": analyzer_tools,
                "input_schema": ["input_data"],
                "output_schema": ["analysis_result"],
                "constraints": {
                    "max_steps": 5,
                    "timeout_seconds": 30
                },
                "permissions": {
                    "can_call_agents": False
                },
                "metadata": {
                    "created_at": timestamp
                }
            },
            {
                "agent_id": new_id("agt_"),
                "name": "Synthesizer",
                "role": "Result Synthesis",
                "goal": "Synthesize analysis into final output",
                "description": "Synthesizes analysis into final output",
                "llm": default_llm.copy(),
                "tools": synthesizer_tools,
                "input_schema": ["analysis_result"],
                "output_schema": ["final_output"],
                "constraints": {
                    "max_steps": 3,
                    "timeout_seconds": 20
                },
                "permissions": {
                    "can_call_agents": False
                },
                "metadata": {
                    "created_at": timestamp
                }
            }
        ]

        return agents

    def _generate_connections(
        self,
        agents: List[Dict[str, Any]],
        execution_model: str
    ) -> List[Dict[str, str]]:
        """Generate workflow connections based on execution model."""
        if execution_model == "sequential":
            connections = []
            for i in range(len(agents) - 1):
                connections.append({
                    "from": agents[i]["agent_id"],
                    "to": agents[i + 1]["agent_id"]
                })
            return connections

        elif execution_model == "parallel":
            return []

        elif execution_model == "hierarchical":
            connections = []
            if len(agents) > 1:
                master_id = agents[0]["agent_id"]
                for agent in agents[1:]:
                    connections.append({
                        "from": master_id,
                        "to": agent["agent_id"]
                    })
            return connections

        else:
            # Default: sequential
            connections = []
            for i in range(len(agents) - 1):
                connections.append({
                    "from": agents[i]["agent_id"],
                    "to": agents[i + 1]["agent_id"]
                })
            return connections

    def modify_from_prompt(
        self,
        user_prompt: str,
        existing_workflow: Dict[str, Any],
        existing_agents: Dict[str, Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing workflow via natural language using LLM-based intent detection.

        The LLM analyzes the user's natural language request to understand their intent
        without relying on keyword matching. It can detect intents like:
        - ADD: Add a new agent to the workflow
        - REMOVE: Remove an agent from the workflow
        - MODIFY: Modify an existing agent's configuration
        - ADD_TOOL: Add a tool to an existing agent
        - REMOVE_TOOL: Remove a tool from an agent
        - REORDER: Change the order of agents
        - CHANGE_TOPOLOGY: Change the execution model
        - CREATE_NEW: Signal to create a fresh workflow
        - EXECUTE: Signal that this is an execution request, not a modification

        Args:
            user_prompt: Natural language modification request
            existing_workflow: Current workflow definition
            existing_agents: Dict of agent_id -> agent definition
            conversation_history: Optional list of previous messages for context

        Returns:
            Dict containing:
            - intent: The detected intent type
            - confidence: Confidence score (0.0-1.0)
            - reasoning: LLM's explanation of interpretation
            - changes: Specific changes to make
            - updated_workflow: Modified workflow (if applicable)
            - updated_agents: Modified agents dict (if applicable)
        """
        if conversation_history is None:
            conversation_history = []

        # Build context for LLM
        workflow_context = self._build_workflow_context(existing_workflow, existing_agents)
        history_context = self._build_history_context(conversation_history)

        # Get LLM to analyze intent and generate changes
        llm_response = self._analyze_modification_intent(
            user_prompt=user_prompt,
            workflow_context=workflow_context,
            history_context=history_context,
            existing_workflow=existing_workflow,
            existing_agents=existing_agents
        )

        # Apply changes based on intent
        result = self._apply_modification(
            llm_response=llm_response,
            existing_workflow=existing_workflow,
            existing_agents=existing_agents,
            user_prompt=user_prompt
        )

        return result

    def _build_workflow_context(
        self,
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> str:
        """Build a textual description of the current workflow for LLM context."""
        workflow_name = workflow.get("name", "Untitled Workflow")
        execution_model = workflow.get("execution_model", "sequential")
        workflow_agents = workflow.get("agents", [])
        connections = workflow.get("connections", [])

        # Build agent list with details
        agent_descriptions = []
        for i, agent_id in enumerate(workflow_agents):
            if isinstance(agent_id, dict):
                agent_id = agent_id.get("agent_id", "")
            agent = agents.get(agent_id, {})
            agent_name = agent.get("name", f"Agent {i+1}")
            agent_role = agent.get("role", "Unknown")
            agent_goal = agent.get("goal", agent.get("description", ""))
            agent_tools = agent.get("tools", [])
            agent_descriptions.append(
                f"  {i+1}. {agent_name} (ID: {agent_id})\n"
                f"     Role: {agent_role}\n"
                f"     Goal/Description: {agent_goal}\n"
                f"     Tools: {agent_tools if agent_tools else 'None'}"
            )

        # Build connections description
        conn_descriptions = []
        for conn in connections:
            from_id = conn.get("from", "")
            to_id = conn.get("to", "")
            from_name = agents.get(from_id, {}).get("name", from_id)
            to_name = agents.get(to_id, {}).get("name", to_id)
            conn_descriptions.append(f"  {from_name} -> {to_name}")

        context = f"""Workflow Name: {workflow_name}
Execution Model: {execution_model}
Number of Agents: {len(workflow_agents)}

Agents:
{chr(10).join(agent_descriptions) if agent_descriptions else '  (No agents)'}

Connections:
{chr(10).join(conn_descriptions) if conn_descriptions else '  (No explicit connections)'}"""

        return context

    def _build_history_context(self, conversation_history: List[Dict[str, str]]) -> str:
        """Build conversation history context for LLM."""
        if not conversation_history:
            return "(No previous conversation)"

        history_lines = []
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role.upper()}: {content}")

        return "\n".join(history_lines)

    def _analyze_modification_intent(
        self,
        user_prompt: str,
        workflow_context: str,
        history_context: str,
        existing_workflow: Dict[str, Any],
        existing_agents: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to analyze the user's intent and generate modification plan."""

        # Get formatted tools list with descriptions for LLM
        available_tools_formatted = _format_tools_for_llm_prompt()

        system_prompt = f"""You are an intelligent workflow modification assistant.
You understand natural language requests to modify existing workflows.
Your job is to UNDERSTAND the user's MEANING, not look for specific keywords.

## CURRENT WORKFLOW
{workflow_context}

{available_tools_formatted}

## CONVERSATION HISTORY
{history_context}

## USER REQUEST
{user_prompt}

## YOUR TASK

Analyze what the user wants. They can express intent in ANY natural language form.
Do NOT look for specific keywords - understand the MEANING and INTENT.

Examples of how users might express different intents:
- "I think we need someone to check the code" -> ADD (add code reviewer agent with tool_code_reviewer)
- "Add a code executor agent" -> ADD (add agent with tool_code_generator - generates executable code)
- "The executor seems redundant, let's remove it" -> REMOVE
- "Make the analyzer focus on security instead" -> MODIFY
- "Can you add web search capability to the reviewer?" -> ADD_TOOL (use tool_web_search)
- "Remove the calculator from the first agent" -> REMOVE_TOOL
- "Switch the order of the last two agents" -> REORDER
- "Let's run them all at once instead of sequentially" -> CHANGE_TOPOLOGY
- "This isn't working, let's start over" -> CREATE_NEW
- "Run this workflow" or "Execute analysis on this data" -> EXECUTE

## RESPONSE FORMAT

Return a JSON object with this EXACT structure:
{{
  "intent": "add|remove|modify|add_tool|remove_tool|reorder|change_topology|create_new|execute",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why you interpreted the request this way",
  "changes": {{
    // Include ONLY the relevant fields based on intent:

    // For ADD intent:
    "new_agent": {{
      "name": "Agent Name",
      "role": "Agent Role",
      "goal": "What this agent should accomplish",
      "description": "Detailed description",
      "position": "after:agent_name|before:agent_name|start|end",
      "tools": ["tool_ids_from_available_tools_list_only"]
    }},

    // For REMOVE intent:
    "agent_to_remove": "agent_id_or_name",

    // For MODIFY intent:
    "agent_to_modify": "agent_id_or_name",
    "updates": {{
      "field_name": "new_value"
    }},

    // For ADD_TOOL intent:
    "agent_name": "target_agent_name_or_id",
    "tool_to_add": "tool_id",

    // For REMOVE_TOOL intent:
    "agent_name": "target_agent_name_or_id",
    "tool_to_remove": "tool_id",

    // For REORDER intent:
    "new_order": ["agent_id_1", "agent_id_2", "agent_id_3"],

    // For CHANGE_TOPOLOGY intent:
    "new_execution_model": "sequential|parallel|hierarchical",

    // For CREATE_NEW intent:
    "reason": "Why starting fresh is recommended",

    // For EXECUTE intent:
    "message": "The input message for workflow execution"
  }}
}}

IMPORTANT RULES:
1. Agent names/IDs in changes MUST match existing agents (check the workflow context)
2. Tool IDs MUST be EXACT matches from the AVAILABLE TOOLS list above (e.g., tool_calculator, tool_code_generator, tool_code_reviewer, tool_file_reader, tool_web_search). NEVER invent tool names like "tool_code_executor" or "tool_data_analyzer" - these DO NOT EXIST.
3. For position in ADD, use existing agent names/IDs
4. Be precise with field names in MODIFY updates
5. If the request is ambiguous, ask for clarification by setting intent to "clarify" with changes.question
6. If it seems like the user wants to run/execute the workflow rather than modify it, use "execute" intent
7. When a user asks for "code execution" or "code executor", use tool_code_generator (it generates executable code). When they ask for "code review" or "code checking", use tool_code_reviewer."""

        llm = self._get_openai_client()

        # Invoke LLM
        response = llm.invoke(system_prompt)

        # Parse LLM response
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            llm_response = json.loads(content)
        except json.JSONDecodeError:
            # Fallback response if LLM doesn't return valid JSON
            llm_response = {
                "intent": "clarify",
                "confidence": 0.3,
                "reasoning": "Could not parse LLM response properly",
                "changes": {
                    "question": "I couldn't understand your request clearly. Could you please rephrase what you'd like to change in the workflow?"
                }
            }

        return llm_response

    def _apply_modification(
        self,
        llm_response: Dict[str, Any],
        existing_workflow: Dict[str, Any],
        existing_agents: Dict[str, Dict[str, Any]],
        user_prompt: str
    ) -> Dict[str, Any]:
        """Apply the modification based on LLM's intent analysis."""

        intent = llm_response.get("intent", "unknown")
        confidence = llm_response.get("confidence", 0.0)
        reasoning = llm_response.get("reasoning", "")
        changes = llm_response.get("changes", {})

        # Deep copy to avoid modifying originals
        import copy
        updated_workflow = copy.deepcopy(existing_workflow)
        updated_agents = copy.deepcopy(existing_agents)

        result = {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "changes": changes,
            "updated_workflow": None,
            "updated_agents": None,
            "success": False,
            "message": ""
        }

        try:
            if intent == "add":
                self._apply_add_agent(changes, updated_workflow, updated_agents, user_prompt)
                result["success"] = True
                result["message"] = f"Added new agent: {changes.get('new_agent', {}).get('name', 'Unknown')}"

            elif intent == "remove":
                agent_removed = self._apply_remove_agent(changes, updated_workflow, updated_agents)
                result["success"] = True
                result["message"] = f"Removed agent: {agent_removed}"

            elif intent == "modify":
                agent_modified = self._apply_modify_agent(changes, updated_workflow, updated_agents)
                result["success"] = True
                result["message"] = f"Modified agent: {agent_modified}"

            elif intent == "add_tool":
                agent_name, tool_id = self._apply_add_tool(changes, updated_agents)
                result["success"] = True
                result["message"] = f"Added tool '{tool_id}' to agent '{agent_name}'"

            elif intent == "remove_tool":
                agent_name, tool_id = self._apply_remove_tool(changes, updated_agents)
                result["success"] = True
                result["message"] = f"Removed tool '{tool_id}' from agent '{agent_name}'"

            elif intent == "reorder":
                self._apply_reorder_agents(changes, updated_workflow, updated_agents)
                result["success"] = True
                result["message"] = "Reordered agents in workflow"

            elif intent == "change_topology":
                new_model = self._apply_change_topology(changes, updated_workflow, updated_agents)
                result["success"] = True
                result["message"] = f"Changed execution model to: {new_model}"

            elif intent == "create_new":
                result["success"] = True
                result["message"] = "Recommendation to create new workflow"
                result["create_new_recommended"] = True
                result["reason"] = changes.get("reason", "User requested to start fresh")
                # Don't update workflow/agents - signal to create new

            elif intent == "execute":
                result["success"] = True
                result["message"] = "This is an execution request, not a modification"
                result["is_execution_request"] = True
                result["execution_message"] = changes.get("message", user_prompt)
                # Don't modify workflow - signal execution intent

            elif intent == "clarify":
                result["success"] = False
                result["message"] = "Clarification needed"
                result["question"] = changes.get("question", "Could you please clarify your request?")

            else:
                result["success"] = False
                result["message"] = f"Unknown intent: {intent}"

        except Exception as e:
            result["success"] = False
            result["message"] = f"Error applying modification: {str(e)}"
            result["error"] = str(e)

        # Only include updated workflow/agents if modification was successful
        if result["success"] and intent not in ["create_new", "execute", "clarify"]:
            result["updated_workflow"] = updated_workflow
            result["updated_agents"] = updated_agents

        return result

    def _find_agent_by_name_or_id(
        self,
        identifier: str,
        agents: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Find agent ID by name or ID."""
        # Direct ID match
        if identifier in agents:
            return identifier

        # Name match (case-insensitive)
        identifier_lower = identifier.lower()
        for agent_id, agent in agents.items():
            if agent.get("name", "").lower() == identifier_lower:
                return agent_id

        return None

    def _apply_add_agent(
        self,
        changes: Dict[str, Any],
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]],
        user_prompt: str
    ) -> None:
        """Apply ADD agent modification."""
        new_agent_spec = changes.get("new_agent", {})

        # Generate new agent
        agent_id = new_id("agt_")
        timestamp = datetime.utcnow().isoformat()

        # Get valid tool IDs from registry (dynamically loaded)
        valid_tool_ids = _get_valid_tool_ids()

        # Auto-select tools based on agent purpose
        specified_tools = new_agent_spec.get("tools", [])

        # Filter to only valid tools (LLM might suggest non-existent tools)
        if specified_tools:
            filtered_tools = [t for t in specified_tools if t in valid_tool_ids]
            if len(filtered_tools) < len(specified_tools):
                # Log warning about invalid tools being filtered
                invalid_tools = [t for t in specified_tools if t not in valid_tool_ids]
                print(f"Warning: Filtered out invalid tool IDs: {invalid_tools}")
            specified_tools = filtered_tools

        # If no valid tools specified, auto-select based on agent purpose
        if not specified_tools:
            specified_tools = self._select_tools_for_agent(
                agent_name=new_agent_spec.get("name", ""),
                agent_role=new_agent_spec.get("role", ""),
                agent_goal=new_agent_spec.get("goal", ""),
                agent_description=new_agent_spec.get("description", ""),
                user_prompt=user_prompt
            )

        new_agent = {
            "agent_id": agent_id,
            "name": new_agent_spec.get("name", "New Agent"),
            "role": new_agent_spec.get("role", "Processing"),
            "goal": new_agent_spec.get("goal", ""),
            "description": new_agent_spec.get("description", ""),
            "llm": {"temperature": 0.3},
            "tools": specified_tools,
            "input_schema": new_agent_spec.get("input_schema", ["input"]),
            "output_schema": new_agent_spec.get("output_schema", ["output"]),
            "constraints": {"max_steps": 10, "timeout_seconds": 60},
            "permissions": {"can_call_agents": False},
            "metadata": {"created_at": timestamp, "created_by": "modify_from_prompt"}
        }

        # Add to agents dict
        agents[agent_id] = new_agent

        # Determine position and update workflow.agents list
        position = new_agent_spec.get("position", "end")
        workflow_agents = workflow.get("agents", [])

        if position == "start":
            workflow_agents.insert(0, agent_id)
        elif position == "end":
            workflow_agents.append(agent_id)
        elif position.startswith("after:"):
            ref_agent = position[6:]
            ref_id = self._find_agent_by_name_or_id(ref_agent, agents)
            if ref_id and ref_id in workflow_agents:
                idx = workflow_agents.index(ref_id)
                workflow_agents.insert(idx + 1, agent_id)
            else:
                workflow_agents.append(agent_id)
        elif position.startswith("before:"):
            ref_agent = position[7:]
            ref_id = self._find_agent_by_name_or_id(ref_agent, agents)
            if ref_id and ref_id in workflow_agents:
                idx = workflow_agents.index(ref_id)
                workflow_agents.insert(idx, agent_id)
            else:
                workflow_agents.insert(0, agent_id)
        else:
            workflow_agents.append(agent_id)

        workflow["agents"] = workflow_agents

        # Regenerate connections based on execution model
        agent_list = [agents[aid] for aid in workflow_agents if aid in agents]
        workflow["connections"] = self._generate_connections(
            agent_list, workflow.get("execution_model", "sequential")
        )

        # Register agent if registry available
        if self.agent_registry:
            try:
                self.agent_registry.register_agent(new_agent)
            except Exception:
                pass

    def _apply_remove_agent(
        self,
        changes: Dict[str, Any],
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> str:
        """Apply REMOVE agent modification."""
        agent_identifier = changes.get("agent_to_remove", "")
        agent_id = self._find_agent_by_name_or_id(agent_identifier, agents)

        if not agent_id:
            raise ValueError(f"Agent not found: {agent_identifier}")

        agent_name = agents[agent_id].get("name", agent_id)

        # Remove from agents dict
        del agents[agent_id]

        # Remove from workflow.agents list
        workflow_agents = workflow.get("agents", [])
        if agent_id in workflow_agents:
            workflow_agents.remove(agent_id)
        workflow["agents"] = workflow_agents

        # Remove from connections
        connections = workflow.get("connections", [])
        workflow["connections"] = [
            conn for conn in connections
            if conn.get("from") != agent_id and conn.get("to") != agent_id
        ]

        # Regenerate connections based on execution model
        agent_list = [agents[aid] for aid in workflow_agents if aid in agents]
        if agent_list:
            workflow["connections"] = self._generate_connections(
                agent_list, workflow.get("execution_model", "sequential")
            )

        return agent_name

    def _apply_modify_agent(
        self,
        changes: Dict[str, Any],
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> str:
        """Apply MODIFY agent modification."""
        agent_identifier = changes.get("agent_to_modify", "")
        updates = changes.get("updates", {})

        agent_id = self._find_agent_by_name_or_id(agent_identifier, agents)
        if not agent_id:
            raise ValueError(f"Agent not found: {agent_identifier}")

        agent = agents[agent_id]
        agent_name = agent.get("name", agent_id)

        # Apply updates
        for field, value in updates.items():
            if field in ["agent_id"]:  # Protected fields
                continue
            agent[field] = value

        # Update metadata
        if "metadata" not in agent:
            agent["metadata"] = {}
        agent["metadata"]["modified_at"] = datetime.utcnow().isoformat()
        agent["metadata"]["modified_by"] = "modify_from_prompt"

        return agent_name

    def _apply_add_tool(
        self,
        changes: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Apply ADD_TOOL modification."""
        agent_identifier = changes.get("agent_name", "")
        tool_id = changes.get("tool_to_add", "")

        agent_id = self._find_agent_by_name_or_id(agent_identifier, agents)
        if not agent_id:
            raise ValueError(f"Agent not found: {agent_identifier}")

        # Validate tool exists (dynamically loaded from registry)
        valid_tool_ids = _get_valid_tool_ids()
        if tool_id not in valid_tool_ids:
            # Provide helpful error message with available tools
            available_tools = sorted(valid_tool_ids)
            raise ValueError(
                f"Unknown tool: '{tool_id}'. Available tools are: {available_tools}"
            )

        agent = agents[agent_id]
        agent_name = agent.get("name", agent_id)

        # Add tool if not already present
        current_tools = agent.get("tools", [])
        if tool_id not in current_tools:
            current_tools.append(tool_id)
            agent["tools"] = current_tools

        return agent_name, tool_id

    def _apply_remove_tool(
        self,
        changes: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, str]:
        """Apply REMOVE_TOOL modification."""
        agent_identifier = changes.get("agent_name", "")
        tool_id = changes.get("tool_to_remove", "")

        agent_id = self._find_agent_by_name_or_id(agent_identifier, agents)
        if not agent_id:
            raise ValueError(f"Agent not found: {agent_identifier}")

        agent = agents[agent_id]
        agent_name = agent.get("name", agent_id)

        # Remove tool if present
        current_tools = agent.get("tools", [])
        if tool_id in current_tools:
            current_tools.remove(tool_id)
            agent["tools"] = current_tools

        return agent_name, tool_id

    def _apply_reorder_agents(
        self,
        changes: Dict[str, Any],
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> None:
        """Apply REORDER agents modification."""
        new_order = changes.get("new_order", [])

        if not new_order:
            raise ValueError("New order not specified")

        # Validate all agents exist
        validated_order = []
        for identifier in new_order:
            agent_id = self._find_agent_by_name_or_id(identifier, agents)
            if agent_id:
                validated_order.append(agent_id)

        if validated_order:
            workflow["agents"] = validated_order

            # Regenerate connections
            agent_list = [agents[aid] for aid in validated_order if aid in agents]
            workflow["connections"] = self._generate_connections(
                agent_list, workflow.get("execution_model", "sequential")
            )

    def _apply_change_topology(
        self,
        changes: Dict[str, Any],
        workflow: Dict[str, Any],
        agents: Dict[str, Dict[str, Any]]
    ) -> str:
        """Apply CHANGE_TOPOLOGY modification."""
        new_model = changes.get("new_execution_model", "sequential")

        valid_models = ["sequential", "parallel", "hierarchical"]
        if new_model not in valid_models:
            raise ValueError(f"Invalid execution model: {new_model}. Must be one of {valid_models}")

        workflow["execution_model"] = new_model

        # Regenerate connections based on new model
        workflow_agents = workflow.get("agents", [])
        agent_list = [agents[aid] for aid in workflow_agents if aid in agents]
        workflow["connections"] = self._generate_connections(agent_list, new_model)

        # Update hierarchy if switching to hierarchical
        if new_model == "hierarchical" and agent_list:
            workflow["hierarchy"] = {
                "master_agent": agent_list[0]["agent_id"],
                "delegation_order": [a["agent_id"] for a in agent_list[1:]],
                "delegation_strategy": "dynamic"
            }
            # Grant master agent permission to call other agents
            agent_list[0]["permissions"] = {"can_call_agents": True}
        else:
            workflow["hierarchy"] = None

        return new_model
