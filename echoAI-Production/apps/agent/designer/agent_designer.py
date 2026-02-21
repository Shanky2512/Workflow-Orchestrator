"""
Agent designer service.
Generates agent definitions from natural language prompts using LLM.

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
import logging
import os
from typing import Dict, Any, List, Optional
from echolib.utils import new_id
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentDesigner:
    """
    Agent designer service.
    Uses LLM to generate agent definitions from natural language prompts.
    """

    # Available tools for auto-selection (must match actual backend tool IDs)
    AVAILABLE_TOOLS = ["tool_web_search", "tool_file_reader", "tool_code_generator", "tool_code_reviewer", "tool_calculator"]

    # Tool selection rules: keyword patterns -> tool IDs
    # IMPORTANT: Keys must match actual backend tool IDs in apps/storage/tools/
    TOOL_SELECTION_RULES = {
        "tool_web_search": [
            # Research & Information
            "research", "analyze", "analysis", "search", "web", "explore", "investigate",
            "find", "lookup", "browse", "internet", "online", "query", "discover",
            "information", "info", "data", "facts", "knowledge", "learn", "learning",
            # News & Updates
            "news", "trends", "trending", "latest", "current", "today", "recent", "update", "updates",
            # Finance & Business
            "financial", "finance", "stock", "stocks", "market", "markets", "trading", "invest", "investment",
            "report", "reports", "analyst", "advisor", "business", "company", "companies", "startup",
            "crypto", "bitcoin", "cryptocurrency", "forex", "currency", "exchange",
            # Travel & Booking
            "travel", "trip", "trips", "vacation", "holiday", "booking", "book", "reserve",
            "flight", "flights", "airline", "airlines", "airport",
            "hotel", "hotels", "accommodation", "stay", "airbnb", "hostel",
            "reservation", "reservations", "destination", "destinations",
            "tour", "tours", "tourism", "tourist", "sightseeing",
            "itinerary", "ticket", "tickets", "pass", "visa",
            "train", "trains", "railway", "bus", "buses", "car", "cars", "rental", "rentals",
            "transport", "transportation", "commute", "route", "routes", "directions",
            # Shopping & Products
            "shop", "shopping", "buy", "purchase", "price", "prices", "cost", "compare", "comparison",
            "product", "products", "item", "items", "store", "stores", "amazon", "ebay", "deal", "deals",
            "review", "reviews", "rating", "ratings", "recommendation", "recommendations",
            # Food & Dining
            "restaurant", "restaurants", "food", "foods", "menu", "dining", "eat", "eating",
            "recipe", "recipes", "cook", "cooking", "cuisine", "delivery", "takeout",
            # Entertainment
            "movie", "movies", "film", "films", "show", "shows", "tv", "television", "netflix", "stream",
            "music", "song", "songs", "artist", "artists", "album", "concert", "concerts",
            "event", "events", "sports", "game", "games", "match", "score", "scores",
            "video", "videos", "youtube", "watch",
            # Location & Maps
            "location", "locations", "address", "map", "maps", "place", "places", "nearby", "local",
            "weather", "forecast", "temperature", "climate",
            # Health & Medical
            "health", "medical", "doctor", "doctors", "hospital", "hospitals", "clinic",
            "medicine", "medication", "symptom", "symptoms", "treatment", "pharmacy",
            # Education
            "education", "school", "schools", "university", "college", "course", "courses",
            "tutorial", "tutorials", "guide", "guides", "how to", "howto", "learn",
            # Jobs & Career
            "job", "jobs", "career", "careers", "employment", "hire", "hiring", "salary", "salaries",
            "resume", "interview", "work", "remote", "freelance",
            # Real Estate
            "real estate", "property", "properties", "house", "houses", "apartment", "apartments",
            "rent", "renting", "lease", "mortgage", "realtor",
            # Social & Communication
            "social media", "twitter", "facebook", "instagram", "linkedin", "tiktok",
            "people", "person", "contact", "email", "phone",
            # Reference & Knowledge
            "wikipedia", "definition", "meaning", "translate", "translation", "language",
            "book", "books", "author", "article", "articles", "blog", "publication"
        ],
        "tool_file_reader": [
            # Document Types
            "file", "files", "document", "documents", "doc", "docs",
            "pdf", "pdfs", "word", "docx", "txt", "text",
            "markdown", "md", "html", "htm", "rtf",
            # Data Files
            "csv", "excel", "xlsx", "xls", "spreadsheet", "spreadsheets",
            "json", "xml", "yaml", "yml", "ini", "config", "configuration",
            # Actions
            "read", "reading", "parse", "parsing", "extract", "extraction",
            "load", "loading", "import", "importing", "open", "opening",
            "analyze", "scan", "scanning",
            # Content Types
            "content", "contents", "data", "table", "tables", "sheet", "sheets",
            "report", "reports", "invoice", "invoices", "receipt", "receipts",
            "contract", "contracts", "agreement", "agreements",
            "resume", "cv", "letter", "letters", "form", "forms",
            # File Operations
            "attachment", "attachments", "upload", "uploaded", "download",
            "log", "logs", "readme", "license", "changelog",
            # Media (text extraction)
            "image", "images", "photo", "photos", "screenshot", "screenshots",
            "scanned", "ocr", "transcript", "transcription"
        ],
        "tool_code_generator": [
            # Languages
            "code", "coding", "program", "programming", "script", "scripting",
            "python", "javascript", "typescript", "java", "csharp", "c#",
            "cpp", "c++", "golang", "go", "rust", "ruby", "php", "swift", "kotlin",
            "scala", "perl", "bash", "shell", "powershell", "sql",
            "html", "css", "react", "angular", "vue", "node", "nodejs",
            # Actions
            "develop", "developer", "development", "build", "building",
            "create", "creating", "generate", "generating", "write", "writing",
            "implement", "implementation", "execute", "execution", "run", "running",
            "compile", "compiling", "debug", "debugging", "fix", "fixing",
            "refactor", "refactoring", "optimize", "optimization",
            # Concepts
            "software", "application", "app", "apps", "api", "apis",
            "backend", "frontend", "fullstack", "full-stack",
            "function", "functions", "method", "methods", "class", "classes",
            "algorithm", "algorithms", "logic", "module", "modules",
            "library", "libraries", "framework", "frameworks", "sdk",
            "package", "packages", "dependency", "dependencies",
            # Testing
            "test", "tests", "testing", "unittest", "unit test", "pytest", "jest",
            "selenium", "automation", "automate", "automated",
            # Web & API
            "rest", "restful", "graphql", "websocket", "http", "https",
            "endpoint", "endpoints", "route", "routes", "request", "response",
            "authentication", "authorization", "oauth", "jwt", "token",
            # Database
            "database", "databases", "db", "query", "queries",
            "mysql", "postgresql", "postgres", "mongodb", "redis", "sqlite",
            # DevOps & Cloud
            "docker", "kubernetes", "k8s", "container", "containers",
            "aws", "azure", "gcp", "cloud", "serverless", "lambda",
            "deploy", "deployment", "ci", "cd", "pipeline", "jenkins", "github actions",
            "git", "github", "gitlab", "bitbucket", "version control",
            # Data & AI
            "scrape", "scraping", "crawler", "crawling", "bot", "bots",
            "parse", "parser", "parsing", "regex", "regular expression",
            "machine learning", "ml", "ai", "artificial intelligence",
            "data science", "pandas", "numpy", "tensorflow", "pytorch"
        ],
        "tool_code_reviewer": [
            # Core Review
            "review", "reviewer", "reviewing", "code review", "peer review",
            "check", "checking", "inspect", "inspection", "examine", "audit", "auditing",
            # Quality
            "quality", "code quality", "clean code", "best practices", "standards",
            "maintainability", "readability", "documentation", "comments",
            "style", "convention", "conventions", "formatting", "lint", "linting", "linter",
            # Analysis
            "static analysis", "analyze", "analysis", "complexity", "metrics",
            "coverage", "test coverage", "code coverage",
            # Issues
            "bug", "bugs", "issue", "issues", "error", "errors", "problem", "problems",
            "smell", "code smell", "anti-pattern", "antipattern",
            "vulnerability", "vulnerabilities", "flaw", "flaws",
            # Security
            "security", "secure", "insecure", "vulnerability", "exploit",
            "injection", "sql injection", "xss", "csrf", "sanitization", "validation",
            "encryption", "authentication", "authorization",
            # Improvement
            "improve", "improvement", "optimize", "optimization", "refactor", "refactoring",
            "suggestion", "suggestions", "feedback", "recommend", "recommendation",
            # Principles
            "solid", "dry", "kiss", "yagni", "separation of concerns",
            "design pattern", "design patterns", "architecture",
            "technical debt", "legacy", "deprecated", "upgrade", "modernize"
        ],
        "tool_calculator": [
            # Basic Math
            "calculate", "calculation", "calculations", "calculator",
            "math", "mathematics", "mathematical", "compute", "computation",
            "add", "addition", "subtract", "subtraction", "multiply", "multiplication",
            "divide", "division", "sum", "total", "difference", "product", "quotient",
            # Statistics
            "average", "mean", "median", "mode", "statistics", "statistical",
            "percentage", "percent", "ratio", "proportion", "fraction", "decimal",
            "probability", "random", "distribution", "variance", "deviation",
            "min", "minimum", "max", "maximum", "range", "count",
            # Financial
            "financial", "finance", "money", "budget", "budgeting",
            "accounting", "accountant", "bookkeeping",
            "interest", "compound interest", "simple interest", "apr", "apy",
            "loan", "loans", "mortgage", "payment", "payments", "amortization",
            "tax", "taxes", "taxation", "income", "expense", "expenses",
            "profit", "loss", "margin", "markup", "discount",
            "roi", "return", "investment", "yield", "dividend",
            "depreciation", "inflation", "gdp", "growth rate",
            "salary", "wage", "hourly", "annual", "monthly",
            "tip", "gratuity", "split", "bill",
            # Conversions
            "convert", "conversion", "unit", "units", "measurement",
            "distance", "length", "weight", "mass", "volume", "area",
            "temperature", "celsius", "fahrenheit", "kelvin",
            "speed", "velocity", "time", "duration", "age",
            "currency", "exchange rate", "forex",
            # Advanced Math
            "equation", "equations", "formula", "formulas", "solve", "solution",
            "algebra", "geometry", "trigonometry", "calculus",
            "derivative", "integral", "limit",
            "exponent", "power", "logarithm", "log", "root", "square root", "sqrt",
            "factorial", "permutation", "combination",
            "matrix", "vector", "linear algebra",
            "graph", "plot", "chart", "function",
            # Numbers
            "number", "numbers", "numeric", "numerical", "digit", "digits",
            "integer", "float", "round", "rounding", "floor", "ceil", "ceiling",
            "absolute", "abs", "positive", "negative", "sign",
            "prime", "even", "odd", "factor", "factors", "multiple", "gcd", "lcm",
            # Body/Health Calculations
            "bmi", "body mass", "calories", "calorie", "nutrition",
            "heart rate", "pace", "distance", "steps"
        ]
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize designer.

        Args:
            api_key: OpenAI API key (optional, reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._llm_client = None
        self._llm_providers = self._load_llm_providers()

    def _load_llm_providers(self) -> Dict[str, Any]:
        """Load LLM provider configurations from llm_provider.json."""
        provider_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "llm_provider.json"
        )
        if os.path.exists(provider_file):
            with open(provider_file, 'r') as f:
                data = json.load(f)
                return {model["id"]: model for model in data.get("models", [])}
        return {}

    def _get_llm_client(self, model_id: str = None):
        """
        Get LLM client using centralized LLM Manager.

        All LLM configuration is now in llm_manager.py
        To change provider/model, edit llm_manager.py

        Args:
            model_id: Optional model ID (ignored, kept for backward compatibility)

        Returns:
            LLM client instance
        """
        from llm_manager import LLMManager

        # Get LLM from centralized manager
        # Uses default configuration from llm_manager.py
        try:
            return LLMManager.get_llm(
                temperature=0.3,
                max_tokens=4000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to get LLM from LLMManager: {e}")

    def _select_tools_for_agent(self, user_prompt: str, intent: Dict[str, Any] = None) -> list:
        """
        Auto-select appropriate tools based on agent purpose and intent keywords.

        Uses a two-phase strategy:
        1. Static keyword matching against built-in tools (TOOL_SELECTION_RULES)
        2. Dynamic search against ToolRegistry for synced API/MCP connector tools

        Connector tools compete with built-in tools in the scoring pool.
        Maximum of 3 tools will be selected (up from 2 to accommodate connectors).

        Args:
            user_prompt: The user's natural language prompt
            intent: Optional pre-analyzed intent dict with keywords

        Returns:
            List of tool IDs (max 3), empty list if no clear match
        """
        # Build keyword set from prompt and intent
        prompt_lower = user_prompt.lower()

        keywords = set()
        # Add words from prompt
        words = [w.strip(".,!?;:()[]{}\"'") for w in prompt_lower.split()]
        keywords.update(w for w in words if len(w) > 2)

        # Add keywords from intent if available
        if intent:
            keywords.update(kw.lower() for kw in intent.get("keywords", []))
            domain = intent.get("domain", "")
            if domain:
                keywords.add(domain.lower())

        if not keywords:
            return []

        # ─── Phase 1: Score built-in tools via static keyword rules ───
        tool_scores = {}

        for tool_id, tool_keywords in self.TOOL_SELECTION_RULES.items():
            score = 0
            for keyword in tool_keywords:
                if keyword in prompt_lower:
                    score += 2  # Higher score for direct prompt match
                elif keyword in keywords:
                    score += 1

            if score > 0:
                tool_scores[tool_id] = score

        # ─── Phase 2: Search ToolRegistry for synced connector tools ───
        connector_tools = self._search_connector_tools(prompt_lower, keywords)
        for tool_id, score in connector_tools.items():
            # Connector tools get a slight boost since they represent
            # specific integrations the user has explicitly configured
            tool_scores[tool_id] = tool_scores.get(tool_id, 0) + score

        if not tool_scores:
            return []

        # Sort by score and take top 3 (allow room for a connector tool)
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool_id for tool_id, score in sorted_tools[:3]]

        return selected_tools

    def _search_connector_tools(self, prompt_lower: str, keywords: set) -> Dict[str, int]:
        """
        Search the ToolRegistry for synced API/MCP connector tools that
        match the user's prompt.

        Connector tools are registered by sync_connectors_as_tools() with
        tool_type API or MCP and tags including "connector" and "synced".

        The search matches against the tool's name, description (which contains
        the auto-generated system prompt), and tags.

        Args:
            prompt_lower: Lowercased user prompt
            keywords: Set of extracted keywords from prompt + intent

        Returns:
            Dict of {tool_id: relevance_score} for matching connector tools
        """
        scores: Dict[str, int] = {}

        try:
            from echolib.di import container
            tool_registry = container.resolve('tool.registry')
        except (KeyError, ImportError):
            # ToolRegistry not available via DI — try direct import
            try:
                from apps.tool.container import container as tool_container
                tool_registry = tool_container.resolve('tool.registry')
            except Exception:
                return scores

        # Get all connector tools (API + MCP)
        try:
            from echolib.types import ToolType
            api_tools = tool_registry.list_by_type(ToolType.API)
            mcp_tools = tool_registry.list_by_type(ToolType.MCP)
            connector_tools = api_tools + mcp_tools
        except Exception:
            # Fallback: search by tag
            try:
                connector_tools = tool_registry.list_by_tags(["connector"])
            except Exception:
                return scores

        for tool in connector_tools:
            # Only consider synced connector tools (not manually registered ones)
            if "synced" not in tool.tags:
                continue

            score = 0
            tool_name_lower = tool.name.lower()
            tool_desc_lower = tool.description.lower()
            connector_name = tool.metadata.get("connector_name", "").lower()

            # Score: tool name appears in prompt
            if tool_name_lower in prompt_lower:
                score += 5  # Strong signal — user mentioned the tool by name

            # Score: connector name appears in prompt
            if connector_name and connector_name in prompt_lower:
                score += 5

            # Score: keyword overlap with tool name words
            name_words = set(tool_name_lower.replace("_", " ").replace("-", " ").split())
            keyword_overlap = keywords & name_words
            score += len(keyword_overlap) * 2

            # Score: keyword overlap with description words
            desc_words = set(tool_desc_lower.replace("_", " ").replace("-", " ").split())
            desc_overlap = keywords & desc_words
            score += len(desc_overlap)

            # Score: tag overlap
            tool_tags_lower = {t.lower() for t in tool.tags}
            tag_overlap = keywords & tool_tags_lower
            score += len(tag_overlap)

            if score > 0:
                scores[tool.tool_id] = score

        return scores

    def _get_connector_tools_description(self) -> str:
        """
        Build a human-readable description of available connector tools
        for inclusion in the LLM system prompt.

        Queries the ToolRegistry for all synced API/MCP connector tools
        and formats them as a bullet list so the LLM knows about external
        integrations the user has configured.

        Returns:
            Formatted string listing connector tools, or empty string if none.
        """
        try:
            from echolib.di import container
            tool_registry = container.resolve('tool.registry')
        except (KeyError, ImportError):
            return ""

        lines = []
        try:
            from echolib.types import ToolType
            api_tools = tool_registry.list_by_type(ToolType.API)
            mcp_tools = tool_registry.list_by_type(ToolType.MCP)
            connector_tools = api_tools + mcp_tools
        except Exception:
            return ""

        synced_tools = [t for t in connector_tools if "synced" in t.tags]
        if not synced_tools:
            return ""

        lines.append("\nThe agent can also use these external integration tools (API/MCP connectors):")
        for tool in synced_tools:
            source = tool.metadata.get("connector_source", "api").upper()
            lines.append(f"- {tool.tool_id}: [{source}] {tool.name} — {tool.description[:120]}")

        return "\n".join(lines)

    def get_available_tools(self) -> Dict[str, Any]:
        """
        Return all available tools (built-in + connector-synced) with metadata.

        Queries the ToolRegistry for the full tool catalogue. Each tool is
        categorized as "builtin" (LOCAL/CREWAI) or "connector" (API/MCP) so
        the frontend can render appropriate UI controls.

        Falls back to the static AVAILABLE_TOOLS list when the ToolRegistry
        is unavailable (e.g. DI container not yet initialized).

        Returns:
            Dict with:
                tools: List of tool dicts with tool_id, name, description,
                       tool_type, tags, source, status
                builtin_count: Number of builtin tools
                connector_count: Number of connector tools
                total: Total tool count
        """
        try:
            from echolib.di import container
            tool_registry = container.resolve('tool.registry')
        except (KeyError, ImportError):
            try:
                from apps.tool.container import container as tool_container
                tool_registry = tool_container.resolve('tool.registry')
            except Exception:
                tool_registry = None

        if tool_registry is not None:
            try:
                return self._build_tools_response_from_registry(tool_registry)
            except Exception as e:
                logger.warning(
                    f"ToolRegistry query failed, falling back to static list: {e}"
                )

        # Fallback: build a minimal response from the static AVAILABLE_TOOLS list
        return self._build_tools_response_from_static()

    def _build_tools_response_from_registry(self, tool_registry) -> Dict[str, Any]:
        """
        Build the available-tools response from ToolRegistry data.

        Args:
            tool_registry: ToolRegistry instance

        Returns:
            Dict with tools list and counts
        """
        from echolib.types import ToolType

        all_tools = tool_registry.list_all()
        builtin_types = {ToolType.LOCAL, ToolType.CREWAI}

        tools_list: List[Dict[str, Any]] = []
        builtin_count = 0
        connector_count = 0

        for tool in all_tools:
            source = "builtin" if tool.tool_type in builtin_types else "connector"
            if source == "builtin":
                builtin_count += 1
            else:
                connector_count += 1

            tools_list.append({
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "tool_type": tool.tool_type.value if hasattr(tool.tool_type, 'value') else str(tool.tool_type),
                "tags": tool.tags or [],
                "source": source,
                "status": tool.status or "active",
                "version": tool.version or "1.0",
            })

        return {
            "tools": tools_list,
            "builtin_count": builtin_count,
            "connector_count": connector_count,
            "total": len(tools_list),
        }

    def _build_tools_response_from_static(self) -> Dict[str, Any]:
        """
        Build a fallback available-tools response from the static AVAILABLE_TOOLS list.

        Returns:
            Dict with tools list (all marked as builtin) and counts
        """
        # Provide human-readable descriptions for the static tool IDs
        static_descriptions = {
            "tool_web_search": "Search the web for information",
            "tool_file_reader": "Read and parse files (PDF, CSV, JSON, etc.)",
            "tool_code_generator": "Generate code from natural language",
            "tool_code_reviewer": "Review code for quality, bugs, and security",
            "tool_calculator": "Mathematical calculations and conversions",
        }

        tools_list: List[Dict[str, Any]] = []
        for tool_id in self.AVAILABLE_TOOLS:
            name = tool_id.replace("tool_", "").replace("_", " ").title()
            tools_list.append({
                "tool_id": tool_id,
                "name": name,
                "description": static_descriptions.get(tool_id, ""),
                "tool_type": "local",
                "tags": [],
                "source": "builtin",
                "status": "active",
                "version": "1.0",
            })

        return {
            "tools": tools_list,
            "builtin_count": len(tools_list),
            "connector_count": 0,
            "total": len(tools_list),
        }

    def create_tool_from_connector(
        self,
        connector_id: str,
        connector_type: str,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new tool from an existing connector and register it.

        Looks up the connector via ConnectorManager, builds a ToolDef following
        the same pattern as ToolRegistry.sync_connectors_as_tools(), and
        registers the resulting tool in the ToolRegistry.

        Args:
            connector_id: ID of the connector to convert
            connector_type: "api" or "mcp"
            tool_name: Optional override for the tool name
            tool_description: Optional override for the tool description

        Returns:
            Dict with created tool metadata (tool_id, name, description, etc.)

        Raises:
            ValueError: If connector_type is invalid, connector not found, or
                        registration fails
        """
        if connector_type not in ("api", "mcp"):
            raise ValueError(
                f"connector_type must be 'api' or 'mcp', got '{connector_type}'"
            )

        # ── Resolve ConnectorManager ──
        from echolib.di import container
        try:
            connector_manager = container.resolve('connector.manager')
        except KeyError:
            raise ValueError(
                "ConnectorManager not available. Ensure 'connector.manager' "
                "is registered in the DI container."
            )

        # ── Fetch the connector ──
        sub_manager = connector_manager.get_manager(connector_type)
        connector = sub_manager.get(connector_id)
        if not connector:
            raise ValueError(
                f"{connector_type.upper()} connector '{connector_id}' not found"
            )

        # ── Resolve ToolRegistry ──
        try:
            tool_registry = container.resolve('tool.registry')
        except KeyError:
            raise ValueError(
                "ToolRegistry not available. Ensure 'tool.registry' "
                "is registered in the DI container."
            )

        # ── Build ToolDef ──
        from echolib.types import ToolDef, ToolType

        connector_name = tool_name or connector.get("connector_name") or connector.get("name", connector_id)
        safe_name = connector_name.lower().replace(" ", "_").replace("-", "_")
        tool_id = f"tool_{connector_type}_{safe_name}"

        # Check if tool already exists
        existing = tool_registry.get(tool_id)
        if existing:
            return {
                "tool_id": existing.tool_id,
                "name": existing.name,
                "description": existing.description,
                "tool_type": existing.tool_type.value,
                "tags": existing.tags,
                "source": "connector",
                "status": existing.status,
                "already_existed": True,
                "message": f"Tool '{tool_id}' already exists for this connector",
            }

        # Derive schemas from creation_payload
        creation_payload = connector.get("creation_payload", {})
        raw_input_schema = creation_payload.get("input_schema", {})
        example_payload = connector.get("example_payload") or creation_payload.get("example_payload", {})

        if raw_input_schema and isinstance(raw_input_schema, dict):
            input_schema = raw_input_schema
        elif example_payload and isinstance(example_payload, dict):
            props = {
                k: {"type": self._infer_json_type(v)}
                for k, v in example_payload.items()
            }
            input_schema = {"type": "object", "properties": props}
        else:
            input_schema = {
                "type": "object",
                "properties": {
                    "payload": {"type": "object", "description": "Request payload"}
                },
            }

        output_schema = creation_payload.get("output_schema", {"type": "object"})
        if not isinstance(output_schema, dict):
            output_schema = {"type": "object"}

        connector_desc = connector.get("description", "") or creation_payload.get("description", "")
        system_prompt = tool_description or (
            f"This tool allows you to interact with {connector_name}. "
            f"{connector_desc + '. ' if connector_desc else ''}"
            f"Use it when you need to call the {connector_name} "
            f"{'API' if connector_type == 'api' else 'MCP service'}."
        )

        tool_type = ToolType.API if connector_type == "api" else ToolType.MCP
        tags = [connector_type, "connector", "synced"]

        tool = ToolDef(
            tool_id=tool_id,
            name=connector_name,
            description=system_prompt,
            tool_type=tool_type,
            input_schema=input_schema,
            output_schema=output_schema,
            execution_config={
                "connector_id": connector_id,
                "connector_source": connector_type,
                "source": "manual_create",
            },
            status="active",
            tags=tags,
            metadata={
                "connector_id": connector_id,
                "connector_name": connector_name,
                "connector_source": connector_type,
                "system_prompt": system_prompt,
            },
        )

        result = tool_registry.register(tool)
        logger.info(
            f"Created tool '{tool_id}' from {connector_type.upper()} "
            f"connector '{connector_id}'"
        )

        return {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "tool_type": tool.tool_type.value,
            "tags": tool.tags,
            "source": "connector",
            "status": tool.status,
            "already_existed": False,
            "message": result.get("message", f"Tool '{tool.name}' registered successfully"),
        }

    @staticmethod
    def _infer_json_type(value: Any) -> str:
        """Infer a JSON Schema type string from a Python value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"

    def design_from_prompt(
        self,
        user_prompt: str,
        default_model: str = "openrouter-devstral",
        icon: str = "",
        tools: list = None,
        variables: list = None
    ) -> Dict[str, Any]:
        """
        Design agent from user prompt using LLM.

        Args:
            user_prompt: Natural language description of agent
            default_model: Model ID to use for agent
            icon: Emoji icon for agent
            tools: List of tool names (if None or empty, auto-selects based on purpose)
            variables: List of variable definitions

        Returns:
            Agent definition dict
        """
        if variables is None:
            variables = []

        # Auto-select tools if not provided or empty
        if tools is None or len(tools) == 0:
            tools = self._select_tools_for_agent(user_prompt)

        # Use LLM to analyze prompt
        try:
            agent_spec = self._design_with_llm(user_prompt, default_model)
        except Exception as e:
            print(f"LLM design failed, using basic structure: {e}")
            agent_spec = self._design_basic(user_prompt)

        # Build agent definition
        agent_id = new_id("agt_")
        timestamp = datetime.utcnow().isoformat()

        # Get LLM-suggested settings or use defaults
        llm_settings = agent_spec.get("settings", {})
        temperature = llm_settings.get("temperature", 0.7)
        max_tokens = llm_settings.get("max_tokens", 2000)
        top_p = llm_settings.get("top_p", 0.9)
        max_iterations = llm_settings.get("max_iterations", 5)

        agent = {
            "agent_id": agent_id,
            "name": agent_spec.get("name", "Agent"),
            "icon": icon,
            "role": agent_spec.get("role", "Processing"),
            "description": agent_spec.get("description", user_prompt[:200]),
            "prompt": agent_spec.get("prompt", user_prompt),
            "model": default_model,
            "tools": tools,
            "variables": variables,
            "settings": {
                "temperature": temperature,
                "max_token": max_tokens,
                "top_p": top_p,
                "max_iteration": max_iterations
            },
            "input_schema": agent_spec.get("input_schema", []),
            "output_schema": agent_spec.get("output_schema", []),
            "constraints": {
                "max_steps": max_iterations,
                "timeout_seconds": 60
            },
            "permissions": {
                "can_call_agents": False,
                "allowed_agents": []
            },
            "metadata": {
                "created_by": "agent_designer",
                "created_at": timestamp,
                "tags": ["auto-generated"]
            }
        }

        return agent

    def _design_with_llm(
        self,
        user_prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """Design agent using LLM analysis."""

        # Build a dynamic list of available connector tools for the LLM prompt
        connector_tools_desc = self._get_connector_tools_description()

        system_prompt = f"""You are an AI agent designer. Analyze the user's request and design a complete agent specification.

Return a JSON response with this exact structure:
{{
  "name": "Creative and Memorable Agent Name",
  "role": "Professional Role/Title",
  "description": "What this agent does (1-2 sentences)",
  "prompt": "Detailed system prompt/instructions for the agent",
  "input_schema": ["list", "of", "input", "keys"],
  "output_schema": ["list", "of", "output", "keys"],
  "settings": {{
    "temperature": 0.7,
    "max_tokens": 2000,
    "top_p": 0.9,
    "max_iterations": 5
  }}
}}

IMPORTANT RULES:

1. NAME: Must be creative, memorable, and professional. Examples:
   - For code review: "CodeCraft Pro", "PyReviewer Elite", "SyntaxMaster"
   - For content writing: "ContentForge", "WordSmith Pro", "NarrativeGenius"
   - For data analysis: "DataWiz", "InsightEngine", "AnalyticsPro"
   - NEVER use generic names like "Custom Agent", "AI Agent", or "New Agent"

2. ROLE: A professional job title (e.g., "Senior Python Developer", "Content Strategist")

3. DESCRIPTION: Clear 1-2 sentence explanation of what the agent does

4. PROMPT: Detailed instructions for the agent to follow when executing tasks.
   If the agent uses external integration tools (API/MCP connectors like Jira, Slack, Salesforce),
   mention those integrations in the prompt so the agent knows it can use them.

5. SETTINGS: Tune based on task type:
   - temperature: 0.1-0.3 for factual/precise tasks (code, math), 0.5-0.7 for balanced tasks, 0.8-1.0 for creative tasks
   - max_tokens: 1000-2000 for short outputs, 2000-4000 for detailed responses
   - top_p: 0.9 default, lower (0.5-0.7) for more focused outputs
   - max_iterations: 3-5 for simple tasks, 5-10 for complex multi-step tasks

AVAILABLE TOOLS:
The agent can use these built-in tools:
- tool_web_search: Search the web for information
- tool_file_reader: Read and parse files (PDF, CSV, JSON, etc.)
- tool_code_generator: Generate code from natural language
- tool_code_reviewer: Review code for quality, bugs, and security
- tool_calculator: Mathematical calculations and conversions
{connector_tools_desc}
When designing the agent's prompt, consider which of these tools the agent might need.
"""

        llm = self._get_llm_client(model_id)

        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object."

        # Invoke LLM
        response = llm.invoke(full_prompt)

        # Parse response
        try:
            # Handle response content - may be string or have content attribute
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to extract JSON from response
            # Sometimes LLM wraps JSON in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            agent_spec = json.loads(content)
            return agent_spec
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            return self._design_basic(user_prompt)

    def _design_basic(self, user_prompt: str) -> Dict[str, Any]:
        """Basic agent structure without LLM."""
        return {
            "name": "Custom Agent",
            "role": "Processing",
            "description": user_prompt[:200],
            "prompt": user_prompt,
            "input_schema": [],
            "output_schema": []
        }

    def update_from_prompt(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Update an existing agent based on user prompt while preserving identity.

        This method detects what fields the user wants to update and only
        modifies those fields, preserving the agent's name, ID, and other
        unchanged attributes.

        Args:
            existing_agent: The current agent definition dict
            user_prompt: User's natural language update request

        Returns:
            Updated agent definition dict with preserved identity
        """
        # Detect what the user wants to update
        update_intent = self._detect_update_fields(user_prompt)

        # Use LLM to generate updates for specified fields only
        try:
            updates = self._generate_field_updates(
                existing_agent, user_prompt, update_intent
            )
        except Exception as e:
            print(f"LLM update failed, applying basic updates: {e}")
            updates = self._apply_basic_updates(existing_agent, user_prompt, update_intent)

        # Merge updates into existing agent, preserving identity
        updated_agent = existing_agent.copy()

        # CRITICAL: Always preserve agent_id and name unless explicitly requested
        preserved_fields = ["agent_id", "name"]
        if "name" not in update_intent.get("fields_to_update", []):
            # Name should not change unless explicitly requested
            updates.pop("name", None)

        # Apply updates
        for key, value in updates.items():
            if key not in preserved_fields or key in update_intent.get("fields_to_update", []):
                updated_agent[key] = value

        # Update metadata
        if "metadata" not in updated_agent:
            updated_agent["metadata"] = {}
        updated_agent["metadata"]["updated_at"] = datetime.utcnow().isoformat()
        updated_agent["metadata"]["update_prompt"] = user_prompt

        return updated_agent

    def _detect_update_fields(self, user_prompt: str) -> Dict[str, Any]:
        """
        Detect which fields the user wants to update based on prompt keywords.

        Args:
            user_prompt: User's update request

        Returns:
            Dict with fields_to_update list and detected intent
        """
        prompt_lower = user_prompt.lower()

        fields_to_update = []

        # Tool-related keywords (includes connector/integration triggers)
        tool_keywords = ["tool", "tools", "add tool", "remove tool", "change tool",
                         "web search", "file reader", "code executor",
                         "connector", "integration", "api", "mcp",
                         "jira", "slack", "salesforce", "servicenow", "github",
                         "confluence", "zendesk", "pagerduty", "datadog"]
        if any(kw in prompt_lower for kw in tool_keywords):
            fields_to_update.append("tools")

        # Description/purpose keywords
        desc_keywords = ["description", "purpose", "what it does", "goal", "objective"]
        if any(kw in prompt_lower for kw in desc_keywords):
            fields_to_update.append("description")

        # Role keywords
        role_keywords = ["role", "position", "job", "title"]
        if any(kw in prompt_lower for kw in role_keywords):
            fields_to_update.append("role")

        # Prompt/behavior keywords
        behavior_keywords = ["prompt", "behavior", "instruction", "system prompt",
                            "how it works", "act", "behave"]
        if any(kw in prompt_lower for kw in behavior_keywords):
            fields_to_update.append("prompt")

        # Settings keywords
        settings_keywords = ["temperature", "max_token", "setting", "configure",
                            "parameter", "iteration"]
        if any(kw in prompt_lower for kw in settings_keywords):
            fields_to_update.append("settings")

        # Name keywords (only if explicitly mentioned)
        name_keywords = ["rename", "change name", "new name", "call it", "named"]
        if any(kw in prompt_lower for kw in name_keywords):
            fields_to_update.append("name")

        # If no specific fields detected, assume description and prompt update
        if not fields_to_update:
            fields_to_update = ["description", "prompt"]

        return {
            "fields_to_update": fields_to_update,
            "original_prompt": user_prompt
        }

    def _generate_field_updates(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str,
        update_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to generate updates for specified fields.

        Args:
            existing_agent: Current agent definition
            user_prompt: User's update request
            update_intent: Dict with fields_to_update

        Returns:
            Dict with updated field values
        """
        fields_to_update = update_intent.get("fields_to_update", [])

        system_prompt = f"""You are an AI agent updater. The user wants to modify an existing agent.

EXISTING AGENT:
- Name: {existing_agent.get('name', 'Unknown')}
- Role: {existing_agent.get('role', 'Processing')}
- Description: {existing_agent.get('description', '')}
- Current Tools: {existing_agent.get('tools', [])}
- System Prompt: {existing_agent.get('prompt', '')[:500]}

FIELDS TO UPDATE: {fields_to_update}

Based on the user's request, provide updates ONLY for the specified fields.
Return a JSON object with ONLY the fields that need updating.

IMPORTANT RULES:
1. NEVER change the agent's name unless "name" is in the fields to update
2. Only return the fields listed in FIELDS TO UPDATE
3. Preserve the agent's core identity and purpose
4. For tools: return the complete new tools list (not just additions/removals)

Example response format:
{{"description": "Updated description", "tools": ["tool1", "tool2"]}}
"""

        llm = self._get_llm_client()
        full_prompt = f"{system_prompt}\n\nUser Request: {user_prompt}\n\nProvide your response as a valid JSON object with only the updated fields."

        response = llm.invoke(full_prompt)

        # Parse response
        try:
            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            updates = json.loads(content)

            # Validate that only requested fields are updated
            validated_updates = {}
            for field in fields_to_update:
                if field in updates:
                    validated_updates[field] = updates[field]

            return validated_updates

        except json.JSONDecodeError:
            return self._apply_basic_updates(existing_agent, user_prompt, update_intent)

    def _apply_basic_updates(
        self,
        existing_agent: Dict[str, Any],
        user_prompt: str,
        update_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply basic updates without LLM (fallback).

        Args:
            existing_agent: Current agent definition
            user_prompt: User's update request
            update_intent: Dict with fields_to_update

        Returns:
            Dict with basic updated values
        """
        fields_to_update = update_intent.get("fields_to_update", [])
        updates = {}

        if "description" in fields_to_update:
            # Append update context to description
            current_desc = existing_agent.get("description", "")
            updates["description"] = f"{current_desc} (Updated: {user_prompt[:100]})"

        if "prompt" in fields_to_update:
            # Append to system prompt
            current_prompt = existing_agent.get("prompt", "")
            updates["prompt"] = f"{current_prompt}\n\nAdditional instructions: {user_prompt}"

        return updates

    # ================================================================
    # Backend-driven conversational flow
    # ================================================================

    def handle_chat_step(
        self,
        message: str,
        step: str,
        agent_state: Dict[str, Any],
        model: str = "openrouter-devstral",
    ) -> Dict[str, Any]:
        """
        Handle one turn of the agent-builder conversation.

        The frontend sends the user message, the current step, and the
        current agent_state.  This method classifies intent, performs
        the appropriate action (design / name confirm / tool ops /
        instruction update / finalize), and returns a structured
        response that the frontend renders without any business logic.

        Steps: initial -> name -> refine -> done

        Args:
            message:     User's natural-language message
            step:        Current conversation step
            agent_state: Current agent definition dict (may be empty on 'initial')
            model:       LLM model id for agent design

        Returns:
            Dict with keys:
                reply        – chat bubble text
                step         – next conversation step
                action       – machine-readable action tag
                agent_state  – (possibly updated) agent definition
                tools_available – list of tool dicts (only on tool-related actions)
        """
        message_lower = message.lower().strip()

        if step == "initial":
            return self._chat_step_initial(message, agent_state, model)
        elif step == "name":
            return self._chat_step_name(message, message_lower, agent_state)
        elif step == "refine":
            return self._chat_step_refine(message, message_lower, agent_state)
        elif step == "done":
            return self._chat_step_done(message, message_lower, agent_state)
        else:
            return {
                "reply": "Something went wrong. Let's start over — describe the agent you'd like to create.",
                "step": "initial",
                "action": "ERROR",
                "agent_state": {},
            }

    # ---- initial step ---------------------------------------------------

    def _chat_step_initial(
        self, message: str, agent_state: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """Design a new agent from the user's first prompt."""
        try:
            agent = self.design_from_prompt(
                user_prompt=message,
                default_model=model,
                icon=agent_state.get("icon", ""),
                tools=agent_state.get("tools"),
                variables=agent_state.get("variables"),
            )
        except Exception as e:
            logger.error(f"Agent design failed: {e}")
            return {
                "reply": f"Sorry, I couldn't design the agent: {e}\nPlease try rephrasing your request.",
                "step": "initial",
                "action": "ERROR",
                "agent_state": agent_state,
            }

        name = agent.get("name", "Agent")
        role = agent.get("role", "")
        desc = agent.get("description", "")

        reply = (
            f'Your agent "{name}" has been designed!\n\n'
            f"Role: {role}\n"
            f"Description: {desc}\n\n"
            f"Next, let's confirm the name. I suggest: {name}.\n\n"
            f"Would you like to use this name, or do you have another name in mind?"
        )
        return {
            "reply": reply,
            "step": "name",
            "action": "AGENT_DESIGNED",
            "agent_state": agent,
        }

    # ---- name step ------------------------------------------------------

    def _chat_step_name(
        self, message: str, message_lower: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle name confirmation / modification."""
        intent = self._classify_intent_safe(
            "name_confirmation", agent_state.get("name", ""), message
        )
        current_name = agent_state.get("name", "Agent")

        if intent["intent"] == "CONFIRMATION":
            reply = (
                f'The agent\'s name is set to "{current_name}".\n\n'
                "Let's move forward and refine what the agent will do. "
                "Could you specify any particular requirements or preferences "
                "for how the agent should work?"
            )
            return {
                "reply": reply,
                "step": "refine",
                "action": "NAME_CONFIRMED",
                "agent_state": agent_state,
            }

        elif intent["intent"] == "MODIFICATION":
            new_name = intent.get("extracted_value") or message.strip()
            agent_state["name"] = new_name
            reply = (
                f'I\'ve updated the name to "{new_name}".\n\n'
                "Let's move forward and refine what the agent will do. "
                "Could you specify any particular requirements or preferences "
                "for how the agent should work?"
            )
            return {
                "reply": reply,
                "step": "refine",
                "action": "NAME_UPDATED",
                "agent_state": agent_state,
            }

        elif intent["intent"] == "REJECTION":
            reply = (
                "No problem! What would you like to name this agent instead?"
            )
            return {
                "reply": reply,
                "step": "name",
                "action": "NAME_REJECTED",
                "agent_state": agent_state,
            }

        else:  # CLARIFICATION
            reply = (
                f'The current suggested name is "{current_name}". '
                "You can accept it, or tell me a different name you'd prefer."
            )
            return {
                "reply": reply,
                "step": "name",
                "action": "CLARIFICATION",
                "agent_state": agent_state,
            }

    # ---- refine step ----------------------------------------------------

    def _chat_step_refine(
        self, message: str, message_lower: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle refinements: tools, instructions, and finalization."""
        intent = self._classify_intent_safe(
            "refinement", "finalize agent", message
        )

        # --- CONFIRMATION → finalize ---
        if intent["intent"] == "CONFIRMATION":
            reply = (
                "Great! Your agent is ready. You can now:\n\n"
                '• Click "Configure" tab to add tools, variables, and fine-tune settings\n'
                '• Or click "Save Agent" to save as-is and start using it'
            )
            return {
                "reply": reply,
                "step": "done",
                "action": "FINALIZED",
                "agent_state": agent_state,
            }

        # --- CLARIFICATION → informational response ---
        if intent["intent"] == "CLARIFICATION":
            return self._handle_clarification(message_lower, agent_state)

        # --- MODIFICATION (or REJECTION treated as modification) ---
        return self._handle_modification(message, message_lower, agent_state)

    def _handle_clarification(
        self, message_lower: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Respond to informational questions during refinement."""
        tool_words = {"tool", "tools", "available", "list", "show", "what"}
        if tool_words & set(message_lower.split()):
            all_tools = self._get_all_tools_for_chat()
            current_ids = agent_state.get("tools", [])

            lines = ["Here are the available tools:\n"]
            for t in all_tools:
                badge = " [Connector]" if t["source"] == "connector" else ""
                lines.append(f'• **{t["name"]}**{badge} — {t["description"]}')

            if current_ids:
                selected_names = self._tool_ids_to_names(current_ids, all_tools)
                lines.append(f"\n**Currently selected:** {', '.join(selected_names)}")

            lines.append(
                '\nTo add tools, say something like "add calculator and web search".'
                '\nTo create a tool from a connector, say "create a [connector name] tool".'
            )
            return {
                "reply": "\n".join(lines),
                "step": "refine",
                "action": "TOOLS_LISTED",
                "agent_state": agent_state,
                "tools_available": all_tools,
            }

        # Generic clarification
        return {
            "reply": (
                "You can:\n\n"
                '• Add or remove tools (say "add web search")\n'
                "• Update the agent's behavior or instructions\n"
                "• Change the agent's role or description\n\n"
                'Or say "looks good" to finalize the agent.'
            ),
            "step": "refine",
            "action": "CLARIFICATION",
            "agent_state": agent_state,
        }

    def _handle_modification(
        self, message: str, message_lower: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle modification requests (tools or instructions)."""

        # Build dynamic tool keywords
        all_tools = self._get_all_tools_for_chat()
        tool_keywords = {"tool", "tools", "add tool", "remove tool",
                         "connector", "integration", "create tool",
                         "new tool", "generate tool"}
        for t in all_tools:
            tool_keywords.add(t["name"].lower())
            tid = t["tool_id"]
            if tid.startswith("tool_"):
                tool_keywords.add(tid[5:].replace("_", " "))

        is_tool_request = any(kw in message_lower for kw in tool_keywords)

        if is_tool_request:
            return self._handle_tool_modification(message_lower, agent_state, all_tools)

        # Regular instruction refinement
        current_prompt = agent_state.get("prompt", "")
        agent_state["prompt"] = f"{current_prompt}\n\nAdditional instructions: {message}"
        return {
            "reply": (
                f'I\'ve updated the agent\'s instructions to include: "{message}"\n\n'
                "Would you like to add any other requirements, or are we ready to finalize?"
            ),
            "step": "refine",
            "action": "INSTRUCTIONS_UPDATED",
            "agent_state": agent_state,
        }

    def _handle_tool_modification(
        self,
        message_lower: str,
        agent_state: Dict[str, Any],
        all_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Parse tool add/remove/create requests and update agent_state."""
        current_tools: list = agent_state.get("tools", [])

        # Build name→id lookup
        tool_map: Dict[str, str] = {}
        for t in all_tools:
            name_l = t["name"].lower()
            tool_map[name_l] = t["tool_id"]
            tool_map[name_l.replace(" ", "_")] = t["tool_id"]
            tool_map[t["tool_id"]] = t["tool_id"]
            if t["tool_id"].startswith("tool_"):
                short = t["tool_id"][5:]
                tool_map[short] = t["tool_id"]
                tool_map[short.replace("_", " ")] = t["tool_id"]

        is_remove = "remove" in message_lower
        matched_ids = []
        for alias, tid in tool_map.items():
            if alias in message_lower:
                matched_ids.append(tid)
        matched_ids = list(dict.fromkeys(matched_ids))  # dedupe, preserve order

        if is_remove and matched_ids:
            new_tools = [t for t in current_tools if t not in matched_ids]
            agent_state["tools"] = new_tools
            removed = ", ".join(self._tool_ids_to_names(matched_ids, all_tools))
            current = ", ".join(self._tool_ids_to_names(new_tools, all_tools)) or "None"
            return {
                "reply": (
                    f"I've removed the following tools: {removed}\n\n"
                    f"Current tools: {current}\n\n"
                    "Would you like to make any other changes?"
                ),
                "step": "refine",
                "action": "TOOLS_REMOVED",
                "agent_state": agent_state,
                "tools_available": all_tools,
            }

        if not is_remove and matched_ids:
            new_tools = list(dict.fromkeys(current_tools + matched_ids))
            agent_state["tools"] = new_tools
            added = ", ".join(self._tool_ids_to_names(matched_ids, all_tools))
            current = ", ".join(self._tool_ids_to_names(new_tools, all_tools))
            return {
                "reply": (
                    f"I've added the following tools: {added}\n\n"
                    f"Current tools: {current}\n\n"
                    "Would you like to make any other changes, or are we ready to finalize?"
                ),
                "step": "refine",
                "action": "TOOLS_ADDED",
                "agent_state": agent_state,
                "tools_available": all_tools,
            }

        # No specific tool matched — check create-from-connector intent
        create_kw = {"create", "generate", "make", "build", "new tool"}
        connector_kw = {
            "connector", "integration", "teams", "slack", "jira",
            "salesforce", "servicenow", "github", "confluence",
            "zendesk", "pagerduty", "datadog",
        }
        wants_new = any(kw in message_lower for kw in create_kw)
        mentions_conn = any(kw in message_lower for kw in connector_kw)

        if wants_new or mentions_conn:
            # Try to auto-create tool from a matching connector
            result = self._try_auto_create_connector_tool(message_lower, agent_state, all_tools)
            if result is not None:
                return result

            # No connector matched — fall back to guidance
            tool_lines = "\n".join(
                f'• {t["name"]}{" [Connector]" if t["source"] == "connector" else ""}'
                for t in all_tools
            )
            return {
                "reply": (
                    "I couldn't find a matching connector for your request. "
                    "Please go to the **Connectors** page to register your "
                    "connector first, then come back and I'll create the tool "
                    "automatically.\n\n"
                    "Available tools:\n\n"
                    f"{tool_lines}\n\n"
                    'Example: "add calculator and web search"'
                ),
                "step": "refine",
                "action": "TOOL_CREATE_GUIDANCE",
                "agent_state": agent_state,
                "tools_available": all_tools,
            }

        # Fallback: show tool list
        tool_lines = "\n".join(
            f'• {t["name"]}{" [Connector]" if t["source"] == "connector" else ""}'
            for t in all_tools
        )
        example = all_tools[0]["name"].lower() if all_tools else "tool name"
        return {
            "reply": (
                "Please specify which tools to add or remove:\n\n"
                f"{tool_lines}\n\n"
                f'Example: "add {example}"'
            ),
            "step": "refine",
            "action": "TOOLS_PROMPT",
            "agent_state": agent_state,
            "tools_available": all_tools,
        }

    def _fetch_all_connectors(self) -> List[Dict[str, Any]]:
        """
        Fetch all registered connectors (API + MCP) from ConnectorManager.

        Returns:
            List of connector dicts, each tagged with '_source' ('api' or 'mcp').
            Empty list if ConnectorManager is unavailable.
        """
        try:
            from echolib.di import container
            connector_manager = container.resolve('connector.manager')
        except (KeyError, ImportError):
            return []

        connectors: List[Dict[str, Any]] = []
        for source, sub in [("api", connector_manager.api), ("mcp", connector_manager.mcp)]:
            try:
                result = sub.list()
                if result.get("success"):
                    for conn in result.get("connectors", []):
                        conn["_source"] = source
                    connectors.extend(result.get("connectors", []))
            except Exception:
                continue
        return connectors

    def _score_connectors(
        self, message_lower: str, connectors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score each connector against the user message. Returns scored list
        sorted by score descending, only entries with score > 0.

        Scoring rules:
        - Full connector_name match in message: +5 (strong signal)
        - Each word (>2 chars) of connector_name found in message: +1
        - Connector_id found in message: +5 (explicit reference)

        Each returned dict: {"connector": <original dict>, "score": int, "name": str}
        """
        scored = []
        for conn in connectors:
            name = (conn.get("connector_name") or conn.get("name", "")).strip()
            if not name:
                continue

            name_lower = name.lower()
            name_normalized = name_lower.replace("_", " ").replace("-", " ")
            connector_id = conn.get("connector_id", "")

            score = 0

            # Full name match (strongest signal)
            if name_normalized in message_lower or name_lower in message_lower:
                score += 5

            # Explicit connector_id mention
            if connector_id and connector_id.lower() in message_lower:
                score += 5

            # Word-level matches
            name_words = [w for w in name_normalized.split() if len(w) > 2]
            score += sum(1 for w in name_words if w in message_lower)

            if score > 0:
                scored.append({"connector": conn, "score": score, "name": name})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def _try_auto_create_connector_tool(
        self,
        message_lower: str,
        agent_state: Dict[str, Any],
        all_tools: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Match user message against registered connectors and either:
        - Auto-create tool if exactly one clear match
        - Return disambiguation options if multiple close matches
        - Return None if no matches (caller shows fallback guidance)

        No tool names are fabricated — everything comes from registered
        connector data (connector_id + connector_name).

        Args:
            message_lower: Lowercased user message
            agent_state: Current agent definition dict
            all_tools: Current list of all available tools

        Returns:
            Response dict with action:
            - TOOL_AUTO_CREATED: single match, tool created
            - TOOL_SELECT_PROMPT: multiple matches, user must pick
            - None: no matches
        """
        connectors = self._fetch_all_connectors()
        if not connectors:
            return None

        scored = self._score_connectors(message_lower, connectors)
        if not scored:
            return None

        top_score = scored[0]["score"]

        # Collect all connectors with scores close to the top
        # "Close" = within 60% of top score (e.g. top=5, threshold=3)
        threshold = max(top_score * 0.6, 1)
        candidates = [s for s in scored if s["score"] >= threshold]

        # ── SINGLE CLEAR WINNER ──
        if len(candidates) == 1:
            return self._create_tool_from_match(candidates[0], agent_state)

        # Check if top candidate is significantly ahead of the rest
        if len(candidates) > 1 and candidates[0]["score"] > candidates[1]["score"] * 1.5:
            return self._create_tool_from_match(candidates[0], agent_state)

        # ── AMBIGUOUS — return options for user to pick ──
        options = []
        for c in candidates[:5]:  # Cap at 5 options
            conn = c["connector"]
            options.append({
                "connector_id": conn.get("connector_id", ""),
                "connector_name": c["name"],
                "connector_type": conn.get("_source", "api"),
                "score": c["score"],
            })

        option_lines = "\n".join(
            f'{i+1}. **{opt["connector_name"]}** ({opt["connector_type"].upper()})'
            for i, opt in enumerate(options)
        )

        reply = (
            f"I found {len(options)} connectors that match your request:\n\n"
            f"{option_lines}\n\n"
            "Which one would you like to add? "
            "Please reply with the name (e.g., the exact connector name)."
        )

        return {
            "reply": reply,
            "step": "refine",
            "action": "TOOL_SELECT_PROMPT",
            "agent_state": agent_state,
            "tools_available": all_tools,
            "connector_options": options,
        }

    def _create_tool_from_match(
        self,
        match: Dict[str, Any],
        agent_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a tool from a single confirmed connector match.

        Args:
            match: Scored match dict with 'connector' and 'name' keys
            agent_state: Current agent definition dict

        Returns:
            Response dict with action TOOL_AUTO_CREATED, or None on failure
        """
        conn = match["connector"]
        connector_id = conn.get("connector_id", "")
        connector_type = conn.get("_source", "api")
        connector_name = match["name"]

        try:
            create_result = self.create_tool_from_connector(
                connector_id=connector_id,
                connector_type=connector_type,
                tool_name=connector_name,
            )
        except Exception as e:
            logger.warning(f"Auto-create tool from connector failed: {e}")
            return None

        # Add the new tool to the agent
        new_tool_id = create_result.get("tool_id", "")
        current_tools: list = agent_state.get("tools", [])
        if new_tool_id and new_tool_id not in current_tools:
            current_tools.append(new_tool_id)
            agent_state["tools"] = current_tools

        # Refresh tool list to include the newly created tool
        refreshed_tools = self._get_all_tools_for_chat()
        tool_display = create_result.get("name", new_tool_id)
        current_names = ", ".join(self._tool_ids_to_names(current_tools, refreshed_tools))

        already = create_result.get("already_existed", False)
        if already:
            reply = (
                f'The tool **{tool_display}** already exists and has been added to your agent.\n\n'
                f"Current tools: {current_names}\n\n"
                "Would you like to make any other changes?"
            )
        else:
            reply = (
                f'I found the **{tool_display}** connector and automatically created a tool from it!\n\n'
                f"Current tools: {current_names}\n\n"
                "Would you like to make any other changes, or are we ready to finalize?"
            )

        return {
            "reply": reply,
            "step": "refine",
            "action": "TOOL_AUTO_CREATED",
            "agent_state": agent_state,
            "tools_available": refreshed_tools,
        }

    # ---- done step ------------------------------------------------------

    def _chat_step_done(
        self, message: str, message_lower: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle messages after agent is finalized."""
        # If user wants more changes, go back to refine
        change_words = {"change", "update", "modify", "add", "remove", "edit"}
        if change_words & set(message_lower.split()):
            return {
                "reply": "Sure! What would you like to change?",
                "step": "refine",
                "action": "REOPEN_REFINE",
                "agent_state": agent_state,
            }

        return {
            "reply": (
                "Your agent is ready! Switch to the \"Configure\" tab to "
                "add tools and customize further, or save the agent to start using it."
            ),
            "step": "done",
            "action": "FINALIZED",
            "agent_state": agent_state,
        }

    # ---- helpers --------------------------------------------------------

    def _classify_intent_safe(
        self, context: str, suggested_value: str, user_message: str
    ) -> Dict[str, Any]:
        """
        Classify intent via AgentService.classify_user_intent, with fallback.
        """
        try:
            from echolib.di import container
            service = container.resolve("agent.service")
            result = service.classify_user_intent(
                context=context,
                suggested_value=suggested_value,
                user_message=user_message,
            )
            return result
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return {"intent": "MODIFICATION", "confidence": 0.0, "reasoning": "fallback"}

    def _get_all_tools_for_chat(self) -> List[Dict[str, Any]]:
        """
        Return the available tools list for chat responses.
        Reuses get_available_tools() and returns just the tools array.
        """
        result = self.get_available_tools()
        return result.get("tools", [])

    @staticmethod
    def _tool_ids_to_names(
        ids: list, all_tools: List[Dict[str, Any]]
    ) -> List[str]:
        """Resolve a list of tool IDs to display names."""
        lookup = {t["tool_id"]: t["name"] for t in all_tools}
        return [lookup.get(tid, tid.replace("_", " ")) for tid in ids]
