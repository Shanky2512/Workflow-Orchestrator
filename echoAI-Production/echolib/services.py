
from typing import List, Callable
from .interfaces import ILogger, IEventBus, ICredentialStore
from .types import *
from .utils import new_id
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Tuple,Union, Generator
from datetime import datetime
import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sys
import traceback
import copy

from fastapi import HTTPException, status
from typing import Dict, Any, Optional


# MCP and API Connectors
from echolib.Get_connector.Get_MCP.http_script import HTTPMCPConnector
from echolib.Get_connector.Get_MCP.sse import SSEMCPConnector
from echolib.Get_connector.Get_MCP.stdio import STDIOMCPConnector
from echolib.Get_connector.Get_MCP.storage import get_storage
from echolib.Get_connector.Get_MCP.validator import validate_and_normalize, ValidationError
from echolib.Get_connector.Get_API.connectors.factory import ConnectorFactory
from echolib.Get_connector.Get_API.models import ConnectorConfig
from echolib.Get_connector.Get_MCP.storage import get_storage, ConnectorStorage
from jsonschema import validate, ValidationError
 
# LLM Details
from echolib.LLM_Details.Core.llm_client_builder import LLMClientBuilder
from echolib.LLM_Details.Core.config_loader import ConfigLoader
from echolib.LLM_Details.Core.unified_model_wrapper import UnifiedModelWrapper
from echolib.LLM_Details.Exeption import ModelNotFoundError, ValidationError, MissingAPIKeyError, ConfigError

# LLM Details
from echolib.llm_factory import LLMFactory
from echolib.LLM_Details.Core import unified_model_wrapper

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO) 
# Create logger instance
logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
    def put(self, doc: Document) -> None:
        self._docs[doc.id] = doc
    def get(self, id: str) -> Document:
        return self._docs[id]
    def search(self, query: str) -> List[Document]:
        q = query.lower()
        return [d for d in self._docs.values() if q in d.title.lower() or q in d.content.lower()]

class StateStore:
    def __init__(self) -> None:
        self._s: dict[str, dict] = {}
    def put(self, key: str, value: dict) -> None:
        self._s[key] = value
    def get(self, key: str) -> dict:
        return self._s.get(key, {})
    def del_(self, key: str) -> None:
        self._s.pop(key, None)

class ToolService:
    def __init__(self, cred_store: ICredentialStore | None = None):
        self._tools: dict[str, ToolDef] = {}
        self._cred = cred_store
    def registerTool(self, tool: ToolDef) -> ToolRef:
        self._tools[tool.name] = tool
        return ToolRef(name=tool.name)
    def listTools(self) -> List[ToolRef]:
        return [ToolRef(name=n) for n in self._tools.keys()]
    def invokeTool(self, name: str, args: dict) -> ToolResult:
        if name not in self._tools:
            raise ValueError('tool not found')
        return ToolResult(name=name, output={'echo': args})

class RAGService:
    """
    Traditional RAG Service supporting both legacy and vector-based stores.
    
    Supports:
    - Legacy DocumentStore (keyword-based search)
    - VectorDocumentStore (semantic vector search with FAISS + OpenAI embeddings)
    """
    def __init__(self, store):
        """
        Initialize RAG service.
        
        Args:
            store: DocumentStore or VectorDocumentStore instance
        """
        self.store = store
        logger.info(f"RAGService initialized with store type: {type(store).__name__}")
    
    def indexDocs(self, docs: List[Document]) -> IndexSummary:
        """
        Index documents into the store.
        
        Args:
            docs: List of documents to index
            
        Returns:
            IndexSummary with count of indexed documents
        """
        indexed_count = 0
        for d in docs:
            result = self.store.put(d)
            # VectorDocumentStore returns bool, DocumentStore returns None
            if result is None or result:
                indexed_count += 1
        
        logger.info(f"Indexed {indexed_count}/{len(docs)} documents")
        return IndexSummary(count=indexed_count)
    
    def queryIndex(self, query: str, filters: dict, top_k: int = 10) -> ContextBundle:
        """
        Query the document index.
        
        Args:
            query: Search query
            filters: Additional filters (not currently used)
            top_k: Number of results to return
            
        Returns:
            ContextBundle with matching documents
        """
        # Check if store supports top_k parameter (VectorDocumentStore)
        if hasattr(self.store, 'search') and 'top_k' in self.store.search.__code__.co_varnames:
            results = self.store.search(query, top_k=top_k)
        else:
            # Fallback for legacy DocumentStore
            results = self.store.search(query)
            results = results[:top_k]
        
        logger.debug(f"Query '{query}' returned {len(results)} results")
        return ContextBundle(documents=results)
    
    def vectorize(self, text: str) -> List[float]:
        """
        Vectorize text (for compatibility).
        
        Args:
            text: Text to vectorize
            
        Returns:
            Embedding vector if VectorDocumentStore is used, otherwise stub
        """
        if hasattr(self.store, '_get_embedding'):
            embedding = self.store._get_embedding(text)
            if embedding is not None:
                return embedding.tolist()
        
        # Fallback stub
        return [float(len(text))]
    
    def get_stats(self) -> dict:
        """
        Get store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        if hasattr(self.store, 'stats'):
            return self.store.stats()
        return {
            "total_documents": len(self.store._docs) if hasattr(self.store, '_docs') else 0,
            "store_type": type(self.store).__name__
        }

class LLMService:
    """
    Service layer for LLM operations.
    
    This is a thin wrapper that delegates to LLMFactory.
    Exists for:
    - Backward compatibility with existing code
    - Dependency injection (receives tool_service from container)
    - Future service-level features (auth, logging, caching, rate limiting)
    """
    
    def __init__(self, tool_service):
        """
        Initialize LLM Service.
        
        Args:
            tool_service: Tool service instance from container
        """
        self.tool_service = tool_service
        
        # Create the factory - this does all the real work
        self.factory = LLMFactory()
        
        logger.info("LLMService initialized (delegates to LLMFactory)")
    
    # ==================== CORE METHODS - Delegate to Factory ====================
    
    def get_model(
        self,
        model_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache: bool = False
    ) -> unified_model_wrapper:
        logger.debug(f"LLMService.get_model({model_id}) -> delegating to factory")
        return self.factory.get_model(model_id, temperature, max_tokens, cache)
    
    def list_models(self) -> Dict:
        logger.debug("LLMService.list_models() -> delegating to factory")
        return self.factory.list_models()
    
    def get_default_model(self) -> unified_model_wrapper:
        logger.debug("LLMService.get_default_model() -> delegating to factory")
        return self.factory.get_default_model()
    
    def register_custom_model(
        self,
        model_id: str,
        name: str,
        provider: str,
        base_url: str,
        api_key_env: str,
        model_name: str,
        description: str = "",
        is_default: bool = False
    ) -> Dict:
        logger.debug(f"LLMService.register_custom_model({model_id}) -> delegating to factory")
        return self.factory.register_custom_model(
            model_id, name, provider, base_url,
            api_key_env, model_name, description, is_default
        )
    
    def remove_custom_model(self, model_id: str) -> Dict:
        logger.debug(f"LLMService.remove_custom_model({model_id}) -> delegating to factory")
        return self.factory.remove_custom_model(model_id)
    
    def refresh_configuration(self):
        logger.debug("LLMService.refresh_configuration() -> delegating to factory")
        self.factory.refresh_configuration()
    
    def get_providers(self) -> list:
        logger.debug("LLMService.get_providers() -> delegating to factory")
        return self.factory.get_providers()
    
    def validate_model_config(self, model_config: Dict) -> Dict:
        logger.debug("LLMService.validate_model_config() -> delegating to factory")
        return self.factory.validate_model_config(model_config)
    
    # ==================== SERVICE-LEVEL FEATURES (Future) ====================
    # These are examples of what you can add at the service layer
    # without touching the factory
    
    def get_model_with_logging(self, model_id: str, user_id: str = None):
        logger.info(f"User {user_id} requesting model {model_id}")
        model = self.factory.get_model(model_id)
        logger.info(f"Model {model_id} provided to user {user_id}")
        return model
    
    def get_model_with_rate_limit(self, model_id: str, user_id: str):
        return self.factory.get_model(model_id)
    
    def get_model_with_auth(self, model_id: str, user_id: str, api_key: str):
        # Verify API key (pseudo-code)
        # if not self._verify_api_key(user_id, api_key):
        #     raise AuthenticationError("Invalid API key")
        
        return self.factory.get_model(model_id)
    
    # ==================== HELPER METHODS ====================
    
    def get_factory(self) -> LLMFactory:
        return self.factory
    
    def clear_cache(self):
        """Clear model cache."""
        self.factory.clear_cache()
    def _build_fallback_model(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> UnifiedModelWrapper:
        """Build a ChatOpenAI fallback model from environment variables."""
        from langchain_openai import ChatOpenAI

        fb_model = os.getenv("FALLBACK_LLM_MODEL", "gpt-oss:20b")
        fb_api_key = os.getenv("FALLBACK_LLM_API_KEY", "ollama")
        fb_base_url = os.getenv("FALLBACK_LLM_BASE_URL", "http://10.188.100.130:8002/v1")
        fb_temp = temperature if temperature is not None else 0.2

        llm_client = ChatOpenAI(
            model=fb_model,
            api_key=fb_api_key,
            base_url=fb_base_url,
            temperature=fb_temp,
            max_tokens=max_tokens or 1000,
        )

        fallback_config = {
            "id": "fallback-chatopenai",
            "name": "ChatOpenAI Fallback",
            "provider": "ollama",
            "model_name": fb_model,
            "base_url": fb_base_url,
        }

        model = UnifiedModelWrapper(
            client=llm_client,
            embeddings_client=None,
            model_id="fallback-chatopenai",
            model_config=fallback_config,
        )
        logger.info(f"Fallback model created: {fb_model} @ {fb_base_url}")
        return model
    
    
    def get_default_model(self) -> UnifiedModelWrapper:
        """Get the default model instance."""
        default_id = self.config_loader.get_default_model_id()
        if not default_id:
            raise ModelNotFoundError("No default model configured")
        
        return self.get_model(default_id)
    
    
    
    def clear_cache(self):
        """Clear the model instance cache."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
    
    def refresh_configuration(self):
        """Reload configuration from file."""
        self.config_loader.refresh_config()
        self.clear_cache()
        logger.info("Configuration refreshed")
    
    



class TemplateRepository:
    def getAgentTemplate(self, name: str) -> AgentTemplate:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "agents"
        if name == "agentCard":
            template_path = data_dir / "agent-card.json"
        else:
            template_path = data_dir / "agent.json"

        if not template_path.exists():
            raise FileNotFoundError(f"Agent template not found at {template_path}")

        return AgentTemplate(name=name, template_path=str(template_path))
    def getWorkflowTemplate(self, name: str) -> Workflow:
        return Workflow(id=new_id('wf_'), name=name)

class LangGraphBuilder:
    def buildFromPrompt(self, prompt: str, template: AgentTemplate):
        return {'graph': 'built', 'prompt': prompt, 'template': template.name}
    def compile(self, graph) -> dict:
        return {'runnable': True}

class AgentService:
    """
    Enhanced AgentService with template matching and update detection.

    When createFromPrompt is called, the service:
    1. Uses LLM to analyze the user's intent (purpose, domain, keywords).
    2. Compares intent against predefined templates via keyword overlap.
    3. If a template matches, builds the agent from that template with user overrides.
    4. If no template matches, falls back to full LLM-powered generation via AgentDesigner.
    5. Registers the resulting agent in the AgentRegistry.

    Update Mode:
    - Detects modification keywords in prompt
    - Preserves agent name and ID when updating
    - Only modifies specified fields
    """

    # Minimum keyword overlap score to consider a template a match
    MATCH_THRESHOLD = 0.25  # Lowered for better fuzzy matching

    # Minimum similarity score to consider an existing agent a match
    AGENT_SIMILARITY_THRESHOLD = 0.25  # Lowered - we use smarter matching now

    # Keywords that indicate user wants to modify rather than create
    UPDATE_KEYWORDS = [
        "change", "modify", "update", "adjust", "edit", "alter",
        "add tool", "remove tool", "refine", "improve", "tweak",
        "fix", "enhance", "revise", "amend"
    ]

    # Tool selection rules: keyword patterns -> tool IDs
    # IMPORTANT: Keys must match actual backend tool IDs in apps/storage/tools/
    # This is synced with agent_designer.py TOOL_SELECTION_RULES
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
        ],
        # Calendar API
        "tool_calendar_api": [
            # Core Functions
            "calendar", "schedule", "scheduling", "availability", "available",
            "pto", "leave", "time off", "vacation", "sick leave", "personal day",
            "time-off", "absence", "out of office",

            # Team Coverage
            "team coverage", "coverage", "staffing", "team", "teams", "workforce",
            "headcount", "minimum coverage", "adequate coverage", "understaffed",
            "overstaffed", "skeleton crew",

            # Leave Types
            "vacation request", "sick day", "sick time", "medical leave",
            "bereavement", "bereavement leave", "parental leave", "maternity",
            "paternity", "family leave", "unpaid leave", "sabbatical",
            "personal leave", "emergency leave", "jury duty", "military leave",

            # Requests & Approval
            "leave request", "time off request", "request", "approval", "approve",
            "deny", "reject", "pending", "submitted", "needs approval",
            "manager approval", "auto-approve", "auto-approval",

            # Conflicts & Issues
            "conflict", "conflicts", "double booked", "overlap", "overlapping",
            "scheduling conflict", "availability conflict", "already scheduled",
            "not available", "unavailable",

            # Checking & Validation
            "check availability", "check coverage", "validate", "verification",
            "verify", "can I take off", "is available", "who is out", "who is off",
            "absence check",

            # Employee Management
            "employee", "employees", "staff", "team member", "team members",
            "colleague", "coworker", "manager", "supervisor", "direct report",
            "employee id", "staff member",

            # Dates & Periods
            "date", "dates", "date range", "period", "duration", "days off",
            "week", "start date", "end date", "from", "to", "until", "through",
            "today", "tomorrow", "next week", "this week", "month",

            # Balance & Tracking
            "pto balance", "remaining", "accrued", "days left", "how many days",
            "time off balance", "vacation days", "sick days", "used",
            "available days",

            # Recommendations
            "recommend", "recommendation", "should approve", "can approve",
            "suggest", "APPROVE", "DENY", "APPROVE_WITH_CAUTION", "flag",
            "review required",

            # Team Queries
            "who is available", "team schedule", "out today", "who's off",
            "absence list", "team calendar", "group schedule", "department schedule",
            "all hands",

            # Holidays & Events
            "holiday", "holidays", "company holiday", "public holiday",
            "blackout dates", "peak season", "busy period", "critical period",
            "freeze period"
        ],# OCR
        "tool_ocr": [
            # Core OCR Functions
            "ocr", "optical character recognition", "text extraction", "extract text",
            "read text", "scan", "scanning", "digitize", "digitization",
            "recognize text", "text recognition",

            # Receipt Processing
            "receipt", "receipts", "expense receipt", "purchase receipt",
            "sales receipt", "transaction receipt", "invoice", "bill", "ticket",
            "voucher", "proof of purchase",

            # Document Types
            "document", "documents", "image", "images", "photo", "photos",
            "picture", "scanned document", "pdf", "jpeg", "jpg", "png", "tiff",
            "scan copy",

            # Extraction Tasks
            "extract", "extraction", "parse", "parsing", "read", "reading",
            "capture", "get text from", "pull data", "data extraction",
            "information extraction",

            # Receipt Data Fields
            "merchant", "merchant name", "store", "vendor", "business name",
            "date", "purchase date", "transaction date", "time", "timestamp",
            "total", "amount", "price", "cost", "subtotal", "tax",
            "total amount", "items", "line items", "products", "purchases",
            "item list", "payment method", "credit card", "cash", "payment type",

            # Image Processing
            "image processing", "preprocessing", "enhance", "clean up",
            "improve quality", "denoise", "sharpen", "contrast", "brightness",
            "rotation", "deskew",

            # Quality & Confidence
            "confidence", "accuracy", "quality", "readable", "legibility",
            "clear", "blurry", "low quality", "poor quality", "unreadable",
            "fuzzy",

            # Batch Operations
            "batch", "multiple", "bulk", "mass processing", "batch processing",
            "several receipts", "many documents", "all receipts", "process all",

            # Text Regions
            "text region", "text block", "bounding box", "coordinates", "location",
            "detect text", "find text", "locate text", "text areas",

            # Tables & Structure
            "table", "tables", "tabular data", "rows", "columns", "grid",
            "structured data", "line items", "itemized", "list",

            # Expense Related
            "expense", "expenses", "spending", "purchase", "transaction",
            "reimbursement", "expense report", "submit expense", "claim",

            # Languages
            "english", "spanish", "french", "german", "multilingual", "language",
            "eng", "spa", "fra", "deu", "translation",

            # Upload & Input
            "upload", "uploaded", "attach", "attachment", "file", "image file",
            "camera", "photo upload", "snap", "picture of", "scan of"
        ],
        # Document Analysis
        "tool_document_analysis": [
            # Core Analysis
            "analyze", "analysis", "analyze receipt", "review", "check", "examine",
            "inspect", "validation", "validate", "verification", "verify",
            "assess", "evaluation",

            # Policy Compliance
            "policy", "policies", "compliance", "compliant", "non-compliant",
            "policy check", "company policy", "expense policy", "rules",
            "guidelines", "regulations", "within policy", "violates policy",
            "policy violation", "follows policy",

            # Approval & Decision
            "approve", "approval", "reject", "rejection", "deny", "accept",
            "auto-approve", "automatic approval", "manual review", "needs review",
            "flag", "flagged", "requires approval", "manager review",

            # Fraud Detection
            "fraud", "fraudulent", "suspicious", "anomaly", "unusual", "abnormal",
            "fraud detection", "fraud score", "risk score", "red flag", "warning",
            "tampered", "altered", "modified", "fake", "forged", "counterfeit",

            # Duplicate Detection
            "duplicate", "duplicates", "duplicate receipt", "already submitted",
            "resubmission", "same receipt", "submitted twice",
            "double submission", "repeat",

            # Categorization
            "category", "categorize", "classify", "classification", "type",
            "expense type", "meals", "transportation", "lodging",
            "office supplies", "travel", "entertainment", "hotel", "flight",
            "uber", "taxi", "restaurant", "gas", "fuel",

            # Amount Validation
            "amount", "amounts", "total", "price", "cost", "validate amount",
            "correct amount", "mismatch", "discrepancy", "doesn't match",
            "wrong total", "math error", "calculation error", "adds up",
            "subtotal", "tax amount",

            # Violations & Issues
            "violation", "violations", "issue", "issues", "problem", "problems",
            "exceeds", "over limit", "too high", "too much", "exceeded",
            "missing", "incomplete", "required", "mandatory",

            # Merchant Verification
            "merchant", "vendor", "business", "store", "verify merchant",
            "known merchant", "approved vendor", "blacklist", "blacklisted",
            "unauthorized", "not allowed",

            # Expense Limits
            "limit", "limits", "cap", "maximum", "threshold", "ceiling",
            "meal limit", "daily limit", "single expense", "per diem",
            "allowance", "budget", "over budget",

            # Receipt Requirements
            "receipt required", "needs receipt", "must have receipt",
            "documentation", "proof", "evidence", "supporting document",
            "backup",

            # Date Validation
            "date", "old", "expired", "too old", "stale", "outdated",
            "recent", "timely", "within timeframe", "submission deadline",
            "days old", "age", "expense age",

            # Risk Assessment
            "risk", "risky", "high risk", "low risk", "confidence", "score",
            "likelihood", "probability", "suspicious pattern", "red flag",

            # Decision Outcomes
            "compliant", "non-compliant", "requires review", "flagged for review",
            "auto-approve", "deny", "reject", "investigate", "escalate",

            # Comparison & Analysis
            "compare", "comparison", "versus", "against", "historical", "typical",
            "average", "normal", "expected", "benchmark", "standard",

            # Recommendations
            "recommendation", "suggest", "advice", "guidance", "action",
            "should approve", "should deny", "review needed", "acceptable"
        ],
        # Code Executor
        "tool_code_executor": [
            # Core Execution
            "execute", "run", "execute code", "run code", "code execution",
            "execute script", "run python", "run analysis", "compute", "calculate",
            "process",

            # Performance Review
            "performance", "performance review", "annual review", "review cycle",
            "evaluation", "appraisal", "assessment", "employee review",
            "staff review", "performance appraisal",

            # Review Components
            "self-assessment", "self assessment", "self evaluation", "self review",
            "peer feedback", "peer review", "360 review", "360 feedback",
            "colleague feedback", "manager review", "supervisor review",
            "manager feedback", "manager assessment", "goals", "goal setting",
            "objectives", "targets", "kpis", "key results",

            # Analysis Types
            "analyze", "analysis", "insights", "generate insights", "analytics",
            "comparison", "compare", "gap analysis", "sentiment analysis",
            "sentiment", "strengths", "weaknesses", "strengths and weaknesses",
            "swot", "goal alignment", "alignment", "correlation", "trends",

            # Data Processing
            "data", "data analysis", "process data", "analyze data",
            "crunch numbers", "statistics", "statistical analysis",
            "metrics", "measurement", "quantitative",

            # Python & Code
            "python", "code", "script", "algorithm", "function",
            "program", "pandas", "numpy", "analysis code", "data science",
            "ml", "machine learning",

            # Insights Generation
            "insights", "findings", "results", "conclusions", "observations",
            "patterns", "themes", "highlights", "key findings", "takeaways",
            "recommendations", "suggestions", "advice", "guidance",

            # Comparison & Benchmarking
            "compare to", "versus", "benchmark", "peer comparison",
            "relative to", "ranking", "percentile", "quartile",
            "top performer", "bottom performer", "above average",
            "below average", "industry standard",

            # Performance Metrics
            "rating", "ratings", "score", "scores", "performance score",
            "competency", "competencies", "skills", "capabilities",
            "proficiency", "achievement", "accomplishment",
            "results", "outcomes", "impact",

            # Employee Data
            "employee", "employee id", "staff member", "team member",
            "individual", "employee data", "performance data",
            "review data", "feedback data",

            # Output & Reports
            "report", "summary", "overview", "dashboard", "visualization",
            "chart", "graph", "plot", "export", "generate report",
            "json", "markdown", "html", "formatted output",

            # Comprehensive Analysis
            "comprehensive", "complete analysis", "full analysis", "detailed",
            "in-depth", "thorough", "holistic", "end-to-end",

            # HR & People Analytics
            "hr", "human resources", "people analytics", "workforce analytics",
            "talent management", "performance management",
            "employee development",

            # Time Periods
            "annual", "yearly", "quarterly", "review period",
            "review cycle", "this year", "last year",
            "year over year", "yoy", "period",

            # Improvement & Growth
            "improvement", "growth", "development", "progress",
            "advancement", "areas for improvement", "development areas",
            "growth opportunities", "coaching", "mentoring",
            "training needs",

            # Execution Context
            "timeout", "execution time", "memory", "performance",
            "optimize", "sandbox", "isolated", "safe execution",
            "resource limits"
        ]
    }

    # Stop words to exclude from similarity matching in agent-building context
    # These words are too generic and cause false positives
    AGENT_STOP_WORDS = {
        "agent", "agents", "assistant", "assistants", "helper", "helpers",
        "bot", "bots", "ai", "system", "systems", "tool", "tools",
        "create", "creating", "build", "building", "make", "making",
        "help", "helping", "want", "need", "please", "would", "could",
        "the", "a", "an", "for", "me", "my", "that", "will", "do", "does"
    }

    # Regex pattern to detect generic agent names like "Agent 1", "Agent 2", etc.
    # These should be skipped for similarity matching
    GENERIC_AGENT_NAME_PATTERN = r'^agent\s*\d+$'

    def __init__(
        self,
        tpl_repo: TemplateRepository,
        graph_builder: LangGraphBuilder,
        cred: ICredentialStore | None = None,
        log: ILogger | None = None,
        registry=None,
        designer=None
    ):
        self.tpl_repo = tpl_repo
        self.graph_builder = graph_builder
        self.agents: dict[str, Agent] = {}
        self.log = log
        self._registry = registry
        self._designer = designer
        self._templates_cache: Optional[List[Dict[str, Any]]] = None

    def _load_templates(self) -> List[Dict[str, Any]]:
        """
        Load predefined agent templates from agent_templates.json.
        Caches after first load.

        Returns:
            List of template dicts from the JSON file.
        """
        if self._templates_cache is not None:
            return self._templates_cache

        templates_path = (
            Path(__file__).parent.parent / "apps" / "storage" / "agent_templates.json"
        )

        if not templates_path.exists():
            logger.warning(f"Templates file not found at {templates_path}")
            self._templates_cache = []
            return self._templates_cache

        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._templates_cache = data.get("templates", [])
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load templates: {e}")
            self._templates_cache = []

        return self._templates_cache

    def _get_llm(self):
        """
        Get an LLM instance via LLMManager for intent analysis.
        Uses low temperature for deterministic extraction.
        """
        from llm_manager import LLMManager
        return LLMManager.get_llm(temperature=0.1, max_tokens=1000)

    def _analyze_intent(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to extract intent keywords from the user prompt.

        Args:
            prompt: The user's natural language agent description.

        Returns:
            Dict with keys: purpose, domain, keywords, matching_roles.
            On failure, returns a basic extraction based on the raw prompt.
        """
        system_prompt = """You are an intent analyzer for an AI agent builder platform.
Given a user's prompt describing an agent they want to create, extract structured intent information.

Return a JSON object with EXACTLY this structure:
{
  "purpose": "one-line purpose of the requested agent",
  "domain": "primary domain (e.g., research, sales, support, data, content, code, hr, finance, project)",
  "keywords": ["list", "of", "relevant", "keywords", "from", "the", "prompt"],
  "matching_roles": ["possible", "role", "titles", "that", "fit"]
}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation."""

        try:
            llm = self._get_llm()
            full_prompt = f"{system_prompt}\n\nUser prompt: {prompt}\n\nJSON:"
            response = llm.invoke(full_prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Strip markdown fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            intent = json.loads(content)

            # Validate expected structure
            if not isinstance(intent, dict):
                raise ValueError("LLM returned non-dict JSON")

            # Ensure required keys exist with defaults
            intent.setdefault("purpose", prompt[:100])
            intent.setdefault("domain", "general")
            intent.setdefault("keywords", [])
            intent.setdefault("matching_roles", [])

            return intent

        except Exception as e:
            logger.warning(f"LLM intent analysis failed, using basic extraction: {e}")
            return self._basic_intent_extraction(prompt)

    def _basic_intent_extraction(self, prompt: str) -> Dict[str, Any]:
        """
        Fallback intent extraction using simple keyword parsing.
        Used when LLM call fails.
        """
        prompt_lower = prompt.lower()

        # Domain keyword mappings
        domain_keywords = {
            "research": ["research", "analyze", "investigate", "study", "explore", "literature"],
            "support": ["support", "customer", "help", "assist", "service", "inquiry"],
            "data": ["data", "analytics", "statistics", "visualization", "dataset", "insights"],
            "content": ["content", "write", "blog", "article", "copy", "creative", "writing"],
            "code": ["code", "review", "programming", "developer", "software", "debug", "python"],
            "project": ["project", "manage", "coordinate", "timeline", "task", "planning"],
            "sales": ["sales", "lead", "prospect", "customer", "product", "revenue"],
            "hr": ["hr", "human resources", "employee", "hiring", "recruitment", "onboarding"],
            "finance": ["finance", "financial", "budget", "revenue", "accounting", "report"],
        }

        # Detect domain
        detected_domain = "general"
        max_matches = 0
        for domain, kws in domain_keywords.items():
            matches = sum(1 for kw in kws if kw in prompt_lower)
            if matches > max_matches:
                max_matches = matches
                detected_domain = domain

        # Extract keywords from prompt (simple word tokenization)
        words = [w.strip(".,!?;:()[]{}\"'") for w in prompt_lower.split()]
        stop_words = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of",
                      "and", "or", "in", "on", "at", "for", "with", "that", "this",
                      "i", "want", "need", "create", "make", "build", "me", "my"}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]

        return {
            "purpose": prompt[:100],
            "domain": detected_domain,
            "keywords": keywords[:15],
            "matching_roles": []
        }

    def _match_template(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compare analyzed intent against all predefined templates.

        Uses keyword overlap scoring across template name, role, and description.
        Returns the best matching template if score exceeds MATCH_THRESHOLD.

        Args:
            intent: Dict from _analyze_intent with purpose, domain, keywords, matching_roles.

        Returns:
            Matched template dict, or None if no sufficient match found.
        """
        templates = self._load_templates()
        if not templates:
            return None

        intent_keywords = set(
            kw.lower() for kw in intent.get("keywords", [])
        )
        intent_domain = intent.get("domain", "").lower()
        intent_roles = set(
            r.lower() for r in intent.get("matching_roles", [])
        )
        intent_purpose = intent.get("purpose", "").lower()

        # Add domain to keyword set for matching
        if intent_domain:
            intent_keywords.add(intent_domain)

        # Add purpose words to keyword set
        purpose_words = {
            w.strip(".,!?;:()[]{}\"'") for w in intent_purpose.split()
            if len(w) > 3
        }
        intent_keywords.update(purpose_words)

        best_template = None
        best_score = 0.0

        for template in templates:
            score = self._score_template(template, intent_keywords, intent_roles)
            if score > best_score:
                best_score = score
                best_template = template

        if best_score >= self.MATCH_THRESHOLD and best_template is not None:
            logger.info(
                f"Template match found: '{best_template.get('name')}' "
                f"(score: {best_score:.2f})"
            )
            return best_template

        logger.info(f"No template match found (best score: {best_score:.2f})")
        return None

    def _score_template(
        self,
        template: Dict[str, Any],
        intent_keywords: set,
        intent_roles: set
    ) -> float:
        """
        Score a template against the intent keywords.

        Scoring factors:
        - Keyword overlap with template name, role, description
        - Role match bonus
        - Tool keyword overlap

        Returns:
            Float score between 0.0 and 1.0.
        """
        if not intent_keywords:
            return 0.0

        # Build template keyword set from name, role, description
        template_text = " ".join([
            template.get("name", ""),
            template.get("role", ""),
            template.get("description", ""),
        ]).lower()

        template_words = {
            w.strip(".,!?;:()[]{}\"'") for w in template_text.split()
            if len(w) > 2
        }

        # Add tool names as keywords
        for tool in template.get("tools", []):
            template_words.update(
                w.lower() for w in tool.split() if len(w) > 2
            )

        # Calculate keyword overlap (Jaccard-like)
        if not template_words:
            return 0.0

        intersection = intent_keywords & template_words
        union = intent_keywords | template_words
        keyword_score = len(intersection) / len(union) if union else 0.0

        # Role match bonus: check if template role matches any intent roles
        role_bonus = 0.0
        template_role = template.get("role", "").lower()
        if template_role:
            for intent_role in intent_roles:
                if (intent_role in template_role) or (template_role in intent_role):
                    role_bonus = 0.25
                    break

        # Name match bonus
        name_bonus = 0.0
        template_name = template.get("name", "").lower()
        for kw in intent_keywords:
            if kw in template_name:
                name_bonus = 0.15
                break

        # Combined score (capped at 1.0)
        total = min(1.0, keyword_score + role_bonus + name_bonus)
        return total

    def createFromPrompt(self, prompt: str, template: AgentTemplate) -> Agent:
        """
        Create an agent from a natural language prompt with template matching.

        Flow:
        1. Analyze intent using LLM.
        2. Attempt template match.
        3. If matched: build from template with user prompt overrides.
        4. If not matched: use AgentDesigner for full LLM generation.
        5. Register in AgentRegistry.
        6. Return the Agent.

        Args:
            prompt: User's natural language description of the desired agent.
            template: AgentTemplate with optional overrides.

        Returns:
            Fully populated Agent instance.
        """
        try:
            # Step 1: Analyze intent
            intent = self._analyze_intent(prompt)
            logger.info(f"Intent analysis: domain={intent.get('domain')}, "
                        f"keywords={intent.get('keywords', [])[:5]}")

            # Step 2: Attempt template match
            matched_template = self._match_template(intent)

            if matched_template:
                # Step 3a: Build from matched template
                agent = self._build_from_template(
                    matched_template, prompt, template, intent
                )
            else:
                # Step 3b: Fall back to LLM generation via AgentDesigner
                agent = self._build_from_llm(prompt, template)

        except Exception as e:
            logger.error(f"Agent creation failed, using minimal fallback: {e}")
            # Ultimate fallback: create a basic agent
            agent = Agent(
                id=new_id('agt_'),
                name=template.name or "Agent",
                description=prompt[:200],
                metadata={
                    "source": "fallback",
                    "created_at": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            )

        # Store internally
        self.agents[agent.id] = agent

        # Register in AgentRegistry if available
        self._register_agent(agent)

        return agent

    def _build_from_template(
        self,
        matched_template: Dict[str, Any],
        prompt: str,
        user_template: AgentTemplate,
        intent: Dict[str, Any]
    ) -> Agent:
        """
        Build an Agent from a matched predefined template, applying user overrides.

        Args:
            matched_template: The template dict from agent_templates.json.
            prompt: Original user prompt (used for description customization).
            user_template: User-provided AgentTemplate with optional overrides.
            intent: Analyzed intent dict.

        Returns:
            Fully populated Agent.
        """
        agent_id = new_id('agt_')
        timestamp = datetime.utcnow().isoformat()

        # Use user template overrides if provided, else fall back to matched template
        name = user_template.name if user_template.name else matched_template.get("name", "Agent")
        icon = user_template.icon if user_template.icon else matched_template.get("icon", "")
        role = user_template.role if user_template.role else matched_template.get("role", "Processing")
        description = (
            user_template.description if user_template.description
            else matched_template.get("description", prompt[:200])
        )
        agent_prompt = (
            user_template.prompt if user_template.prompt
            else matched_template.get("prompt", prompt)
        )

        # Tools: user override > template tools > auto-select based on intent
        tools = user_template.tools if user_template.tools else matched_template.get("tools", [])
        if not tools:
            # Auto-select tools based on prompt/intent if neither provided
            tools = self._auto_select_tools(prompt, intent)

        variables = (
            user_template.variables if user_template.variables
            else matched_template.get("variables", [])
        )
        settings = (
            user_template.settings if user_template.settings
            else matched_template.get("settings", {})
        )

        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            tools=tools,
            metadata={
                "source": "template",
                "template_name": matched_template.get("name"),
                "icon": icon,
                "prompt": agent_prompt,
                "variables": variables,
                "settings": settings,
                "original_prompt": prompt,
                "intent_domain": intent.get("domain", "general"),
                "created_by": "agent_service",
                "created_at": timestamp,
            }
        )

        return agent

    def _build_from_llm(self, prompt: str, template: AgentTemplate) -> Agent:
        """
        Build an Agent using AgentDesigner for full LLM-powered generation.

        Args:
            prompt: User's natural language description.
            template: AgentTemplate with optional overrides.

        Returns:
            Agent built from LLM-designed spec.
        """
        if self._designer is None:
            # Import and create designer if not injected
            from apps.agent.designer.agent_designer import AgentDesigner
            self._designer = AgentDesigner()

        # Use AgentDesigner to generate full agent spec
        # Pass None for tools if not provided to trigger auto-selection
        agent_dict = self._designer.design_from_prompt(
            user_prompt=prompt,
            default_model="openrouter-devstral",
            icon=template.icon or "",
            tools=template.tools if template.tools else None,
            variables=template.variables or []
        )

        # Tag as LLM-generated
        if "metadata" not in agent_dict:
            agent_dict["metadata"] = {}
        agent_dict["metadata"]["source"] = "llm_generated"
        agent_dict["metadata"]["original_prompt"] = prompt

        # Convert dict to Agent model
        agent = Agent(
            id=agent_dict.get("agent_id", new_id('agt_')),
            name=agent_dict.get("name", template.name or "Agent"),
            role=agent_dict.get("role"),
            description=agent_dict.get("description"),
            tools=agent_dict.get("tools"),
            metadata=agent_dict.get("metadata")
        )

        return agent

    def _register_agent(self, agent: Agent) -> None:
        """
        Register the agent in the AgentRegistry for persistence.

        Args:
            agent: The Agent to register.
        """
        if self._registry is None:
            return

        try:
            # Build registry-compatible dict
            agent_dict = {
                "agent_id": agent.id,
                "name": agent.name,
                "role": agent.role or "Processing",
                "description": agent.description or "",
                "icon": (agent.metadata or {}).get("icon", ""),
                "prompt": (agent.metadata or {}).get("prompt", ""),
                "tools": agent.tools or [],
                "variables": (agent.metadata or {}).get("variables", []),
                "settings": (agent.metadata or {}).get("settings", {}),
                "input_schema": agent.input_schema or [],
                "output_schema": agent.output_schema or [],
                "metadata": agent.metadata or {},
            }
            self._registry.register_agent(agent_dict)
            logger.info(f"Agent {agent.id} registered in registry")
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")

    def _auto_select_tools(self, prompt: str, intent: Dict[str, Any] = None) -> list:
        """
        Auto-select appropriate tools based on prompt and intent keywords.

        Uses keyword matching to determine which tools are relevant for the agent.
        Maximum of 2 tools will be selected.

        Args:
            prompt: The user's natural language prompt
            intent: Optional pre-analyzed intent dict with keywords

        Returns:
            List of tool IDs (max 2), empty list if no clear match
        """
        prompt_lower = prompt.lower()

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

        # Score each tool based on keyword matches
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

        if not tool_scores:
            return []

        # Sort by score and take top 2
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool_id for tool_id, score in sorted_tools[:2]]

        return selected_tools

    def _detect_update_intent(self, prompt: str) -> bool:
        """
        Detect if the user's prompt indicates modification of an existing agent
        rather than creation of a new one.

        Args:
            prompt: User's natural language prompt

        Returns:
            True if prompt indicates update/modification intent
        """
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.UPDATE_KEYWORDS)

    def classify_user_intent(
        self,
        context: str,
        suggested_value: str,
        user_message: str,
        conversation_history: list = None
    ) -> Dict[str, Any]:
        """
        Classify user intent using LLM reasoning.

        This method enables natural language understanding for conversational flows.
        Instead of pattern matching, it uses LLM to understand what the user means.

        Args:
            context: The conversation context (e.g., "name_confirmation", "refinement")
            suggested_value: The value being confirmed/modified (e.g., agent name)
            user_message: The user's natural language response
            conversation_history: Optional list of previous messages

        Returns:
            Dict with:
            - intent: CONFIRMATION | MODIFICATION | REJECTION | CLARIFICATION
            - confidence: float between 0 and 1
            - reasoning: explanation of the classification
            - extracted_value: new value if intent is MODIFICATION
        """
        system_prompt = """You are an intent classifier for an AI agent builder conversation.

Your task is to analyze the user's message and classify their intent.

CONTEXT TYPES:
- name_confirmation: User is responding to "Would you like to use this name?"
- refinement: User is responding to "Any changes or are we ready to finalize?"
- tool_selection: User is responding to tool configuration
- general: General conversation

INTENT TYPES:
1. CONFIRMATION - User is accepting/approving the suggested value
   Examples: "yes", "ok", "keep this name", "sounds good", "that's perfect", "love it", "this name is great"

2. MODIFICATION - User wants to change/replace the value
   Examples: "call it X instead", "change to Y", "I prefer Z", "rename it to..."

3. REJECTION - User is declining/refusing
   Examples: "no", "I don't like it", "start over", "cancel"

4. CLARIFICATION - User is asking a question or needs more info
   Examples: "what does this mean?", "can you explain?", "what options do I have?"

CRITICAL RULES:
- If user expresses ANY form of approval, agreement, satisfaction, or acceptance -> CONFIRMATION
- Natural language variations like "oh yes this is great", "keep this name", "this name is the best" -> CONFIRMATION
- Only classify as MODIFICATION if user EXPLICITLY provides a new value or asks to change
- When in doubt between CONFIRMATION and MODIFICATION, prefer CONFIRMATION if no new value is provided

Return JSON only:
{
  "intent": "CONFIRMATION" | "MODIFICATION" | "REJECTION" | "CLARIFICATION",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "extracted_value": null or "the new value if MODIFICATION"
}"""

        user_prompt = f"""Context: {context}
Suggested value: "{suggested_value}"
User message: "{user_message}"

Classify the user's intent. Return JSON only."""

        try:
            llm = self._get_llm()
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = llm.invoke(full_prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # Strip markdown fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Validate and ensure required fields
            valid_intents = ["CONFIRMATION", "MODIFICATION", "REJECTION", "CLARIFICATION"]
            if result.get("intent") not in valid_intents:
                result["intent"] = "CONFIRMATION"  # Safe default

            result.setdefault("confidence", 0.8)
            result.setdefault("reasoning", "Intent classified by LLM")
            result.setdefault("extracted_value", None)

            logger.info(f"Intent classified: {result['intent']} (confidence: {result['confidence']})")
            return result

        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            # Fallback: simple heuristic
            return self._fallback_intent_classification(user_message, suggested_value)

    def _fallback_intent_classification(self, user_message: str, suggested_value: str) -> Dict[str, Any]:
        """
        Fallback intent classification when LLM is unavailable.
        Uses simple heuristics as last resort.
        """
        msg_lower = user_message.lower().strip()

        # Check for obvious rejections
        rejection_words = ["no", "nope", "cancel", "stop", "quit", "nevermind", "never mind"]
        if any(msg_lower == word or msg_lower.startswith(word + " ") for word in rejection_words):
            return {
                "intent": "REJECTION",
                "confidence": 0.7,
                "reasoning": "Fallback: detected rejection keyword",
                "extracted_value": None
            }

        # Check for questions
        if "?" in user_message or any(msg_lower.startswith(q) for q in ["what", "how", "why", "can", "could", "would"]):
            return {
                "intent": "CLARIFICATION",
                "confidence": 0.6,
                "reasoning": "Fallback: detected question pattern",
                "extracted_value": None
            }

        # Check for modification indicators
        modification_patterns = ["call it", "name it", "rename", "change to", "change it to", "prefer", "instead"]
        for pattern in modification_patterns:
            if pattern in msg_lower:
                # Try to extract new value
                extracted = user_message
                for p in modification_patterns:
                    if p in msg_lower:
                        parts = msg_lower.split(p)
                        if len(parts) > 1:
                            extracted = parts[1].strip().strip("\"'")
                            break
                return {
                    "intent": "MODIFICATION",
                    "confidence": 0.7,
                    "reasoning": "Fallback: detected modification pattern",
                    "extracted_value": extracted if extracted != user_message else None
                }

        # Default: treat as confirmation (most common case)
        return {
            "intent": "CONFIRMATION",
            "confidence": 0.6,
            "reasoning": "Fallback: no rejection/modification detected, assuming confirmation",
            "extracted_value": None
        }

    def updateFromPrompt(
        self,
        agent_id: str,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Update an existing agent based on user prompt.

        This method preserves the agent's name and ID while only updating
        the fields specified in the user's prompt.

        Args:
            agent_id: ID of the agent to update
            prompt: User's natural language update request

        Returns:
            Dict with action type and updated agent definition

        Raises:
            ValueError: If agent not found
        """
        if self._registry is None:
            raise ValueError("Registry not available")

        # Get existing agent
        existing_agent = self._registry.get_agent(agent_id)
        if not existing_agent:
            raise ValueError(f"Agent '{agent_id}' not found")

        # Use designer to generate updates while preserving identity
        if self._designer is None:
            from apps.agent.designer.agent_designer import AgentDesigner
            self._designer = AgentDesigner()

        # Call update_from_prompt which preserves name and ID
        updated_agent = self._designer.update_from_prompt(
            existing_agent=existing_agent,
            user_prompt=prompt
        )

        # Save updated agent to registry
        self._registry.update_agent(agent_id, updated_agent)

        logger.info(f"Agent {agent_id} updated via prompt")

        return {
            "action": "UPDATE_AGENT",
            "agent_id": agent_id,
            "agent_name": updated_agent.get("name"),
            "agent": updated_agent
        }

    def _normalize_word(self, word: str) -> str:
        """
        Simple word normalization to handle common variations.
        Maps related words to a common root for better matching.
        """
        word = word.lower().strip(".,!?;:()[]{}\"'")

        # Common word family mappings
        normalizations = {
            # Analysis family
            "analyst": "analy", "analysis": "analy", "analyze": "analy",
            "analyzes": "analy", "analyzing": "analy", "analytical": "analy",
            # Finance family
            "financial": "financ", "finance": "financ", "finances": "financ",
            # Research family
            "research": "research", "researcher": "research", "researching": "research",
            # Code family
            "code": "code", "coding": "code", "coder": "code", "coded": "code",
            # Data family
            "data": "data", "dataset": "data", "datasets": "data",
            # Support family
            "support": "support", "supporting": "support", "supporter": "support",
            # Content family
            "content": "content", "contents": "content",
            # Write family
            "write": "write", "writer": "write", "writing": "write", "written": "write",
            # Project family
            "project": "project", "projects": "project",
            # Manage family
            "manage": "manag", "manager": "manag", "management": "manag", "managing": "manag",
            # Sales family
            "sales": "sale", "sale": "sale", "selling": "sale",
            # Customer family
            "customer": "custom", "customers": "custom",
            # Review family
            "review": "review", "reviewer": "review", "reviewing": "review", "reviews": "review",
            # HR family
            "hr": "hr", "human": "human", "resources": "resource", "resource": "resource",
        }

        return normalizations.get(word, word)

    def _score_template_match(self, template: Dict[str, Any], user_prompt: str, intent_keywords: set) -> float:
        """
        Score how well a template matches the user's prompt using multiple strategies.

        Scoring:
        1. Direct name match in prompt: HIGH score (0.8+)
        2. All name words found in prompt: HIGH score (0.7+)
        3. Normalized word overlap: MEDIUM score
        4. Jaccard similarity: baseline

        Returns score between 0.0 and 1.0
        """
        template_name = template.get("name", "").lower()
        template_role = template.get("role", "").lower()
        user_prompt_lower = user_prompt.lower()

        # PRE-CHECK: Skip generic agent names like "Agent 1", "Agent 2", "Agent 3"
        # These should not be used for similarity matching as they're placeholder names
        if re.match(self.GENERIC_AGENT_NAME_PATTERN, template_name, re.IGNORECASE):
            logger.debug(f"Skipping generic agent name: '{template_name}'")
            return 0.0

        # Strategy 1: Direct name match (template name appears in prompt)
        # e.g., "financial analyst" in "create a financial analyst agent"
        # But skip if the name is just a stop word
        if template_name and template_name in user_prompt_lower:
            # Verify it's not just matching stop words
            name_significant_words = [w for w in template_name.split()
                                      if w not in self.AGENT_STOP_WORDS and len(w) > 2]
            if len(name_significant_words) >= 1:
                return 0.95  # Near-perfect match

        # Strategy 2: All significant words from template name appear in prompt
        # IMPORTANT: Filter out stop words to avoid false positives like "Agent 3" matching any prompt with "agent"
        name_words = [w.strip() for w in template_name.split()
                      if len(w) > 2 and w.strip() not in self.AGENT_STOP_WORDS]

        if name_words:
            # Check if all name words (or their normalized forms) appear in prompt
            # Also filter stop words from prompt to avoid matching on common words
            prompt_words = {w for w in user_prompt_lower.split()
                          if w not in self.AGENT_STOP_WORDS}
            prompt_normalized = {self._normalize_word(w) for w in prompt_words}

            name_matches = 0
            for nw in name_words:
                nw_normalized = self._normalize_word(nw)
                # Check exact match or normalized match
                if nw in prompt_words or nw_normalized in prompt_normalized:
                    name_matches += 1
                # Check partial match (word contains or is contained)
                elif any(nw in pw or pw in nw for pw in prompt_words if len(pw) > 3):
                    name_matches += 0.7

            # Require at least 2 significant words for high-confidence match
            # This prevents single-word matches from getting 0.85 score
            if len(name_words) >= 2:
                name_match_ratio = name_matches / len(name_words)
                if name_match_ratio >= 0.8:
                    return 0.85
                elif name_match_ratio >= 0.5:
                    return 0.6 + (name_match_ratio * 0.2)
            elif len(name_words) == 1 and name_matches >= 1:
                # Single significant word match gets lower score (needs role/description confirmation)
                # This prevents "Agent 3" (after filtering) from matching everything
                return 0.4  # Lower score for single-word matches

        # Strategy 3: Role match (also filter stop words)
        if template_role:
            role_words = [w.strip() for w in template_role.split()
                         if len(w) > 2 and w.strip() not in self.AGENT_STOP_WORDS]
            prompt_words_filtered = {w for w in user_prompt_lower.split()
                                    if w not in self.AGENT_STOP_WORDS}
            role_matches = sum(1 for rw in role_words if rw in prompt_words_filtered or
                             self._normalize_word(rw) in {self._normalize_word(w) for w in prompt_words_filtered})
            if role_words and role_matches / len(role_words) >= 0.5:
                return 0.5 + (role_matches / len(role_words) * 0.3)

        # Strategy 4: Normalized keyword overlap (fallback, also filter stop words)
        template_text = f"{template_name} {template_role} {template.get('description', '')}".lower()
        template_words = {self._normalize_word(w) for w in template_text.split()
                        if len(w) > 2 and w not in self.AGENT_STOP_WORDS}
        intent_normalized = {self._normalize_word(kw) for kw in intent_keywords
                           if kw.lower() not in self.AGENT_STOP_WORDS}

        if template_words and intent_normalized:
            intersection = template_words & intent_normalized
            union = template_words | intent_normalized
            jaccard = len(intersection) / len(union) if union else 0.0
            return jaccard * 0.8  # Scale Jaccard to max 0.8

        return 0.0

    def _check_existing_agents(self, intent: Dict[str, Any], user_prompt: str = "") -> Optional[Dict[str, Any]]:
        """
        Check if a semantically similar agent already exists in templates or registry.

        Uses smart matching that handles word variations (analyst/analysis/analyze).
        First checks predefined templates, then the registry.

        Args:
            intent: Analyzed intent dict with keywords, domain, purpose
            user_prompt: Original user prompt for direct matching

        Returns:
            Dict with matching agent/template info if found, None otherwise
        """
        # Build intent keyword set
        intent_keywords = set(kw.lower() for kw in intent.get("keywords", []))
        intent_domain = intent.get("domain", "").lower()
        intent_purpose = intent.get("purpose", "").lower()

        # Use purpose as prompt if not provided
        if not user_prompt:
            user_prompt = intent_purpose

        # Add domain and purpose words to keywords
        if intent_domain:
            intent_keywords.add(intent_domain)
        purpose_words = {
            w.strip(".,!?;:()[]{}\"'") for w in intent_purpose.split()
            if len(w) > 3
        }
        intent_keywords.update(purpose_words)

        if not intent_keywords and not user_prompt:
            return None

        # --- Step 1: Check templates FIRST ---
        templates = self._load_templates()
        best_template = None
        best_template_score = 0.0

        for template in templates:
            score = self._score_template_match(template, user_prompt, intent_keywords)

            if score > best_template_score:
                best_template_score = score
                best_template = template

        # Match threshold: 0.4 for smart matching (lower than before since scoring is smarter)
        if best_template_score >= 0.4 and best_template is not None:
            logger.info(
                f"Template match found: '{best_template.get('name')}' "
                f"(score: {best_template_score:.2f})"
            )
            # Auto-select tools based on template purpose to get proper tool IDs
            template_text = f"{best_template.get('name', '')} {best_template.get('role', '')} {best_template.get('description', '')}"
            auto_tools = self._auto_select_tools(template_text, None)

            # Create a copy of the template with auto-selected tool IDs
            template_with_tools = dict(best_template)
            template_with_tools["tools"] = auto_tools

            # Return template as pseudo-agent with correct tool IDs
            return {
                "action": "AGENT_EXISTS",
                "agent_id": best_template.get("id", f"tpl_{best_template.get('name', 'unknown').lower().replace(' ', '_')}"),
                "agent_name": best_template.get("name"),
                "similarity_score": round(best_template_score, 2),
                "message": "A similar agent template already exists. You can configure or modify it.",
                "agent": template_with_tools,
                "source": "template"
            }

        # --- Step 2: Check registry ---
        if self._registry is None:
            return None

        existing_agents = self._registry.list_agents()
        if not existing_agents:
            return None

        best_match = None
        best_score = 0.0

        for agent in existing_agents:
            # Build a pseudo-template from agent for scoring
            agent_as_template = {
                "name": agent.get("name", ""),
                "role": agent.get("role", ""),
                "description": agent.get("description", "")
            }
            score = self._score_template_match(agent_as_template, user_prompt, intent_keywords)

            if score > best_score:
                best_score = score
                best_match = agent

        if best_score >= 0.4 and best_match is not None:
            logger.info(
                f"Similar agent found: '{best_match.get('name')}' "
                f"(similarity: {best_score:.2f})"
            )
            return {
                "action": "AGENT_EXISTS",
                "agent_id": best_match.get("agent_id"),
                "agent_name": best_match.get("name"),
                "similarity_score": round(best_score, 2),
                "message": "A similar agent already exists. You can configure or modify it.",
                "agent": best_match,
                "source": "registry"
            }

        return None

    def createFromCanvasCard(self, cardJSON: dict, template: AgentTemplate) -> Agent:
        """
        Build a proper Agent from canvas card JSON (template data).

        Maps all template fields into the Agent structure and applies
        user-provided overrides from the template parameter.

        Args:
            cardJSON: Dict with template/card data (name, icon, role, description,
                      prompt, tools, variables, settings).
            template: AgentTemplate with optional overrides.

        Returns:
            Fully populated Agent.
        """
        agent_id = new_id('agt_')
        timestamp = datetime.utcnow().isoformat()

        # Resolve each field: user override > card data > default
        name = template.name if template.name else cardJSON.get("name", "Agent")
        icon = template.icon if template.icon else cardJSON.get("icon", "")
        role = template.role if template.role else cardJSON.get("role", "Processing")
        description = (
            template.description if template.description
            else cardJSON.get("description", "")
        )
        agent_prompt = (
            template.prompt if template.prompt
            else cardJSON.get("prompt", "")
        )
        tools = (
            template.tools if template.tools
            else cardJSON.get("tools", [])
        )
        variables = (
            template.variables if template.variables
            else cardJSON.get("variables", [])
        )
        settings = (
            template.settings if template.settings
            else cardJSON.get("settings", {})
        )

        agent = Agent(
            id=agent_id,
            name=name,
            role=role,
            description=description,
            tools=tools,
            metadata={
                "source": cardJSON.get("source", "canvas_card"),
                "icon": icon,
                "prompt": agent_prompt,
                "variables": variables,
                "settings": settings,
                "created_by": "agent_service",
                "created_at": timestamp,
            }
        )

        # Store and register
        self.agents[agent.id] = agent
        self._register_agent(agent)

        return agent

    def validateA2A(self, agent: Agent) -> ValidationResult:
        """Validate agent-to-agent communication compatibility."""
        return ValidationResult(ok=True)

    def listAgents(self) -> List[Agent]:
        """List all agents created by this service instance."""
        return list(self.agents.values())

class WorkflowService:
    def __init__(self, agentsvc: AgentService, bus: IEventBus | None = None):
        self.agentsvc = agentsvc
        self.bus = bus
    def createFromPrompt(self, prompt: str, agents: List[Agent]) -> Workflow:
        wf = Workflow(id=new_id('wf_'), name='wf_from_prompt')
        return wf
    def createFromCanvas(self, canvasJSON: dict) -> Workflow:
        return Workflow(id=new_id('wf_'), name='wf_from_canvas')
    def validate(self, workflow: Workflow) -> ValidationResult:
        return ValidationResult(ok=True)
    def publish(self, workflow: Workflow) -> None:
        pass


class ConnectorManager:
    """Unified manager routing to API or MCP connectors."""
    
    def __init__(self):
        self.api = APIConnector()
        self.mcp = MCPConnector()
    
    def get_manager(self, connector_type: str):
        """Route to appropriate manager."""
        if connector_type == "api":
            return self.api
        elif connector_type == "mcp":
            return self.mcp
        else:
            raise ValueError("connector_type must be 'api' or 'mcp'")

        

class MCPConnector:
    """Manages MCP connectors (HTTP/SSE/STDIO) with storage."""
    def __init__(self, storage_dir: str | None = None):  
        # Choose a directory
        default_dir = Path(__file__).resolve().parent / "Get_connector" / "Get_MCP" / "connectors"
        target_dir = storage_dir or os.getenv("MCP_STORAGE_DIR") or str(default_dir)

        # NEW: singleton init
        self.storage = get_storage(target_dir)


    async def create(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create connector ONLY after example_payload test passes.
        Returns rich test output on success; no persistence on failure.
        """
        try:
            # ----- 0) Extract & require example_payload -----
            # Keep a full copy for persistence (with example_payload)
            creation_payload = dict(config)  # shallow copy is fine here

            example_payload = creation_payload.get("example_payload")
            if not isinstance(example_payload, dict) or not example_payload:
                return {
                    "success": False,
                    "error": "Field 'example_payload' is required and must be a non-empty object",
                    "message": "Provide an example_payload so the connector can be validated before saving."
                }

            # Normalize connector config (exclude example_payload from normalization)
            config_without_example = dict(creation_payload)
            config_without_example.pop("example_payload", None)

            normalized = validate_and_normalize(config_without_example)
            transport = normalized["transport_type"]

            # ----- 1) Construct connector in-memory (not saved yet) -----
            if transport == "http":
                connector = HTTPMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in HTTPMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "sse":
                connector = SSEMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in SSEMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "stdio":
                connector = STDIOMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in STDIOMCPConnector.__init__.__code__.co_varnames
                })
            else:
                return {"success": False, "error": f"Unsupported transport: {transport}"}

            # ----- 2) Validate connector config (schema-level) -----
            is_valid, errors = connector.validate_config()
            if not is_valid:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "errors": errors,
                    "message": "Connector configuration is invalid. Fix errors and try again."
                }

            # ----- 3) RUN TEST (gatekeeper) -----
            # Only save if this test passes
            try:
                test_result = await connector.test(example_payload)
            except Exception as e:
                return {
                    "success": False,
                    "error": "example_payload_test_exception",
                    "message": f"Testing the connector raised an exception: {str(e)}",
                    "trace": repr(e)
                }

            test_success = bool(test_result.get("success"))
            status_code = test_result.get("status_code")
            elapsed     = test_result.get("elapsed_seconds")
            # Prefer common keys for body
            sample_body = (
                test_result.get("data",
                test_result.get("body",
                test_result.get("response",
                test_result)))
            )

            if not test_success:
                # DO NOT PERSIST ANYTHING
                return {
                    "success": False,
                    "error": "Connector validation failed",
                    "message": "example_payload test failed. The connector was NOT created/saved.",
                    "validation": {
                        "status": "failed",
                        "status_code": status_code,
                        "elapsed_seconds": elapsed,
                        "test_result": test_result  # echo entire test result for debugging
                    }
                }

            # ----- 4) Persist ONLY AFTER SUCCESS -----
            connector_id = connector.connector_id
            now_iso = datetime.utcnow().isoformat()

            
            serialized = connector.serialize()
            serialized.pop("test_results", None)   # remove saved test output
            serialized.pop("sample_response", None)
            serialized.pop("last_response", None)

            # Save serialized connector + metadata   
            # Build minimal config exactly like API connector style

            stored_config = {
                "connector_id": connector_id,
                "connector_name": connector.name,
                "creation_payload": creation_payload,   # contains example_payload ✔️
                "validation_status": "validated",
                "validation_error": None,
                "tested_at": now_iso,
                "created_at": now_iso,
                "updated_at": now_iso
            }

            self.storage.save(
                stored_config,
                creation_payload=creation_payload,
                validation_status="validated",
                validation_error=None,
                tested_at=now_iso
            )
            # ----- 5) Return rich success payload -----
            # Include test result fields and a sample response
            return {
                "success": True,
                "connector_id": connector_id,
                "name": connector.name,
                "transport_type": connector.transport_type.value,
                "validation_status": "validated",
                "message": "✅ Connector created and example_payload validated successfully",
                "validation": {
                    "status": "validated",
                    "status_code": status_code,
                    "elapsed_seconds": elapsed,
                    "test_result": test_result,     # full test result (frontend can collapse)
                    "sample_response": sample_body   # convenient shortcut for UI/agents
                },
                "created_at": now_iso
            }

        except ValidationError as e:
            return {"success": False, "error": "Validation error", "errors": e.errors}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def invoke_async(self, connector_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke connector - always loads fresh from storage, no caching."""
        try:
            # Load connector data from storage (always fresh)
            data = self.storage.load(connector_id)
            if not data:
                return {"success": False, "error": f"Connector not found: {connector_id}"}
            
            # Determine payload to use
            if payload is None:
                # Try to use stored example_payload
                creation_payload = data.get("creation_payload", {})
                stored_example = creation_payload.get("example_payload")
                validation_status = data.get("validation_status")
                
                if stored_example and validation_status == "validated":
                    payload = stored_example
                else:
                    return {
                        "success": False,
                        "error": "No payload provided and no validated example payload stored"
                    }
            
            # Extract connector config - stored flat at root level
            config = data.copy()
            
            # Get transport_type
            transport = config.get("creation_payload",{}).get("transport_type")
            
            if not transport:
                return {
                    "success": False,
                    "error": "transport_type missing in connector configuration"
                }
            
            # Handle dict/Enum if needed
            if isinstance(transport, dict):
                transport = transport.get("value")
            elif hasattr(transport, "value"):
                transport = transport.value
            
            transport = str(transport).lower()
            
            # Create connector instance fresh from storage
            if transport == "http":
                connector = HTTPMCPConnector.from_dict(config)
            elif transport == "sse":
                connector = SSEMCPConnector.from_dict(config)
            elif transport == "stdio":
                connector = STDIOMCPConnector.from_dict(config)
            else:
                return {
                    "success": False, 
                    "error": f"Invalid transport type: {transport}"
                }
            
            # Execute test (connector is used once and discarded)
            result = await connector.test(payload)
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False, 
                "error": str(e),
                "traceback": traceback.format_exc()
            }

        
    def get(self, connector_id: str) -> Optional[Dict[str, Any]]:
        """Get connector with full metadata."""
        data = self.storage.load(connector_id)
        if not data:
            return None
        
        # Return everything including creation_payload and metadata
        return {
            "connector_id": data.get("connector_id"),
            "creation_payload": data.get("creation_payload"),
            "validation_status": data.get("validation_status"),
            "validation_error": data.get("validation_error"),
            "tested_at": data.get("tested_at"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at")
        }
    
    def list(self) -> Dict[str, Any]:
        """List all connectors with metadata."""
        connectors = self.storage.list_all()
        return {"success": True, "count": len(connectors), "connectors": connectors}
    
    async def update(self, connector_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update connector with hybrid validation, deep merge, and mandatory testing."""
        try:
            # 1. Validate updates not empty
            if not updates:
                return {"success": False, "error": "Updates cannot be empty"}
            
            # 2. Load existing connector
            data = self.storage.load(connector_id)
            if not data:
                return {"success": False, "error": f"Connector not found: {connector_id}"}
            
            # 3. Get existing creation_payload
            creation_payload = data.get("creation_payload", {})
            if not creation_payload:
                return {"success": False, "error": "creation_payload missing in stored connector"}
            
            # 4. Check forbidden fields - only error if values CHANGED
            forbidden_fields = ["transport_type", "input_schema", "output_schema", "connector_id"]
            forbidden_changes = []
            
            for field in forbidden_fields:
                if field in updates:
                    old_value = creation_payload.get(field)
                    new_value = updates[field]
                    
                    # Deep comparison for dicts
                    if old_value != new_value:
                        forbidden_changes.append(field)
            
            if forbidden_changes:
                return {
                    "success": False,
                    "error": f"Cannot change the following fields: {', '.join(forbidden_changes)}"
                }
            
            # 5. Check auth type change
            if "auth_config" in updates and isinstance(updates["auth_config"], dict):
                if "type" in updates["auth_config"]:
                    old_type = creation_payload.get("auth_config", {}).get("type")
                    new_type = updates["auth_config"]["type"]
                    
                    if old_type and new_type and new_type != old_type:
                        return {
                            "success": False,
                            "error": f"Cannot change auth type from '{old_type}' to '{new_type}'"
                        }
            
            # 6. Check example_payload removal
            if "example_payload" in updates and updates["example_payload"] is None:
                return {"success": False, "error": "Cannot remove example_payload"}
            
            # 7. Deep merge updates into creation_payload
            merged_payload = self._deep_merge(creation_payload.copy(), updates)
            
            # 8. Detect changed fields
            changed_fields = self._get_changed_fields(creation_payload, merged_payload)
            
            # 9. If nothing changed (or only forbidden fields that are same), error
            if not changed_fields:
                return {"success": False, "error": "No valid fields to update"}
            
            # 10. Get example_payload (guaranteed to exist)
            example_payload = merged_payload.get("example_payload")
            if not example_payload:
                return {"success": False, "error": "example_payload missing after merge"}
            
            # 11. Validate merged config
            try:
                normalized = validate_and_normalize(merged_payload)
            except ValidationError as e:
                return {"success": False, "error": "Validation failed", "errors": e.errors}
            except Exception as e:
                return {"success": False, "error": f"Validation error: {str(e)}"}
            
            # 12. Get transport type
            transport = normalized.get("transport_type", merged_payload.get("transport_type"))
            
            if isinstance(transport, dict):
                transport = transport.get("value")
            elif hasattr(transport, "value"):
                transport = transport.value
            
            transport = str(transport).lower()
            
            # 13. Create temp connector for testing
            if transport == "http":
                temp_connector = HTTPMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in HTTPMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "sse":
                temp_connector = SSEMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in SSEMCPConnector.__init__.__code__.co_varnames
                })
            elif transport == "stdio":
                temp_connector = STDIOMCPConnector(**{
                    k: v for k, v in normalized.items()
                    if k in STDIOMCPConnector.__init__.__code__.co_varnames
                })
            else:
                return {"success": False, "error": f"Invalid transport: {transport}"}
            
            # 14. MANDATORY TEST with timeout limit
            try:
                # Set timeout (use connector's timeout + 5 seconds buffer)
                connector_timeout = merged_payload.get("timeout", 30)
                test_timeout = connector_timeout + 5
                
                test_result = await asyncio.wait_for(
                    temp_connector.test(example_payload),
                    timeout=test_timeout
                )
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Update test timed out after {test_timeout} seconds - changes not saved",
                    "test_result": {
                        "success": False,
                        "error": "Timeout",
                        "duration_ms": test_timeout * 1000
                    }
                }
            
            # 15. If test fails, reject update
            if not test_result.get("success"):
                return {
                    "success": False,
                    "error": "Update test failed - changes not saved",
                    "test_result": test_result
                }
            
            # 16. Test succeeded - save updated connector
            self.storage.save(
                temp_connector.serialize(),
                creation_payload=merged_payload,
                validation_status="validated",
                tested_at=datetime.utcnow().isoformat()
            )
            
            return {
                "success": True,
                "connector_id": connector_id,
                "message": "Connector updated and validated successfully",
                "changed_fields": changed_fields,
                "test_result": test_result
            }
            
        except ValidationError as e:
            return {"success": False, "error": "Validation error", "errors": e.errors}
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge updates into base dict, removing null values."""
        result = base.copy()
        
        for key, value in updates.items():
            if value is None:
                # Remove field if set to null
                result.pop(key, None)
            elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Deep merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Direct assignment for primitives, lists, or new keys
                result[key] = value
        
        return result
    
    def _get_changed_fields(self, old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
        """
        Recursively compare old and new dicts, return list of changed field paths.
        
        Example:
            old = {"auth_config": {"token": "abc"}, "timeout": 10}
            new = {"auth_config": {"token": "xyz"}, "timeout": 10}
            Returns: ["auth_config.token"]
        """
        changed = []
        
        # Check all keys in new dict
        for key, new_value in new.items():
            old_value = old.get(key)
            field_path = f"{prefix}{key}" if prefix else key
            
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                # Recursively check nested dicts
                nested_changes = self._get_changed_fields(old_value, new_value, f"{field_path}.")
                changed.extend(nested_changes)
            elif new_value != old_value:
                # Value changed (or new field)
                changed.append(field_path)
        
        # Check for removed keys (in old but not in new)
        for key in old.keys():
            if key not in new:
                field_path = f"{prefix}{key}" if prefix else key
                changed.append(field_path)
        
        return changed


    
    def delete(self, connector_id: str) -> Dict[str, Any]:
        """Delete connector."""
        try:
            success = self.storage.delete(connector_id)

        
            return {
                "success": success,
                "message": f"Connector {connector_id} deleted" if success else "Deletion failed"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class APIConnector:
    """
    Manages API connectors with comprehensive validation and metadata storage.
    
    Features:
    - 3-level validation (definition → connector creation → example testing)
    - Stores full creation payload for transparency
    - Optional example_payload testing during creation
    - Test endpoint with stored example or custom payload
    - Detailed error categorization
    """
    
    def __init__(self):
        """Initialize connector manager."""
        from echolib.Get_connector.Get_API.connectors.factory import ConnectorFactory
        self.factory = ConnectorFactory
        self._connectors = {}  # In-memory storage: {connector_id: metadata}

        # Setup storage
        current_file = Path(__file__) # Path to services.py
        api_folder = current_file.parent / "Get_connector" / "Get_API"
        self.storage_dir = api_folder / "connectors_data"
        self.storage_file = self.storage_dir / "api_connectors.json"

        #Load existing connectors from disk
        self._load_from_file()

    def _build_creation_payload_from_config(
            self, config: ConnectorConfig
    ) -> Dict[str, Any]:
        """
        Build a normalized, cannoical creation payload from ConnectroConfig
        this payload is safe to store and reuse for update/load operations"""
        return {
            "name": config.name,
            "description" : config.description or "",
            "base_url": config.base_url,
            "auth_config":config.auth,
            "default_headers": config.default_headers or {},
            "timeout" : config.timeout,
            "verify_ssl" : config.verify_ssl,
        }

    def _load_from_file(self):
        """Load all connectors from individual files."""
        try:
            if not self.storage_dir.exists():
                return
            
            # Find all .json files
            json_files = list(self.storage_dir.glob("*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    connector_id = data["connector_id"]
                    
                    # Recreate connector
                    from echolib.Get_connector.Get_API.models.config import ConnectorConfig
                    
                    creation_payload = data["creation_payload"]
                    
                    config = ConnectorConfig(
                        id=connector_id,
                        name=creation_payload["name"],
                        description=creation_payload.get("description", ""),
                        base_url=creation_payload["base_url"],
                        auth=creation_payload["auth_config"],
                        default_headers=creation_payload.get("default_headers", {}),
                        timeout=float(creation_payload.get("timeout", 30)),
                        verify_ssl=creation_payload.get("verify_ssl", False)
                    )
                    
                    connector = self.factory.create(config)
                    
                    # Store in memory
                    self._connectors[connector_id] = {
                        "connector": connector,
                        "connector_id": data["connector_id"],
                        "connector_name": data["connector_name"],
                        "creation_payload": data["creation_payload"],
                        "example_payload": data.get("example_payload"),
                        "validation": data.get("validation", {"status": "not_tested"}),
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"]
                    }
                
                except Exception as e:
                    logging.warning(f"Failed to load {json_file.name}: {str(e)}")
                    continue
        
        except Exception as e:
            logging.error(f"Error loading connectors: {str(e)}")
    
    def _delete_connector_file(self, connector_id: str):
        """Delete connector file from disk."""
        try:
            connector_file = self.storage_dir / f"{connector_id}.json"
            if connector_file.exists():
                connector_file.unlink()
        except Exception as e:
            logging.error(f"Failed to delete connector file {connector_id}: {str(e)}")

    def _save_connector(self, connector_id: str):
        """Save single connector to its own file."""
        try:
            connector_file = self.storage_dir / f"{connector_id}.json"
            connector_data = self._connectors.get(connector_id)
            
            if not connector_data:
                return
            
            # Serialize (exclude connector object)
            serializable_data = {
                "connector_id": connector_data["connector_id"],
                "connector_name": connector_data["connector_name"],
                "creation_payload": connector_data["creation_payload"],
                "example_payload": connector_data.get("example_payload"),
                "validation": connector_data.get("validation"),
                "created_at": connector_data["created_at"],
                "updated_at": connector_data["updated_at"]
            }
            
            with open(connector_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save connector {connector_id}: {str(e)}")

    def _validate_auth_values(self, auth_config: Dict[str, Any]) -> Optional[str]:
        """
        Validate auth config values are not placeholders.
        Returns error message if invalid, None if valid.
        """
        auth_type = auth_config.get("type")
        
        # Placeholder patterns to reject
        PLACEHOLDER_PATTERNS = [
            "your_token",
            "your_key",
            "replace",
            "xxx",
            "token_here",
            "api_key_here",
            "your-token",
            "your-key",
            "<",
            ">",
            "...",
        ]
        
        if auth_type == "bearer":
            token = auth_config.get("token", "")
            
            if not token:
                return "Bearer token is required"
            
            # Check for placeholders
            token_lower = token.lower()
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern in token_lower:
                    return f"Bearer token appears to be a placeholder. Please provide a real token."
            
            # Check minimum length
            if len(token) < 10:
                return "Bearer token is too short. Please provide a valid token."
        
        elif auth_type == "api_key":
            key = auth_config.get("key", "")
            
            if not key:
                return "API key is required"
            
            key_lower = key.lower()
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern in key_lower:
                    return f"API key appears to be a placeholder. Please provide a real key."
            
            if len(key) < 10:
                return "API key is too short. Please provide a valid key."
        
        elif auth_type == "basic":
            username = auth_config.get("username", "")
            password = auth_config.get("password", "")
            
            if not username or not password:
                return "Username and password are required for basic auth"
        
        elif auth_type == "oauth2":
            client_id = auth_config.get("client_id", "")
            
            if not client_id:
                return "OAuth2 requires client_id"
        
        return None  # Valid
    
    def create(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create API connector with strict validation:
        - example_payload is REQUIRED
        - connector is ONLY saved if example_payload test PASSES
        """
        try:
            # ---------- LEVEL 1: Validate Definition ----------
            required_fields = ("name", "base_url", "auth_config", "example_payload")
            for f in required_fields:
                if not definition.get(f):
                    return {"success": False, "error": f"Field '{f}' is required"}

            auth_config = definition.get("auth_config", {})
            if not auth_config.get("type"):
                return {"success": False, "error": "Field 'auth_config.type' is required"}

            example_payload = definition.get("example_payload")
            if not isinstance(example_payload, dict) or not example_payload.get("endpoint"):
                return {"success": False, "error": "example_payload must be an object with at least 'endpoint' and 'method' (default GET)"}

            auth_validation_error = self._validate_auth_values(auth_config=auth_config)
            if auth_validation_error:
                return {"success": False, "error": auth_validation_error}

            # ---------- LEVEL 2: Build Config & Create In-Memory Connector ----------
            try:
                config = ConnectorConfig(
                    name=definition["name"],
                    description=definition.get("description", ""),
                    base_url=definition["base_url"],
                    auth=auth_config,
                    default_headers=definition.get("default_headers", {}),
                    timeout=float(definition.get("timeout", 30)),
                    verify_ssl=bool(definition.get("verify_ssl", False)),
                )
            except Exception as e:
                return {"success": False, "error": f"Invalid connector configuration: {e}"}

            try:
                connector = self.factory.create(config)
            except Exception as e:
                return {"success": False, "error": f"Failed to create connector: {e}"}

            # ---------- LEVEL 3: Test Example Payload (Gatekeeper) ----------
            logger.info(f"Testing example_payload for connector: {config.name}")
            test_result = self._test_example_payload(connector, example_payload)

            if not test_result.get("success"):
                # Do NOT persist anything on failure
                error_details = test_result.get("error_details", {})
                return {
                    "success": False,
                    "error": "Connector validation failed",
                    "message": "Cannot create connector: example_payload test failed. Please check your configuration and try again.",
                    "status_code": test_result.get("status_code"),
                    "response_body": test_result.get("response_body"),
                    "details": error_details,
                }

            # ---------- LEVEL 4: Persist ONLY AFTER SUCCESS ----------
            now = datetime.utcnow().isoformat()
            creation_payload = self._build_creation_payload_from_config(config)

            self._connectors[config.id] = {
                "connector": connector,
                "connector_id": config.id,
                "connector_name": config.name,
                "creation_payload": creation_payload,
                "example_payload": example_payload,
                "validation": {
                    "status": "validated",
                    "tested_at": datetime.utcnow().isoformat(),
                    "error": None,
                },
                "created_at": now,
                "updated_at": now,
            }

            # Persist to disk AFTER in-memory registry is set
            self._save_connector(config.id)

            # ---------- Response ----------
            return {
                "success": True,
                "connector_id": config.id,
                "connector_name": config.name,
                "created_at": now,
                "message": "✅ Connector created and example payload validated successfully",
                "validation": {
                    "status": "validated",
                    "response_time_ms": int(test_result.get("elapsed_seconds", 0) * 1000),
                    "status_code": test_result.get("status_code"),
                    "sample_response": test_result.get("response_body"),
                },
            }

        except Exception as e:
            logger.exception("Unexpected error during creation")
            return {"success": False, "error": f"Unexpected error during creation: {e}"}

    def _test_example_payload(self, connector_or_id, example_payload: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Testing example payload...")
        logger.debug("Example payload: %s", example_payload)

        try:
            # accept connector instance or id
            if hasattr(connector_or_id, "execute"):
                connector = connector_or_id
            else:
                data = self._connectors.get(connector_or_id)
                if not data:
                    return {"success": False, "status_code": None,
                            "error_details": {"type":"not_found","message":"Connector not found"}}
                connector = data["connector"]

            endpoint = example_payload.get("endpoint", "/")
            method   = example_payload.get("method", "GET").upper()
            params   = example_payload.get("params", {}) or {}
            body     = example_payload.get("body")

            logger.info("Testing: %s %s", method, endpoint)
            logger.debug("Params: %s", params)
            logger.debug("Body  : %s", body)

            result = connector.execute(
                method=method,
                endpoint=endpoint,
                query_params=params if method == "GET" else None,
                body=body if method != "GET" else None
            )

            # Normalize result (support dict or response-like)
            status_code = getattr(result, "status_code", None)
            elapsed     = getattr(result, "elapsed_seconds", None)
            body_val    = getattr(result, "body", None)
            success     = getattr(result, "success", None)

            if isinstance(result, dict):
                status_code = result.get("status_code", status_code)
                elapsed     = result.get("elapsed_seconds", elapsed)
                body_val    = result.get("body", body_val)
                success     = result.get("success", success)

            if success:
                return {
                    "success": True,
                    "status_code": status_code,
                    "elapsed_seconds": elapsed,
                    "response_body": body_val
                }

            # Non-success: try to categorize and include response body
            error_details = self._categorize_error(result, endpoint, method, params, body)
            return {
                "success": False,
                "status_code": status_code,
                "response_body": body_val,
                "error_details": error_details
            }

        except Exception as e:
            logger.error("Exception during test: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return {
                "success": False,
                "status_code": None,
                "error_details": {
                    "type": "unexpected_error",
                    "category": "exception",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }

    
    def _categorize_error(self, result, endpoint, method, params, body) -> Dict[str, Any]:
        """
        Categorize HTTP error with detailed information.
        
        Returns dict with: type, category, status_code, message, details, can_retry, suggestion
        """
        status_code = result.status_code
        error_message = result.error or "Unknown error"
        
        # Rate Limit
        if status_code == 429:
            return {
                "type": "rate_limit",
                "category": "api",
                "status_code": 429,
                "message": "API rate limit exceeded",
                "details": error_message,
                "can_retry": True,
                "suggestion": "Wait and retry after rate limit resets"
            }
        
        # Service Unavailable
        if status_code == 503:
            return {
                "type": "service_unavailable",
                "category": "api",
                "status_code": 503,
                "message": "API is temporarily unavailable",
                "details": error_message,
                "can_retry": True,
                "suggestion": "API might be down. Try again later."
            }
        
        # Server Error
        if status_code >= 500:
            return {
                "type": "server_error",
                "category": "api",
                "status_code": status_code,
                "message": "API server error",
                "details": error_message,
                "can_retry": True,
                "suggestion": "This is an API server issue, not your fault."
            }
        
        # Unauthorized
        if status_code == 401:
            return {
                "type": "unauthorized",
                "category": "auth",
                "status_code": 401,
                "message": "Authentication failed",
                "details": error_message,
                "can_retry": False,
                "suggestion": "Check your API key or authentication credentials."
            }
        
        # Forbidden
        if status_code == 403:
            return {
                "type": "forbidden",
                "category": "auth",
                "status_code": 403,
                "message": "Access forbidden",
                "details": error_message,
                "can_retry": False,
                "suggestion": "Check API permissions or subscription level."
            }
        
        # Bad Request
        if status_code == 400:
            return {
                "type": "bad_request",
                "category": "payload",
                "status_code": 400,
                "message": "Invalid request parameters",
                "details": error_message,
                "attempted_request": {
                    "endpoint": endpoint,
                    "method": method,
                    "params": params,
                    "body": body
                },
                "can_retry": False,
                "suggestion": "Fix the parameters in example_payload."
            }
        
        # Not Found
        if status_code == 404:
            return {
                "type": "not_found",
                "category": "payload",
                "status_code": 404,
                "message": "Endpoint not found",
                "details": f"GET {endpoint} returned 404",
                "can_retry": False,
                "suggestion": "Check the endpoint path in example_payload."
            }
        
        # Timeout
        if status_code == 408:
            return {
                "type": "connection_timeout",
                "category": "network",
                "status_code": 408,
                "message": "Request timed out",
                "details": error_message,
                "can_retry": True,
                "suggestion": "API might be slow. Try increasing timeout."
            }
        
        # Connection Error
        if status_code == 503:
            return {
                "type": "connection_error",
                "category": "network",
                "status_code": 503,
                "message": "Could not connect to API",
                "details": error_message,
                "can_retry": True,
                "suggestion": "Check if base_url is correct and API is reachable."
            }
        
        # Generic Error
        return {
            "type": "unknown",
            "category": "unknown",
            "status_code": status_code,
            "message": "Unknown error",
            "details": error_message,
            "can_retry": False
        }

    logger = logging.getLogger(__name__)

    
    def invoke(
        self,
        connector_id: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke a connector endpoint.

        Behavior:
        - If `payload` is None: use the saved example_payload as-is.
        - If `payload` is provided: use ONLY that payload; do NOT merge.
            But validate that all fields/paths that exist in example_payload
            are present in payload (values may differ; extra fields allowed).

        Returns:
        {
            "success": bool,
            "status_code": int | None,
            "data": Any | None,
            "error": dict | None,
            "elapsed_seconds": float | None
        }
        """

        def _collect_required_paths(obj: Any, prefix: str = "") -> List[str]:
            """
            Collect all required 'paths' from the example_payload.
            We treat any dict key path as required presence.
            Lists are treated as atomic (we require the list key itself if present in example).
            """
            required = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    path = f"{prefix}{k}"
                    required.append(path)  # the key itself must exist
                    # Recurse into dicts to require nested keys as well
                    required.extend(_collect_required_paths(v, path + "."))
            # For lists/scalars, we don't enforce internal structure—only presence of the key itself above.
            return required

        def _missing_paths(required_paths: List[str], candidate: Dict[str, Any]) -> List[str]:
            """
            Check which required paths are missing from candidate.
            A path like 'params.q' means candidate['params']['q'] must exist.
            """
            missing = []
            for path in required_paths:
                parts = path.split(".")
                cur = candidate
                exists = True
                for p in parts:
                    if not isinstance(cur, dict) or p not in cur:
                        exists = False
                        break
                    cur = cur[p]
                if not exists:
                    missing.append(path)
            return missing

        def _normalize_result(result: Any) -> Tuple[bool, Optional[int], Any, Optional[float], Optional[Any]]:
            """Normalize connector.execute result whether object-like or dict-like."""
            success         = getattr(result, "success", None)
            status_code     = getattr(result, "status_code", None)
            body            = getattr(result, "body", None)
            elapsed_seconds = getattr(result, "elapsed_seconds", None)
            err             = getattr(result, "error", None)

            if isinstance(result, dict):
                success         = result.get("success", success)
                status_code     = result.get("status_code", status_code)
                elapsed_seconds = result.get("elapsed_seconds", elapsed_seconds)
                body            = result.get("body", result.get("data", body))
                err             = result.get("error", err)

            if success is None and isinstance(status_code, int):
                success = 200 <= status_code < 300

            return bool(success), status_code, body, elapsed_seconds, err

        try:
            # ---- Load connector ----
            connector_data = self._connectors.get(connector_id)
            if not connector_data:
                return {
                    "success": False,
                    "status_code": None,
                    "data": None,
                    "error": {"type": "not_found", "message": f"Connector '{connector_id}' not found"},
                    "elapsed_seconds": None
                }

            connector = connector_data["connector"]
            example_payload = connector_data.get("example_payload") or {}
            if not example_payload:
                return {
                    "success": False,
                    "status_code": None,
                    "data": None,
                    "error": {"type": "configuration_error", "message": "No example_payload defined for this connector"},
                    "elapsed_seconds": None
                }

            # ---- Decide which request to use ----
            if payload is None:
                # Use example_payload as-is
                request_payload = example_payload
            else:
                # Use ONLY the given payload, but ensure it includes all paths present in example_payload
                required_paths = _collect_required_paths(example_payload)
                missing = _missing_paths(required_paths, payload)
                if missing:
                    return {
                        "success": False,
                        "status_code": None,
                        "data": None,
                        "error": {
                            "type": "validation_failed",
                            "policy": "presence_required_from_example",
                            "message": "Payload is missing required fields that exist in example_payload.",
                            "violations": [
                                {
                                    "path": p,
                                    "rule": "required",
                                    "expected": "present (as in example_payload)",
                                    "actual": "missing",
                                    "hint": "Provide this field or omit payload to use the example_payload."
                                } for p in missing
                            ]
                        },
                        "elapsed_seconds": None
                    }
                request_payload = payload

            # ---- Extract endpoint/method/params/body ----
            endpoint = request_payload.get("endpoint")
            method   = (request_payload.get("method") or "GET").upper()
            params   = request_payload.get("params")
            body     = request_payload.get("body")
            headers  = request_payload.get("headers")  # optional

            if not endpoint:
                return {
                    "success": False,
                    "status_code": None,
                    "data": None,
                    "error": {
                        "type": "validation_failed",
                        "policy": "required",
                        "violations": [{
                            "path": "endpoint",
                            "rule": "required",
                            "expected": "non-empty string",
                            "actual": endpoint
                        }]
                    },
                    "elapsed_seconds": None
                }

            # Map GET to query params; non-GET to body
            query_params = params if method == "GET" else None
            req_body     = body   if method != "GET" else None

            # ---- Execute ----
            exec_kwargs = {
                "method": method,
                "endpoint": endpoint,
                "query_params": query_params,
                "body": req_body
            }
            if headers is not None:
                exec_kwargs["headers"] = headers  # only if your connector supports headers

            result = connector.execute(**exec_kwargs)

            success, status_code, resp_body, elapsed_seconds, err = _normalize_result(result)

            return {
                "success": success,
                "status_code": status_code,
                "data": resp_body if success else None,
                "error": None if success else (err or {
                    "type": "upstream_error",
                    "message": "Request failed",
                    "response": resp_body
                }),
                "elapsed_seconds": elapsed_seconds
            }

        except Exception as e:
            logger.error("Exception during invoke for connector_id=%s: %s", connector_id, e)
            logger.debug("Traceback:\n%s", traceback.format_exc())
            return {
                "success": False,
                "status_code": None,
                "data": None,
                "error": {
                    "type": "unexpected_error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                },
                "elapsed_seconds": None
            }

    def get(self, connector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get connector details including full creation payload.
        
        Args:
            connector_id: Connector ID
        
        Returns:
            Dict with connector_id, creation_payload, validation status, timestamps
        """
        try:
            # Direct lookup by ID
            connector_data = self._connectors.get(connector_id)
            
            if not connector_data:
                return None
            
            # Mask sensitive fields in auth
            creation_payload = connector_data["creation_payload"].copy()
            example_payload = connector_data["example_payload"].copy()
            if "auth_config" in creation_payload:
                auth = creation_payload["auth_config"].copy()
                if "key" in auth:
                    auth["key"] = "****" + auth["key"][-3:] if len(auth["key"]) > 3 else "****"
                if "token" in auth:
                    auth["token"] = "****" + auth["token"][-3:] if len(auth["token"]) > 3 else "****"
                if "password" in auth:
                    auth["password"] = "****" + auth["password"][-3:] if len(auth.get("password", "")) > 3 else "****"
                if "client_secret" in auth:
                    auth["client_secret"] = "****" + auth["client_secret"][-3:] if len(auth.get("client_secret", "")) > 3 else "****"
                creation_payload["auth_config"] = auth
            
            return {
                "success": True,
                "connector_id": connector_data["connector_id"],
                "creation_payload": creation_payload,
                "example_payload" : example_payload,
                "validation": connector_data["validation"],
                "created_at": connector_data["created_at"],
                "updated_at": connector_data["updated_at"]
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}

    def test(
        self,
        connector_id: str,
        custom_payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Test connector with stored example or custom payload.
        
        Args:
            connector_name: Connector name
            custom_payload: Optional custom test payload
        
        Returns:
            Dict with test results
        """
        try:
            # Find connector by name
            connector_data = self._connectors.get(connector_id)
            if not connector_data:
                return {"success": False, "error": f"Connector ID '{connector_id}' not found"}
            
            # Determine which payload to use
            if custom_payload:
                # Use custom payload
                test_payload = custom_payload
            else:
                # Try to use stored example
                stored_example = connector_data.get("example_payload")
                validation_status = connector_data["validation"]["status"]
                
                if not stored_example:
                    return {
                        "success": False,
                        "error": "No example payload stored. Please provide custom_payload.",
                        "requires_custom_payload": True
                    }
                
                if validation_status == "failed":
                    return {
                        "success": False,
                        "error": "Stored example payload failed validation during creation. Please provide custom_payload.",
                        "validation_error": connector_data["validation"]["error"],
                        "requires_custom_payload": True
                    }
                
                test_payload = stored_example
            
            # Execute test
            result = self._test_example_payload(connector_data["connector_id"], test_payload)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "✅ Connector tested successfully",
                    "status_code": result["status_code"],
                    "response_time_ms": result.get("elapsed_seconds", 0) * 1000,
                    "test_method": "custom_payload" if custom_payload else "stored_example",
                    "result" : result["response_body"]
                }
            else:
                return {
                    "success": False,
                    "message": "❌ Connector test failed",
                    "error": result["error_details"],
                    "test_method": "custom_payload" if custom_payload else "stored_example"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list(self) -> Dict[str, Any]:
        """
        List all connectors.
        
        Returns:
            Dict with success, count, and list of connectors
        """
        try:
            connectors = [
                {
                    "connector_id": data["connector_id"],
                    "connector_name": data["connector_name"],
                    "example_payload" : data["example_payload"],
                    "creation_payload" : data["creation_payload"],
                    "validation_status": data["validation"]["status"],
                    "created_at": data["created_at"]
                }
                for connector_id, data in self._connectors.items()
            ]
            
            return {
                "success": True,
                "count": len(connectors),
                "connectors": connectors
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "connectors": []
            }
    
    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update connector configuration.
        
        Frontend sends FULL creation_payload with changes.
        Backend detects changes, filters restricted fields, validates, and saves.
        
        Args:
            updates: Dict containing connector_id and full creation_payload
            
        Returns:
            Dict with success status, changes applied, fields ignored, and test results
        """
        try:
            # ========== VALIDATION: Input ==========
            if not updates:
                return {"success": False, "error": "No updates provided"}
            
            connector_id = updates.get("connector_id")
            if not connector_id:
                return {"success": False, "error": "Field 'connector_id' is required"}
            
            received_payload = {k: v for k, v in updates.items() if k != "connector_id"}
            
            if not received_payload:
                return {"success": False, "error": "Updates object is empty"}
            
            logger.info(f"Processing update for connector: {connector_id}")
            
            # ========== Load Existing Connector ==========
            connector_data = self._connectors.get(connector_id)
            
            if not connector_data:
                return {
                    "success": False,
                    "error": f"Connector ID '{connector_id}' not found"
                }
            
            connector_name = connector_data["connector_name"]
            existing_payload = connector_data["creation_payload"]
            existing_example = connector_data.get("example_payload")
            
            logger.info(f"Loaded existing connector: {connector_name}")
            
            # ========== RESTRICTED FIELDS ==========
            RESTRICTED_FIELDS = ["connector_id", "connector_name"]
            
            # ========== Detect Changes ==========
            change_analysis = self._detect_changes(existing_payload, received_payload, RESTRICTED_FIELDS)
            
            allowed_changes = change_analysis["allowed_changes"]
            restricted_changes = change_analysis["restricted_changes"]
            unchanged_fields = change_analysis["unchanged_fields"]
            
            logger.info(f"Allowed changes: {len(allowed_changes)}")
            logger.info(f"Restricted changes: {len(restricted_changes)}")
            logger.info(f"Unchanged fields: {len(unchanged_fields)}")
            
            # ========== Check: Only Restricted Changes? ==========
            if not allowed_changes and restricted_changes:
                return {
                    "success": False,
                    "error": "No allowed changes detected",
                    "message": "All attempted changes are to restricted fields.",
                    "fields_ignored": {
                        field: f"Cannot modify '{field}' (restricted field)"
                        for field in restricted_changes.keys()
                    }
                }
            
            # ========== Check: No Changes at All? ==========
            if not allowed_changes and not restricted_changes:
                return {
                    "success": False,
                    "error": "No changes detected",
                    "message": "The received configuration is identical to the existing configuration."
                }
            
            # ========== Validate: Auth Type Change ==========
            if "auth" in allowed_changes or "auth_config" in allowed_changes:
                auth_field = "auth" if "auth" in received_payload else "auth_config"
                existing_auth = existing_payload.get(auth_field, existing_payload.get("auth" if auth_field == "auth_config" else "auth_config", {}))
                received_auth = received_payload.get(auth_field, {})
                
                existing_type = existing_auth.get("type")
                received_type = received_auth.get("type")
                
                if received_type and received_type != existing_type:
                    return {
                        "success": False,
                        "error": "Cannot change authentication type",
                        "message": f"Changing auth type from '{existing_type}' to '{received_type}' is not allowed. Create a new connector instead.",
                        "current_auth_type": existing_type,
                        "attempted_auth_type": received_type
                    }
            
            # ========== Apply Allowed Changes ==========
            logger.info("Applying allowed changes to configuration...")
            
            # Start with existing payload
            import copy
            updated_payload = copy.deepcopy(existing_payload)
            
            # Apply only allowed changes
            for change_path, change_info in allowed_changes.items():
                # Navigate to the field and update it
                path_parts = change_path.split(".")
                current = updated_payload
                
                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[path_parts[-1]] = change_info["to"]
            
            logger.debug(f"Updated payload: {updated_payload}")
            
            # ========== Handle example_payload ==========
            # Check if example_payload was changed
            new_example = received_payload.get("example_payload")
            
            if new_example is None and "example_payload" in allowed_changes:
                # User tried to remove example_payload
                return {
                    "success": False,
                    "error": "Cannot remove example_payload",
                    "message": "Every connector must have an example_payload for validation."
                }
            
            # Use new example if provided and changed, else keep existing
            example_to_test = new_example if new_example and new_example != existing_example else existing_example
            
            if not example_to_test:
                return {
                    "success": False,
                    "error": "No example_payload available for validation"
                }
            
            # ========== Validate Merged Config ==========
            if not updated_payload.get("name"):
                return {"success": False, "error": "Updated config missing required field: 'name'"}
            
            if not updated_payload.get("base_url"):
                return {"success": False, "error": "Updated config missing required field: 'base_url'"}
            
            auth_config = updated_payload.get("auth") or updated_payload.get("auth_config")
            if not auth_config:
                return {"success": False, "error": "Updated config missing required field: 'auth'"}
            
            if not auth_config.get("type"):
                return {"success": False, "error": "Updated auth config missing required field: 'type'"}
            
            # ========== Create Temporary Connector ==========
            logger.info("Creating temporary connector for testing...")
            
            config_dict = updated_payload.copy()
            config_dict["id"] = connector_id
            
            # Normalize auth field name
            if "auth_config" in config_dict and "auth" not in config_dict:
                config_dict["auth"] = config_dict.pop("auth_config")
            
            try:
                config = ConnectorConfig(**config_dict)
                temp_connector = self.factory.create(config)
            except Exception as e:
                logger.error(f"Failed to create connector: {str(e)}")
                return {
                    "success": False,
                    "error": f"Invalid updated configuration: {str(e)}"
                }
            
            # ========== Test Updated Connector ==========
            logger.info("Testing updated connector with example_payload...")
            logger.debug(f"Example payload: {example_to_test}")
            
            test_result = self._test_example_payload(temp_connector, example_to_test)
            
            logger.info(f"Test result: {'PASS' if test_result.get('success') else 'FAIL'}")
            
            if not test_result.get("success"):
                # Test FAILED - don't save
                logger.warning("Update validation failed - changes NOT saved")
                
                # Mask sensitive fields in changes
                masked_changes = {}
                for path, info in allowed_changes.items():
                    if any(sensitive in path.lower() for sensitive in ["password", "token", "key", "secret"]):
                        masked_changes[path] = "changed (value hidden for security)"
                    else:
                        masked_changes[path] = {
                            "from": info["from"],
                            "to": info["to"]
                        }
                
                return {
                    "success": False,
                    "error": "Update validation failed - changes not saved",
                    "message": "The updated configuration failed validation test. Original connector unchanged.",
                    "attempted_changes": masked_changes,
                    "test_result": {
                        "success": False,
                        "status_code": test_result.get("status_code"),
                        "error": test_result.get("error_details", {}).get("message", "Unknown error"),
                        "error_category": test_result.get("error_category")
                    }
                }
            
            # ========== Test PASSED - Save ==========
            logger.info("Validation passed - saving updated connector")

            now = datetime.utcnow().isoformat()

            # Restore auth_config for storage (if needed)
            if "auth" in updated_payload and "auth_config" not in updated_payload:
                updated_payload["auth_config"] = updated_payload["auth"]

            # ========== SYNC connector_name with creation_payload.name ==========
            # If name in creation_payload changed, update connector_name to match
            updated_connector_name = updated_payload.get("name", connector_name)

            if updated_connector_name != connector_name:
                logger.info(f"Connector name changed: '{connector_name}' → '{updated_connector_name}'")

            self._connectors[connector_id] = {
                "connector": temp_connector,
                "connector_id": connector_id,
                "connector_name": updated_connector_name,  # ← Use UPDATED name!
                "creation_payload": updated_payload,
                "example_payload": example_to_test,
                "validation": {
                    "status": "validated",
                    "tested_at": now,
                    "error": None
                },
                "created_at": connector_data.get("created_at", now),
                "updated_at": now
            }  
            
            self._save_connector(connector_id)
            
            logger.info(f"✅ Connector '{connector_name}' updated successfully")
            
            # ========== Prepare Success Response ==========
            
            # Mask sensitive fields in response
            changes_applied = {}
            for path, info in allowed_changes.items():
                if any(sensitive in path.lower() for sensitive in ["password", "token", "key", "secret"]):
                    changes_applied[path] = "changed (value hidden for security)"
                else:
                    changes_applied[path] = {
                        "from": info["from"],
                        "to": info["to"]
                    }
            
            # Prepare ignored fields message
            fields_ignored = {}
            for path in restricted_changes.keys():
                fields_ignored[path] = f"Cannot modify '{path}' (restricted field)"
            
            response = {
                "success": True,
                "connector_id": connector_id,
                "connector_name": connector_name,
                "message": f"Connector '{connector_name}' updated and validated successfully",
                "changes_applied": changes_applied,
                "fields_unchanged": unchanged_fields,
                "test_result": {
                    "success": True,
                    "status_code": test_result.get("status_code"),
                    "response_time_ms": test_result.get("elapsed_seconds", 0) * 1000
                },
                "updated_at": now
            }
            
            # Add ignored fields if any
            if fields_ignored:
                response["fields_ignored"] = fields_ignored
                response["message"] = f"Connector '{connector_name}' updated successfully (some fields ignored)"
            
            return response
        
        except Exception as e:
            logger.error(f"Unexpected error during update: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Unexpected error during update: {str(e)}"
            }

        
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge updates into base dict, removing null values.
        
        Args:
            base: Base dictionary
            updates: Updates to merge in
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in updates.items():
            if value is None:
                # Remove field if set to null
                result.pop(key, None)
            elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Direct assignment for primitives and lists
                result[key] = value
        
        return result
    
    def _detect_changes(self, existing: Dict[str, Any], received: Dict[str, Any], restricted_fields: list) -> Dict[str, Any]:
        """
        Detect changes between existing and received configurations.
        
        Args:
            existing: Existing configuration
            received: Received configuration
            restricted_fields: List of fields that cannot be changed
            
        Returns:
            Dict with allowed_changes, restricted_changes, and unchanged fields
        """
        import copy
        
        def _deep_compare(path: str, old_val: Any, new_val: Any, changes: dict, category: str):
            """Recursively compare values and record changes."""
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Compare nested dicts
                all_keys = set(old_val.keys()) | set(new_val.keys())
                for key in all_keys:
                    new_path = f"{path}.{key}" if path else key
                    _deep_compare(
                        new_path,
                        old_val.get(key),
                        new_val.get(key),
                        changes,
                        category
                    )
            else:
                # Compare primitive values
                if old_val != new_val:
                    changes[path] = {
                        "from": old_val,
                        "to": new_val,
                        "category": category
                    }
        
        all_changes = {}
        
        # Compare all fields in received
        for key in received.keys():
            _deep_compare(
                key,
                existing.get(key),
                received.get(key),
                all_changes,
                "restricted" if key in restricted_fields else "allowed"
            )
        
        # Separate changes by category
        allowed_changes = {k: v for k, v in all_changes.items() if v["category"] == "allowed"}
        restricted_changes = {k: v for k, v in all_changes.items() if v["category"] == "restricted"}
        
        # Find unchanged fields (top-level only)
        unchanged = []
        for key in existing.keys():
            if key not in all_changes and key in received:
                unchanged.append(key)
        
        return {
            "allowed_changes": allowed_changes,
            "restricted_changes": restricted_changes,
            "unchanged_fields": unchanged
        }


    
    def delete(self, connector_id: str) -> Dict[str, Any]:
        """
        Delete connector by id.
        
        Args:
            connector_id: Connector id
        
        Returns:
            Dict with success status
        """
        try:
            # Find and delete by name
            connector_data = self._connectors.get(connector_id)
            if not connector_id:
                return {"success": False, "error": f"Connector '{connector_id}' not found"}
            
            del self._connectors[connector_id]
            
            # Persist to disk
            self._delete_connector_file(connector_id)

            return {
                "success": True,
                "message": f"Connector '{connector_id}' deleted successfully"
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}

class ChatOrchestrator:
    def __init__(self, bus: IEventBus, llm: LLMService, rag: RAGService, wf: WorkflowService, ag: AgentService, conns: ConnectorManager, log: ILogger | None = None):
        self.bus = bus
        self.llm = llm
        self.rag = rag
        self.wf = wf
        self.ag = ag
        self.conns = conns
        self.log = log
    def orchestrate(self, msg: dict, session: Session) -> dict:
        ctx = self.rag.queryIndex(msg.get('content',''), {})
        out = self.llm.generate(msg.get('content',''), [])
        self.bus.publish('chat.events', Event(type='chat.completion', data={'text': out.text}))
        return {'reply': out.text, 'ctx_docs': [d.id for d in ctx.documents]}
    def routeToAgent(self, msg: dict) -> Agent:
        return Agent(id='agt_default', name='default')
    def routeToService(self, intent: str):
        return {'service': intent}
    def publishEvent(self, topic: str, event: Event) -> None:
        self.bus.publish(topic, event)
