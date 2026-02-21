
import json
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional

class Event(BaseModel):
    type: str
    data: Dict[str, Any]

class Session(BaseModel):
    id: str
    user_id: str
    data: Dict[str, Any] = {}

class UserContext(BaseModel):
    user_id: str
    email: str

class Document(BaseModel):
    id: str
    title: str
    content: str

class IndexSummary(BaseModel):
    count: int

class ContextBundle(BaseModel):
    documents: List[Document]

class LLMOutput(BaseModel):
    text: str
    tokens: int = 0

# ==================== GRAPH RAG TYPES ====================

class Entity(BaseModel):
    """
    Represents an entity extracted from documents.
    
    Entities are key concepts, people, organizations, locations, etc.
    that form nodes in the knowledge graph.
    """
    id: str = Field(..., description="Unique entity identifier")
    name: str = Field(..., description="Entity name/label")
    type: str = Field(..., description="Entity type (person, organization, concept, etc.)")
    description: Optional[str] = Field(default=None, description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    source_doc_ids: List[str] = Field(default_factory=list, description="Documents this entity appears in")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class Relationship(BaseModel):
    """
    Represents a relationship between two entities in the knowledge graph.
    
    Relationships form edges connecting entity nodes and capture
    semantic connections between concepts.
    """
    id: str = Field(..., description="Unique relationship identifier")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: str = Field(..., description="Type of relationship (related_to, part_of, etc.)")
    description: Optional[str] = Field(default=None, description="Relationship description")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength (0-1)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    source_doc_ids: List[str] = Field(default_factory=list, description="Documents this relationship appears in")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class GraphDocument(BaseModel):
    """
    Enhanced document with extracted graph structure.
    
    Contains both the original document and extracted entities/relationships.
    """
    document: Document = Field(..., description="Original document")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


class GraphQueryResult(BaseModel):
    """
    Result of a graph-based query including relevant subgraph.
    
    Returns documents along with their entity/relationship context
    for enhanced understanding.
    """
    documents: List[Document] = Field(default_factory=list, description="Relevant documents")
    entities: List[Entity] = Field(default_factory=list, description="Related entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Related relationships")
    paths: List[List[str]] = Field(default_factory=list, description="Graph paths (entity ID chains)")
    score: float = Field(default=0.0, description="Overall relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""
    total_documents: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    entity_types: Dict[str, int] = Field(default_factory=dict)
    relationship_types: Dict[str, int] = Field(default_factory=dict)


class GraphIndexSummary(BaseModel):
    """Summary of graph indexing operation."""
    indexed_documents: int = 0
    extracted_entities: int = 0
    extracted_relationships: int = 0
    processing_time_ms: float = 0.0
    stats: Optional[GraphStats] = None



# ==================== TOOL SYSTEM TYPES ====================

class ToolType(str, Enum):
    """
    Classification of tool execution types.

    LOCAL: Python function in tools folder (executed locally)
    MCP: MCP connector endpoint (executed via MCP protocol)
    API: Direct HTTP API call (executed via HTTP request)
    CREWAI: CrewAI-native tool (executed within CrewAI context)
    """
    LOCAL = "local"
    MCP = "mcp"
    API = "api"
    CREWAI = "crewai"


class ToolDef(BaseModel):
    """
    Complete tool definition with execution configuration.

    This model defines everything needed to register, discover, validate,
    and execute a tool within the EchoAI system.

    Attributes:
        tool_id: Unique identifier (e.g., "tool_calculator", "tool_web_search")
        name: Human-readable tool name
        description: Detailed description of tool functionality
        tool_type: How the tool is executed (LOCAL, MCP, API, CREWAI)
        input_schema: JSON Schema for validating tool input
        output_schema: JSON Schema for validating tool output
        execution_config: Type-specific configuration for tool execution
        version: Semantic version of the tool
        tags: Categorization tags for discovery and filtering
        status: Current status (active, deprecated, disabled)
        metadata: Additional metadata (author, created_at, etc.)
    """
    # Required identification fields
    tool_id: str = Field(default="", description="Unique tool identifier")
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(..., description="Tool functionality description")

    # Tool type and configuration
    tool_type: ToolType = Field(default=ToolType.LOCAL, description="Execution type")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for input validation"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for output validation"
    )
    execution_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific execution configuration"
    )

    # Metadata fields
    version: str = Field(default="1.0", description="Tool version")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    status: str = Field(default="active", description="Tool status")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @field_validator('tool_id', mode='before')
    @classmethod
    def ensure_tool_id(cls, v: str, info) -> str:
        """Generate tool_id from name if not provided."""
        if not v and info.data.get('name'):
            # Generate tool_id from name: "Calculator" -> "tool_calculator"
            name = info.data['name']
            return f"tool_{name.lower().replace(' ', '_').replace('-', '_')}"
        return v

    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Ensure status is one of the allowed values."""
        allowed = {"active", "deprecated", "disabled"}
        if v.lower() not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator('input_schema', 'output_schema')
    @classmethod
    def validate_schema_structure(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that schema has basic JSON Schema structure if provided."""
        if v and 'type' not in v:
            # Add default type if schema has content but no type
            v['type'] = 'object'
        return v


class ToolRef(BaseModel):
    """Lightweight reference to a registered tool."""
    name: str
    tool_id: Optional[str] = None


class ToolResult(BaseModel):
    """
    Result of a tool execution.

    Attributes:
        name: Tool name that was executed
        tool_id: Tool identifier
        output: Execution output data
        success: Whether execution succeeded
        error: Error message if execution failed
        metadata: Execution metadata (timing, context, etc.)
    """
    name: str = Field(..., description="Tool name")
    tool_id: str = Field(default="", description="Tool identifier")
    output: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    success: bool = Field(default=True, description="Execution success flag")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata"
    )

class AppDef(BaseModel):
    name: str
    config: Dict[str, Any]

class App(BaseModel):
    id: str
    name: str
    config: Dict[str, Any]

class DeployResult(BaseModel):
    app_id: str
    env: str
    status: str

class AgentTemplate(BaseModel):
    """
    AgentTemplate loads and manages the agent.json template file.
    Returns the template as JSON/JSON object to be consumed by AgentService.

    Also exposes commonly accessed fields (name, icon, role, etc.) as
    direct attributes for compatibility with AgentService template matching.
    """
    template_path: Optional[Path] = None
    template_data: Optional[Dict[str, Any]] = None

    # Direct-access fields used by AgentService (populated from template_data)
    name: Optional[str] = None
    icon: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    prompt: Optional[str] = None
    tools: Optional[List[str]] = None
    variables: Optional[List[Dict[str, Any]]] = None
    settings: Optional[Dict[str, Any]] = None
    source: Optional[str] = None  # "template", "llm_generated", "user"

    def __init__(self, name: Optional[str] = None, template_path: Optional[str] = None, **kwargs):
        """
        Initialize AgentTemplate.

        Args:
            name: Optional agent name (for backward compatibility)
            template_path: Optional path to template file. Defaults to backend/data/templates/agent.json
        """
        # Determine template file path
        if template_path:
            path = Path(template_path)
        else:
            # Default to backend/data/templates/agent.json relative to echolib/types.py
            current_file = Path(__file__)  # backend/echolib/types.py
            path = current_file.parent.parent / "data" / "templates" / "agent.json"  # backend/data/templates/agent.json

        # Load the template JSON
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # Update name if provided
        if name and template_data:
            template_data['name'] = name

        # Extract direct-access fields from template_data
        direct_fields = {}
        if template_data:
            direct_fields['name'] = template_data.get('name', name)
            direct_fields['icon'] = template_data.get('icon')
            direct_fields['description'] = template_data.get('description')
            direct_fields['role'] = template_data.get('role')
            direct_fields['prompt'] = template_data.get('prompt')
            direct_fields['tools'] = template_data.get('tools')
            direct_fields['variables'] = template_data.get('variables')
            direct_fields['settings'] = template_data.get('settings')
            direct_fields['source'] = template_data.get('source')
        elif name:
            direct_fields['name'] = name

        # Allow kwargs to override direct fields
        for key in ['icon', 'description', 'role', 'prompt', 'tools', 'variables', 'settings', 'source']:
            if key in kwargs:
                direct_fields[key] = kwargs.pop(key)

        # Initialize with Pydantic
        super().__init__(template_path=path, template_data=template_data, **direct_fields, **kwargs)

    def to_json(self) -> Dict[str, Any]:
        """
        Returns the template as a JSON-compatible dictionary.
        This can be consumed by the AgentService to update all Agent fields.

        Returns:
            Dict[str, Any]: The complete template as a dictionary
        """
        if self.template_data is None:
            raise ValueError("Template data not loaded")
        return self.template_data.copy()

    def to_json_string(self, indent: int = 2) -> str:
        """
        Returns the template as a JSON string.

        Args:
            indent: Number of spaces for indentation

        Returns:
            str: The template as a formatted JSON string
        """
        return json.dumps(self.to_json(), indent=indent)

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'AgentTemplate':
        """
        Create an AgentTemplate instance from a JSON dictionary.

        Args:
            json_data: Dictionary containing the template data

        Returns:
            AgentTemplate: Instance with the provided template data
        """
        instance = cls.__new__(cls)
        instance.template_path = None
        instance.template_data = json_data.copy()
        # Populate direct-access fields
        instance.name = json_data.get('name')
        instance.icon = json_data.get('icon')
        instance.description = json_data.get('description')
        instance.role = json_data.get('role')
        instance.prompt = json_data.get('prompt')
        instance.tools = json_data.get('tools')
        instance.variables = json_data.get('variables')
        instance.settings = json_data.get('settings')
        instance.source = json_data.get('source')
        super(AgentTemplate, instance).__init__(
            template_path=None,
            template_data=instance.template_data,
            name=instance.name,
            icon=instance.icon,
            description=instance.description,
            role=instance.role,
            prompt=instance.prompt,
            tools=instance.tools,
            variables=instance.variables,
            settings=instance.settings,
            source=instance.source,
        )
        return instance

# Enhanced LLM Configuration
class LLMConfig(BaseModel):
    provider: str  # openai, anthropic, azure, local
    model: str
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None


class PromptRequest(BaseModel):
    prompt: str

# Enhanced Agent with full orchestrator support
class Agent(BaseModel):
    id: str
    name: str
    # NEW FIELDS (optional for backward compatibility)
    role: Optional[str] = None
    description: Optional[str] = None
    llm: Optional[LLMConfig] = None
    tools: Optional[List[str]] = None  # Tool IDs
    input_schema: Optional[List[Dict[str, Any]]] = None
    output_schema: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    permissions: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Workflow Connection (for graph topology)
class WorkflowConnection(BaseModel):
    from_agent: str
    to_agent: str
    condition: Optional[str] = None

# Workflow Hierarchy (for hierarchical execution)
class WorkflowHierarchy(BaseModel):
    master_agent: str
    delegation_order: List[str]

# HITL Configuration
class HITLConfig(BaseModel):
    enabled: bool = False
    review_points: Optional[List[str]] = None

# Workflow Validation Info
class WorkflowValidation(BaseModel):
    validated_by: Optional[str] = None
    validated_at: Optional[str] = None
    validation_hash: Optional[str] = None

# Enhanced Workflow with full orchestrator support
class Workflow(BaseModel):
    id: str
    name: str
    # NEW FIELDS (optional for backward compatibility)
    description: Optional[str] = None
    status: Optional[str] = "draft"  # draft, validated, testing, final
    version: Optional[str] = "0.1"
    execution_model: Optional[str] = None  # sequential, parallel, hierarchical, hybrid
    agents: Optional[List[str]] = None  # Agent IDs
    connections: Optional[List[WorkflowConnection]] = None
    hierarchy: Optional[WorkflowHierarchy] = None
    state_schema: Optional[Dict[str, str]] = None  # key -> type
    human_in_loop: Optional[HITLConfig] = None
    validation: Optional[WorkflowValidation] = None
    metadata: Optional[Dict[str, Any]] = None

# Enhanced Validation Result
class ValidationResult(BaseModel):
    ok: bool
    details: Optional[str] = None
    # NEW FIELDS
    valid: Optional[bool] = None  # Alias for ok
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

class ConnectorDef(BaseModel):
    name: str
    config: Dict[str, Any]

class ConnectorRef(BaseModel):
    name: str

class ConnectorResult(BaseModel):
    name: str
    result: Dict[str, Any]

class Health(BaseModel):
    status: str

# ==================== ORCHESTRATOR TYPES ====================

# Graph Visualization
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # agent, master_agent, start, end
    metadata: Optional[Dict[str, Any]] = None

class GraphEdge(BaseModel):
    source: str
    target: str
    condition: Optional[str] = None

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

# Workflow Execution
class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str
    mode: str  # "test" or "final"
    version: Optional[str] = None
    input_payload: Optional[Dict[str, Any]] = {}

class ExecutionResponse(BaseModel):
    run_id: str
    status: str
    output: Optional[Dict[str, Any]] = None

# Workflow Lifecycle
class WorkflowValidationRequest(BaseModel):
    workflow: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]

class SaveFinalRequest(BaseModel):
    workflow: Dict[str, Any]

class CloneWorkflowRequest(BaseModel):
    workflow_id: str
    from_version: str

# MCP Tool Definition (Enhanced)
class MCPToolConfig(BaseModel):
    server: str  # MCP server name or URL
    endpoint: str  # MCP endpoint
    version: Optional[str] = None

class MCPToolDefinition(BaseModel):
    tool_id: str
    name: str
    description: str
    mcp: MCPToolConfig
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    permissions: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    status: Optional[str] = "active"  # active, deprecated
    metadata: Optional[Dict[str, Any]] = None

# Workflow Metrics (Telemetry)
class WorkflowMetrics(BaseModel):
    workflow_id: str
    version: str
    total_duration_ms: float
    agent_metrics: Dict[str, Dict[str, Any]]


class CardCreate(BaseModel):
    agent_name: str = Field(..., min_length=1, description="Human-readable name of the agent")
    purpose: str = Field(..., min_length=1, description="What this agent is for")
    goals: Optional[List[str]] = Field(default=None, description="High-level goals (optional)")
    input_assumptions: Optional[str] = Field(default=None, description="Assumptions about inputs (optional)")
    output_definitions: Optional[str] = Field(default=None, description="Expected outputs / format (optional)")
    tools_apis_used: List[str] = Field(default_factory=list, description="Tools/APIs the agent uses")
    reasoning_working_style: Optional[str] = Field(default=None, description="How the agent reasons/works (optional)")
    error_handling_patterns: Optional[str] = Field(default=None, description="How to handle errors (optional)")
    example_workflows: Optional[List[str]] = Field(default=None, description="Example workflows (optional)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "agent_name": "Procurement Helper",
                "purpose": "Automate vendor onboarding and RFQ triage",
                "goals": [
                    "Reduce average onboarding time by 30%",
                    "Ensure RFQs are classified and routed correctly"
                ],
                "input_assumptions": "CSV uploads or SharePoint folders; user provides cost center",
                "output_definitions": "JSON summary plus XLSX export; posts status to Teams channel",
                "tools_apis_used": ["SharePoint", "SAP RFC", "OpenAI Embeddings"],
                "reasoning_working_style": "Reason step-by-step; prefer structured outputs",
                "error_handling_patterns": "Retry transient network errors up to 3 times; escalate with ticket",
                "example_workflows": [
                    "User uploads vendor CSV → validate → create SAP vendor → send summary",
                    "Classify RFQ email → extract entities → route to commodity owner"
                ]
            }
        }
    }

# ==================== TRADITIONAL RAG TYPES ====================

class TraditionalRAGLoadRequest(BaseModel):
    path: str = Field(..., description="File path or directory path to load documents from")

class TraditionalRAGLoadResponse(BaseModel):
    status: str = Field(..., description="Load status (success/error)")
    documents_loaded: int = Field(default=0)
    chunks_created: int = Field(default=0)
    sources: List[str] = Field(default_factory=list)
    path: str = Field(..., description="Upload summary or path that was loaded")

class TraditionalRAGQueryRequest(BaseModel):
    query: str = Field(..., description="User query/question")

class TraditionalRAGQueryResponse(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list)
    chunks_used: int = Field(default=0)

class TraditionalRAGStats(BaseModel):
    initialized: bool = Field(default=False)
    documents_loaded: int = Field(default=0)
    chunks_indexed: int = Field(default=0)
    sources: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)


class CardResponse(BaseModel):
    id: str
    agent_name: str
    purpose: str
    status: str = "created"