
from echolib.di import container
from echolib.adapters import KafkaEventBus, OTelLogger
from echolib.services import TemplateRepository, LangGraphBuilder, AgentService, WorkflowService

# Import new orchestrator services
from .validator.validator import WorkflowValidator
from .storage.filesystem import WorkflowStorage
from .designer.designer import WorkflowDesigner
from .designer.compiler import WorkflowCompiler
from .runtime.executor import WorkflowExecutor
from .runtime.guards import RuntimeGuards
from .runtime.hitl import HITLManager, HITLDBManager
from .visualization.graph_mapper import GraphMapper
from .visualization.graph_editor import GraphEditor
from .visualization.node_mapper import NodeMapper

# Import agent registry from agent app
from apps.agent.registry.registry import AgentRegistry

# Existing services (keep for backward compatibility)
_bus = KafkaEventBus()
_tpl = TemplateRepository()
_graph = LangGraphBuilder()
_agentsvc = AgentService(_tpl, _graph)
_wfsvc = WorkflowService(_agentsvc, _bus)

# New orchestrator services
_storage = WorkflowStorage()
_agent_registry = AgentRegistry()


def _get_tool_registry():
    """Lazily get tool registry from DI container to avoid circular imports."""
    return container.resolve('tool.registry')


def _get_tool_registry_dict():
    """Get tool registry as dict for validator (tool_id -> tool_def)."""
    try:
        registry = _get_tool_registry()
        return {t.tool_id: t.model_dump() for t in registry.list_all()}
    except Exception:
        return {}


_validator = WorkflowValidator(tool_registry_provider=_get_tool_registry_dict)
_designer = WorkflowDesigner(agent_registry=_agent_registry)
_compiler = WorkflowCompiler()
_guards = RuntimeGuards()
_hitl = HITLDBManager()  # Extends HITLManager with DB persistence for HITL Node system
_graph_mapper = GraphMapper(storage=_storage)
_graph_editor = GraphEditor()
_node_mapper = NodeMapper(tool_registry=_get_tool_registry())
_executor = WorkflowExecutor(
    storage=_storage,
    compiler=_compiler,
    agent_registry=_agent_registry,
    guards=_guards
)

# Register existing services
container.register('event.bus', lambda: _bus)
container.register('agent.service', lambda: _agentsvc)
container.register('workflow.service', lambda: _wfsvc)
container.register('logger', lambda: OTelLogger())

# Register new orchestrator services
container.register('workflow.storage', lambda: _storage)
container.register('workflow.validator', lambda: _validator)
container.register('workflow.designer', lambda: _designer)
container.register('workflow.compiler', lambda: _compiler)
container.register('workflow.guards', lambda: _guards)
container.register('workflow.hitl', lambda: _hitl)
container.register('workflow.graph_mapper', lambda: _graph_mapper)
container.register('workflow.graph_editor', lambda: _graph_editor)
container.register('workflow.node_mapper', lambda: _node_mapper)
container.register('workflow.executor', lambda: _executor)
