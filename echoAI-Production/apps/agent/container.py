from echolib.di import container
from echolib.adapters import OTelLogger, KeyVaultClient
from echolib.services import TemplateRepository, LangGraphBuilder, ToolService, AgentService

# Import new orchestrator services
from .registry.registry import AgentRegistry
from .factory.factory import AgentFactory
from .permissions.permissions import AgentPermissions
from .designer.agent_designer import AgentDesigner

# Import real ToolRegistry from the tool container
from apps.tool.container import get_registry as get_tool_registry

# Existing services (keep for backward compatibility)
_tpl = TemplateRepository()
_tool = ToolService()
_graph = LangGraphBuilder()

# New orchestrator services
_registry = AgentRegistry()
_factory = AgentFactory(tool_registry=get_tool_registry())
_permissions = AgentPermissions()
_designer = AgentDesigner()

# AgentService with registry and designer for template matching and update detection
_agentsvc = AgentService(
    tpl_repo=_tpl,
    graph_builder=_graph,
    registry=_registry,
    designer=_designer
)

# Register existing services
container.register('agent.service', lambda: _agentsvc)
container.register('tool.service', lambda: _tool)
container.register('cred.store', lambda: KeyVaultClient())
container.register('logger', lambda: OTelLogger())

# Register new orchestrator services
container.register('agent.registry', lambda: _registry)
container.register('agent.factory', lambda: _factory)
container.register('agent.permissions', lambda: _permissions)
container.register('agent.designer', lambda: _designer)
