
from echolib.di import container
from echolib.adapters import KafkaEventBus, OTelLogger
from echolib.services import DocumentStore, RAGService, ToolService, LLMService, AgentService, TemplateRepository, LangGraphBuilder, WorkflowService, ConnectorManager, MCPConnector,APIConnector, ChatOrchestrator

_bus = KafkaEventBus()
_store = DocumentStore()
_rag = RAGService(_store)
_tool = ToolService()
_llm = LLMService(_tool)
_tpl = TemplateRepository()
_graph = LangGraphBuilder()
_agent = AgentService(_tpl, _graph)
_wf = WorkflowService(_agent, _bus)
_mgr = ConnectorManager()
_orch = ChatOrchestrator(_bus, _llm, _rag, _wf, _agent, _mgr)

container.register('chat.orchestrator', lambda: _orch)
container.register('event.bus', lambda: _bus)
container.register('logger', lambda: OTelLogger())
