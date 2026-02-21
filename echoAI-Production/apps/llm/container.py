from echolib.di import container
from echolib.services import ToolService, LLMService
from echolib.adapters import OTelLogger

_tool = ToolService()
_llm = LLMService(_tool)
container.register('tool.service', lambda: _tool)
container.register('llm.service', lambda: _llm)
container.register('logger', lambda: OTelLogger())
