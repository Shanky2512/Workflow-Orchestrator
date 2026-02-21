
from abc import ABC, abstractmethod
from typing import Callable, Optional
from .types import *  # noqa

class IEventBus(ABC):
    @abstractmethod
    def publish(self, topic: str, event: Event) -> None: ...
    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None: ...
    @abstractmethod
    def createTopic(self, topic: str) -> None: ...

class ISessionStore(ABC):
    @abstractmethod
    def createSession(self, userId: str, data: dict) -> Session: ...
    @abstractmethod
    def getSession(self, sessionId: str) -> Session | None: ...
    @abstractmethod
    def invalidateSession(self, sessionId: str) -> None: ...

class IAuthProvider(ABC):
    @abstractmethod
    def validateToken(self, jwt_str: str) -> UserContext: ...
    @abstractmethod
    def getLoginUrl(self) -> str: ...

class ICredentialStore(ABC):
    @abstractmethod
    def getSecret(self, name: str) -> str: ...
    @abstractmethod
    def setSecret(self, name: str, value: str) -> None: ...
    @abstractmethod
    def rotateKey(self, name: str) -> None: ...

class ILogger(ABC):
    @abstractmethod
    def info(self, msg: str, ctx: dict) -> None: ...
    @abstractmethod
    def warn(self, msg: str, ctx: dict) -> None: ...
    @abstractmethod
    def error(self, msg: str, ctx: dict) -> None: ...
    @abstractmethod
    def trace(self, span: str, ctx: dict) -> None: ...

class IAgentService(ABC):
    @abstractmethod
    def createFromPrompt(self, prompt: str, template: Optional["AgentTemplate"] = None) -> "Agent":
        """Create an Agent from a freeform prompt and a template."""
        ...
    @abstractmethod
    def createFromCanvasCard(self, cardJSON: dict, template: Optional["AgentTemplate"] = None) -> "Agent":
        """Create an Agent from a canvas card JSON using an optional template."""
        ...
    def validateA2A(self, agent: "Agent") -> "ValidationResult":
        """Validate agent-to-agent (A2A) configuration."""
        ...
    @abstractmethod
    def listAgents(self) -> List["Agent"]:
        """Return all registered agents."""