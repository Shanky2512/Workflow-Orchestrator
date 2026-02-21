
from typing import Callable, Dict, List
from .interfaces import IEventBus, ISessionStore, IAuthProvider, ICredentialStore, ILogger
from .types import Event, Session, UserContext
from .config import settings
from .logging import logger
from .security import decode_token

class KafkaEventBus(IEventBus):
    def __init__(self) -> None:
        self._topics: Dict[str, List[Callable[[Event], None]]] = {}
    def publish(self, topic: str, event: Event) -> None:
        for h in self._topics.get(topic, []):
            try:
                h(event)
            except Exception as ex:
                logger.error(f'bus handler error: {ex}')
    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        self._topics.setdefault(topic, []).append(handler)
    def createTopic(self, topic: str) -> None:
        self._topics.setdefault(topic, [])

class MemcachedSessionStore(ISessionStore):
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
    def createSession(self, userId: str, data: dict) -> Session:
        sid = f'sess_{len(self._sessions)+1}'
        s = Session(id=sid, user_id=userId, data=data)
        self._sessions[sid] = s
        return s
    def getSession(self, sessionId: str) -> Session | None:
        return self._sessions.get(sessionId)
    def invalidateSession(self, sessionId: str) -> None:
        self._sessions.pop(sessionId, None)

class AzureADAuth(IAuthProvider):
    def validateToken(self, jwt_str: str) -> UserContext:
        payload = decode_token(jwt_str)
        if not payload:
            raise ValueError('invalid token')
        return UserContext(user_id=payload['sub'], email=payload.get('email',''))
    def getLoginUrl(self) -> str:
        return 'https://login.microsoftonline.com/common/oauth2/authorize'

class KeyVaultClient(ICredentialStore):
    def __init__(self) -> None:
        self._secrets: Dict[str, str] = {}
    def getSecret(self, name: str) -> str:
        return self._secrets.get(name, '')
    def setSecret(self, name: str, value: str) -> None:
        self._secrets[name] = value
    def rotateKey(self, name: str) -> None:
        self._secrets[name] = self._secrets.get(name, '') + '_rotated'

class OTelLogger(ILogger):
    def info(self, msg: str, ctx: dict) -> None:
        logger.info(msg + ' ' + str(ctx))
    def warn(self, msg: str, ctx: dict) -> None:
        logger.warning(msg + ' ' + str(ctx))
    def error(self, msg: str, ctx: dict) -> None:
        logger.error(msg + ' ' + str(ctx))
    def trace(self, span: str, ctx: dict) -> None:
        logger.debug(f'TRACE {span} {ctx}')
