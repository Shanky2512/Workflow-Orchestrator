
from typing import Callable, Dict, Any

class Container:
    def __init__(self) -> None:
        self._p: Dict[str, Callable[[], Any]] = {}
    def register(self, key: str, provider: Callable[[], Any]) -> None:
        self._p[key] = provider
    def resolve(self, key: str) -> Any:
        if key not in self._p:
            raise KeyError(f'no provider for {key}')
        return self._p[key]()

container = Container()
