from abc import ABC, abstractmethod
from typing import AsyncIterator, Any, NamedTuple, Callable, Awaitable

from wake_ai.core.verbose_formatter import VerboseFormatter


class FunctionTool(NamedTuple):
    name: str
    input_schema: dict[str, Any]
    description: str | None
    handler: Callable[..., Awaitable[Any]]


class SessionABC(ABC):
    @property
    @abstractmethod
    def session_id(self) -> str | None:
        pass

    @abstractmethod
    def query(self, prompt: str, model: str, max_cost: float | None, formatter: VerboseFormatter) -> AsyncIterator[Any]:  # TODO
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
