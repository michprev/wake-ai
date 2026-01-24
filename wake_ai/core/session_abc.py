from abc import ABC, abstractmethod
from typing import AsyncIterator, Any

from wake_ai.core.verbose_formatter import VerboseFormatter


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
