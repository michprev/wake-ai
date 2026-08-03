from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Literal, NamedTuple

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent
from openai import APIError, AsyncOpenAI, omit

from .api_session_utils import (
    SHELL_INPUT_SCHEMA,
    SSEServerParameters,
    StreamableHTTPServerParameters,
    compute_backoff_time,
    normalize_mcp_schema,
    open_mcp_clients,
    resolve_mcp_alias,
    run_sandboxed,
)
from .session_abc import SessionABC, FunctionTool
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_COMPACTIONS = 5
MAX_COMPACT_TRUNCATIONS = 3


class OpenRouterResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]
    final_message: str | None = None
    context_tokens: int | None = None  # prompt tokens of the latest request ≈ current context size
    compaction_count: int = 0


class OpenRouterModelUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost: float

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.cost = 0.0


class OpenRouterTotalTokenUsage:
    """Per-model token/cost accumulation. Cost is OpenRouter's authoritative
    usage.cost (USD) — never computed from token counts client-side."""

    usage: dict[str, OpenRouterModelUsage]

    def __init__(self) -> None:
        self.usage = {}

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage.values())

    def update(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
        cost: float,
    ) -> None:
        u = self.usage.setdefault(model, OpenRouterModelUsage())
        u.prompt_tokens += prompt_tokens
        u.completion_tokens += completion_tokens
        u.cached_tokens += cached_tokens
        u.reasoning_tokens += reasoning_tokens
        u.cost += cost

    def format_summary(self) -> str:
        lines = ["Token usage:"]
        for model, u in self.usage.items():
            lines.append(
                f"  {model}: input={u.prompt_tokens} (cached {u.cached_tokens}), "
                f"output={u.completion_tokens} (reasoning {u.reasoning_tokens}), "
                f"cost=${u.cost:.4f}"
            )
        lines.append(
            f"  total: input={sum(u.prompt_tokens for u in self.usage.values())} "
            f"(cached {sum(u.cached_tokens for u in self.usage.values())}), "
            f"output={sum(u.completion_tokens for u in self.usage.values())}, "
            f"cost=${self.total_cost:.4f}"
        )
        return "\n".join(lines)
