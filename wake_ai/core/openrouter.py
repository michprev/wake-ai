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


def _to_plain(value: Any) -> Any:
    """OpenRouter extra fields normally arrive as plain dicts/lists, but be
    defensive about pydantic models leaking through."""
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


class ChatTurnAccumulator:
    """Accumulates chat-completions stream deltas into one complete assistant turn.

    OpenRouter extensions (reasoning, reasoning_details, annotations) are not in
    the SDK's typed Delta model — they surface via pydantic extra fields, hence
    the getattr access.
    """

    content: str | None
    reasoning: str | None
    finish_reason: str | None
    annotations: list[dict[str, Any]]

    def __init__(self) -> None:
        self.content = None
        self.reasoning = None
        self.finish_reason = None
        self.annotations = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._reasoning_details: list[dict[str, Any]] = []
        self._reasoning_by_index: dict[int, dict[str, Any]] = {}

    @property
    def reasoning_details(self) -> list[dict[str, Any]]:
        return self._reasoning_details

    def add_chunk(self, chunk: Any) -> None:
        if not chunk.choices:
            return  # usage-only final chunk
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            self.finish_reason = choice.finish_reason
        delta = choice.delta
        if delta is None:
            return

        if delta.content:
            self.content = (self.content or "") + delta.content

        reasoning = getattr(delta, "reasoning", None)
        if isinstance(reasoning, str) and reasoning:
            self.reasoning = (self.reasoning or "") + reasoning

        details = getattr(delta, "reasoning_details", None)
        if details:
            for detail in details:
                self._merge_reasoning_detail(dict(_to_plain(detail)))

        annotations = getattr(delta, "annotations", None)
        if annotations:
            self.annotations.extend(_to_plain(a) for a in annotations)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                entry = self._tool_calls.setdefault(tc.index, {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if tc.id:
                    entry["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        entry["function"]["name"] += tc.function.name
                    if tc.function.arguments:
                        entry["function"]["arguments"] += tc.function.arguments

    def _merge_reasoning_detail(self, detail: dict[str, Any]) -> None:
        index = detail.get("index")
        if index is not None and index in self._reasoning_by_index:
            existing = self._reasoning_by_index[index]
            for key, value in detail.items():
                if key in ("text", "data", "summary") and isinstance(value, str):
                    existing[key] = existing.get(key, "") + value
                elif value is not None:
                    existing[key] = value
        else:
            self._reasoning_details.append(detail)
            if index is not None:
                self._reasoning_by_index[index] = detail

    def tool_calls(self) -> list[dict[str, Any]]:
        return [self._tool_calls[i] for i in sorted(self._tool_calls)]

    def assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        calls = self.tool_calls()
        if calls:
            message["tool_calls"] = calls
        if self._reasoning_details:
            # replayed verbatim & unreordered — required for reasoning models
            message["reasoning_details"] = self._reasoning_details
        return message
