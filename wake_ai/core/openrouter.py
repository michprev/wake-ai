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


class OpenRouterSession(SessionABC):
    execution_dir: Path
    fork_session: "OpenRouterSession | None"
    instructions: str | None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    reasoning_max_tokens: int | None
    max_retries: int
    request_timeout: float | None
    tool_call_timeout: float | None
    mcp_call_timeout: float | None
    tools: dict[str, FunctionTool]
    shell: bool
    web_search: bool
    web_search_engine: Literal["native", "exa", "firecrawl", "parallel", "perplexity"] | None
    web_search_max_results: int | None
    mcps: dict[str, ClientSession | StdioServerParameters | SSEServerParameters | StreamableHTTPServerParameters]
    shell_network_access: bool
    writable_roots: list[Path | str]
    provider: dict[str, Any] | None
    extra_body: dict[str, Any] | None

    client: AsyncOpenAI
    conversation: list[dict[str, Any]]
    total_token_usage: OpenRouterTotalTokenUsage

    def __init__(
        self,
        execution_dir: Path,
        *,
        fork_session: "OpenRouterSession | None" = None,
        instructions: str | None = None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None,
        reasoning_max_tokens: int | None = None,
        max_retries: int = 5,
        request_timeout: float | None = 1200,  # 20 minutes
        tool_call_timeout: float | None = 300,  # 5 minutes
        mcp_call_timeout: float | None = 300,  # 5 minutes
        tools: list[FunctionTool] | None = None,
        shell: bool = True,
        web_search: bool = False,
        web_search_engine: Literal["native", "exa", "firecrawl", "parallel", "perplexity"] | None = None,
        web_search_max_results: int | None = None,
        mcps: dict[str, ClientSession | StdioServerParameters | SSEServerParameters | StreamableHTTPServerParameters] | None = None,
        shell_network_access: bool = False,
        writable_roots: list[Path | str] | None = None,
        provider: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        if reasoning_effort is not None and reasoning_max_tokens is not None:
            raise ValueError("reasoning_effort and reasoning_max_tokens are mutually exclusive")

        self.execution_dir = execution_dir
        self.fork_session = fork_session
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.reasoning_max_tokens = reasoning_max_tokens
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.tool_call_timeout = tool_call_timeout
        self.mcp_call_timeout = mcp_call_timeout
        self.shell = shell
        self.web_search = web_search
        self.web_search_engine = web_search_engine
        self.web_search_max_results = web_search_max_results
        self.mcps = mcps or {}
        self.shell_network_access = shell_network_access
        self.writable_roots = writable_roots or []
        self.provider = provider
        self.extra_body = extra_body

        self.tools = {}
        if tools is not None:
            for tool in tools:
                self.tools[tool.name] = tool

        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=self.request_timeout,
            default_headers={
                "HTTP-Referer": "https://github.com/wakehacker/wake-ai",
                "X-Title": "Wake AI",
            },
        )
        if fork_session is not None:
            if not fork_session.conversation:
                raise ValueError("Forking from OpenRouterSession with empty conversation")
            self.conversation = fork_session.conversation.copy()
        else:
            self.conversation = []
        self._session_id: str | None = None
        self.total_token_usage = OpenRouterTotalTokenUsage()
        self._last_message: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def reset(self) -> None:
        """Reset conversation state and session ID."""
        self.conversation = []
        self._session_id = None

    def _request_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.instructions is not None:
            messages.append({"role": "system", "content": self.instructions})
        return messages + self.conversation

    def _build_extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            body["reasoning"] = {"effort": self.reasoning_effort}
        elif self.reasoning_max_tokens is not None:
            body["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
        if self.provider is not None:
            body["provider"] = self.provider
        if self.extra_body:
            body.update(self.extra_body)
        return body

    async def _collect_tools(
        self, mcp_clients: dict[str, ClientSession]
    ) -> tuple[list[dict[str, Any]], dict[str, tuple[ClientSession, str]]]:
        """Build the chat-completions tools array + MCP alias registry."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools.values()
        ]

        reserved: set[str] = set(self.tools.keys()) | {"shell"}
        mcp_tools: dict[str, tuple[ClientSession, str]] = {}
        for server_name, client in mcp_clients.items():
            cursor = None
            while True:
                response = await client.list_tools(cursor=cursor)
                for tool in response.tools:
                    alias = resolve_mcp_alias(server_name, tool.name, reserved, mcp_tools)
                    if alias is None:
                        continue
                    mcp_tools[alias] = (client, tool.name)
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": alias,
                            "description": tool.description,
                            "parameters": normalize_mcp_schema(tool.inputSchema),
                        },
                    })
                if response.nextCursor is None:
                    break
                cursor = response.nextCursor

        if self.shell:
            tools.append({
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Execute multiple shell commands in parallel",
                    "parameters": SHELL_INPUT_SCHEMA,
                },
            })

        if self.web_search:
            # OpenRouter server tool: executed server-side, agentic (0-N calls).
            web_tool: dict[str, Any] = {"type": "openrouter:web_search"}
            parameters: dict[str, Any] = {}
            if self.web_search_engine is not None:
                parameters["engine"] = self.web_search_engine
            if self.web_search_max_results is not None:
                parameters["max_results"] = self.web_search_max_results
            if parameters:
                web_tool["parameters"] = parameters
            tools.append(web_tool)

        return tools, mcp_tools

    def _record_usage(self, model: str, usage: Any) -> int:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_tokens = usage.prompt_tokens or 0
        self.total_token_usage.update(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.completion_tokens or 0,
            cached_tokens=(getattr(prompt_details, "cached_tokens", 0) or 0) if prompt_details is not None else 0,
            reasoning_tokens=(getattr(completion_details, "reasoning_tokens", 0) or 0) if completion_details is not None else 0,
            cost=float(getattr(usage, "cost", 0.0) or 0.0),
        )
        return prompt_tokens

    async def query(self, prompt, model, max_cost, formatter):  # implemented in a later task
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator per SessionABC
