from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import AsyncIterator, NamedTuple, Literal, Any, cast

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.types import TextContent
from openai import APIError, AsyncOpenAI, omit
from openai.types.responses import EasyInputMessageParam, FunctionToolParam, ResponseIncompleteEvent, ResponseAudioDeltaEvent, ResponseCompletedEvent, ResponseCreatedEvent, ResponseFunctionCallArgumentsDeltaEvent, ResponseFunctionCallArgumentsDoneEvent, ResponseFunctionShellCallOutputContentParam, ResponseFunctionShellToolCall, ResponseFunctionToolCall, ResponseInProgressEvent, ResponseOutputItemAddedEvent, ResponseOutputItemDoneEvent, ResponseOutputMessage, ResponseOutputRefusal, ResponseOutputText, ResponseReasoningItem, ResponseTextConfigParam, ResponseTextDeltaEvent, ResponseTextDoneEvent, ToolParam, WebSearchToolParam
from openai.types.responses.function_shell_tool import FunctionShellTool
from openai.types.responses.response_function_shell_call_output_content_param import OutcomeExit, OutcomeTimeout
from openai.types.responses.response_function_web_search import ResponseFunctionWebSearch
from openai.types.responses.response_input_item_param import ResponseInputItemParam
from openai.types.responses.response_input_param import FunctionCallOutput, ShellCallOutput
from openai.types.shared_params import ResponseFormatText
from openai.types.shared_params.reasoning import Reasoning

from .api_session_utils import (
    SHELL_INPUT_SCHEMA as LEGACY_SHELL_INPUT_SCHEMA,
    SSEServerParameters,
    StreamableHTTPServerParameters,
    compute_backoff_time as _compute_backoff_time,
    mcp_tool_alias as _mcp_tool_alias,
    normalize_mcp_schema as _normalize_mcp_schema,
    open_mcp_clients,
    resolve_mcp_alias as _resolve_mcp_alias,
    run_sandboxed,
)
from .codex_pricing import GPT_PRICING
from .session_abc import SessionABC, FunctionTool
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]
    final_message: str | None = None
    context_tokens: int | None = None  # input tokens of the latest request ≈ current context size
    compaction_count: int = 0


MAX_COMPACTIONS = 5

# OpenAI bills the hosted `web_search` tool per call ($10 / 1,000 calls). This
# fee is NOT reflected in token usage, so we count web-search calls separately
# and add it into total_cost. Each ResponseFunctionWebSearch item (search /
# open_page / find action) is one billable call.
WEB_SEARCH_COST_PER_CALL = 0.01

# Models that support OpenAI's native shell tool (FunctionShellTool).
# All others fall back to the legacy "shell" function workaround.
_NATIVE_SHELL_MODEL_PREFIXES = ("gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5.6")


class OpenAITokenUsage:
    input_tokens_total: int
    input_tokens_cached: int
    input_tokens_cache_write: int
    output_tokens: int
    cost: float

    def __init__(self, input_tokens_total: int = 0, input_tokens_cached: int = 0, output_tokens: int = 0, input_tokens_cache_write: int = 0) -> None:
        self.input_tokens_total = input_tokens_total
        self.input_tokens_cached = input_tokens_cached
        self.input_tokens_cache_write = input_tokens_cache_write
        self.output_tokens = output_tokens
        self.cost = 0.0

    def update(self, model: str, tier: Literal["flex", "standard", "priority"], input_tokens_total: int, input_tokens_cached: int, output_tokens: int, input_tokens_cache_write: int = 0) -> None:
        self.input_tokens_total += input_tokens_total
        self.input_tokens_cached += input_tokens_cached
        self.input_tokens_cache_write += input_tokens_cache_write
        self.output_tokens += output_tokens

        pricing = GPT_PRICING[model][tier]
        if pricing.cache_write_mtoken_cost is not None:
            # GPT-5.6+: cache writes are billed separately (1.25x input rate).
            regular = input_tokens_total - input_tokens_cached - input_tokens_cache_write
            input_cost = regular * pricing.input_mtoken_cost / 1e6
            cached_input_cost = input_tokens_cached * pricing.cached_input_mtoken_cost / 1e6
            cache_write_cost = input_tokens_cache_write * pricing.cache_write_mtoken_cost / 1e6
        else:
            # Pre-5.6: cache writes are folded into the regular input price.
            input_cost = (input_tokens_total - input_tokens_cached) * pricing.input_mtoken_cost / 1e6
            cached_input_cost = input_tokens_cached * pricing.cached_input_mtoken_cost / 1e6
            cache_write_cost = 0.0
        output_cost = output_tokens * pricing.output_mtoken_cost / 1e6

        if model in {"gpt-5.4", "gpt-5.4-pro", "gpt-5.5", "gpt-5.5-pro", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} and input_tokens_total >= 272_000:
            input_cost *= 2
            cached_input_cost *= 2
            cache_write_cost *= 2
            output_cost *= 1.5

        self.cost += input_cost + cached_input_cost + cache_write_cost + output_cost


class OpenAITotalTokenUsage:
    usage: dict[str, dict[Literal["flex", "standard", "priority"], OpenAITokenUsage]]

    def __init__(self) -> None:
        self.usage = {}
        self.web_search_calls = 0

    @property
    def total_tokens(self) -> OpenAITokenUsage:
        return OpenAITokenUsage(
            input_tokens_total=sum(usage.input_tokens_total for tiered_usage in self.usage.values() for usage in tiered_usage.values()),
            input_tokens_cached=sum(usage.input_tokens_cached for tiered_usage in self.usage.values() for usage in tiered_usage.values()),
            output_tokens=sum(usage.output_tokens for tiered_usage in self.usage.values() for usage in tiered_usage.values()),
            input_tokens_cache_write=sum(usage.input_tokens_cache_write for tiered_usage in self.usage.values() for usage in tiered_usage.values()),
        )

    @property
    def total_cost(self) -> float:
        total_cost = self.web_search_calls * WEB_SEARCH_COST_PER_CALL
        for model, tiered_usage in self.usage.items():
            for tier, usage in tiered_usage.items():
                total_cost += usage.cost
        return total_cost

    def update(
        self,
        input_tokens_total: int,
        input_tokens_cached: int,
        output_tokens: int,
        model: str,
        tier: Literal["flex", "standard", "priority"],
        input_tokens_cache_write: int = 0,
    ) -> None:
        if model not in self.usage:
            self.usage[model] = {
                "flex": OpenAITokenUsage(),
                "standard": OpenAITokenUsage(),
                "priority": OpenAITokenUsage(),
            }
        self.usage[model][tier].update(model, tier, input_tokens_total, input_tokens_cached, output_tokens, input_tokens_cache_write)

    def format_summary(self) -> str:
        lines = ["Token usage:"]
        for model, tiered_usage in self.usage.items():
            for tier, usage in tiered_usage.items():
                if usage.input_tokens_total == 0 and usage.output_tokens == 0:
                    continue
                cached_str = f"cached {usage.input_tokens_cached}"
                if usage.input_tokens_cache_write:
                    cached_str += f", written {usage.input_tokens_cache_write}"
                lines.append(
                    f"  {model} ({tier}): input={usage.input_tokens_total} "
                    f"({cached_str}), output={usage.output_tokens}, "
                    f"cost=${usage.cost:.4f}"
                )
        total = self.total_tokens
        total_cached_str = f"cached {total.input_tokens_cached}"
        if total.input_tokens_cache_write:
            total_cached_str += f", written {total.input_tokens_cache_write}"
        summary = (
            f"  total: input={total.input_tokens_total} "
            f"({total_cached_str}), output={total.output_tokens}"
        )
        if self.web_search_calls:
            summary += f", web_search_calls={self.web_search_calls}"
        summary += f", cost=${self.total_cost:.4f}"
        lines.append(summary)
        return "\n".join(lines)


class ResponseIncompleteError(Exception):
    reason: str | None

    def __init__(self, reason: str | None) -> None:
        self.reason = reason

    def __str__(self) -> str:
        return f"Response incomplete: {self.reason}"


class OpenAISession(SessionABC):
    execution_dir: Path
    fork_session: OpenAISession | None
    instructions: str | None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None
    output_verbosity: Literal["low", "medium", "high"] | None
    max_retries: int
    request_timeout: float | None
    tool_call_timeout: float | None
    mcp_call_timeout: float | None
    tools: dict[str, FunctionTool]
    shell: bool
    web_search: bool
    web_search_context_size: Literal["low", "medium", "high"] | None
    mcps: dict[str, ClientSession | StdioServerParameters | SSEServerParameters | StreamableHTTPServerParameters]
    shell_network_access: bool
    writable_roots: list[Path | str]

    client: AsyncOpenAI
    _session_id: str | None
    conversation: list[ResponseInputItemParam]
    total_token_usage: OpenAITotalTokenUsage
    _last_message: str | None

    def __init__(
        self,
        execution_dir: Path,
        *,
        fork_session: OpenAISession | None = None,
        instructions: str | None = None,
        service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None,
        reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None,
        reasoning_summary: Literal["auto", "concise", "detailed"] | None = None,
        output_verbosity: Literal["low", "medium", "high"] | None = None,
        max_retries: int = 5,
        request_timeout: float | None = 1200,  # 20 minutes
        tool_call_timeout: float | None = 300,  # 5 minutes
        mcp_call_timeout: float | None = 300,  # 5 minutes
        tools: list[FunctionTool] | None = None,
        shell: bool = True,
        web_search: bool = False,
        web_search_context_size: Literal["low", "medium", "high"] | None = None,
        mcps: dict[str, ClientSession | StdioServerParameters | SSEServerParameters | StreamableHTTPServerParameters] | None = None,
        shell_network_access: bool = False,
        writable_roots: list[Path | str] | None = None,
    ):
        self.execution_dir = execution_dir
        self.fork_session = fork_session
        self.instructions = instructions
        self.service_tier = service_tier
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.output_verbosity = output_verbosity
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.tool_call_timeout = tool_call_timeout
        self.mcp_call_timeout = mcp_call_timeout
        self.shell = shell
        self.web_search = web_search
        self.web_search_context_size = web_search_context_size
        self.mcps = mcps or {}
        self.shell_network_access = shell_network_access
        self.writable_roots = writable_roots or []

        self.tools = {}
        if tools is not None:
            for tool in tools:
                self.tools[tool.name] = tool

        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=self.request_timeout,
        )
        if fork_session is not None:
            if not fork_session.conversation:
                raise ValueError("Forking from OpenAISession with empty conversation")
            self.conversation = fork_session.conversation.copy()
        else:
            self.conversation = []
        self._session_id = None
        self.total_token_usage = OpenAITotalTokenUsage()
        self._last_message = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def _call_tool(self, tool_call: ResponseFunctionToolCall) -> Any:
        tool = self.tools[tool_call.name]

        if tool_call.arguments:
            input = json.loads(tool_call.arguments)
        else:
            input = {}

        return await asyncio.wait_for(tool.handler(**input), timeout=self.tool_call_timeout)

    async def _call_shell(self, commands: list[str], timeout_ms: int | None, max_output_length: int | None) -> list[ResponseFunctionShellCallOutputContentParam]:
        timeout = timeout_ms / 1000.0 if timeout_ms is not None else 10 * 60  # seconds (10 min default)
        max_length = max_output_length if max_output_length is not None else 1000000

        output: list[ResponseFunctionShellCallOutputContentParam] = []

        for command in commands:
            try:
                writable_roots = [Path(root) for root in self.writable_roots]
                stdout, stderr, returncode = await run_sandboxed(command, self.shell_network_access, writable_roots, timeout, self.execution_dir)

                output.append(ResponseFunctionShellCallOutputContentParam(
                    outcome=OutcomeExit(
                        exit_code=returncode,
                        type="exit",
                    ),
                    stderr=stderr[:max_length],
                    stdout=stdout[:max_length],
                ))
            except asyncio.TimeoutError:
                output.append(ResponseFunctionShellCallOutputContentParam(
                    outcome=OutcomeTimeout(
                        type="timeout",
                    ),
                    stderr="",
                    stdout="",
                ))

        return output

    async def _call_legacy_shell(self, arguments: str) -> list[ResponseFunctionShellCallOutputContentParam]:
        args = json.loads(arguments)
        if "commands" not in args:
            raise ValueError("commands not found in arguments")
        if "timeout_ms" not in args:
            raise ValueError("timeout_ms not found in arguments")

        return await self._call_shell(args["commands"], args["timeout_ms"], args.get("max_output_length", None))

    async def _call_mcp_tool(self, client: ClientSession, tool_name: str, arguments: str) -> Any:
        args = json.loads(arguments) if arguments else {}

        result = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=self.mcp_call_timeout)

        if result.structuredContent is not None:
            return {"isError": result.isError, "structuredContent": result.structuredContent}
        else:
            return {"isError": result.isError, "content": "\n".join(c.text for c in result.content if isinstance(c, TextContent))}

    async def _iter_response_events(self, stream: Any, timeout: float | None) -> AsyncIterator[Any]:
        """Iterate a streaming response, enforcing an idle timeout between events.

        The SDK/httpx ``timeout`` is not reliably enforced for streaming
        responses: when OpenAI's backend keeps the connection alive but stalls
        mid-response, no read gap is observed and the iterator blocks forever.
        We wrap each ``__anext__`` in ``asyncio.wait_for`` so a stalled stream
        raises ``asyncio.TimeoutError`` (recoverable via the retry logic) and
        always close the underlying stream to release the connection.
        """
        try:
            it = stream.__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(it.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    return
                yield event
        finally:
            await stream.close()

    async def _stream_messages(self, prompt: str, model: str, mcp_clients: dict[str, ClientSession], formatter: VerboseFormatter) -> AsyncIterator[tuple[float, int | None, int]]:
        tools: list[ToolParam] = [
            FunctionToolParam(
                name=tool.name,
                parameters=tool.input_schema,
                description=tool.description,
                type="function",
                strict=False,
            ) for tool in self.tools.values()
        ]

        _use_native_shell = model.startswith(_NATIVE_SHELL_MODEL_PREFIXES)

        _reserved_aliases: set[str] = set(self.tools.keys())
        mcp_tools: dict[str, tuple[ClientSession, str]] = {}  # alias -> (client, original_tool_name)

        for server_name, client in mcp_clients.items():
            cursor = None
            while True:
                response = await client.list_tools(cursor=cursor)
                for tool in response.tools:
                    alias = _resolve_mcp_alias(
                        server_name, tool.name, _reserved_aliases, mcp_tools
                    )
                    if alias is None:
                        continue

                    mcp_tools[alias] = (client, tool.name)
                    tools.append(FunctionToolParam(
                        name=alias,
                        parameters=_normalize_mcp_schema(tool.inputSchema),
                        description=tool.description,
                        type="function",
                        strict=False,
                    ))
                if response.nextCursor is None:
                    break
                cursor = response.nextCursor

        if self.shell:
            if not _use_native_shell:
                tools.append(FunctionToolParam(
                    name="shell",
                    parameters=LEGACY_SHELL_INPUT_SCHEMA,
                    description="Execute multiple shell commands in parallel",
                    type="function",
                    strict=False,
                ))
            else:
                tools.append(FunctionShellTool(type="shell"))

        if self.web_search:
            tool = WebSearchToolParam(
                type="web_search",
            )
            if self.web_search_context_size is not None:
                tool["search_context_size"] = self.web_search_context_size
            tools.append(tool)

        retry = 0
        service_tier = self.service_tier or "standard"
        last_flex_successful = True
        compact_reason: str | None = None
        compact_count = 0
        context_tokens: int | None = None
        self._last_message = None

        self.conversation.append(EasyInputMessageParam(content=prompt, role="user", type="message"))

        while True:
            # Defined before the try so the except handler can always cancel
            # any tasks spawned mid-stream, even if create() itself failed.
            tool_calls: dict[str, asyncio.Task[Any]] = {}
            shell_calls: dict[str, tuple[asyncio.Task[list[ResponseFunctionShellCallOutputContentParam]], ResponseFunctionShellToolCall]] = {}
            try:
                if compact_reason is not None:
                    compact_count += 1
                    if compact_count > MAX_COMPACTIONS:
                        raise RuntimeError("max_compactions_reached")

                    formatter.print_system_message(f"Compacting conversation ({compact_count}/{MAX_COMPACTIONS})")

                    compacted = await self.client.responses.compact(
                        model=model,
                        input=self.conversation,
                        instructions=self.instructions or omit,
                        timeout=self.request_timeout,
                    )
                    self.total_token_usage.update(
                        input_tokens_total=compacted.usage.input_tokens,
                        input_tokens_cached=compacted.usage.input_tokens_details.cached_tokens,
                        output_tokens=compacted.usage.output_tokens,
                        model=model,
                        tier="standard",
                        input_tokens_cache_write=compacted.usage.input_tokens_details.cache_write_tokens,
                    )
                    self.conversation = cast(
                        list[ResponseInputItemParam],
                        [item.model_dump(mode="json", exclude_none=True) for item in compacted.output],
                    )
                    if compact_reason == "max_output_tokens":
                        self.conversation.append(EasyInputMessageParam(content="continue", role="user", type="message"))
                    compact_reason = None

                stream = await asyncio.wait_for(
                    self.client.responses.create(
                        input=self.conversation,
                        model=model,
                        instructions=self.instructions or omit,
                        service_tier="default" if service_tier == "standard" else service_tier,
                        tools=tools or omit,
                        stream=True,
                        store=False,
                        reasoning=Reasoning(
                            effort=self.reasoning_effort,
                            summary=self.reasoning_summary,
                        ),
                        text=ResponseTextConfigParam(
                            format=ResponseFormatText(
                                type="text",
                            ),
                            verbosity=self.output_verbosity,
                        ),
                        timeout=self.request_timeout,
                        include=["reasoning.encrypted_content"],
                    ),
                    timeout=self.request_timeout,
                )

                last_event = None

                async for event in self._iter_response_events(stream, self.request_timeout):
                    last_event = event
                    if isinstance(event, ResponseCreatedEvent):
                        pass
                    elif isinstance(event, ResponseInProgressEvent):
                        pass
                    elif isinstance(event, ResponseTextDeltaEvent):
                        pass
                    elif isinstance(event, ResponseTextDoneEvent):
                        pass  # already printed in ResponseOutputItemDoneEvent - ResponseOutputMessage
                    elif isinstance(event, ResponseOutputItemAddedEvent):
                        pass
                    elif isinstance(event, ResponseOutputItemDoneEvent):
                        if isinstance(event.item, ResponseOutputMessage):
                            for content in event.item.content:
                                if isinstance(content, ResponseOutputText):
                                    self._last_message = content.text
                                    formatter.print_agent_message(content.text)
                                elif isinstance(content, ResponseOutputRefusal):
                                    formatter.print_agent_message(content.refusal)
                        elif isinstance(event.item, ResponseFunctionToolCall):
                            formatter.print_tool_use(event.item.name, event.item.arguments)

                            # TODO: workaround for bug on OpenAI's server side
                            if event.item.name == "shell" and True:
                            # if event.item.name == "shell" and not _use_native_shell:
                                tool_calls[event.item.call_id] = asyncio.create_task(self._call_legacy_shell(event.item.arguments))
                            elif event.item.name in mcp_tools:
                                _mcp_client, _mcp_original_name = mcp_tools[event.item.name]
                                tool_calls[event.item.call_id] = asyncio.create_task(self._call_mcp_tool(_mcp_client, _mcp_original_name, event.item.arguments))
                            else:
                                tool_calls[event.item.call_id] = asyncio.create_task(self._call_tool(event.item))
                        elif isinstance(event.item, ResponseReasoningItem):
                            for summary in event.item.summary:
                                formatter.print_thinking(summary.text)
                        elif isinstance(event.item, ResponseFunctionShellToolCall):
                            for command in event.item.action.commands:
                                if event.item.action.timeout_ms is not None:
                                    tool_name = f"bash (timeout: {event.item.action.timeout_ms / 1000.0}s)"
                                else:
                                    tool_name = "bash"
                                formatter.print_tool_use(tool_name, command)
                            shell_calls[event.item.call_id] = (
                                asyncio.create_task(
                                    self._call_shell(
                                        event.item.action.commands,
                                        event.item.action.timeout_ms,
                                        event.item.action.max_output_length,
                                    )
                                ),
                                event.item,
                            )
                        elif isinstance(event.item, ResponseFunctionWebSearch):
                            if event.item.action is not None:
                                formatter.print_tool_use("web_search", event.item.action.to_dict())
                                # Per-call fee not captured by token-based pricing.
                                # Only completed searches are billed (failed ones aren't).
                                if event.item.status == "completed":
                                    self.total_token_usage.web_search_calls += 1
                        else:
                            assert False, f"Unexpected item type: {type(event.item)}"

                    elif isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                        response = event.response
                        if response.usage is not None:
                            # input_tokens = full prompt sent this request (system + tools +
                            # entire conversation, since store=False) ≈ current context size
                            context_tokens = response.usage.input_tokens
                            self.total_token_usage.update(
                                input_tokens_total=response.usage.input_tokens,
                                input_tokens_cached=response.usage.input_tokens_details.cached_tokens,
                                output_tokens=response.usage.output_tokens,
                                model=model,
                                tier=service_tier,
                                input_tokens_cache_write=response.usage.input_tokens_details.cache_write_tokens,
                            )

                        if response.incomplete_details is not None:
                            if response.incomplete_details.reason != "max_output_tokens":
                                raise ResponseIncompleteError(response.incomplete_details.reason)
                            else:
                                logger.warning(f"Output tokens exceeded, compacting conversation")
                                compact_reason = "max_output_tokens"
                    elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                        pass
                    elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                        pass
                    elif isinstance(event, ResponseAudioDeltaEvent):
                        pass
                    else:
                        pass
            except (APIError, httpx.RemoteProtocolError, httpx.TimeoutException, asyncio.TimeoutError, ResponseIncompleteError) as e:
                # Cancel tasks spawned mid-stream before this attempt failed.
                # Their results are never sent back to the model, and the retry
                # regenerates the turn from scratch, so leaving them running
                # risks duplicate side effects (e.g. a shell command run twice).
                for task in tool_calls.values():
                    task.cancel()
                for task, _ in shell_calls.values():
                    task.cancel()

                if isinstance(e, APIError) and e.code == "context_length_exceeded":
                    logger.warning(f"Context length exceeded, compacting conversation")
                    compact_reason = "context_length_exceeded"
                    continue

                formatter.print_error(f"Request failed with tier {service_tier}: {e}")

                if service_tier == "flex" and compact_reason is None:
                    if last_flex_successful:
                        max_retries = self.max_retries
                    else:
                        max_retries = 0
                else:
                    max_retries = self.max_retries

                if retry < max_retries:
                    backoff = _compute_backoff_time(retry)
                    logger.warning(f"Request failed with tier {service_tier}, retrying... ({retry+1}/{max_retries}) with backoff time {backoff}")
                    await asyncio.sleep(backoff)
                    retry += 1
                    continue
                else:
                    if service_tier == "flex" and compact_reason is None:
                        # reset retries, set flag & switch to standard
                        last_flex_successful = False
                        retry = 0
                        service_tier = "standard"
                        logger.warning(f"Switching to standard tier after {max_retries} retries")
                        continue
                    raise e

            retry = 0
            if service_tier == "flex":
                last_flex_successful = True
            if self.service_tier == "flex":
                service_tier = "flex"

            yield self.total_token_usage.total_cost, context_tokens, compact_count

            assert response is not None, f"Expected response, got last event: {last_event}"

            # Persist assistant outputs (messages, tool calls, reasoning, ...)
            # so future requests can include complete conversation history.
            self.conversation.extend(
                cast(
                    list[ResponseInputItemParam],
                    [item.model_dump(mode="json", exclude_none=True) for item in response.output],
                )
            )

            if not tool_calls and not shell_calls and compact_reason is None:
                break

            shell_tasks = [task for task, _ in shell_calls.values()]
            await asyncio.gather(*tool_calls.values(), *shell_tasks, return_exceptions=True)

            for call_id, task in tool_calls.items():
                exc = task.exception()
                if exc is not None:
                    error_msg = "Tool call timed out" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                    formatter.print_tool_result(error_msg, True)
                    output = json.dumps(error_msg)
                else:
                    formatter.print_tool_result(task.result(), False)
                    output = json.dumps(task.result())

                self.conversation.append(FunctionCallOutput(
                    call_id=call_id,
                    output=output,
                    type="function_call_output",
                ))

            for call_id, (task, shell_call) in shell_calls.items():
                for output in task.result():
                    if output["outcome"]["type"] == "exit":
                        if output["outcome"]["exit_code"] != 0:
                            formatter.print_tool_result(output["stderr"], True)
                        else:
                            formatter.print_tool_result(output["stdout"], False)
                    elif output["outcome"]["type"] == "timeout":
                        formatter.print_tool_result("Shell command timed out", True)
                    else:
                        assert False, f"Unexpected outcome: {output['outcome']}"

                self.conversation.append(ShellCallOutput(
                    call_id=call_id,
                    output=task.result(),
                    type="shell_call_output",
                    max_output_length=shell_call.action.max_output_length,
                ))

    async def query(self, prompt: str, model: str, max_cost: float | None, formatter: VerboseFormatter) -> AsyncIterator[OpenAIResponse]:
        if model.lower() not in GPT_PRICING:
            raise ValueError(f"Model '{model}' not supported")

        if self._session_id is None:
            self._session_id = uuid.uuid4().hex

        async with open_mcp_clients(self.mcps, self.request_timeout) as mcp_clients:
            formatter.print_user_message(prompt)

            initial_cost = self.total_token_usage.total_cost

            if max_cost is not None and max_cost <= 0:
                yield OpenAIResponse(cost=0.0, status="terminating_on_max_cost")
                return

            context_tokens: int | None = None
            compaction_count = 0
            async for total_cost, context_tokens, compaction_count in self._stream_messages(prompt, model, mcp_clients, formatter):
                current_cost = total_cost - initial_cost
                yield OpenAIResponse(cost=current_cost, status="running", context_tokens=context_tokens, compaction_count=compaction_count)
                if max_cost is not None and current_cost >= max_cost:
                    formatter.print_error(f"Max cost reached ({current_cost:.4f} >= {max_cost:.4f}). Stopping query.")
                    formatter.print_system_message(self.total_token_usage.format_summary())
                    yield OpenAIResponse(cost=current_cost, status="terminating_on_max_cost", final_message=self._last_message, context_tokens=context_tokens, compaction_count=compaction_count)
                    return

            formatter.print_system_message(self.total_token_usage.format_summary())
            yield OpenAIResponse(cost=self.total_token_usage.total_cost - initial_cost, status="succeeded", final_message=self._last_message, context_tokens=context_tokens, compaction_count=compaction_count)


    def reset(self) -> None:
        """
        Reset the session ID
        """
        self.conversation = []
        self._session_id = None
