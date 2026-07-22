from __future__ import annotations

import asyncio
import contextvars
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Generator, TypedDict, Any, NamedTuple, Literal

from .codex_pricing import CodexTokenPricing, GPT_PRICING
from .session_abc import SessionABC, FunctionTool
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Reserved MCP server name under which in-process Python `tools` are exposed to codex.
_TOOLS_MCP_SERVER_NAME = "local_tools"
# Generous startup timeout so codex-side MCP init doesn't miss the server under load.
_TOOLS_MCP_STARTUP_TIMEOUT_SEC = 120


TERMINATION_PROMPT = (
    "You are approaching the cost limit. Please finish the task as quickly as possible."
)


class SimpleTotalTokenUsage(TypedDict):
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    output_tokens: int


class TieredTotalTokenUsage(TypedDict):
    flex: SimpleTotalTokenUsage
    standard: SimpleTotalTokenUsage
    priority: SimpleTotalTokenUsage


def _compute_cost(
    usage: SimpleTotalTokenUsage | TieredTotalTokenUsage,
    costs: dict[Literal["flex", "standard", "priority"], CodexTokenPricing],
    tier: Literal["flex", "standard", "priority"] = "standard",
) -> float:
    if "input_tokens" not in usage:
        cost = 0.0
        for tier in ["flex", "standard", "priority"]:
            if tier in usage:
                cost += _compute_cost(usage[tier], costs, tier)
        return cost

    pricing = costs[tier]
    cached = usage["cached_input_tokens"]
    cache_write = usage.get("cache_write_input_tokens", 0)
    if pricing.cache_write_mtoken_cost is not None:
        # GPT-5.6+: cache writes are billed separately (see CodexTokenPricing),
        # mirroring OpenAITokenUsage.update().
        regular = usage["input_tokens"] - cached - cache_write
        cache_write_cost = cache_write * pricing.cache_write_mtoken_cost / 1e6
    else:
        # Pre-5.6: cache writes are folded into the regular input price.
        regular = usage["input_tokens"] - cached
        cache_write_cost = 0.0
    return (
        regular * pricing.input_mtoken_cost / 1e6
        + cached * pricing.cached_input_mtoken_cost / 1e6
        + cache_write_cost
        + usage["output_tokens"] * pricing.output_mtoken_cost / 1e6
    )


def _normalize_usage(usage: dict[str, Any]) -> SimpleTotalTokenUsage:
    """Flatten a codex `turn.completed` `usage` dict into a SimpleTotalTokenUsage.

    codex reports the thread's LIFETIME-cumulative usage on every turn (prior
    turns are reloaded from the rollout on `resume`), as a single untiered
    object — NOT the size of the current context window.
    """
    return SimpleTotalTokenUsage(
        input_tokens=usage.get("input_tokens", 0),
        cached_input_tokens=usage.get("cached_input_tokens", 0),
        cache_write_input_tokens=usage.get("cache_write_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


def _usage_delta(current: SimpleTotalTokenUsage, previous: SimpleTotalTokenUsage) -> SimpleTotalTokenUsage:
    """Component-wise ``current - previous`` (both are LIFETIME totals)."""
    return SimpleTotalTokenUsage(
        input_tokens=current["input_tokens"] - previous["input_tokens"],
        cached_input_tokens=current["cached_input_tokens"] - previous["cached_input_tokens"],
        cache_write_input_tokens=current["cache_write_input_tokens"] - previous["cache_write_input_tokens"],
        output_tokens=current["output_tokens"] - previous["output_tokens"],
    )


class CodexResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]
    final_message: str | None = None
    context_tokens: int | None = None  # always None: codex exec exposes no per-request/context-window size
    compaction_count: int = 0


@dataclass
class StdioMcpServer:
    """A stdio-transport MCP server, passed to codex per-invocation (no config file)."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass
class StreamableHttpMcpServer:
    """A streamable-HTTP-transport MCP server, passed to codex per-invocation (no config file)."""
    url: str
    bearer_token_env_var: str | None = None
    http_headers: dict[str, str] | None = None
    env_http_headers: dict[str, str] | None = None


McpServer = StdioMcpServer | StreamableHttpMcpServer


class CodexSession(SessionABC):
    execution_dir: Path
    _session_id: str | None
    reasoning_effort: str
    models_pricing: dict[str, dict[Literal["flex", "standard", "priority"], CodexTokenPricing]] | None
    instructions: str | None
    reasoning_summary: Literal["auto", "concise", "detailed", "none"] | None
    output_verbosity: Literal["low", "medium", "high"] | None
    web_search: bool
    web_search_context_size: Literal["low", "medium", "high"] | None
    tools: list[FunctionTool]
    mcps: dict[str, McpServer]
    mcp_call_timeout: float | None
    sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"]
    bypass_sandbox: bool
    shell_network_access: bool
    codex_executable: str
    additional_options: dict[str, Any]

    token_usage: TieredTotalTokenUsage
    _last_message: str | None
    _instructions_file: str | None
    _tools_mcp_url: str | None

    def __init__(
        self,
        execution_dir: Path,
        *,
        session_id: str | None = None,
        reasoning_effort: str = "high",
        reasoning_summary: Literal["auto", "concise", "detailed", "none"] | None = None,
        output_verbosity: Literal["low", "medium", "high"] | None = None,
        models_pricing: dict[str, dict[Literal["flex", "standard", "priority"], CodexTokenPricing]] | None = None,
        instructions: str | None = None,
        web_search: bool = False,
        web_search_context_size: Literal["low", "medium", "high"] | None = None,
        tools: list[FunctionTool] | None = None,
        mcps: dict[str, McpServer] | None = None,
        mcp_call_timeout: float | None = None,
        sandbox_mode: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write",
        bypass_sandbox: bool = False,
        shell_network_access: bool = False,
        codex_executable: str = "codex",
        additional_options: dict[str, Any] | None = None,
    ):
        self.execution_dir = execution_dir
        self._session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.output_verbosity = output_verbosity
        self.models_pricing = models_pricing
        self.instructions = instructions
        self.web_search = web_search
        self.web_search_context_size = web_search_context_size
        self.tools = list(tools) if tools is not None else []
        self.mcps = mcps or {}
        self.mcp_call_timeout = mcp_call_timeout
        self.sandbox_mode = sandbox_mode
        self.bypass_sandbox = bypass_sandbox
        self.shell_network_access = shell_network_access
        self.codex_executable = codex_executable
        self.additional_options = additional_options or {}

        self.token_usage = TieredTotalTokenUsage(
            flex=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, cache_write_input_tokens=0, output_tokens=0),
            standard=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, cache_write_input_tokens=0, output_tokens=0),
            priority=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, cache_write_input_tokens=0, output_tokens=0),
        )
        self._last_message = None
        self._instructions_file = None
        self._tools_mcp_url = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _set_lifetime_usage(self, lifetime: SimpleTotalTokenUsage) -> None:
        """Record codex's latest LIFETIME-cumulative usage.

        codex `exec` is single-tier from our side, so the untiered figure is
        stored under ``standard``. This is a SET (not an accumulate): the value
        codex reports already includes every prior turn on the thread.
        """
        self.token_usage["standard"] = lifetime

    def _write_instructions_file(self) -> str:
        """Write the system prompt to a temp file for codex's `model_instructions_file` config (once)."""
        if self._instructions_file is None:
            assert self.instructions is not None
            fd, path = tempfile.mkstemp(prefix="codex_instructions_", suffix=".md")
            with os.fdopen(fd, "w") as f:
                f.write(self.instructions)
            self._instructions_file = path
        return self._instructions_file

    async def _setup_process(self, prompt: str, model: str, formatter: VerboseFormatter) -> asyncio.subprocess.Process:
        args = [self.codex_executable, "exec", "--json"]

        args.append("--model")
        args.append(model)

        #args.append("--include-plan-tool")

        args.append("--cd")
        args.append(str(self.execution_dir))

        args.append("--skip-git-repo-check")

        if self.bypass_sandbox:
            # For sessions already wrapped in an external sandbox (e.g. beast's nested
            # sandbox) or when bwrap is unavailable: skip codex's own sandboxing.
            # (Full access already includes network.)
            args.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            args.append("--sandbox")
            args.append(self.sandbox_mode)
            # Grant network to shell commands under the workspace-write sandbox
            # (has no effect under read-only).
            if self.shell_network_access:
                args.append("-c")
                args.append("sandbox_workspace_write.network_access=true")

        args.append("-c")
        args.append(f'model_reasoning_effort="{self.reasoning_effort}"')

        if self.reasoning_summary is not None:
            args.append("-c")
            args.append(f"model_reasoning_summary={json.dumps(self.reasoning_summary)}")

        if self.output_verbosity is not None:
            args.append("-c")
            args.append(f"model_verbosity={json.dumps(self.output_verbosity)}")

        if self.instructions is not None:
            args.append("-c")
            args.append(f'model_instructions_file="{self._write_instructions_file()}"')

        if self.web_search:
            args.append("-c")
            args.append("tools_web_search_request=true")
            if self.web_search_context_size is not None:
                args.append("-c")
                args.append(f"web_search_config.search_context_size={json.dumps(self.web_search_context_size)}")

        # MCP servers are passed per-invocation as `-c mcp_servers.<name>.*` overrides
        # (not via a config file) so each session can carry its own toolset. Values are
        # JSON-encoded, which is valid TOML for the scalar/array cases; maps (env, headers)
        # are emitted as dotted sub-keys. codex infers the transport from the fields present
        # (`command` -> stdio, `url` -> streamable_http).
        for name, server in self.mcps.items():
            prefix = f"mcp_servers.{name}"
            if isinstance(server, StdioMcpServer):
                args.append("-c")
                args.append(f"{prefix}.command={json.dumps(server.command)}")
                if server.args:
                    args.append("-c")
                    args.append(f"{prefix}.args={json.dumps(server.args)}")
                for k, v in (server.env or {}).items():
                    args.append("-c")
                    args.append(f"{prefix}.env.{k}={json.dumps(v)}")
            elif isinstance(server, StreamableHttpMcpServer):
                args.append("-c")
                args.append(f"{prefix}.url={json.dumps(server.url)}")
                if server.bearer_token_env_var is not None:
                    args.append("-c")
                    args.append(f"{prefix}.bearer_token_env_var={json.dumps(server.bearer_token_env_var)}")
                for k, v in (server.http_headers or {}).items():
                    args.append("-c")
                    args.append(f"{prefix}.http_headers.{k}={json.dumps(v)}")
                for k, v in (server.env_http_headers or {}).items():
                    args.append("-c")
                    args.append(f"{prefix}.env_http_headers.{k}={json.dumps(v)}")
            else:
                raise TypeError(f"Unsupported MCP server type for CodexSession: {type(server)}")

            if self.mcp_call_timeout is not None:
                args.append("-c")
                args.append(f"{prefix}.tool_timeout_sec={json.dumps(self.mcp_call_timeout)}")
            # Headless `exec` has no approver: without this, tools whose approval mode
            # resolves to "requires approval" (the default for un-annotated tools under
            # a non-bypass sandbox) get auto-cancelled ("user cancelled MCP tool call").
            args.append("-c")
            args.append(f'{prefix}.default_tools_approval_mode="approve"')

        # In-process Python `tools` are exposed via a streamable-HTTP MCP server
        # started for the duration of the query (see `query`). `required` makes codex
        # exit with an error (rather than silently dropping the tools) if it can't
        # initialize the server, so failures surface loudly and retry; a generous
        # startup timeout guards against codex-side init timing under load.
        if self._tools_mcp_url is not None:
            prefix = f"mcp_servers.{_TOOLS_MCP_SERVER_NAME}"
            args.append("-c")
            args.append(f"{prefix}.url={json.dumps(self._tools_mcp_url)}")
            args.append("-c")
            args.append(f"{prefix}.required=true")
            args.append("-c")
            args.append(f"{prefix}.startup_timeout_sec={_TOOLS_MCP_STARTUP_TIMEOUT_SEC}")
            args.append("-c")
            args.append(f'{prefix}.default_tools_approval_mode="approve"')
            if self.mcp_call_timeout is not None:
                args.append("-c")
                args.append(f"{prefix}.tool_timeout_sec={json.dumps(self.mcp_call_timeout)}")

        for key, value in self.additional_options.items():
            args.append("-c")
            if isinstance(value, bool):
                args.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (list, int)):
                args.append(f"{key}={value}")
            else:
                args.append(f"{key}='{value}'")

        if self._session_id is not None:
            args.append("resume")
            args.append(self._session_id)

        formatter.print_system_message(f"Running {' '.join(args)}")

        proc = await asyncio.create_subprocess_exec(
            args[0],
            *args[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=128 * 1024 * 1024,  # 128MB
        )

        assert proc.stdin is not None

        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.close()

        return proc

    async def _receive_messages(self, proc: asyncio.subprocess.Process) -> AsyncIterator[dict[str, Any]]:
        assert proc.stdout is not None
        assert proc.stderr is not None

        def get_chunk_size(buffer_size: int) -> int:
            """Exponentially scale chunk size with buffer size for efficient reading of both small and huge messages."""
            if buffer_size < 512 * 1024:
                return 64 * 1024       # 64KB - normal messages
            elif buffer_size < 10 * 1024 * 1024:
                return 1024 * 1024     # 1MB - large messages
            elif buffer_size < 100 * 1024 * 1024:
                return 16 * 1024 * 1024  # 16MB - very large messages
            else:
                return 64 * 1024 * 1024  # 64MB - truly massive messages (e.g., 1GB)

        def process_line(line: bytes) -> dict[str, Any] | None:
            if not line.strip():
                return None

            try:
                return json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                preview_size = 500
                if len(line) <= preview_size * 2:
                    preview = line
                else:
                    preview = line[:preview_size] + b" ... " + line[-preview_size:]

                logger.error(f"Failed to parse JSON (line size: {len(line)} bytes): {e}")
                logger.error(f"Line preview: {preview}")
                return None

        def extract_messages(buffer: bytearray) -> Generator[dict[str, Any], None, None]:
            while True:
                newline_pos = buffer.find(b"\n")
                if newline_pos == -1:
                    break

                line = bytes(buffer[:newline_pos])
                del buffer[:newline_pos + 1]  # remove line + newline
                msg = process_line(line)
                if msg is not None:
                    yield msg

        buffer = bytearray()

        while True:
            chunk_size = get_chunk_size(len(buffer))

            try:
                chunk = await proc.stdout.read(chunk_size)
            except Exception as e:
                logger.error(f"Failed to read from Codex stdout: {e}")
                logger.error(f"Buffer size at failure: {len(buffer)} bytes")
                raise

            if not chunk:  # EOF
                break

            buffer.extend(chunk)

            for msg in extract_messages(buffer):
                yield msg

        # process remaining buffer
        for msg in extract_messages(buffer):
            yield msg

        buffer = buffer.strip()
        if buffer:
            logger.error(f"Unprocessed Codex buffer: {buffer}")

        await proc.wait()

        if proc.returncode != 0:
            err = await proc.stderr.read()
            stdout = await proc.stdout.read()
            raise RuntimeError(
                f"Process failed with exit code: {proc.returncode}\nStderr:\n{err.decode('utf-8')}\nStdout:\n{stdout.decode('utf-8')}"
            )

    def _process_message(self, msg: dict[str, Any], formatter: VerboseFormatter) -> None:
        if msg["type"] == "error":
            formatter.print_error(msg["message"])
        elif msg["type"] == "item.started":
            if msg["item"]["type"] == "command_execution":
                formatter.print_tool_use(msg["item"]["command"], {})
            elif msg["item"]["type"] == "mcp_tool_call":
                formatter.print_tool_use(
                    msg["item"]["server"] + "." + msg["item"]["tool"],
                    msg["item"].get("arguments", {}),
                )
            elif msg["item"]["type"] == "todo_list":
                formatter.print_todo([
                    {
                        "status": "pending" if not i["completed"] else "completed",
                        "text": i["text"],
                    }
                    for i in msg["item"]["items"]
                ])
            else:
                logger.warning(f"Unexpected Codex item.started message: {msg}")
        elif msg["type"] == "item.updated":
            if msg["item"]["type"] == "todo_list":
                formatter.print_todo([
                    {
                        "status": "pending" if not i["completed"] else "completed",
                        "text": i["text"],
                    }
                    for i in msg["item"]["items"]
                ])
            else:
                logger.warning(f"Unexpected Codex item.updated message: {msg}")
        elif msg["type"] == "item.completed":
            if msg["item"]["type"] == "agent_message":
                self._last_message = msg["item"]["text"]
                formatter.print_agent_message(msg["item"]["text"])
            elif msg["item"]["type"] == "reasoning":
                formatter.print_thinking(msg["item"]["text"])
            elif msg["item"]["type"] == "command_execution":
                formatter.print_tool_result(
                    msg["item"]["aggregated_output"],
                    msg["item"]["exit_code"] != 0,
                )
            elif msg["item"]["type"] == "file_change":
                changes = msg["item"]["changes"]
                message = "Changed files:\n" + "\n".join(ch["path"] + " " + ch["kind"] for ch in changes)
                formatter.print_system_message(message)
            elif msg["item"]["type"] == "mcp_tool_call":
                if msg["item"]["status"] == "completed":
                    if "result" in msg["item"]:
                        formatter.print_tool_result(
                            msg["item"]["result"]["content"],
                            False,
                        )
                elif msg["item"]["status"] == "failed":
                    if msg["item"].get("error", None) is not None and "message" in msg["item"]["error"]:
                        formatter.print_tool_result(msg["item"]["error"]["message"], True)
                    elif msg["item"].get("result", None) is not None and "content" in msg["item"]["result"]:
                        formatter.print_tool_result(msg["item"]["result"]["content"], True)
                    else:
                        formatter.print_tool_result("MCP tool call failed", True)
                else:
                    logger.warning(f"Unexpected Codex item.completed message for MCP tool call: {msg['item']}")
            elif msg["item"]["type"] == "todo_list":
                formatter.print_todo([
                    {
                        "status": "pending" if not i["completed"] else "completed",
                        "text": i["text"],
                    }
                    for i in msg["item"]["items"]
                ])
            else:
                logger.warning(f"Unexpected Codex item.completed message: {msg}")
        elif msg["type"] == "thread.started":
            if self._session_id is None:
                self._session_id = msg["thread_id"]
            else:
                assert self._session_id == msg["thread_id"]
        elif msg["type"] == "turn.started":
            pass
        elif msg["type"] == "turn.completed":
            pass
        elif msg["type"] == "turn.failed":
            formatter.print_error(msg["error"]["message"])
        else:
            logger.warning(f"Unexpected Codex message: {msg}")

    async def query(self, prompt: str, model: str, max_cost: float | None, formatter: VerboseFormatter) -> AsyncIterator[CodexResponse]:
        if self.models_pricing is not None and model in self.models_pricing:
            model_pricing = self.models_pricing[model]
        elif model.lower() in GPT_PRICING:
            model_pricing = GPT_PRICING[model.lower()]
        else:
            raise ValueError(f"No pricing found for model '{model}'. Please provide models_pricing.")

        # codex reports the thread's LIFETIME-cumulative usage on every turn (see
        # `_normalize_usage`), reloading prior turns on `resume`. This query's
        # incremental cost is therefore the DELTA of that lifetime total across
        # the query — not the raw reported value, which on a resumed session
        # already includes (and would re-bill) every previous turn.
        query_start_usage = deepcopy(self.token_usage["standard"])
        self._last_message = None

        # Expose in-process Python `tools` to codex via a short-lived HTTP MCP server.
        # Handlers run in the context captured here so context-vars (e.g. the flow
        # engine's current-host, enabling nested subagents) propagate into them.
        tools_server = None
        if self.tools:
            from .codex_tools import InProcessToolsServer
            tools_server = InProcessToolsServer(self.tools, name=_TOOLS_MCP_SERVER_NAME)
            tools_server.set_context(contextvars.copy_context())
            self._tools_mcp_url = await tools_server.start()

        try:
            # On `resume`, codex reloads the thread and its running token totals,
            # so each turn's reported `usage` is the thread's LIFETIME total — we
            # bill the per-query delta against `query_start_usage` (captured above).
            proc = await self._setup_process(prompt, model, formatter)

            if self.instructions is not None:
                formatter.print_system_message(f"Custom instructions:\n{self.instructions}")

            formatter.print_user_message(prompt)
            terminated = False
            cost = 0.0

            async for msg in self._receive_messages(proc):
                self._process_message(msg, formatter)

                if msg["type"] == "turn.completed":
                    lifetime = _normalize_usage(msg["usage"])
                    self._set_lifetime_usage(lifetime)
                    cost = _compute_cost(_usage_delta(lifetime, query_start_usage), model_pricing)
                    yield CodexResponse(cost=cost, status="running", final_message=self._last_message)

                    if max_cost is not None and cost > max_cost:
                        proc.terminate()
                        terminated = True

            assert self._session_id is not None

            main_cost = cost
            cost = 0.0

            await proc.wait()

            if terminated:
                formatter.print_user_message(TERMINATION_PROMPT)
                # Re-baseline: the continuation turn is billed as ITS delta only;
                # `main_cost` already covers everything up to this point.
                query_start_usage = deepcopy(self.token_usage["standard"])
                proc = await self._setup_process(TERMINATION_PROMPT, model, formatter)

                async for msg in self._receive_messages(proc):
                    self._process_message(msg, formatter)

                    if msg["type"] == "turn.completed":
                        lifetime = _normalize_usage(msg["usage"])
                        self._set_lifetime_usage(lifetime)
                        cost = _compute_cost(_usage_delta(lifetime, query_start_usage), model_pricing)
                        yield CodexResponse(cost=main_cost + cost, status="terminating_on_max_cost", final_message=self._last_message)

            yield CodexResponse(cost=main_cost + cost, status="succeeded", final_message=self._last_message)
        finally:
            self._tools_mcp_url = None
            if tools_server is not None:
                await tools_server.stop()

    def reset(self) -> None:
        """
        Reset the session ID
        """
        self._session_id = None
        self._last_message = None
        # keep token_usage and the instructions temp file
