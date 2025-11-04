from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Generator, TypedDict, Any, NamedTuple, Literal

from rich.console import Console
from rich.rule import Rule

from .codex_pricing import CodexTokenPricing, GPT_PRICING
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger

logger = get_logger(__name__)


TERMINATION_PROMPT = (
    "You are approaching the cost limit. Please finish the task as quickly as possible."
)


class SimpleTotalTokenUsage(TypedDict):
    input_tokens: int
    cached_input_tokens: int
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
    else:
        non_cached_input_tokens = (
            usage["input_tokens"] - usage["cached_input_tokens"]
        )
        cost = (
            non_cached_input_tokens * costs[tier].input_mtoken_cost / 1e6
            + usage["cached_input_tokens"] * costs[tier].cached_input_mtoken_cost / 1e6
            + usage["output_tokens"] * costs[tier].output_mtoken_cost / 1e6
        )
        return cost


class CodexResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]


class CodexSession:
    execution_dir: Path
    session_id: str | None
    fork_session: str | CodexSession | None
    reasoning_effort: str
    models_pricing: dict[str, dict[Literal["flex", "standard", "priority"], CodexTokenPricing]] | None
    service_tier: Literal["flex", "standard", "priority"] | None
    codex_executable: str
    additional_options: dict[str, Any]

    token_usage: TieredTotalTokenUsage

    def __init__(
        self,
        execution_dir: Path,
        *,
        session_id: str | None = None,
        fork_session: str | CodexSession | None = None,
        reasoning_effort: str = "high",
        models_pricing: dict[str, dict[Literal["flex", "standard", "priority"], CodexTokenPricing]] | None = None,
        service_tier: Literal["flex", "standard", "priority"] | None = None,
        codex_executable: str = "codex",
        additional_options: dict[str, Any] | None = None,
    ):
        if session_id is not None and fork_session is not None:
            raise ValueError("session_id and fork_session cannot be used together")

        self.execution_dir = execution_dir
        self.session_id = session_id
        self.fork_session = fork_session
        self.reasoning_effort = reasoning_effort
        self.models_pricing = models_pricing
        self.service_tier = service_tier
        self.codex_executable = codex_executable
        self.additional_options = additional_options or {}

        self.token_usage = TieredTotalTokenUsage(
            flex=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0),
            standard=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0),
            priority=SimpleTotalTokenUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0),
        )

    def _update_token_usage(self, usage: SimpleTotalTokenUsage | TieredTotalTokenUsage) -> None:
        if "input_tokens" in usage:
            self.token_usage["standard"]["input_tokens"] += usage["input_tokens"]
            self.token_usage["standard"]["cached_input_tokens"] += usage["cached_input_tokens"]
            self.token_usage["standard"]["output_tokens"] += usage["output_tokens"]
        else:
            for tier in ["flex", "standard", "priority"]:
                self.token_usage[tier]["input_tokens"] += usage[tier]["input_tokens"]
                self.token_usage[tier]["cached_input_tokens"] += usage[tier]["cached_input_tokens"]
                self.token_usage[tier]["output_tokens"] += usage[tier]["output_tokens"]

    async def _setup_process(self, prompt: str, model: str, formatter: VerboseFormatter) -> asyncio.subprocess.Process:
        args = [self.codex_executable, "exec", "--json"]

        if self.service_tier is not None:
            args.append("--service-tier")
            args.append(self.service_tier)

        args.append("--model")
        args.append(model)

        #args.append("--include-plan-tool")

        args.append("--cd")
        args.append(str(self.execution_dir))

        args.append("--skip-git-repo-check")

        args.append("--sandbox")
        args.append("workspace-write")

        args.append("-c")
        args.append(f'model_reasoning_effort="{self.reasoning_effort}"')

        for key, value in self.additional_options.items():
            args.append("-c")
            if isinstance(value, bool):
                args.append(f"{key}={str(value).lower()}")
            elif isinstance(value, (list, int)):
                args.append(f"{key}={value}")
            else:
                args.append(f"{key}='{value}'")

        if self.session_id is not None:
            args.append("resume")
            args.append(self.session_id)
        elif self.fork_session is not None:
            args.append("resume")

            if isinstance(self.fork_session, str):
                args.append(self.fork_session)
            else:
                if self.fork_session.session_id is None:
                    raise RuntimeError("Forking from CodexSession without assigned session id")
                args.append(self.fork_session.session_id)

            args.append("--fork")

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
            if self.session_id is None:
                self.session_id = msg["thread_id"]
            else:
                assert self.session_id == msg["thread_id"]
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

        # each launched process starts counting tokens from the beginning
        # (even if we continue an existing session)
        proc = await self._setup_process(prompt, model, formatter)

        if "experimental_instructions_file" in self.additional_options:
            instructions = Path(self.additional_options["experimental_instructions_file"]).read_text()
            formatter.print_system_message(f"Custom instructions:\n{instructions}")

        formatter.print_user_message(prompt)
        terminated = False
        cost = 0.0

        async for msg in self._receive_messages(proc):
            self._process_message(msg, formatter)

            if msg["type"] == "turn.completed":
                self._update_token_usage(msg["usage"])
                cost = _compute_cost(
                    msg["usage"],
                    model_pricing,
                )
                yield CodexResponse(cost=cost, status="running")

                if max_cost is not None and cost > max_cost:
                    proc.terminate()
                    terminated = True

        assert self.session_id is not None

        main_cost = cost

        await proc.wait()

        if terminated:
            formatter.print_user_message(TERMINATION_PROMPT)
            proc = await self._setup_process(TERMINATION_PROMPT, model, formatter)
            cost = 0.0

            async for msg in self._receive_messages(proc):
                self._process_message(msg, formatter)

                if msg["type"] == "turn.completed":
                    self._update_token_usage(msg["usage"])
                    cost = _compute_cost(
                        msg["usage"],
                        model_pricing,
                    )
                    yield CodexResponse(cost=main_cost + cost, status="terminating_on_max_cost")

        yield CodexResponse(cost=main_cost + cost, status="succeeded")

    def reset(self) -> None:
        """
        Reset the session ID
        """
        self.session_id = None
        # keep token_usage
