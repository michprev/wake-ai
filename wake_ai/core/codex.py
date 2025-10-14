from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, TypedDict, Any, NamedTuple, Literal

from rich.console import Console
from rich.rule import Rule

from .codex_pricing import CodexTokenPricing, GPT_PRICING
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger

logger = get_logger(__name__)


TERMINATION_PROMPT = (
    "You are approaching the cost limit. Please finish the task as quickly as possible."
)


class TotalTokenUsage(TypedDict):
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


def _compute_cost(usage: TotalTokenUsage, costs: CodexTokenPricing) -> float:
    non_cached_input_tokens = (
        usage["input_tokens"] - usage["cached_input_tokens"]
    )
    return (
        non_cached_input_tokens * costs.input_mtoken_cost / 1e6
        + usage["cached_input_tokens"] * costs.cached_input_mtoken_cost / 1e6
        + usage["output_tokens"] * costs.output_mtoken_cost / 1e6
    )


class CodexResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]


class CodexSession:
    execution_dir: Path
    session_id: str | None
    reasoning_effort: str
    models_pricing: dict[str, CodexTokenPricing] | None

    def __init__(
        self,
        execution_dir: Path,
        *,
        session_id: str | None = None,
        reasoning_effort: str = "high",
        models_pricing: dict[str, CodexTokenPricing] | None = None,
    ):
        self.execution_dir = execution_dir
        self.session_id = session_id
        self.reasoning_effort = reasoning_effort
        self.models_pricing = models_pricing

    async def _setup_process(self, prompt: str, model: str) -> asyncio.subprocess.Process:
        args = ["codex", "exec", "--json"]
        args.append("--model")
        args.append(model)

        args.append("--include-plan-tool")

        args.append("--cd")
        args.append(str(self.execution_dir))

        args.append("--skip-git-repo-check")

        args.append("--sandbox")
        args.append("workspace-write")

        args.append("-c")
        args.append(f'model_reasoning_effort="{self.reasoning_effort}"')

        if self.session_id is not None:
            args.append("resume")
            args.append(self.session_id)

        logger.info(f"Running {' '.join(args)}")

        proc = await asyncio.create_subprocess_exec(
            args[0],
            *args[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=100 * 1024 * 1024,  # 100MB
        )

        assert proc.stdin is not None

        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.write_eof()
        proc.stdin.close()

        return proc

    async def _receive_messages(self, proc: asyncio.subprocess.Process) -> AsyncIterator[dict[str, Any]]:
        assert proc.stdout is not None
        assert proc.stderr is not None

        while True:
            try:
                line = await proc.stdout.readline()
            except ValueError:
                tmp = await proc.stdout.read(1024 * 1024)
                logger.error(f"Failed to read from Codex stdout")
                logger.error(f"artifact data:\n{tmp}")
                raise

            if not line:
                break

            msg = json.loads(line.decode("utf-8"))
            yield msg

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
                    {},  # TODO!!
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
                # no new info, pass for now
                pass  # TODO!!
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
        proc = await self._setup_process(prompt, model)

        formatter.print_user_message(prompt)
        terminated = False
        cost = 0.0

        async for msg in self._receive_messages(proc):
            self._process_message(msg, formatter)

            if msg["type"] == "turn.completed":
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
            proc = await self._setup_process(TERMINATION_PROMPT, model)
            cost = 0.0

            async for msg in self._receive_messages(proc):
                self._process_message(msg, formatter)

                if msg["type"] == "turn.completed":
                    cost = _compute_cost(
                        msg["usage"],
                        model_pricing,
                    )
                    yield CodexResponse(cost=main_cost + cost, status="terminating_on_max_cost")

        yield CodexResponse(cost=main_cost + cost, status="succeeded")

    def clone(self) -> CodexSession:
        """
        Clone the session configuration, but start a new session.
        """
        return CodexSession(
            execution_dir=self.execution_dir,
            session_id=None,
            reasoning_effort=self.reasoning_effort,
            models_pricing=self.models_pricing,
        )
