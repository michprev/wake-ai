from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from contextlib import nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from io import StringIO
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from typing import Any, Awaitable, Literal, Callable, Coroutine, TypeVar, Generic, cast

import rich_click as click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback

from .claude import ClaudeSession
from .codex import CodexSession
from .codex_pricing import GPT_PRICING
from .openrouter import OpenRouterSession
from .session_abc import SessionABC
from .verbose_formatter import VerboseFormatter
from ..results import AIResult
from ..utils.common import render_template
from ..utils.logging import get_verbosity_level


# Portable, filesystem-safe subagent name. Excludes path separators (`/`, `\`),
# Windows-reserved chars (`<>:"|?*`), whitespace, and leading dot. Allowed
# characters: ASCII letters, digits, `_`, `-`, `.`. Length 1..64.
# Use externally with pydantic as e.g.:
#   SubagentName = Annotated[str, StringConstraints(pattern=SUBAGENT_NAME_PATTERN,
#                                                    min_length=1, max_length=64)]
SUBAGENT_NAME_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$"
SUBAGENT_NAME_MAX_LENGTH = 64
_SUBAGENT_NAME_RE = re.compile(SUBAGENT_NAME_PATTERN)


def _validate_subagent_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"Subagent name must be a string, got {type(name).__name__}")
    if not name:
        raise ValueError("Subagent name cannot be empty")
    if len(name) > SUBAGENT_NAME_MAX_LENGTH:
        raise ValueError(
            f"Subagent name '{name}' exceeds {SUBAGENT_NAME_MAX_LENGTH} characters"
        )
    if not _SUBAGENT_NAME_RE.match(name):
        raise ValueError(
            f"Subagent name '{name}' is invalid; must match {SUBAGENT_NAME_PATTERN} "
            f"(ASCII letters/digits/_-., no leading dot, no path separators or "
            f"reserved chars, no whitespace)"
        )


def _default_session(model: str, execution_dir: Path, working_dir: Path) -> SessionABC:
    """Pick the backend for a step that didn't pass an explicit session.

    OpenRouter slugs are always ``vendor/model`` — neither bare GPT names nor
    Claude model names contain a slash.
    """
    if "/" in model:
        return OpenRouterSession(execution_dir, writable_roots=[working_dir])
    if model.lower() in GPT_PRICING:
        return CodexSession(execution_dir)
    return ClaudeSession(execution_dir, working_dir)


# Per-task slot holding the WorkflowStep whose session is currently executing.
# Set by AIWorkflow._execute_step around `session.query(...)` so tool handlers
# invoked from inside that query can locate the step they belong to without
# threading it through `partial(...)` (which is impossible at step-construction
# time since the step is being created by the same `add_step` call that takes
# the session as input).
#
# ContextVars are per-asyncio-task: `asyncio.create_task` snapshots the parent's
# context, so concurrent parallel steps each see their own value. This is the
# only safe primitive for this pattern — `threading.local` would be wrong.
#
# Read via `current_host()`. Will gain `SubStep` to its type once nested
# subagents land.
_current_host: ContextVar[WorkflowStep | SubStep | None] = ContextVar(
    "wake_ai_current_host", default=None
)


def get_current_host() -> WorkflowStep | SubStep:
    """Return the WorkflowStep or SubStep whose session is currently executing.

    Intended for tool handlers that need to invoke `host.run_subagent(...)`.
    At top-level step execution, returns the WorkflowStep. Inside a subagent's
    tool handler, returns the enclosing SubStep — so nested `run_subagent`
    calls register children under the right node.

    Raises RuntimeError if called outside an active step execution.
    """
    host = _current_host.get()
    if host is None:
        raise RuntimeError(
            "current_host() called outside an active step execution"
        )
    return host


def _format_duration(seconds: float) -> str:
    """Format duration dynamically based on the time scale."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if days > 0:
        return f"{days:d}d {hours:02d}h {minutes:02d}m {secs:05.2f}s"
    elif hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:05.2f}s"
    elif minutes > 0:
        return f"{minutes:d}m {secs:05.2f}s"
    else:
        return f"{secs:.2f}s"


def _format_context(tokens: int | None, compactions: int = 0) -> str:
    """Format context size (e.g. '252k') with an optional compaction count ('252k ⟳2')."""
    if not tokens:
        return "-"
    if tokens >= 1000:
        size = f"{tokens / 1000:.0f}k"
    else:
        size = str(tokens)
    if compactions:
        size += f" ⟳{compactions}"
    return size


def require_initialized(func):
    """Decorator to ensure __init__ was called on AIWorkflow instances."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(self, AIWorkflow) and not getattr(self, "_init_called", False):
            raise RuntimeError(
                f"AIWorkflow.__init__() was not called. "
                f"Make sure to call super().__init__(...) in {self.__class__.__name__}.__init__()"
            )
        return func(self, *args, **kwargs)

    return wrapper


T = TypeVar("T")


class DynamicStepResult(Generic[T]):
    def __init__(self, step: DynamicWorkflowStep[T]):
        self._step = step

    def __await__(self):  # type: ignore[override]
        async def _wait() -> T:
            await self._step.done_event.wait()
            if self._step.status == "skipped":
                raise RuntimeError(f"Step '{self._step.name}' was skipped")
            if self._step.error is not None:
                raise self._step.error
            return cast(T, self._step._return_value)

        return _wait().__await__()


@dataclass
class DynamicWorkflowStep(Generic[T]):
    # immutable
    name: str
    handler: Callable[[DynamicWorkflowStep[T]], Coroutine[Any, Any, T]]
    requires: list[WorkflowStep | DynamicWorkflowStep[Any]]
    condition: Callable[[DynamicWorkflowStep[T]], bool] | None
    done_event: asyncio.Event
    result: Awaitable[T]

    # mutable
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    _return_value: T | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: BaseException | None = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DynamicWorkflowStep):
            return False
        return self.name == other.name


# Default max nesting depth for subagent trees. Configurable via
# `AIWorkflow.max_subagent_depth` (subclass attribute or instance value).
# Cheap insurance against accidental cycles; raise/lower as needed.
DEFAULT_MAX_SUBAGENT_DEPTH = 4


@dataclass
class SubStep:
    """
    Tool-invoked subagent record. Identity = (parent_host, name).

    SubSteps form a tree: each SubStep may itself host child SubSteps via
    `run_subagent`. Identity lookup is local to the parent's `substeps` list,
    so two different parents may each have a child named "helper" without
    collision.

    Stats accumulate across multiple `run_subagent` invocations of the same
    name on the same parent (supports resumed sessions and validation
    retries that re-trigger the same subagent).

    Staleness on root-WorkflowStep retry is structural: the parent's prior
    `substeps` list is moved into `FailedWorkflowStep.substeps`, and a fresh
    list takes its place. SubStep itself stores no "attempt" field.

    No status field: visual status mirrors the root WorkflowStep. The only
    sub-specific state is `current_call_start`, which gates the live-ticking
    duration and the in-call glyph.
    """
    name: str
    model: str
    depth: int  # 1 for direct children of a WorkflowStep, 2 for grandchildren, ...
    formatter: VerboseFormatter  # own formatter; also used as parent context for children
    # Back-ref to the host that created this SubStep. Excluded from repr/eq to
    # avoid infinite recursion (parent.substeps contains this SubStep).
    parent: WorkflowStep | SubStep = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    cost: float = 0.0  # cumulative across all calls
    accumulated_duration: float = 0.0  # cumulative seconds of completed calls
    current_call_start: datetime | None = None  # set during an active call
    invocation_count: int = 0
    last_error: BaseException | None = None
    context_tokens: int | None = None  # context size of the latest call (last-wins)
    compaction_count: int = 0  # cumulative across all calls
    substeps: list[SubStep] = field(default_factory=list)

    @property
    def log_file_path(self) -> Path | None:
        return self.formatter.log_file

    async def run_subagent(
        self,
        name: str,
        session: SessionABC,
        *,
        prompt: str,
        model: str,
        max_cost: float | None = None,
        formatter: VerboseFormatter | None = None,
        max_depth: int = DEFAULT_MAX_SUBAGENT_DEPTH,
    ) -> str:
        """Launch a nested subagent under this SubStep. See module docs."""
        return await _run_subagent(
            parent=self,
            depth=self.depth + 1,
            max_depth=max_depth,
            name=name,
            session=session,
            prompt=prompt,
            model=model,
            max_cost=max_cost,
            formatter=formatter,
        )


@dataclass
class FailedWorkflowStep:
    cost: float
    start_time: datetime
    end_time: datetime
    error: BaseException
    substeps: list[SubStep] = field(default_factory=list)


async def _run_subagent(
    *,
    parent: WorkflowStep | SubStep,
    depth: int,
    max_depth: int,
    name: str,
    session: SessionABC,
    prompt: str,
    model: str,
    max_cost: float | None,
    formatter: VerboseFormatter | None,
) -> str:
    from ..utils.logging import get_verbosity_level

    _validate_subagent_name(name)
    if depth > max_depth:
        raise RuntimeError(
            f"Subagent nesting depth {depth} exceeds max ({max_depth}); "
            f"chain: {parent.formatter.step_name}_{name}"
        )

    sub = next((s for s in parent.substeps if s.name == name), None)
    if sub is None:
        chain = f"{parent.formatter.step_name}_{name}"
        parent_log = parent.formatter.log_file
        log_file = parent_log.parent / f"{chain}.log" if parent_log is not None else None
        sub_formatter = formatter or VerboseFormatter(
            parent.formatter.console,
            chain,
            log_file,
            get_verbosity_level(),
            reset_log_file=True,  # first ever call → reset (rotates, doesn't delete)
        )
        sub = SubStep(
            name=name,
            model=model,
            depth=depth,
            formatter=sub_formatter,
            parent=parent,
        )
        parent.substeps.append(sub)
    else:
        # Concurrency guard, scoped to (parent_host, name). Different parents
        # with same-named children don't collide (separate substeps lists).
        if sub.current_call_start is not None:
            raise RuntimeError(
                f"Subagent '{name}' is already running under "
                f"'{parent.formatter.step_name}'"
            )
        if formatter is not None:
            sub.formatter = formatter

    starting_cost = sub.cost
    starting_compactions = sub.compaction_count
    sub.invocation_count += 1
    sub.current_call_start = datetime.now()

    # Re-set the host ContextVar so any deeper tool handler sees THIS sub as
    # its host (enables grandchild subagents). Per-task scoping means parallel
    # branches stay isolated.
    host_token = _current_host.set(sub)
    last_info = None
    try:
        async for info in session.query(prompt, model, max_cost, sub.formatter):
            # info.cost / info.compaction_count are running totals for THIS query
            # call only; accumulate across calls. context_tokens is an absolute
            # current size, so last value wins. Backends may omit these fields.
            sub.cost = starting_cost + info.cost
            sub.context_tokens = getattr(info, "context_tokens", None)
            sub.compaction_count = starting_compactions + getattr(info, "compaction_count", 0)
            last_info = info
    except BaseException as e:
        sub.last_error = e
        raise
    finally:
        sub.accumulated_duration += (
            datetime.now() - sub.current_call_start
        ).total_seconds()
        sub.current_call_start = None
        _current_host.reset(host_token)

    if last_info is None or last_info.final_message is None:
        raise ValueError(f"No final message received from subagent '{name}'")
    return last_info.final_message


def _tree_cost(substeps: list[SubStep]) -> float:
    return sum(s.cost + _tree_cost(s.substeps) for s in substeps)


def _tree_in_call(substeps: list[SubStep]) -> bool:
    return any(
        s.current_call_start is not None or _tree_in_call(s.substeps)
        for s in substeps
    )


def _count_subagents(substeps: list[SubStep]) -> int:
    """Total number of SubSteps in the subtree."""
    return sum(1 + _count_subagents(s.substeps) for s in substeps)


def _collect_in_call(substeps: list[SubStep], depth: int = 1) -> list[tuple[int, SubStep]]:
    """The live frontier: every in-call SubStep in the subtree, with its depth."""
    out: list[tuple[int, SubStep]] = []
    for sub in substeps:
        if sub.current_call_start is not None:
            out.append((depth, sub))
        out.extend(_collect_in_call(sub.substeps, depth + 1))
    return out


def _substep_snapshot(sub: SubStep) -> tuple:
    return (
        sub.name,
        sub.cost,
        sub.current_call_start is not None,
        tuple(_substep_snapshot(c) for c in sub.substeps),
    )


@dataclass
class WorkflowStep:
    # immutable
    name: str
    prompt: str | Path
    model: str
    session: SessionABC
    retries: int
    max_cost: float | None
    requires: list[WorkflowStep | DynamicWorkflowStep]
    priority: int
    validator: Callable[[WorkflowStep], list[str]] | None
    validation_retry_model: str | None
    max_validation_retries: int
    max_validation_retry_cost: float | None
    # Nudge: after the session finishes on its own terms, `nudge(step)` returns a
    # follow-up prompt (or None to stop) to poke the model into continuing /
    # self-checking. Unlike the validator, a nudge never fails the step.
    nudge: Callable[[WorkflowStep], str | None] | None
    max_nudges: int
    max_nudge_cost: float | None
    nudge_model: str | None
    pre_hook: Callable[[WorkflowStep], Any] | None
    post_hook: Callable[[WorkflowStep], Any] | None
    condition: Callable[[WorkflowStep], bool] | None
    formatter: VerboseFormatter
    additional_context: dict[str, Any]
    done_event: asyncio.Event

    # mutable
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    attempt: int = 0  # counterpart to retries
    nudge_count: int = 0  # counterpart to max_nudges; reset on full-step retry
    final_message: str | None = None  # terminal message from the latest query on this step's session
    failed_attempts: list[FailedWorkflowStep] = field(default_factory=list)
    substeps: list[SubStep] = field(default_factory=list)
    cost: float = 0.0
    context_tokens: int | None = None  # context size of the latest call (last-wins)
    compaction_count: int = 0  # cumulative across all calls (incl. validation retries)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: BaseException | None = None

    async def run_subagent(
        self,
        name: str,
        session: SessionABC,
        *,
        prompt: str,
        model: str,
        max_cost: float | None = None,
        formatter: VerboseFormatter | None = None,
        max_depth: int = DEFAULT_MAX_SUBAGENT_DEPTH,
    ) -> str:
        """
        Run a tool-invoked subagent on `session` and record it under this step.

        Identity is (this step, name) — looked up in `self.substeps`. Repeat
        invocations of the same name within the same attempt accumulate stats
        onto the same SubStep (supports resumed sessions and validation
        retries that re-trigger the subagent). On step retry, the prior
        `substeps` list is moved into the new `FailedWorkflowStep.substeps`
        and `self.substeps` starts fresh.

        Default log file: `{working_dir}/{step.name}_{subagent.name}.log`,
        appended on subsequent invocations.
        """
        return await _run_subagent(
            parent=self,
            depth=1,
            max_depth=max_depth,
            name=name,
            session=session,
            prompt=prompt,
            model=model,
            max_cost=max_cost,
            formatter=formatter,
        )

    @property
    def parent(self) -> None:
        """WorkflowStep is the root of the host tree; has no parent."""
        return None

    def format_prompt(self, context: dict[str, Any]) -> str:
        return render_template(
            self.prompt,
            {**context, **self.additional_context},
            name=f"step '{self.name}'",
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WorkflowStep):
            return False
        return self.name == other.name


class AIWorkflow(ABC):
    console: Console
    working_dir: Path
    execution_dir: Path
    cleanup_working_dir: bool
    show_progress: bool

    start_time: datetime
    steps: list[WorkflowStep | DynamicWorkflowStep[Any]]
    context: dict[str, Any]
    status_title: str

    _init_called: bool
    _last_status_snapshot: tuple[Any, ...] | None

    def __init__(
        self,
        working_dir: Path | str | None = None,
        execution_dir: Path | str | None = None,
        show_progress: bool | None = None,
        cleanup_working_dir: bool | None = None,
        console: Console | None = None,
    ):
        ctx = click.get_current_context(silent=True)
        if ctx is None:
            cli = {}
        else:
            ctx.ensure_object(dict)
            cli = ctx.obj

        if working_dir is None:
            working_dir = cli.get("working_dir", None)
        if execution_dir is None:
            execution_dir = cli.get("execution_dir", None)

        self.execution_dir = (
            Path(execution_dir).resolve() if execution_dir else Path.cwd()
        )

        # Set up working directory
        if working_dir is not None:
            self.working_dir = Path(working_dir).resolve()
        else:
            # Generate session ID for working directory
            import random
            import string
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=6)
            )
            session_id = f"{timestamp}_{suffix}"
            self.working_dir = self.execution_dir / ".wake" / "ai" / session_id

        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.steps_log_file = self.working_dir / "steps.log"

        # Back up an existing steps.log to steps.{i}.log, same as VerboseFormatter
        if self.steps_log_file.exists():
            i = 0
            while self.steps_log_file.with_suffix(f".{i}.log").exists():
                i += 1
            self.steps_log_file.rename(self.steps_log_file.with_suffix(f".{i}.log"))

        self.show_progress = (
            show_progress
            if show_progress is not None
            else cli.get("show_progress", True)
        )

        self.cleanup_working_dir = (
            cleanup_working_dir
            if cleanup_working_dir is not None
            else cli.get("cleanup_working_dir", False)
        )

        if console is not None:
            self.console = console
        else:
            cli_console = cli.get("console")
            if cli_console is not None:
                self.console = cli_console
            else:
                self.console = Console()

        self.steps = []
        self.context = {}
        self.status_title = ""
        self._init_called = True
        self._last_status_snapshot = None

        # Pause/resume gate. Set = workflow may launch new steps; cleared = paused.
        # Pausing withholds only NEW launches — in-flight steps run to completion,
        # so pausing never interrupts a step mid-flight, and never triggers a retry
        # or from-scratch restart. Toggle via toggle_pause() (spacebar / SIGUSR1).
        self._run_gate = asyncio.Event()
        self._run_gate.set()

        # Live-adjustable cap on concurrently running WorkflowSteps (None = unlimited).
        # Seeded from execute(max_parallel_steps=...); changed mid-run via ↑/↓ keys.
        self._max_parallel_steps: int | None = None

        # Total-time clock pausing: cumulative seconds spent fully paused (idle),
        # plus the start of the current idle interval (None when not idle).
        self._paused_accumulated: float = 0.0
        self._quiescent_since: datetime | None = None

    @property
    def cumulative_cost(self) -> float:
        return sum(
            step.cost
            + sum(f.cost + _tree_cost(f.substeps) for f in step.failed_attempts)
            + _tree_cost(step.substeps)
            for step in self.steps if isinstance(step, WorkflowStep)
        )

    @property
    def is_paused(self) -> bool:
        """True when paused — no new steps will be launched (in-flight ones drain)."""
        return not self._run_gate.is_set()

    def toggle_pause(self) -> None:
        """Flip pause state. Safe from a keypress/signal handler. Drains, never cancels.

        Pausing lets in-flight steps finish; it does not cancel or retry them.
        Resuming relaunches whatever was withheld.
        """
        if self._run_gate.is_set():
            self._run_gate.clear()
            if not self.show_progress:
                self.console.print("[yellow]⏸ Pausing — finishing in-flight steps…[/yellow]")
        else:
            self._run_gate.set()
            if not self.show_progress:
                self.console.print("[green]▶ Resuming[/green]")

    def _running_workflow_steps(self) -> int:
        return sum(
            1 for s in self.steps
            if isinstance(s, WorkflowStep) and s.status == "running"
        )

    def _elapsed_seconds(self) -> float:
        """Wall-clock since start, minus time spent fully paused (idle).

        Draining (paused with steps still running) keeps counting — real work is
        happening; only the quiescent PAUSED state freezes the clock.
        """
        elapsed = (datetime.now() - self.start_time).total_seconds() - self._paused_accumulated
        if self._quiescent_since is not None:
            elapsed -= (datetime.now() - self._quiescent_since).total_seconds()
        return max(0.0, elapsed)

    def adjust_parallelism(self, delta: int) -> None:
        """Raise/lower the concurrent-step cap mid-run. Safe from a key handler.

        Lowering only stops NEW launches — already-running steps are never
        cancelled, so concurrency drains down to the new cap. From unlimited, a
        decrease first snaps to the current concurrency, then steps down.
        """
        cur = self._max_parallel_steps
        if cur is None:
            if delta >= 0:
                return  # already unlimited
            cur = max(self._running_workflow_steps(), 1)
        self._max_parallel_steps = max(1, cur + delta)
        if not self.show_progress:
            self.console.print(f"[cyan]∥ max parallel steps → {self._max_parallel_steps}[/cyan]")

    def _install_pause_controls(self) -> Callable[[], None]:
        """Wire spacebar (interactive TTY) and SIGUSR1 to toggle_pause; return a cleanup fn.

        No-ops gracefully when stdin isn't a TTY (piped/headless/Docker) or the
        platform lacks termios/SIGUSR1. SIGUSR1 still works headless:
        ``kill -USR1 <pid>`` toggles pause.
        """
        cleanups: list[Callable[[], None]] = []
        loop = asyncio.get_running_loop()

        # Spacebar on an interactive TTY.
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_attrs = termios.tcgetattr(fd)
                tty.setcbreak(fd)  # ICANON+ECHO off; ISIG stays on so Ctrl+C still works

                def _on_key() -> None:
                    try:
                        data = os.read(fd, 8)  # an arrow key is a 3-byte escape sequence
                    except Exception:
                        return
                    if data == b" ":
                        self.toggle_pause()
                    elif data in (b"\x1b[A", b"\x1bOA"):  # Up
                        self.adjust_parallelism(+1)
                    elif data in (b"\x1b[B", b"\x1bOB"):  # Down
                        self.adjust_parallelism(-1)

                loop.add_reader(fd, _on_key)

                def _restore_tty() -> None:
                    with suppress(Exception):
                        loop.remove_reader(fd)
                    with suppress(Exception):
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

                cleanups.append(_restore_tty)
        except Exception:
            pass

        # SIGUSR1 for headless toggling (e.g. the dockerized beast-run service).
        try:
            loop.add_signal_handler(signal.SIGUSR1, self.toggle_pause)
            cleanups.append(lambda: loop.remove_signal_handler(signal.SIGUSR1))
        except (NotImplementedError, AttributeError, ValueError):
            pass

        def _cleanup() -> None:
            for fn in cleanups:
                with suppress(Exception):
                    fn()

        return _cleanup

    @require_initialized
    def add_step(
        self,
        name: str,
        prompt: str | Path,
        model: str,
        *,
        requires: list[WorkflowStep | DynamicWorkflowStep[Any]] | None = None,
        priority: int = 0,
        max_cost: float | None = None,
        retries: int = 0,
        validator: Callable[[WorkflowStep], list[str]] | None = None,
        validation_retry_model: str | None = None,
        max_validation_retries: int = 3,
        max_validation_retry_cost: float | None = None,
        nudge: Callable[[WorkflowStep], str | None] | None = None,
        max_nudges: int = 1,
        max_nudge_cost: float | None = None,
        nudge_model: str | None = None,
        session: SessionABC | None = None,
        pre_hook: Callable[[WorkflowStep], Any] | None = None,
        post_hook: Callable[[WorkflowStep], Any] | None = None,
        condition: Callable[[WorkflowStep], bool] | None = None,
        formatter: VerboseFormatter | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> WorkflowStep:
        """
        Validation retry always uses the same session.

        IMPORTANT: Requiring a dynamic step doesn't wait for steps possibly created by that dynamic step.
        """
        if any(s.name == name for s in self.steps):
            raise ValueError(f"Step with name '{name}' already exists")

        if ("\\" in name or "/" in name):
            raise ValueError(f"Step contains invalid characters: '{name}'")

        if session is None:
            session = _default_session(model, self.execution_dir, self.working_dir)
        else:
            if retries > 0 and session.session_id is not None:
                raise ValueError("Non-zero retries are not possible when continuing a session")

        step = WorkflowStep(
            name=name,
            status="pending",
            prompt=prompt,
            model=model,
            session=session,
            retries=retries,
            max_cost=max_cost,
            requires=requires or [],
            priority=priority,
            validator=validator,
            validation_retry_model=validation_retry_model,
            max_validation_retries=max_validation_retries,
            max_validation_retry_cost=max_validation_retry_cost,
            nudge=nudge,
            max_nudges=max_nudges,
            max_nudge_cost=max_nudge_cost,
            nudge_model=nudge_model,
            pre_hook=pre_hook,
            post_hook=post_hook,
            condition=condition,
            formatter=(
                formatter or VerboseFormatter(self.console, name, self.working_dir / f"{name}.log", get_verbosity_level())
            ),
            additional_context=additional_context or {},
            done_event=asyncio.Event(),
        )
        self.steps.append(step)

        # Update sorted steps using our own topological sort
        self.steps = self._find_topological_order()

        return step

    def add_dynamic_step(
        self,
        name: str,
        handler: Callable[[DynamicWorkflowStep[T]], Coroutine[Any, Any, T]],
        *,
        requires: list[WorkflowStep | DynamicWorkflowStep[Any]] | None = None,
        condition: Callable[[DynamicWorkflowStep[T]], bool] | None = None,
    ) -> DynamicWorkflowStep[T]:
        """
        IMPORTANT: Requiring a dynamic step doesn't wait for steps possibly created by that dynamic step.
        """
        if any(s.name == name for s in self.steps):
            raise ValueError(f"Step with name '{name}' already exists")

        step = DynamicWorkflowStep(
            name=name,
            status="pending",
            handler=handler,
            requires=requires or [],
            condition=condition,
            done_event=asyncio.Event(),
            result=None,  # temporary
        )
        step.result = DynamicStepResult(step)
        self.steps.append(step)
        return step

    def _find_topological_order(self) -> list[WorkflowStep | DynamicWorkflowStep[Any]]:
        """Find topological ordering of steps using Kahn's algorithm."""
        # Build in-degree map
        in_degree = {step: len(step.requires) for step in self.steps}

        # Find steps with no dependencies
        queue = sorted(
            [step for step in self.steps if in_degree[step] == 0],
            key=lambda s: getattr(s, "priority", 0),
            reverse=True,
        )
        result: list[WorkflowStep | DynamicWorkflowStep[Any]] = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # For each step that depends on current step
            for step in self.steps:
                if current in step.requires:
                    in_degree[step] -= 1
                    if in_degree[step] == 0:
                        queue.append(step)

            queue.sort(key=lambda s: getattr(s, "priority", 0), reverse=True)

        # Check for cycles
        if len(result) != len(self.steps):
            raise ValueError("Steps form a cycle")

        pending = [step for step in result if step.status == "pending"]
        running = sorted((step for step in result if step.status == "running"), key=lambda x: x.start_time)
        others = sorted((step for step in result if step.status not in ["pending", "running"]), key=lambda x: x.end_time)

        return others + running + pending

    # running in a separate asyncio task
    async def _execute_step(self, step: WorkflowStep) -> None:
        if step.pre_hook is not None:
            step.pre_hook(step)

        # Tag this asyncio task's context so tool handlers fired from within
        # `step.session.query(...)` can locate the step via `current_host()`.
        # Per-task scoping: parallel _execute_step tasks each have their own.
        host_token = _current_host.set(step)
        try:
            # main prompt query
            last_info = None
            async for info in step.session.query(
                step.format_prompt(self.context),
                step.model,
                step.max_cost,
                step.formatter,
            ):
                # context_tokens: absolute current size (last-wins). cost /
                # compaction_count: per-call running totals accumulated across the
                # query calls that make up this row. Backends may omit these fields.
                step.cost = info.cost
                step.context_tokens = getattr(info, "context_tokens", None)
                step.compaction_count = getattr(info, "compaction_count", 0)
                last_info = info

            step.final_message = getattr(last_info, "final_message", None)
            total_cost = step.cost
            total_compactions = step.compaction_count

            # nudge loop — poke a session that finished on its own terms to keep
            # working / self-check. Runs before validation (nudge in task-space,
            # then let the validator be the final gate). Only nudge on a natural
            # "succeeded"; never a session that bailed on max cost/turns. Unlike
            # validation, a nudge never fails the step — the callback returns the
            # follow-up prompt, or None to stop. Cost accumulates like validation.
            if step.nudge is not None and getattr(last_info, "status", None) == "succeeded":
                while step.nudge_count < step.max_nudges:
                    nudge_prompt = step.nudge(step)
                    if nudge_prompt is None:
                        break
                    step.nudge_count += 1

                    async for info in step.session.query(
                        nudge_prompt,
                        step.nudge_model or step.model,
                        step.max_nudge_cost,
                        step.formatter,
                    ):
                        step.cost = total_cost + info.cost
                        step.context_tokens = getattr(info, "context_tokens", None)
                        step.compaction_count = total_compactions + getattr(info, "compaction_count", 0)
                        last_info = info

                    step.final_message = getattr(last_info, "final_message", None)
                    total_cost = step.cost
                    total_compactions = step.compaction_count

            # validation + fixing attempts
            if step.validator is not None:
                errors = step.validator(step)

                retry_count = 0
                while errors and retry_count < step.max_validation_retries:
                    retry_count += 1
                    prompt = "The following errors occurred, please fix them:\n-" + "\n-".join(errors)

                    async for info in step.session.query(
                        prompt,
                        step.validation_retry_model or step.model,
                        step.max_validation_retry_cost,
                        step.formatter,
                    ):
                        step.cost = total_cost + info.cost
                        step.context_tokens = getattr(info, "context_tokens", None)
                        step.compaction_count = total_compactions + getattr(info, "compaction_count", 0)
                        last_info = info

                    step.final_message = getattr(last_info, "final_message", None)
                    total_cost = step.cost
                    total_compactions = step.compaction_count

                    errors = step.validator(step)

                if errors:
                    step.status = "failed"
                    error_msg = f"Step '{step.name}' validation failed after {step.max_validation_retries} retries. Errors: {'; '.join(errors)}"
                    raise RuntimeError(error_msg)
        finally:
            _current_host.reset(host_token)

        if step.post_hook is not None:
            step.post_hook(step)

    @abstractmethod
    def collect_result(self) -> AIResult:
        ...

    @require_initialized
    async def execute(
        self,
        *,
        max_parallel_steps: int | None = None,
    ) -> AIResult:
        try:
            self._max_parallel_steps = max_parallel_steps
            key_cleanup = self._install_pause_controls()
            running: dict[asyncio.Task, WorkflowStep | DynamicWorkflowStep[Any]] = {}

            self.start_time = datetime.now()

            ctx = nullcontext() if not self.show_progress else Live(
                self._get_status_display(),
                refresh_per_second=10,
                auto_refresh=True,
                console=self.console,
                transient=True,
                get_renderable=self._get_status_display,
            )

            with ctx as live:
                while any(step.status in ["pending", "running"] for step in self.steps):
                    # Freeze the total-time clock once fully quiescent (paused with
                    # no running step). Draining (paused, steps still finishing)
                    # keeps ticking. Once paused, no new steps launch, so the
                    # running count only falls — the idle interval runs until resume.
                    if self.is_paused and self._running_workflow_steps() == 0:
                        if self._quiescent_since is None:
                            self._quiescent_since = datetime.now()
                    elif self._quiescent_since is not None:
                        self._paused_accumulated += (datetime.now() - self._quiescent_since).total_seconds()
                        self._quiescent_since = None

                    # Start all ready steps that can run (withheld while paused;
                    # in-flight steps still drain to completion below).
                    for step in list(self.steps):
                        if not self._run_gate.is_set():
                            break
                        # only count WorkflowSteps; read the live (mutable) cap
                        limit = self._max_parallel_steps
                        if (
                            limit is not None
                            and len([s for s in running.values() if isinstance(s, WorkflowStep)]) >= limit
                        ):
                            break

                        if (
                            step.status == "pending" and
                            all(r.status in ["completed", "skipped"] for r in step.requires)
                        ):
                            if isinstance(step, WorkflowStep):
                                same_session_steps = [
                                    s for s in running.values()
                                    if isinstance(s, WorkflowStep) and s.session == step.session
                                ]
                                if same_session_steps:
                                    raise RuntimeError(f"Step {step.name} shares the same session with steps: {', '.join(s.name for s in same_session_steps)}; and so cannot run in parallel")

                            skipped_requires = [r for r in step.requires if r.status == "skipped"]
                            if skipped_requires:
                                raise RuntimeError(f"Step {step.name} has skipped requires: {', '.join(r.name for r in skipped_requires)}; and so cannot run")

                            if step.condition is not None and not step.condition(step):
                                step.status = "skipped"
                                step.start_time = datetime.now()
                                step.end_time = step.start_time
                                step.done_event.set()
                            else:
                                step.status = "running"
                                step.start_time = datetime.now()

                                if isinstance(step, WorkflowStep):
                                    step.attempt += 1
                                    task = asyncio.create_task(self._execute_step(step))
                                else:
                                    dynamic_step = cast(DynamicWorkflowStep[Any], step)
                                    task = asyncio.create_task(dynamic_step.handler(dynamic_step))
                                running[task] = step

                    if not running:
                        # Nothing in flight. If paused, this is the quiescent
                        # "safe to suspend/kill" state — park until resumed.
                        # Otherwise yield briefly (nothing launchable this tick).
                        if not self._run_gate.is_set():
                            await self._run_gate.wait()
                        else:
                            await asyncio.sleep(0.05)
                        continue

                    # dynamic steps may produce new children steps at runtime and wait for them to complete
                    done, _ = await asyncio.wait(running.keys(), timeout=1, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        step = running.pop(task)
                        exception = task.exception()
                        if exception is None:
                            step.status = "completed"
                            step.end_time = datetime.now()
                            if isinstance(step, DynamicWorkflowStep):
                                step._return_value = task.result()
                            step.done_event.set()
                        else:
                            if isinstance(step, WorkflowStep) and step.attempt <= step.retries:
                                if isinstance(exception, BaseExceptionGroup):
                                    for i, sub in enumerate(exception.exceptions, 1):
                                        self.console.print(f"[red]Subexception {i}:[/red] {sub!r}")
                                        self.console.print(
                                            Traceback.from_exception(type(sub), sub, sub.__traceback__)
                                        )
                                else:
                                    self.console.print(
                                        Traceback.from_exception(type(exception), exception, exception.__traceback__)
                                    )
                                self.console.print(f"Exception happened during step {step.name}, retrying...")

                                assert step.start_time is not None
                                # Move this attempt's substeps into the
                                # failed-attempt record (structural staleness)
                                # and start a fresh list for the next attempt.
                                step.failed_attempts.append(FailedWorkflowStep(
                                    cost=step.cost,
                                    start_time=step.start_time,
                                    end_time=datetime.now(),
                                    error=exception,
                                    substeps=step.substeps,
                                ))
                                step.substeps = []

                                step.cost = 0.0
                                step.nudge_count = 0
                                step.final_message = None
                                step.status = "pending"
                                step.start_time = None
                                step.session.reset()
                                step.formatter.reset_log_file()

                                # wait for 60 seconds to avoid overwhelming the server
                                await asyncio.sleep(60)
                            else:
                                step.status = "failed"
                                step.end_time = datetime.now()
                                step.error = exception
                                step.done_event.set()
                                raise exception

            return self.collect_result()
        finally:
            with suppress(Exception):
                key_cleanup()
            self.console.print(self._get_status_display())

            if self.cleanup_working_dir and self.working_dir.exists():
                shutil.rmtree(self.working_dir)

    @classmethod
    def _render_substep_tree(
        cls,
        table: Table,
        sub: SubStep,
        parent_status: str,
        stale: bool,
    ) -> None:
        """Render a SubStep and its descendants. Status mirrors root WorkflowStep."""
        in_call = sub.current_call_start is not None and not stale

        parent_glyph = {
            "pending": " ",
            "running": "⟳",
            "completed": "✓",
            "failed": "✗",
            "skipped": "○",
        }.get(parent_status, " ")

        if stale:
            glyph = "✗" if parent_status == "failed" else parent_glyph
            style = "dim"
        elif in_call:
            glyph = "▶"
            style = "bright_yellow"
        else:
            glyph = parent_glyph
            if parent_status == "completed":
                style = "green"
            elif parent_status == "failed":
                style = "red"
            elif parent_status == "running":
                style = "bright_cyan"
            elif parent_status == "skipped":
                style = "yellow"
            else:
                style = "dim"

        elapsed = sub.accumulated_duration
        if in_call and sub.current_call_start is not None:
            elapsed += (datetime.now() - sub.current_call_start).total_seconds()
        sub_duration = _format_duration(elapsed) if elapsed > 0 else "-"
        sub_cost = f"${sub.cost:.4f}" if sub.cost > 0 else "-"

        suffix = f" ×{sub.invocation_count}" if sub.invocation_count > 1 else ""
        indent = "  " * sub.depth
        table.add_row(
            f"{indent}└─ {glyph} {sub.name}{suffix}",
            "",
            sub.model,
            _format_context(sub.context_tokens, sub.compaction_count),
            sub_duration,
            sub_cost,
            style=style,
        )

        # Recurse into children
        for child in sub.substeps:
            cls._render_substep_tree(table, child, parent_status, stale)

    def _make_status_table(self) -> Table:
        # Every column no_wrap so a logical row always renders as exactly one
        # terminal line — that keeps the compact view's row budget an exact line
        # budget, so it can never overflow vertically (it crops/ellipsizes width).
        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("Step", no_wrap=True, width=60, overflow="ellipsis")
        table.add_column("Attempt", no_wrap=True, width=9)
        table.add_column("Model", no_wrap=True, width=18, overflow="ellipsis")
        table.add_column("Context", justify="right", no_wrap=True, width=12)
        table.add_column("Time", justify="right", no_wrap=True, width=12)
        table.add_column("Cost", justify="right", no_wrap=True, width=10)
        return table

    def _step_row(self, step: WorkflowStep) -> tuple[list[str], str]:
        """(cells, style) for a WorkflowStep's own row."""
        prefix, row_style = {
            "pending": ("  ", "dim"),
            "completed": ("✓ ", "green"),
            "skipped": ("○ ", "yellow"),
            "running": ("⟳ ", "bright_cyan"),
            "failed": ("✗ ", "red"),
        }.get(step.status, ("  ", "dim"))

        if step.status == "running" and step.start_time:
            duration = _format_duration((datetime.now() - step.start_time).total_seconds())
        elif step.end_time and step.start_time:
            duration = _format_duration((step.end_time - step.start_time).total_seconds())
        else:
            duration = "-"

        cost = f"${step.cost:.4f}" if step.cost > 0 else "-"
        attempt = "-" if step.attempt == 0 else f"{step.attempt}/{step.retries + 1}"
        cells = [
            f"{prefix}{step.name}",
            attempt,
            step.model,
            _format_context(step.context_tokens, step.compaction_count),
            duration,
            cost,
        ]
        return cells, row_style

    def _substep_row_cells(self, sub: SubStep, depth: int) -> list[str]:
        """Cells for an in-call (live) substep row in the compact view."""
        elapsed = sub.accumulated_duration
        if sub.current_call_start is not None:
            elapsed += (datetime.now() - sub.current_call_start).total_seconds()
        suffix = f" ×{sub.invocation_count}" if sub.invocation_count > 1 else ""
        return [
            f"{'  ' * depth}└─ ▶ {sub.name}{suffix}",
            "",
            sub.model,
            _format_context(sub.context_tokens, sub.compaction_count),
            _format_duration(elapsed) if elapsed > 0 else "-",
            f"${sub.cost:.4f}" if sub.cost > 0 else "-",
        ]

    def _add_total_row(self, table: Table) -> None:
        table.add_section()
        table.add_row(
            "Total", "", "", "",
            _format_duration(self._elapsed_seconds()),
            f"${self.cumulative_cost:.4f}",
            style="bold",
        )

    def _build_full_table(self) -> Table:
        """The complete table (every step, every substep) — written to steps.log."""
        table = self._make_status_table()
        for step in self.steps:
            if not isinstance(step, WorkflowStep):
                continue
            for i, failed_attempt in enumerate(step.failed_attempts):
                table.add_row(
                    f"✗ {step.name}",
                    f"{i + 1}/{step.retries + 1}",
                    step.model, "",
                    _format_duration((failed_attempt.end_time - failed_attempt.start_time).total_seconds()),
                    f"${failed_attempt.cost:.4f}",
                    style="dim red",
                )
                for sub in failed_attempt.substeps:
                    self._render_substep_tree(table, sub, parent_status="failed", stale=True)
            cells, row_style = self._step_row(step)
            table.add_row(*cells, style=row_style)
            for sub in step.substeps:
                self._render_substep_tree(table, sub, parent_status=step.status, stale=False)
        self._add_total_row(table)
        return table

    @staticmethod
    def _counts_summary(counts: dict[str, int]) -> str:
        parts = [
            f"{glyph}{counts[status]}"
            for status, glyph in (
                ("completed", "✓"), ("running", "⟳"), ("pending", "⋯"),
                ("failed", "✗"), ("skipped", "○"),
            )
            if counts.get(status)
        ]
        return "   ".join(parts) or "no steps"

    def _render_frontier(self, table: Table, step: WorkflowStep, budget: int) -> int:
        """Render a running step's live (in-call) subagents, up to `budget` rows;
        collapse everything else into one summary row. Returns rows added."""
        total = _count_subagents(step.substeps)
        if total == 0 or budget <= 0:
            return 0
        in_call = _collect_in_call(step.substeps)
        used = 0
        for depth, sub in in_call[: max(0, budget - 1)]:  # leave a row for the summary
            table.add_row(*self._substep_row_cells(sub, depth), style="bright_yellow")
            used += 1
        unshown_live = len(in_call) - used
        done = total - len(in_call)
        remaining = unshown_live + done
        if remaining > 0 and used < budget:
            label = (
                f"    └─ +{remaining} more ({done} done)" if unshown_live
                else f"    └─ +{done} subagents done"
            )
            table.add_row(label, "", "", "", "", "", style="dim")
            used += 1
        return used

    def _build_compact_table(self) -> Table:
        """Auto-fitting 'live frontier' view: status summary + failed + running
        (showing only each step's in-call subagents, the rest counted) + pinned
        Total. Sized to the terminal so it never overflows; full detail in steps.log.
        """
        table = self._make_status_table()

        height = self.console.size.height or 40
        # reserve: borders(2) header(1) summary(1) section+total(2) subtitle(1)
        # safety(1) + 1 extra so the top border survives a terminal overlay
        budget = max(4, height - 9)

        wf_steps = [s for s in self.steps if isinstance(s, WorkflowStep)]
        counts: dict[str, int] = {}
        for s in wf_steps:
            counts[s.status] = counts.get(s.status, 0) + 1
        running = [s for s in wf_steps if s.status == "running"]
        failed = [s for s in wf_steps if s.status == "failed"]

        table.add_row(self._counts_summary(counts), "", "", "", "", "", style="bold dim")
        rows_left = budget

        # Failed steps (no substeps) — surfaced first.
        hidden_failed = 0
        for s in failed:
            if rows_left <= 1 and running:
                hidden_failed += 1
                continue
            cells, _ = self._step_row(s)
            table.add_row(*cells, style="red")
            rows_left -= 1
        if hidden_failed:
            table.add_row(f"  ✗ +{hidden_failed} more failed (see steps.log)", "", "", "", "", "", style="dim red")
            rows_left -= 1

        # Running steps with fair-share live frontier.
        hidden_running = 0
        n = len(running)
        for i, s in enumerate(running):
            if rows_left <= 0:
                hidden_running = n - i
                break
            cells, style = self._step_row(s)
            table.add_row(*cells, style=style)
            rows_left -= 1
            remaining_steps = n - i
            slice_budget = rows_left // remaining_steps if remaining_steps else 0
            rows_left -= self._render_frontier(table, s, slice_budget)
        if hidden_running:
            table.add_row(f"  ⟳ +{hidden_running} more running (see steps.log)", "", "", "", "", "", style="dim cyan")
            rows_left -= 1

        # Pending count, if room.
        if counts.get("pending") and rows_left > 0:
            table.add_row(f"  ⋯ {counts['pending']} pending", "", "", "", "", "", style="dim")

        self._add_total_row(table)
        return table

    def _get_status_display(self) -> Panel:
        """Compact 'live frontier' panel for the screen; the full table goes to steps.log."""
        # Write the FULL table to the log only when status changes (not on time ticks).
        current_snapshot = tuple(
            (
                step.name,
                step.status,
                step.attempt if isinstance(step, WorkflowStep) else None,
                step.cost if isinstance(step, WorkflowStep) else None,
                tuple(
                    (f.cost, tuple(_substep_snapshot(s) for s in f.substeps))
                    for f in step.failed_attempts
                ) if isinstance(step, WorkflowStep) else None,
                tuple(_substep_snapshot(s) for s in step.substeps)
                if isinstance(step, WorkflowStep) else None,
            )
            for step in self.steps
        )
        if current_snapshot != self._last_status_snapshot:
            buffer = StringIO()
            # Wide, fixed width so the full table renders without compression/cropping.
            log_console = Console(file=buffer, force_terminal=False, width=200)
            log_console.print(self._build_full_table())
            with open(self.steps_log_file, "w") as f:
                f.write(buffer.getvalue())
            self._last_status_snapshot = current_snapshot

        # Title shows live concurrency (running/cap) and pause state; subtitle lists
        # controls. status_title (e.g. beast's progress %) is preserved, never clobbered.
        running_n = self._running_workflow_steps()
        limit = self._max_parallel_steps
        par = f"∥ {running_n}/{'∞' if limit is None else limit}"
        base = f"{self.status_title}  │  {par}" if self.status_title else par
        if self.is_paused:
            note = (
                "⏸ DRAINING — finishing in-flight steps…" if running_n
                else "⏸ PAUSED — safe to suspend or kill"
            )
            title = f"{note}  │  {base}"
            border = "yellow"
        else:
            title = base
            border = "blue"

        return Panel(
            self._build_compact_table(),
            title=title,
            subtitle="SPACE pause   ↑/↓ parallel",
            subtitle_align="right",
            border_style=border,
        )
