from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from io import StringIO
from pathlib import Path
import shutil
from typing import Any, Awaitable, Literal, Callable, NamedTuple, Coroutine, TypeVar, Generic, cast

import jinja2
import jinja2.meta
import rich_click as click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback

from .claude import ClaudeSession
from .codex import CodexSession
from .codex_pricing import GPT_PRICING
from .verbose_formatter import VerboseFormatter
from ..results import AIResult
from ..utils.logging import get_verbosity_level


def _format_duration(seconds: float) -> str:
    """Format duration dynamically based on the time scale."""
    days = int(seconds // 86400)
    hours = int(seconds // 3600)
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


class FailedWorkflowStep(NamedTuple):
    cost: float
    start_time: datetime
    end_time: datetime
    error: BaseException


@dataclass
class WorkflowStep:
    # immutable
    name: str
    prompt: str | Path
    model: str
    session: ClaudeSession | CodexSession
    retries: int
    max_cost: float | None
    requires: list[WorkflowStep | DynamicWorkflowStep]
    validator: Callable[[WorkflowStep], list[str]] | None
    validation_retry_model: str | None
    max_validation_retries: int
    max_validation_retry_cost: float | None
    pre_hook: Callable[[WorkflowStep], Any] | None
    post_hook: Callable[[WorkflowStep], Any] | None
    condition: Callable[[WorkflowStep], bool] | None
    formatter: VerboseFormatter
    additional_context: dict[str, Any]
    done_event: asyncio.Event

    # mutable
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    attempt: int = 0  # counterpart to retries
    failed_attempts: list[FailedWorkflowStep] = field(default_factory=list)
    cost: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: BaseException | None = None

    def format_prompt(self, context: dict[str, Any]) -> str:
        env = jinja2.Environment(
            undefined=jinja2.StrictUndefined,
            loader=jinja2.FileSystemLoader(str(self.prompt.parent)) if isinstance(self.prompt, Path) else None
        )
        prompt_text = self.prompt.read_text() if isinstance(self.prompt, Path) else self.prompt

        # Parse the template to find all variables
        ast = env.parse(prompt_text)
        prompt_context_keys = jinja2.meta.find_undeclared_variables(ast)

        # Warn if there are context keys that are not in the context
        for key in prompt_context_keys:
            if key not in context and key not in self.additional_context:
                print(f"Context key '{key}' used in step '{self.name}' not provided")

        template = env.from_string(prompt_text)
        return template.render(**context, **self.additional_context)

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
            self.working_dir = Path.cwd() / ".wake" / "ai" / session_id

        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.steps_log_file = self.working_dir / "steps.log"

        self.execution_dir = (
            Path(execution_dir).resolve() if execution_dir else Path.cwd()
        )

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
        self._init_called = True
        self._last_status_snapshot = None

    @property
    def cumulative_cost(self) -> float:
        return sum(step.cost + sum(f.cost for f in step.failed_attempts) for step in self.steps if isinstance(step, WorkflowStep))

    @require_initialized
    def add_step(
        self,
        name: str,
        prompt: str | Path,
        model: str,
        *,
        requires: list[WorkflowStep | DynamicWorkflowStep[Any]] | None = None,
        max_cost: float | None = None,
        retries: int = 0,
        validator: Callable[[WorkflowStep], list[str]] | None = None,
        validation_retry_model: str | None = None,
        max_validation_retries: int = 3,
        max_validation_retry_cost: float | None = None,
        session: ClaudeSession | CodexSession | None = None,
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
            if model.lower() in GPT_PRICING:
                session = CodexSession(self.execution_dir)
            else:
                session = ClaudeSession(self.execution_dir, self.working_dir)
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
            validator=validator,
            validation_retry_model=validation_retry_model,
            max_validation_retries=max_validation_retries,
            max_validation_retry_cost=max_validation_retry_cost,
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
        queue = [step for step in self.steps if in_degree[step] == 0]
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

        # main prompt query
        async for info in step.session.query(
            step.format_prompt(self.context),
            step.model,
            step.max_cost,
            step.formatter,
        ):
            step.cost = info.cost

        total_cost = step.cost

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

                total_cost = step.cost

                errors = step.validator(step)

            if errors:
                step.status = "failed"
                error_msg = f"Step '{step.name}' validation failed after {step.max_validation_retries} retries. Errors: {'; '.join(errors)}"
                raise RuntimeError(error_msg)

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
                    # Start all ready steps that can run
                    for step in list(self.steps):
                        # only count WorkflowSteps
                        if (
                            max_parallel_steps is not None
                            and len([s for s in running.values() if isinstance(s, WorkflowStep)]) >= max_parallel_steps
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
                                traceback = Traceback.from_exception(type(exception), exception, exception.__traceback__)
                                self.console.print(traceback)
                                self.console.print(f"Exception happened during step {step.name}, retrying...")

                                assert step.start_time is not None
                                step.failed_attempts.append(FailedWorkflowStep(
                                    cost=step.cost,
                                    start_time=step.start_time,
                                    end_time=datetime.now(),
                                    error=exception,
                                ))

                                step.cost = 0.0
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
            self.console.print(self._get_status_display())

            if self.cleanup_working_dir and self.working_dir.exists():
                shutil.rmtree(self.working_dir)

    def _get_status_display(self) -> Panel:
        """Build the status display with table and progress."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            min_width=125,
        )
        table.add_column("Step", no_wrap=True, width=40)
        table.add_column("Attempt", no_wrap=True, width=20)
        table.add_column("Model", no_wrap=True, width=25)
        table.add_column("Time", justify="right", width=20)
        table.add_column("Cost", justify="right", width=20)

        # Add rows for each step
        for step in self.steps:
            if isinstance(step, DynamicWorkflowStep):
                continue

            # Determine row style based on status
            if step.status == "pending":
                row_style = "dim"
                step_name = f"  {step.name}"
            elif step.status == "completed":
                row_style = "green"
                step_name = f"✓ {step.name}"
            elif step.status == "skipped":
                row_style = "yellow"
                step_name = f"○ {step.name}"
            elif step.status == "running":
                row_style = "bright_cyan"
                step_name = f"⟳ {step.name}"
            else:
                row_style = "red"
                step_name = f"✗ {step.name}"

            # Format duration (show live time for running steps)
            if step.status == "running" and step.start_time:
                # Calculate current running time
                running_time = (datetime.now() - step.start_time).total_seconds()
                duration = _format_duration(running_time)
            elif step.end_time and step.start_time:
                duration = _format_duration(
                    (step.end_time - step.start_time).total_seconds()
                )
            else:
                duration = "-"

            if isinstance(step, DynamicWorkflowStep):
                cost = "-"
                model = "-"
                attempt = "-"
            else:
                # Format cost
                cost = f"${step.cost:.4f}" if step.cost > 0 else "-"
                model = step.model
                if step.attempt == 0:
                    attempt = "-"
                else:
                    attempt = f"{step.attempt}/{step.retries + 1}"

                # print failed attempts first
                for i, failed_attempt in enumerate(step.failed_attempts):
                    table.add_row(
                        f"✗ {step.name}",
                        f"{i + 1}/{step.retries + 1}",
                        model,
                        _format_duration((failed_attempt.end_time - failed_attempt.start_time).total_seconds()),
                        f"${failed_attempt.cost:.4f}",
                        style="dim red",
                    )

            table.add_row(step_name, attempt, model, duration, cost, style=row_style)

        # Add total row
        total_cost = sum(
            step.cost + sum(f.cost for f in step.failed_attempts)
            for step in self.steps if isinstance(step, WorkflowStep)
        )

        table.add_section()
        table.add_row(
            "Total",
            "",
            "",
            f"{_format_duration((datetime.now() - self.start_time).total_seconds())}",
            f"${total_cost:.4f}",
            style="bold",
        )

        # Write rendered status to log file only when status changes (not time)
        current_snapshot = tuple(
            (
                step.name,
                step.status,
                step.attempt if isinstance(step, WorkflowStep) else None,
                step.cost if isinstance(step, WorkflowStep) else None,
                tuple(f.cost for f in step.failed_attempts) if isinstance(step, WorkflowStep) else None,
            )
            for step in self.steps
        )

        if current_snapshot != self._last_status_snapshot:
            buffer = StringIO()
            log_console = Console(file=buffer, force_terminal=False, width=84)
            log_console.print(table)
            with open(self.steps_log_file, "w") as f:
                f.write(buffer.getvalue())
            self._last_status_snapshot = current_snapshot

        # Return panel with table and status
        return Panel(table, title="", border_style="blue", width=84)
