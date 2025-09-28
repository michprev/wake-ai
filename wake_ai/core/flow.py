from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
import shutil
from typing import Any, Literal, Callable

import jinja2
import jinja2.meta
import rich_click as click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

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


def conditionally_cleanup(func):
    @wraps(func)
    def wrapper(self: AIWorkflow, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.cleanup_working_dir and self.working_dir.exists():
                shutil.rmtree(self.working_dir)

    return wrapper


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


@dataclass()
class WorkflowStep:
    # immutable
    name: str
    prompt: str
    model: str
    session: ClaudeSession | CodexSession
    max_cost: float | None
    requires: list[WorkflowStep]
    validator: Callable[[WorkflowStep], list[str]] | None
    validation_retry_model: str | None
    max_validation_retries: int
    max_validation_retry_cost: float | None
    pre_hook: Callable[[WorkflowStep], None] | None
    post_hook: Callable[[WorkflowStep], None] | None
    condition: Callable[[WorkflowStep], bool] | None
    formatter: VerboseFormatter

    # mutable
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    cost: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: Exception | None = None

    def format_prompt(self, context: dict[str, Any]) -> str:
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)

        # Parse the template to find all variables
        ast = env.parse(self.prompt)
        prompt_context_keys = jinja2.meta.find_undeclared_variables(ast)

        # Warn if there are context keys that are not in the context
        for key in prompt_context_keys:
            if key not in context:
                print(f"Context key '{key}' used in step '{self.name}' not provided")

        template = env.from_string(self.prompt)
        return template.render(**context)

    def __hash__(self) -> int:
        # Hash based on immutable fields only
        return hash((self.name, self.prompt))

    def __eq__(self, other) -> bool:
        if not isinstance(other, WorkflowStep):
            return False
        return self.name == other.name and self.prompt == other.prompt


class AIWorkflow(ABC):
    console: Console
    working_dir: Path
    execution_dir: Path

    start_time: datetime
    steps: list[WorkflowStep]
    context: dict[str, Any]

    # cached topologically sorted steps
    _sorted_steps: list[WorkflowStep]
    _init_called: bool

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
        self._sorted_steps = []
        self._init_called = True

    @property
    def cumulative_cost(self) -> float:
        return sum(step.cost for step in self.steps)

    def _pre_step_hook(self, step: WorkflowStep) -> None:
        pass

    def _post_step_hook(self, step: WorkflowStep) -> None:
        pass

    @require_initialized
    def add_step(
        self,
        name: str,
        prompt: str,
        model: str,
        *,
        requires: list[WorkflowStep] | None = None,
        max_cost: float | None = None,
        validator: Callable[[WorkflowStep], list[str]] | None = None,
        validation_retry_model: str | None = None,
        max_validation_retries: int = 3,
        max_validation_retry_cost: float | None = None,
        session: ClaudeSession | CodexSession | None = None,
        pre_hook: Callable[[WorkflowStep], None] | None = None,
        post_hook: Callable[[WorkflowStep], None] | None = None,
        condition: Callable[[WorkflowStep], bool] | None = None,
        formatter: VerboseFormatter | None = None,
    ) -> WorkflowStep:
        """
        Validation retry always uses the same session.
        """
        if any(s.name == name for s in self.steps):
            raise ValueError(f"Step with name '{name}' already exists")

        if session is None:
            if model.lower() in GPT_PRICING:
                session = CodexSession(self.execution_dir, self.console)
            else:
                session = ClaudeSession(self.execution_dir, self.working_dir)

        step = WorkflowStep(
            name=name,
            status="pending",
            prompt=prompt,
            model=model,
            session=session,
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
        )
        self.steps.append(step)

        # Update sorted steps using our own topological sort
        self._sorted_steps = self._find_topological_order()

        return step

    def _find_topological_order(self) -> list[WorkflowStep]:
        """Find topological ordering of steps using Kahn's algorithm."""
        # Build in-degree map
        in_degree = {step: len(step.requires) for step in self.steps}

        # Find steps with no dependencies
        queue = [step for step in self.steps if in_degree[step] == 0]
        result = []

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

        # Reorder: non-pending steps first, then pending steps, maintaining relative order
        non_pending = [step for step in result if step.status != "pending"]
        pending = [step for step in result if step.status == "pending"]

        return non_pending + pending

    # running in a separate asyncio task
    async def _execute_step(self, step: WorkflowStep) -> None:
        if step.pre_hook is not None:
            step.pre_hook(step)

        self._pre_step_hook(step)

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

        self._post_step_hook(step)

    @abstractmethod
    def collect_result(self) -> AIResult:
        ...

    @require_initialized
    @conditionally_cleanup
    async def execute(
        self,
        *,
        max_parallel_steps: int | None = None,
    ) -> AIResult:
        running: dict[asyncio.Task, WorkflowStep] = {}

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
            while any(step.status in ["pending", "running"] for step in self._sorted_steps):
                # Start all ready steps that can run
                for step in self._sorted_steps:
                    if max_parallel_steps is not None and len(running) >= max_parallel_steps:
                        break

                    if (
                        step.status == "pending" and
                        all(r.status in ["completed", "skipped"] for r in step.requires)
                    ):
                        same_session_steps = [s for s in running.values() if s.session == step.session]
                        if same_session_steps:
                            raise RuntimeError(f"Step {step.name} shares the same session with steps: {', '.join(s.name for s in same_session_steps)}; and so cannot run in parallel")

                        skipped_requires = [r for r in step.requires if r.status == "skipped"]
                        if skipped_requires:
                            raise RuntimeError(f"Step {step.name} has skipped requires: {', '.join(r.name for r in skipped_requires)}; and so cannot run")

                        if step.condition is not None and not step.condition(step):
                            step.status = "skipped"
                            step.start_time = datetime.now()
                            step.end_time = step.start_time
                        else:
                            task = asyncio.create_task(self._execute_step(step))
                            step.status = "running"
                            step.start_time = datetime.now()
                            running[task] = step

                # Wait for at least one to complete
                if running:
                    done, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        step = running.pop(task)
                        step.status = "completed" if task.exception() is None else "failed"
                        step.end_time = datetime.now()
                        if task.exception():
                            raise task.exception()

        self.console.print(self._get_status_display())

        return self.collect_result()

    def _get_status_display(self) -> Panel:
        """Build the status display with table and progress."""
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
            width=80,
        )
        table.add_column("Step", no_wrap=True, width=40)
        table.add_column("Time", justify="right", width=15)
        table.add_column("Cost", justify="right", width=12)

        # Add rows for each step
        for step in self._sorted_steps:
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

            # Format cost
            cost = f"${step.cost:.4f}" if step.cost > 0 else "-"

            table.add_row(step_name, duration, cost, style=row_style)

        # Add total row
        total_cost = sum(step.cost for step in self.steps)

        table.add_section()
        table.add_row(
            "Total",
            f"{_format_duration((datetime.now() - self.start_time).total_seconds())}",
            f"${total_cost:.4f}",
            style="bold",
        )

        # Return panel with table and status
        return Panel(table, title="", border_style="blue", width=84)
