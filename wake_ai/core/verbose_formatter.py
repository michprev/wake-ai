import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, Literal
from contextlib import contextmanager

from rich.console import Console
from rich.rule import Rule

from ..utils.logging import should_verbose_log

COLORS = {
    "todo_header": "bold blue",
    "todo_complete": "bold green",
    "todo_progress": "yellow",
    "todo_pending": "dim white",
    "tool_use": "bright_magenta",
    "tool_input": "magenta",
    "tool_result": "bright_cyan",
    "tool_result_json": "cyan",
    "tool_error": "bold red",
    "system": "bright_blue",
    "user": "bold white",
    "agent": "white",
    "thinking": "dim white",
    "error": "bold red",
    "unknown": "dim red",
}
# TODO customizable style


class TodoItem(TypedDict):
    status: Literal["completed", "in_progress", "pending"]
    text: str


class VerboseFormatter:
    console: Console
    log_file: Path | None
    verbosity: int
    step_name: str
    log_with_console_formatting: bool

    def __init__(
        self,
        console: Console,
        step_name: str,
        log_file: Path | None,
        verbosity: int,
        log_with_console_formatting: bool = True,
        reset_log_file: bool = True,
    ):
        self.console = console
        self.log_file = log_file
        self.verbosity = verbosity
        self.step_name = step_name
        self.log_with_console_formatting = log_with_console_formatting

        if reset_log_file:
            self.reset_log_file()

    @contextmanager
    def _file_console(self):
        if self.log_file is None:
            yield None
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            fc = Console(
                file=f,
                force_terminal=self.log_with_console_formatting,
                width=self.console.width,
            )
            yield fc

    def _get_splitter(self) -> Rule:
        return Rule(title=f"[{datetime.now().strftime('%H:%M:%S')}] {self.step_name}", style="dim white")

    def reset_log_file(self) -> None:
        if self.log_file is not None and self.log_file.exists():
            i = 0
            while self.log_file.with_suffix(f".{i}.log").exists():
                i += 1
            self.log_file.rename(self.log_file.with_suffix(f".{i}.log"))

    def print_error(self, message: str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print("Error: " + message, style=COLORS["error"], markup=False)

        self.console.print(splitter)
        self.console.print("Error: " + message, style=COLORS["error"], markup=False)

    def print_user_message(self, message: str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print("User: " + message, style=COLORS["user"], markup=False)

        if self.verbosity == 0 or not should_verbose_log("user"):
            return

        self.console.print(splitter)
        self.console.print("User: " + message, style=COLORS["user"], markup=False)

    def print_agent_message(self, message: str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print("Agent: " + message, style=COLORS["agent"], markup=False)

        if self.verbosity == 0 or not should_verbose_log("agent"):
            return

        self.console.print(splitter)
        self.console.print("Agent: " + message, style=COLORS["agent"], markup=False)

    def print_thinking(self, message: str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print("Thinking: " + message, style=COLORS["thinking"], markup=False)

        if self.verbosity == 0 or not should_verbose_log("thinking"):
            return

        self.console.print(splitter)
        self.console.print("Thinking: " + message, style=COLORS["thinking"], markup=False)

    def print_system_message(self, message: str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print("System: " + message, style=COLORS["system"], markup=False)

        if self.verbosity == 0 or not should_verbose_log("system"):
            return

        self.console.print(splitter)
        self.console.print("System: " + message, style=COLORS["system"], markup=False)

    def print_tool_use(self, name: str, input: dict[str, Any] | str) -> None:
        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                fc.print(f"Using tool: {name}", style=COLORS["tool_use"], markup=False)
                if input:
                    fc.print(input, style=COLORS["tool_input"], markup=False)

        if self.verbosity == 0 or not should_verbose_log("tool"):
            return

        self.console.print(splitter)
        self.console.print(f"Using tool: {name}", style=COLORS["tool_use"], markup=False)
        if input:
            self.console.print(input, style=COLORS["tool_input"], markup=False)

    def print_tool_result(self, result: str | dict[str, Any] | list[dict[str, Any]], is_error: bool) -> None:
        def print_single(console: Console, result: str | dict[str, Any]) -> None:
            if isinstance(result, dict):
                if "type" in result and result["type"] == "text" and "text" in result:
                    try:
                        console.print_json(result["text"])
                    except json.JSONDecodeError:
                        console.print(result["text"], style=style, markup=False)
                else:
                    console.print(result, style=style, markup=False)
            else:
                console.print(result, style=style, markup=False)

        style = COLORS["tool_result"] if not is_error else COLORS["tool_error"]
        splitter = self._get_splitter()

        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                if isinstance(result, list):
                    for item in result:
                        print_single(fc, item)
                else:
                    print_single(fc, result)

        if (self.verbosity < 2 or not should_verbose_log("tool_result")) and not is_error:
            return

        self.console.print(splitter)
        if isinstance(result, list):
            for item in result:
                print_single(self.console, item)
        else:
            print_single(self.console, result)

    def print_todo(self, todos: list[TodoItem]) -> None:
        def print_todo_list(console: Console, todos: list[TodoItem]) -> None:
            console.print(
                f"📋 [{COLORS['todo_header']}]Todo list:[/{COLORS['todo_header']}]",
                highlight=False,
            )
            for todo in todos:
                # Select appropriate visual indicators for each status type
                if todo["status"] == "completed":
                    icon = "✅"
                    style = COLORS["todo_complete"]
                elif todo["status"] == "in_progress":
                    icon = "🔄"
                    style = COLORS["todo_progress"]
                else:  # pending
                    icon = "⌛"
                    style = COLORS["todo_pending"]

                console.print(
                    f"  {icon} {todo['text']}", highlight=False, style=style
                )

        splitter = self._get_splitter()
        with self._file_console() as fc:
            if fc is not None:
                fc.print(splitter)
                print_todo_list(fc, todos)

        if self.verbosity == 0 or not should_verbose_log("todo"):
            return

        self.console.print(splitter)
        print_todo_list(self.console, todos)
