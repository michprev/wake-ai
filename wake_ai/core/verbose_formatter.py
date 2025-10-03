import json
from pathlib import Path
from typing import Any, TypedDict, Literal

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
    "system": "purple",
    "user": "bold white",
    "agent": "white",
    "thinking": "dim white",
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
    splitter: Rule
    file_splitter: str

    def __init__(self, console: Console, step_name: str, log_file: Path | None, verbosity: int):
        self.console = console
        self.log_file = log_file
        self.verbosity = verbosity
        self.splitter = Rule(title=step_name, style="dim white")
        self.file_splitter = "─" * 100 + "\n"

    def print_user_message(self, message: str) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"User: {message}\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("user"):
            return

        self.console.print(self.splitter)
        self.console.print("User: " + message, style=COLORS["user"], markup=False)

    def print_agent_message(self, message: str) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"Agent: {message}\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("agent"):
            return

        self.console.print(self.splitter)
        self.console.print("Agent: " + message, style=COLORS["agent"], markup=False)

    def print_thinking(self, message: str) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"Thinking: {message}\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("thinking"):
            return

        self.console.print(self.splitter)
        self.console.print("Thinking: " + message, style=COLORS["thinking"], markup=False)

    def print_system_message(self, message: str) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"System: {message}\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("system"):
            return

        self.console.print(self.splitter)
        self.console.print("System: " + message, style=COLORS["system"], markup=False)

    def print_tool_use(self, name: str, input: dict[str, Any]) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"Using tool: {name}\n")
                if input:
                    f.write(json.dumps(input, indent=2) + "\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("tool"):
            return

        self.console.print(self.splitter)
        self.console.print(f"Using tool: {name}", style=COLORS["tool_use"], markup=False)
        if input:
            self.console.print(input, style=COLORS["tool_input"], markup=False)

    def print_tool_result(self, result: str | dict[str, Any] | list[dict[str, Any]], is_error: bool) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(f"Tool result:\n")
                if is_error:
                    f.write("Error: True\n")
                else:
                    f.write("Error: False\n")
                if isinstance(result, str):
                    f.write(result + "\n")
                else:
                    f.write(json.dumps(result, indent=2) + "\n")
                f.write(self.file_splitter)

        if self.verbosity < 2 or not should_verbose_log("tool_result"):
            return

        style = COLORS["tool_result"] if not is_error else COLORS["tool_error"]

        def print_single(result: str | dict[str, Any]) -> None:
            if isinstance(result, str):
                self.console.print(result, style=style, markup=False)
            else:
                if "type" in result and result["type"] == "text" and "text" in result:
                    try:
                        self.console.print_json(result["text"])
                    except json.JSONDecodeError:
                        self.console.print(result["text"], style=style, markup=False)
                else:
                    self.console.print(result, style=style, markup=False)

        self.console.print(self.splitter)
        if isinstance(result, list):
            for item in result:
                print_single(item)
        else:
            print_single(result)

    def print_todo(self, todos: list[TodoItem]) -> None:
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write("Todo: list\n")
                for todo in todos:
                    f.write(f"  {todo['status']}: {todo['text']}\n")
                f.write(self.file_splitter)

        if self.verbosity == 0 or not should_verbose_log("todo"):
            return

        self.console.print(self.splitter)
        self.console.print(
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

            self.console.print(
                f"  {icon} {todo['text']}", highlight=False, style=style
            )
