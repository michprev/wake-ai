from pathlib import Path
from typing import AsyncIterator, NamedTuple, Literal

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, Message, ResultMessage, SystemMessage, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage, query
from claude_agent_sdk.types import SystemPromptPreset

from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger


logger = get_logger(__name__)

TURN_STEP = 25
MAX_TERMINATION_ATTEMPTS = 3

TERMINATION_PROMPT = (
    "You are approaching the cost limit. Please finish the task as quickly "
    "as possible. This is attempt {finish_tries}/{max_finish_tries}. "
    "After reaching {max_finish_tries} attempts, the task will be terminated."
)

DEFAULT_ALLOWED_TOOLS = (
    # Read-only tools (always safe)
    "Read",
    "Grep",
    "Glob",
    "LS",
    "Task",
    "TodoWrite",
    # Wake MCP
    "mcp__wake",
    # Write tools (needed for results - cannot be path-restricted)
    "Write(/{working_dir}/**)",
    "Edit(/{working_dir}/**)",
    "MultiEdit(/{working_dir}/**)",
    # Essential bash commands for codebase analysis
    "Bash(wake:*)",  # Wake framework commands
    "Bash(cd:*)",  # Directory navigation
    "Bash(pwd)",  # Print working directory
    "Bash(ls:*)",  # List files (though LS tool is preferred)
    "Bash(find:*)",  # Find files by pattern
    "Bash(tree:*)",  # Directory structure visualization
    "Bash(diff:*)",  # Compare files
    "Bash(mkdir:*)",  # Create directories
    "Bash(mv:*)",  # Move/rename files
    "Bash(cp:*)",  # Copy files
)


class ClaudeResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]


class ClaudeSession:
    execution_dir: Path
    working_dir: Path
    session_id: str | None
    allowed_tools: list[str]
    disallowed_tools: list[str]

    def __init__(
        self,
        execution_dir: Path,
        working_dir: Path,
        *,
        session_id: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
    ):
        self.execution_dir = execution_dir
        self.working_dir = working_dir
        self.session_id = session_id

        if allowed_tools is None:
            allowed_tools = list(DEFAULT_ALLOWED_TOOLS)
        self.allowed_tools = [t.format(working_dir=working_dir) for t in allowed_tools]

        if disallowed_tools is None:
            self.disallowed_tools = []
        else:
            self.disallowed_tools = disallowed_tools

    def _process_message(self, message: Message, formatter: VerboseFormatter) -> None:
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                formatter.print_user_message(message.content)
            else:
                for content in message.content:
                    if isinstance(content, TextBlock):
                        formatter.print_user_message(content.text)
                    elif isinstance(content, ToolResultBlock):
                        if content.content is None:
                            logger.warning(f"Unexpected Claude tool result content: {content}")
                            continue
                        formatter.print_tool_result(content.content, content.is_error or False)
                    else:
                        logger.warning(f"Unexpected Claude user content type: {type(content)}")
        elif isinstance(message, AssistantMessage):
            for content in message.content:
                if isinstance(content, TextBlock):
                    formatter.print_agent_message(content.text)
                elif isinstance(content, ThinkingBlock):
                    formatter.print_thinking(content.thinking)
                elif isinstance(content, ToolUseBlock):
                    if content.name == "TodoWrite":
                        formatter.print_todo(content.input.get("todos", []))
                    else:
                        formatter.print_tool_use(content.name, content.input)
                elif isinstance(content, ToolResultBlock):
                    if content.content is None:
                        logger.warning(f"Unexpected Claude tool result content: {content}")
                        continue
                    formatter.print_tool_result(content.content, content.is_error or False)
                else:
                    logger.warning(f"Unexpected Claude assistant content type: {type(content)}")
        elif isinstance(message, SystemMessage):
            if message.subtype == "init":
                if "session_id" in message.data:
                    if self.session_id is None:
                        self.session_id = message.data["session_id"]
                    else:
                        # sanity check
                        assert self.session_id == message.data["session_id"]

                status = (
                    f"System: {message.subtype}\n"
                    f"CWD: {message.data.get('cwd', 'N/A')}\n"
                    f"Session: {message.data.get('session_id', 'N/A')}"
                )
                formatter.print_system_message(status)
            else:
                breakpoint()
            pass
        elif isinstance(message, ResultMessage):
            pass
        else:
            logger.warning(f"Unexpected Claude message type: {type(message)}")

    async def query(self, prompt: str, model: str, max_cost: float | None, formatter: VerboseFormatter) -> AsyncIterator[ClaudeResponse]:
        options = ClaudeAgentOptions(
            system_prompt=SystemPromptPreset(type="preset", preset="claude_code"),
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            resume=self.session_id,
            model=model,
            cwd=str(self.execution_dir),  # Set working directory for command execution
            permission_mode="default",
            max_turns=TURN_STEP,
        )

        total_cost = 0.0
        result: ResultMessage | None = None

        formatter.print_user_message(prompt)

        # initial query
        async for message in query(prompt=prompt, options=options):
            self._process_message(message, formatter)
            # ResultMessage indicates the response is complete.
            if isinstance(message, ResultMessage):
                result = message

        assert result is not None
        total_cost += result.total_cost_usd or 0.0

        assert self.session_id is not None
        if options.resume is None:
            # from now on we must keep using the same session id
            options.resume = self.session_id

        while result.subtype == "error_max_turns" and (max_cost is None or total_cost < max_cost):
            yield ClaudeResponse(cost=total_cost, status="running")

            formatter.print_user_message("continue")

            result = None
            async for message in query(prompt="continue", options=options):
                self._process_message(message, formatter)
                # ResultMessage indicates the response is complete.
                if isinstance(message, ResultMessage):
                    result = message

            assert result is not None
            total_cost += result.total_cost_usd or 0.0

        termination_attempt = 0
        while result.subtype == "error_max_turns" and termination_attempt < MAX_TERMINATION_ATTEMPTS:
            termination_attempt += 1
            yield ClaudeResponse(cost=total_cost, status="terminating_on_max_cost")

            termination_prompt = TERMINATION_PROMPT.format(finish_tries=termination_attempt, max_finish_tries=MAX_TERMINATION_ATTEMPTS)

            formatter.print_user_message(termination_prompt)

            result = None
            async for message in query(prompt=termination_prompt, options=options):
                self._process_message(message, formatter)
                # ResultMessage indicates the response is complete.
                if isinstance(message, ResultMessage):
                    result = message

            assert result is not None
            total_cost += result.total_cost_usd or 0.0

        if result.subtype == "success":
            yield ClaudeResponse(cost=total_cost, status="succeeded")
        elif result.subtype == "error_max_turns":
            yield ClaudeResponse(cost=total_cost, status="terminated")
        else:
            yield ClaudeResponse(cost=total_cost, status="errored")
            raise RuntimeError(f"Claude Code returned an unexpected subtype: {result.subtype}")
