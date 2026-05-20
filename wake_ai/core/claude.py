from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import AsyncIterator, NamedTuple, Literal, Callable, Awaitable, Any

from claude_agent_sdk import AgentDefinition, AssistantMessage, ClaudeAgentOptions, McpServerConfig, Message, ResultMessage, SandboxIgnoreViolations, SandboxNetworkConfig, SandboxSettings, SystemMessage, TaskNotificationMessage, TaskProgressMessage, TaskStartedMessage, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock, UserMessage, query, create_sdk_mcp_server, SdkMcpTool
from claude_agent_sdk.types import SystemPromptPreset

from .session_abc import SessionABC, FunctionTool
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
    "LSP",
    "Agent",
    "ListMcpResourcesTool",
    # Wake MCP
    "mcp__wake",
    "mcp__rust-analysis",
    "mcp___internal",
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


async def wrap_prompt(text):
    yield {"type": "user", "message": {"role": "user", "content": text}}


async def tool_wrapper(handler: Callable[..., Awaitable[Any]], args: dict[str, Any]) -> Any:
    return {
        "content": [
            {
                "type": "text",
                "text": str(await handler(**args))
            }
        ]
    }


class ClaudeSession(SessionABC):
    execution_dir: Path
    working_dir: Path
    _session_id: str | None
    allowed_tools: list[str]
    disallowed_tools: list[str]
    mcp_servers: dict[str, McpServerConfig]
    agents: dict[str, AgentDefinition]
    system_prompt: str | SystemPromptPreset | None
    fork_session: str | ClaudeSession | None
    tools: list[FunctionTool]
    env: dict[str, str]
    effort: Literal["low", "medium", "high", "max"] | None
    turn_step: int | None
    shell_network_access: bool
    ignore_skills: bool

    def __init__(
        self,
        execution_dir: Path,
        working_dir: Path,
        *,
        session_id: str | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        mcp_servers: dict[str, McpServerConfig] | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        system_prompt: str | SystemPromptPreset | None = None,
        fork_session: str | ClaudeSession | None = None,
        tools: list[FunctionTool] | None = None,
        env: dict[str, str] | None = None,
        effort: Literal["low", "medium", "high", "max"] | None = None,
        turn_step: int | None = TURN_STEP,
        shell_network_access: bool = False,
        ignore_skills: bool = True,
    ):
        if session_id is not None and fork_session is not None:
            raise ValueError("session_id and fork_session cannot be used together")

        self.execution_dir = execution_dir
        self.working_dir = working_dir
        self._session_id = session_id

        if allowed_tools is None:
            allowed_tools = list(DEFAULT_ALLOWED_TOOLS)
        self.allowed_tools = [t.format(working_dir=working_dir) for t in allowed_tools]

        if disallowed_tools is None:
            self.disallowed_tools = []
        else:
            self.disallowed_tools = disallowed_tools

        if mcp_servers is None:
            mcp_servers = {}
        self.mcp_servers = mcp_servers

        if agents is None:
            agents = {}
        self.agents = agents

        self.system_prompt = system_prompt
        self.fork_session = fork_session

        if tools is None:
            self.tools = []
        else:
            self.tools = list(tools)

        if env is None:
            self.env = {}
        else:
            self.env = env

        self.env["CLAUDECODE"] = ""

        self.effort = effort
        self.turn_step = turn_step
        self.shell_network_access = shell_network_access
        self.ignore_skills = ignore_skills

    @property
    def session_id(self) -> str | None:
        return self._session_id

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
                        formatter.print_todo([
                            {
                                "status": i["status"],
                                "text": (
                                    i.get("activeForm", i["content"])
                                    if i["status"] == "in_progress" else i["content"]
                                ),
                            }
                            for i in content.input.get("todos", [])
                        ])
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
                    if self._session_id is None:
                        self._session_id = message.data["session_id"]
                    else:
                        # sanity check
                        assert self._session_id == message.data["session_id"]

                status = (
                    f"System: {message.subtype}\n"
                    f"CWD: {message.data.get('cwd', 'N/A')}\n"
                    f"Session: {message.data.get('session_id', 'N/A')}"
                )
                formatter.print_system_message(status)
            elif isinstance(message, TaskStartedMessage):
                formatter.print_system_message(f"Task started: {message.description}")
            elif isinstance(message, TaskProgressMessage):
                formatter.print_system_message(f"Task progress: {message.description}")
            elif isinstance(message, TaskNotificationMessage):
                formatter.print_system_message(f"Task {message.status}: {message.summary}")
            else:
                logger.warning(f"Unexpected Claude system message subtype: {message.subtype}")
        elif isinstance(message, ResultMessage):
            pass
        else:
            logger.warning(f"Unexpected Claude message type: {type(message)}")

    async def query(self, prompt: str, model: str, max_cost: float | None, formatter: VerboseFormatter) -> AsyncIterator[ClaudeResponse]:
        if self.fork_session is not None:
            if isinstance(self.fork_session, str):
                fork_session_id = self.fork_session
            else:
                if self.fork_session._session_id is None:
                    raise RuntimeError("Forking from ClaudeSession without assigned session id")
                fork_session_id = self.fork_session._session_id
        else:
            fork_session_id = None

        if self.tools:
            internal_mcp = create_sdk_mcp_server(
                name="_internal",
                version="1.0.0",
                tools=[
                    SdkMcpTool(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.input_schema,
                        handler=partial(tool_wrapper, tool.handler),
                    )
                    for tool in self.tools
                ]
            )
            mcps = {**self.mcp_servers, "_internal": internal_mcp}
        else:
            mcps = self.mcp_servers

        options = ClaudeAgentOptions(
            add_dirs=[Path.home() / ".config/wake", Path.home() / ".cache/wake/explorers"],
            system_prompt=self.system_prompt,
            allowed_tools=self.allowed_tools,
            disallowed_tools=self.disallowed_tools,
            resume=self._session_id or fork_session_id,
            model=model,
            cwd=str(self.execution_dir),  # Set working directory for command execution
            permission_mode="default",
            max_turns=self.turn_step,
            mcp_servers=mcps,
            agents=self.agents,
            fork_session=fork_session_id is not None,
            stderr=formatter.print_error,
            env=self.env,
            max_buffer_size=10 * 1024 * 1024 * 1024,  # 10GB
            effort=self.effort,
            sandbox=SandboxSettings(
                enabled=True,
                autoAllowBashIfSandboxed=True,
                excludedCommands=[],
                allowUnsandboxedCommands=False,
                network=SandboxNetworkConfig(
                    allowAllUnixSockets=self.shell_network_access,
                    allowLocalBinding=self.shell_network_access,
                    allowedDomains=["*"],
                ),
                ignoreViolations=SandboxIgnoreViolations(
                    file=[self.working_dir.resolve().as_posix()],
                ),
                enableWeakerNestedSandbox=False,
            ),
            skills=[] if self.ignore_skills else None,
        )

        total_cost = 0.0
        result: ResultMessage | None = None

        if isinstance(self.system_prompt, str):
            formatter.print_system_message(f"System prompt:\n{self.system_prompt}")
        elif isinstance(self.system_prompt, dict) and isinstance(self.system_prompt.get("append", None), str):
            formatter.print_system_message(f"Appended system prompt:\n{self.system_prompt['append']}")

        formatter.print_user_message(prompt)

        # initial query
        async for message in query(prompt=wrap_prompt(prompt), options=options):
            self._process_message(message, formatter)
            # ResultMessage indicates the response is complete.
            if isinstance(message, ResultMessage):
                result = message

        assert result is not None
        total_cost += result.total_cost_usd or 0.0

        assert self._session_id is not None
        # from now on we must keep using the same session id
        options.resume = self._session_id
        options.fork_session = False

        while result.subtype == "error_max_turns" and (max_cost is None or total_cost < max_cost):
            yield ClaudeResponse(cost=total_cost, status="running")

            formatter.print_user_message("continue")

            result = None
            async for message in query(prompt=wrap_prompt("continue"), options=options):
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
            async for message in query(prompt=wrap_prompt(termination_prompt), options=options):
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

    def reset(self) -> None:
        """
        Reset the session ID
        """
        self._session_id = None
