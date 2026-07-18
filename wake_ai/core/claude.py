from __future__ import annotations

import os
import tempfile
from functools import partial
from pathlib import Path
from typing import AsyncIterator, NamedTuple, Literal, Callable, Awaitable, Any

from claude_agent_sdk import AgentDefinition, AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, HookContext, HookInput, HookJSONOutput, HookMatcher, McpServerConfig, Message, ResultMessage, SandboxIgnoreViolations, SandboxNetworkConfig, SandboxSettings, SystemMessage, TaskNotificationMessage, TaskProgressMessage, TaskStartedMessage, TextBlock, ThinkingBlock, ThinkingConfigAdaptive, ThinkingConfigDisabled, ThinkingConfigEnabled, ToolResultBlock, ToolUseBlock, UserMessage, create_sdk_mcp_server, SdkMcpTool
from claude_agent_sdk.types import HookEvent, SystemPromptPreset

from .session_abc import SessionABC, FunctionTool
from .verbose_formatter import VerboseFormatter
from ..utils.logging import get_logger


logger = get_logger(__name__)

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
    "mcp___internal",  # internal MCP server for native tools
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


# When the bash sandbox runs with network access, the Claude Code CLI launches an
# internal HTTP(S)/SOCKS proxy and injects proxy env vars (HTTP_PROXY/HTTPS_PROXY/
# ALL_PROXY/GIT_SSH_COMMAND/...) into every sandboxed command. That proxy enforces
# sandbox.network.allowedDomains and refuses any non-allowlisted host; there is no
# "allow all domains" value (a bare "*" matches nothing), so the only way to reach
# arbitrary hosts is to bypass the proxy per command. This PreToolUse hook does that
# centrally, so agents never need to prepend `no_proxy="*"` themselves.
#
# `export ...; <cmd>` (not an inline `A=b <cmd>` prefix) is used so the bypass spans
# the WHOLE command line, including compound commands (`a && b`, pipes, subshells);
# the inline prefix only covers the first simple command. Reaching the network still
# requires the OS sandbox to permit the direct socket (shell_network_access=True sets
# allowLocalBinding); this hook alone does not open the network.
SANDBOX_PROXY_BYPASS_PREFIX = (
    "export no_proxy='*' NO_PROXY='*' "
    "http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' "
    "all_proxy='' ALL_PROXY='' GIT_SSH_COMMAND=''; "
)


async def sandbox_proxy_bypass_hook(input: HookInput, tool_use_id: str | None, context: HookContext) -> HookJSONOutput:
    """PreToolUse hook: transparently prepend the proxy-bypass env to Bash commands.

    Returns ``permissionDecision: "allow"`` alongside ``updatedInput``. This is
    required, not a broadening: with ``autoAllowBashIfSandboxed`` the CLI already
    auto-runs any sandboxed command the model writes, but that free pass does NOT
    survive a hook rewrite -- the rewritten (now multi-operation) command would fall
    back to the explicit allowlist and be denied. The explicit ``allow`` re-grants
    exactly what auto-allow was already granting; it does NOT override ``deny`` rules
    (those still win) and the OS sandbox remains the real boundary.
    """
    if input.get("tool_name") != "Bash":
        return {}
    tool_input = dict(input.get("tool_input") or {})
    command = tool_input.get("command", "")
    if not command or "NO_PROXY='*'" in command:  # idempotent: never rewrite twice
        return {}
    tool_input["command"] = SANDBOX_PROXY_BYPASS_PREFIX + command
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": tool_input,
        }
    }


class ClaudeResponse(NamedTuple):
    cost: float
    status: Literal["running", "terminating_on_max_cost", "succeeded", "terminated", "errored"]
    final_message: str | None = None
    context_tokens: int | None = None
    compaction_count: int = 0


def _accumulate_model_usage(acc: dict[str, dict[str, float]], model_usage: dict[str, Any] | None) -> None:
    """Sum per-model numeric token counters from a turn's ResultMessage into ``acc``."""
    if not model_usage:
        return
    for model, usage in model_usage.items():
        if not isinstance(usage, dict):
            continue
        model_acc = acc.setdefault(model, {})
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            model_acc[key] = model_acc.get(key, 0.0) + value


def _format_token_usage(acc: dict[str, dict[str, float]]) -> str:
    lines = ["Token usage:"]
    for model, usage in acc.items():
        parts = ", ".join(
            f"{k}={int(v) if float(v).is_integer() else round(v, 6)}"
            for k, v in usage.items()
        )
        lines.append(f"  {model}: {parts}")
    return "\n".join(lines)


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
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    turn_step: int | None
    sandbox: bool
    shell_network_access: bool
    extra_shell_writable_roots: list[Path | str]
    weaker_nested_sandbox: bool
    bypass_sandbox_proxy: bool
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
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        thinking: ThinkingConfigAdaptive | ThinkingConfigEnabled | ThinkingConfigDisabled | None = None,
        turn_step: int | None = None,
        sandbox: bool = True,
        shell_network_access: bool = False,
        extra_shell_writable_roots: list[Path | str] | None = None,
        weaker_nested_sandbox: bool = False,
        bypass_sandbox_proxy: bool = False,
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
        # Native thinking config -> forwarded to the CLI as --thinking / --thinking-display.
        # Prefer this over injecting {"thinking": ...} via CLAUDE_CODE_EXTRA_BODY: the env-body
        # override forces thinking onto EVERY request (incl. the WebSearch server tool's internal
        # forced-tool_choice call, which then 400s), whereas this is applied per-call by the CLI
        # and coexists with web search.
        self.thinking = thinking
        self.turn_step = turn_step
        self.sandbox = sandbox
        self.shell_network_access = shell_network_access
        self.extra_shell_writable_roots = extra_shell_writable_roots or []
        self.weaker_nested_sandbox = weaker_nested_sandbox
        self.bypass_sandbox_proxy = bypass_sandbox_proxy
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
            elif message.subtype == "thinking_tokens":
                pass  # not interesting
            elif isinstance(message, TaskStartedMessage):
                formatter.print_system_message(f"Task started: {message.description}")
            elif isinstance(message, TaskProgressMessage):
                formatter.print_system_message(f"Task progress: {message.description}")
            elif isinstance(message, TaskNotificationMessage):
                formatter.print_system_message(f"Task {message.status}: {message.summary}")
            else:
                logger.warning(f"Unexpected Claude system message subtype: {message.subtype}; {message.data}")
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

        compact_count = 0

        async def pre_compact_hook(input: HookInput, tool_use_id: str | None, context: HookContext) -> HookJSONOutput:
            nonlocal compact_count
            compact_count += 1
            return {}

        network_config = SandboxNetworkConfig(
            allowAllUnixSockets=self.shell_network_access,
            allowLocalBinding=self.shell_network_access,
            deniedDomains=[],
        )
        if self.shell_network_access:
            # macOS specifics for granting outbound network access:
            # - allowLocalBinding (above) lets the seatbelt sandbox permit direct
            #   outbound TCP; without it direct connections fail with EPERM.
            # - allowMachLookup for trustd lets Go/system-TLS clients (e.g. gh) reach
            #   the macOS trust daemon to verify certificates; without it they fail
            #   with `x509: OSStatus -26276`.
            # The CLI still injects HTTP(S)/SOCKS proxy env vars (which we cannot
            # override via `env`) that 403 every non-allowlisted host. There is no
            # "allow all domains" value, so to reach arbitrary hosts each command must
            # bypass the proxy with `no_proxy='*' NO_PROXY='*'`. Set
            # bypass_sandbox_proxy=True to have the sandbox_proxy_bypass_hook do that
            # automatically instead of instructing agents to prepend it themselves.
            network_config["allowMachLookup"] = ["com.apple.trustd*"]
        else:
            # Any non-None allowedDomains (even []) activates the proxy allowlist; an
            # empty list matches nothing, so all outbound traffic is blocked. Combined
            # with allowLocalBinding=False this leaves no path to the network.
            network_config["allowedDomains"] = []

        hooks: dict[HookEvent, list[HookMatcher]] = {"PreCompact": [HookMatcher(hooks=[pre_compact_hook])]}
        if self.bypass_sandbox_proxy:
            hooks["PreToolUse"] = [HookMatcher(matcher="Bash", hooks=[sandbox_proxy_bypass_hook])]

        sandbox_settings: SandboxSettings | None = None
        if self.sandbox:
            extra_writable = [Path(root).resolve().as_posix() for root in self.extra_shell_writable_roots]
            sandbox_settings = SandboxSettings(
                enabled=True,
                autoAllowBashIfSandboxed=True,
                excludedCommands=[],
                allowUnsandboxedCommands=False,
                network=network_config,
                ignoreViolations=SandboxIgnoreViolations(
                    file=[
                        self.working_dir.resolve().as_posix(),
                        Path(os.environ.get("TMPDIR") or tempfile.gettempdir()).resolve().as_posix(),
                        "/private/tmp",
                    ] + extra_writable,
                ),
                enableWeakerNestedSandbox=self.weaker_nested_sandbox,
            )
            # Default sandbox writes are limited to cwd + $TMPDIR. extra_shell_writable_roots
            # must be granted via filesystem.allowWrite -- listing them only under
            # ignoreViolations suppresses the violation report but does NOT permit the write.
            # `filesystem` isn't in the SDK's SandboxSettings TypedDict yet, but the CLI honors
            # it (verified), so set it as a passthrough key.
            if extra_writable:
                sandbox_settings["filesystem"] = {"allowWrite": extra_writable}  # type: ignore[typeddict-unknown-key]

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
            thinking=self.thinking,
            skills=[] if self.ignore_skills else None,
            hooks=hooks,
            sandbox=sandbox_settings,
        )

        total_cost = 0.0
        result: ResultMessage | None = None
        accumulated_usage: dict[str, dict[str, float]] = {}
        context_size: int | None = None
        refreshed_compact_count = -1  # forces a refresh on the very first message

        if isinstance(self.system_prompt, str):
            formatter.print_system_message(f"System prompt:\n{self.system_prompt}")
        elif isinstance(self.system_prompt, dict) and isinstance(self.system_prompt.get("append", None), str):
            formatter.print_system_message(f"Appended system prompt:\n{self.system_prompt['append']}")

        formatter.print_user_message(prompt)

        # A single client is kept alive for the whole query: the initial turn plus
        # any "continue"/termination turns share one streaming session. This also
        # lets us read accurate, live context usage via get_context_usage() after
        # each message (it's a local IPC round-trip to the CLI, not an API call).
        client = ClaudeSDKClient(options=options)
        await client.connect()
        try:
            async def context_tokens() -> int | None:
                """Accurate context window size (matches the CLI `/context` command)."""
                try:
                    return (await client.get_context_usage())["totalTokens"]
                except Exception as e:
                    logger.warning(f"Failed to read Claude context usage: {e}")
                    return None

            async def run_turn(turn_prompt: str, status: Literal["running", "terminating_on_max_cost"]) -> AsyncIterator[ClaudeResponse]:
                """Run one turn, yielding a live progress update after each message."""
                nonlocal result, total_cost, context_size, refreshed_compact_count
                await client.query(turn_prompt)
                async for message in client.receive_response():
                    self._process_message(message, formatter)
                    # ResultMessage indicates the turn is complete; fold in its cost.
                    if isinstance(message, ResultMessage):
                        result = message
                        total_cost += message.total_cost_usd or 0.0
                        _accumulate_model_usage(accumulated_usage, message.model_usage)
                    # Progress only changes when an API call completes (top-level
                    # assistant output, turn result) or after a compaction; other
                    # messages would repeat the previous values, so skip both the
                    # context-usage IPC round-trip and the yield.
                    if (
                        isinstance(message, ResultMessage)
                        or (isinstance(message, AssistantMessage) and message.parent_tool_use_id is None)
                        or compact_count != refreshed_compact_count
                    ):
                        context_size = await context_tokens()
                        refreshed_compact_count = compact_count
                        yield ClaudeResponse(cost=total_cost, status=status, context_tokens=context_size, compaction_count=compact_count)

            # initial query
            async for info in run_turn(prompt, "running"):
                yield info
            assert result is not None
            assert self._session_id is not None

            while result.subtype == "error_max_turns" and (max_cost is None or total_cost < max_cost):
                formatter.print_user_message("continue")
                async for info in run_turn("continue", "running"):
                    yield info

            termination_attempt = 0
            while result.subtype == "error_max_turns" and termination_attempt < MAX_TERMINATION_ATTEMPTS:
                termination_attempt += 1
                termination_prompt = TERMINATION_PROMPT.format(finish_tries=termination_attempt, max_finish_tries=MAX_TERMINATION_ATTEMPTS)

                formatter.print_user_message(termination_prompt)
                async for info in run_turn(termination_prompt, "terminating_on_max_cost"):
                    yield info
        finally:
            await client.disconnect()

        formatter.print_system_message(_format_token_usage(accumulated_usage))

        if result.subtype == "success":
            yield ClaudeResponse(cost=total_cost, status="succeeded", final_message=result.result, context_tokens=context_size, compaction_count=compact_count)
        elif result.subtype == "error_max_turns":
            yield ClaudeResponse(cost=total_cost, status="terminated", final_message=result.result, context_tokens=context_size, compaction_count=compact_count)
        else:
            yield ClaudeResponse(cost=total_cost, status="errored", final_message=result.result, context_tokens=context_size, compaction_count=compact_count)
            raise RuntimeError(f"Claude Code returned an unexpected subtype: {result.subtype}")

    def reset(self) -> None:
        """
        Reset the session ID
        """
        self._session_id = None
