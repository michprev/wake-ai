"""In-process MCP server that exposes Python `FunctionTool`s to the codex CLI.

`ClaudeSession` exposes Python tools in-process via the Agent SDK's
`create_sdk_mcp_server`. codex is an external CLI, so the equivalent is a small
streamable-HTTP MCP server hosted in this process (on an ephemeral localhost
port) whose tool handlers are the same in-process coroutines. `CodexSession`
starts one per query and wires codex to it with a `-c mcp_servers.<name>.url`
override.

Tool handlers run inside a copy of the context captured where the query was
issued, so context-vars (e.g. the flow engine's current-host, which enables
nested subagents) propagate into handlers just as they do for the in-process
sessions.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
from typing import Any

import uvicorn
import mcp.types as mcp_types
from mcp.server.fastmcp import FastMCP

from .session_abc import FunctionTool


class InProcessToolsServer:
    def __init__(self, tools: list[FunctionTool], *, name: str = "local_tools"):
        self.name = name
        self._tools: dict[str, FunctionTool] = {t.name: t for t in tools}
        self._context: contextvars.Context | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._task: asyncio.Task[Any] | None = None
        self.url: str | None = None

    def set_context(self, context: contextvars.Context | None) -> None:
        """Context to run tool handlers in (captured where the query is issued)."""
        self._context = context

    async def _dispatch(self, name: str, arguments: dict[str, Any] | None) -> Any:
        tool = self._tools[name]
        coro = tool.handler(**(arguments or {}))
        try:
            if self._context is not None:
                # ensure_future runs synchronously inside ctx.run, so the created
                # Task copies `ctx` (each call gets its own copy — safe for
                # concurrent tool calls) and the handler sees the captured vars.
                task = self._context.run(asyncio.ensure_future, coro)
            else:
                task = asyncio.ensure_future(coro)
            result = await task
        except Exception as e:  # surfaced to the model as a tool error
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=str(e))],
                isError=True,
            )
        return [mcp_types.TextContent(type="text", text=str(result))]

    def _build_app(self):
        # Reuse FastMCP's transport scaffolding (session manager, routing, lifespan)
        # but register our own explicit-schema handlers on its low-level server.
        fmcp = FastMCP(self.name, stateless_http=True, json_response=True)
        server = fmcp._mcp_server
        tools = self._tools
        dispatch = self._dispatch

        @server.list_tools()
        async def _list_tools() -> list[mcp_types.Tool]:
            return [
                mcp_types.Tool(
                    name=t.name,
                    description=t.description or "",
                    inputSchema=t.input_schema,
                )
                for t in tools.values()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
            return await dispatch(name, arguments)

        return fmcp.streamable_http_app()

    async def start(self) -> str:
        config = uvicorn.Config(
            self._build_app(), host="127.0.0.1", port=0, log_level="warning"
        )
        self._uvicorn = uvicorn.Server(config)
        self._task = asyncio.ensure_future(self._uvicorn.serve())
        while not self._uvicorn.started:
            await asyncio.sleep(0.02)
        port = self._uvicorn.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self.url

    async def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await self._task
        self._uvicorn = None
        self._task = None
        self.url = None
