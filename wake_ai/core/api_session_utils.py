"""Backend-agnostic plumbing shared by direct-API sessions (OpenAI, OpenRouter)."""

from __future__ import annotations

import hashlib
import platform
import random
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from .landlock import run_under_landlock
from .seatbelt import run_under_seatbelt
from ..utils.logging import get_logger

logger = get_logger(__name__)

MAX_TOOL_NAME_LEN = 64
_ALIAS_HASH_LEN = 8  # hex chars appended when the slug must be shortened

SHELL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "description": "A list of shell commands to execute. Execution in order is NOT guaranteed.",
            "items": {
                "type": "string",
            },
        },
        "timeout_ms": {
            "type": "number",
            "description": "The timeout in milliseconds for each command.",
        },
        "max_output_length": {
            "type": "integer",
            "description": "The maximum output length in characters for each command.",
        },
    },
    "required": ["commands", "timeout_ms"],
}


@dataclass
class SSEServerParameters:
    url: str
    headers: dict[str, str] | None = None
    timeout: float = 5
    sse_read_timeout: float = 60 * 5


@dataclass
class StreamableHTTPServerParameters:
    url: str
    headers: dict[str, str] | None = None
    timeout: float = 5
    sse_read_timeout: float = 60 * 5
    terminate_on_close: bool = False


def compute_backoff_time(retry: int) -> float:
    # base time is 20 seconds, exponential growth; 10% jitter
    # retry starts at 0
    exp = 2.0 ** retry
    base = 20 * exp
    return base * random.uniform(0.9, 1.1)


def normalize_mcp_schema(schema: Any) -> dict[str, Any]:
    """Ensure MCP inputSchema includes 'properties' — required by OpenAI for object schemas."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return {"type": "object", "properties": {}}
    if "properties" not in schema:
        return {**schema, "properties": {}}
    return schema


def slugify(s: str) -> str:
    """Replace characters outside [A-Za-z0-9_-] with underscore."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


def mcp_tool_alias(server_name: str, tool_name: str) -> str:
    """Return a stable OpenAI-safe function name for an MCP tool.

    Format: ``<server>__<tool>`` (both parts slugified). If the result exceeds
    64 characters, the slug is truncated and an 8-hex-char SHA-256 suffix of
    the original names is appended so the alias stays both within the limit
    and stable across restarts.
    """
    slug = f"{slugify(server_name)}__{slugify(tool_name)}"
    if len(slug) <= MAX_TOOL_NAME_LEN:
        return slug
    h = hashlib.sha256(f"{server_name}\x00{tool_name}".encode()).hexdigest()[:_ALIAS_HASH_LEN]
    return f"{slug[:MAX_TOOL_NAME_LEN - _ALIAS_HASH_LEN - 1]}_{h}"


def resolve_mcp_alias(
    server_name: str,
    tool_name: str,
    reserved: set[str],
    registered: dict[str, Any],
) -> str | None:
    """Return a unique OpenAI-safe alias for an MCP tool, or ``None`` if impossible.

    Tries the primary slug first; if it collides, a hash-disambiguated form is
    used. Returns ``None`` (with error log) only if both forms are already taken.
    """
    alias = mcp_tool_alias(server_name, tool_name)
    if alias not in reserved and alias not in registered:
        return alias

    h = hashlib.sha256(f"{server_name}\x00{tool_name}".encode()).hexdigest()[:_ALIAS_HASH_LEN]
    disambiguated = f"{alias[:MAX_TOOL_NAME_LEN - _ALIAS_HASH_LEN - 1]}_{h}"
    logger.warning(
        "MCP tool '%s' from '%s' has conflicting alias '%s'; "
        "registering as '%s' instead.",
        tool_name, server_name, alias, disambiguated,
    )
    if disambiguated in reserved or disambiguated in registered:
        logger.error(
            "Cannot register MCP tool '%s' from '%s': "
            "disambiguated alias '%s' also taken. Skipping.",
            tool_name, server_name, disambiguated,
        )
        return None
    return disambiguated


async def run_sandboxed(
    command: str,
    network_access: bool,
    writable_roots: list[Path],
    timeout: float | None,
    cwd: Path,
) -> tuple[str, str, int]:
    """Run one shell command under the OS sandbox (seatbelt/landlock)."""
    system = platform.system()
    if system == "Darwin":
        return await run_under_seatbelt(command, network_access, writable_roots, timeout, cwd)
    if system == "Linux":
        return await run_under_landlock(command, network_access, writable_roots, timeout, cwd)
    raise NotImplementedError("Shell tools are only supported on macOS and Linux")


@asynccontextmanager
async def open_mcp_clients(
    mcps: dict[str, ClientSession | StdioServerParameters | SSEServerParameters | StreamableHTTPServerParameters],
    read_timeout: float | None,
) -> AsyncIterator[dict[str, ClientSession]]:
    """Open/initialize MCP clients; tear them down in reverse order on exit.

    Moved from OpenAISession.query(); already-connected ClientSession values
    are passed through untouched.
    """
    mcp_clients: dict[str, ClientSession] = {}
    opened_clients: list[Any] = []
    try:
        for server_name, info in mcps.items():
            if isinstance(info, ClientSession):
                mcp_clients[server_name] = info
            else:
                if isinstance(info, StdioServerParameters):
                    handle = stdio_client(info)
                    read, write = await handle.__aenter__()
                elif isinstance(info, SSEServerParameters):
                    handle = sse_client(info.url, info.headers, info.timeout, info.sse_read_timeout)
                    read, write = await handle.__aenter__()
                elif isinstance(info, StreamableHTTPServerParameters):
                    handle = streamablehttp_client(info.url, info.headers, 60, 60, False)
                    read, write, _ = await handle.__aenter__()
                else:
                    raise ValueError(f"Unknown MCP server type: {type(info)}")

                opened_clients.append(handle)

                client = ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=read_timeout) if read_timeout is not None else None,
                )
                opened_clients.append(client)

                session = await client.__aenter__()
                await session.initialize()
                mcp_clients[server_name] = session
        yield mcp_clients
    finally:
        for client in reversed(opened_clients):
            await client.__aexit__(None, None, None)
