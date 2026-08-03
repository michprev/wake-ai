import pytest

from wake_ai.core.api_session_utils import (
    SHELL_INPUT_SCHEMA,
    compute_backoff_time,
    mcp_tool_alias,
    normalize_mcp_schema,
    resolve_mcp_alias,
    slugify,
)


def test_slugify_replaces_invalid_chars():
    assert slugify("my.tool name!") == "my_tool_name_"
    assert slugify("Already-ok_123") == "Already-ok_123"


def test_mcp_tool_alias_short_names():
    assert mcp_tool_alias("srv", "tool") == "srv__tool"


def test_mcp_tool_alias_long_names_are_stable_and_within_limit():
    long_tool = "t" * 100
    a1 = mcp_tool_alias("server", long_tool)
    a2 = mcp_tool_alias("server", long_tool)
    assert a1 == a2
    assert len(a1) <= 64


def test_resolve_mcp_alias_collision_falls_back_to_hash():
    reserved = {"srv__tool"}
    alias = resolve_mcp_alias("srv", "tool", reserved, {})
    assert alias is not None
    assert alias != "srv__tool"
    assert len(alias) <= 64


def test_normalize_mcp_schema_adds_properties():
    assert normalize_mcp_schema({"type": "object"}) == {"type": "object", "properties": {}}
    assert normalize_mcp_schema(None) == {"type": "object", "properties": {}}
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert normalize_mcp_schema(schema) == schema


def test_compute_backoff_time_grows_exponentially():
    # base 20s with ±10% jitter, doubling per retry
    assert 18.0 <= compute_backoff_time(0) <= 22.0
    assert 36.0 <= compute_backoff_time(1) <= 44.0


def test_shell_input_schema_shape():
    assert SHELL_INPUT_SCHEMA["type"] == "object"
    assert "commands" in SHELL_INPUT_SCHEMA["properties"]
    assert SHELL_INPUT_SCHEMA["required"] == ["commands", "timeout_ms"]
