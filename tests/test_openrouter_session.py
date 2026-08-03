from pathlib import Path
from types import SimpleNamespace

import pytest

from wake_ai.core.openrouter import OpenRouterSession


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def make_session(tmp_path: Path, **kwargs) -> OpenRouterSession:
    return OpenRouterSession(tmp_path, **kwargs)


def test_reasoning_params_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        make_session(tmp_path, reasoning_effort="high", reasoning_max_tokens=2000)


def test_system_message_prepended_not_stored(tmp_path):
    session = make_session(tmp_path, instructions="Audit contracts.")
    session.conversation.append({"role": "user", "content": "hi"})
    messages = session._request_messages()
    assert messages[0] == {"role": "system", "content": "Audit contracts."}
    assert messages[1] == {"role": "user", "content": "hi"}
    assert all(m["role"] != "system" for m in session.conversation)


def test_extra_body_reasoning_provider_and_escape_hatch(tmp_path):
    session = make_session(
        tmp_path,
        reasoning_effort="high",
        provider={"order": ["deepseek"], "allow_fallbacks": False},
        extra_body={"transforms": []},
    )
    body = session._build_extra_body()
    assert body["reasoning"] == {"effort": "high"}
    assert body["provider"] == {"order": ["deepseek"], "allow_fallbacks": False}
    assert body["transforms"] == []
    assert body["usage"] == {"include": True}


def test_extra_body_reasoning_max_tokens(tmp_path):
    session = make_session(tmp_path, reasoning_max_tokens=2000)
    assert session._build_extra_body()["reasoning"] == {"max_tokens": 2000}


def test_extra_body_usage_accounting_default(tmp_path):
    session = make_session(tmp_path)
    assert session._build_extra_body()["usage"] == {"include": True}


def test_extra_body_usage_accounting_overridable(tmp_path):
    session = make_session(tmp_path, extra_body={"usage": {"include": False}})
    assert session._build_extra_body()["usage"] == {"include": False}


async def test_collect_tools_shell_web_search_and_function(tmp_path):
    async def handler(x: int) -> int:
        return x

    from wake_ai.core.session_abc import FunctionTool
    session = make_session(
        tmp_path,
        tools=[FunctionTool(name="mytool", input_schema={"type": "object", "properties": {}},
                            description="my tool", handler=handler)],
        shell=True,
        web_search=True,
        web_search_engine="exa",
        web_search_max_results=3,
    )
    tools, mcp_tools = await session._collect_tools({})
    assert mcp_tools == {}
    by_type = {}
    for t in tools:
        by_type.setdefault(t["type"], []).append(t)
    function_names = [t["function"]["name"] for t in by_type["function"]]
    assert "mytool" in function_names
    assert "shell" in function_names
    (ws,) = by_type["openrouter:web_search"]
    assert ws["parameters"] == {"engine": "exa", "max_results": 3}


async def test_web_search_disabled_by_default(tmp_path):
    session = make_session(tmp_path, shell=False)
    tools, _ = await session._collect_tools({})
    assert tools == []


def test_fork_copies_conversation(tmp_path):
    parent = make_session(tmp_path)
    parent.conversation.append({"role": "user", "content": "hi"})
    child = make_session(tmp_path, fork_session=parent)
    assert child.conversation == parent.conversation
    assert child.conversation is not parent.conversation


def test_fork_empty_conversation_raises(tmp_path):
    parent = make_session(tmp_path)
    with pytest.raises(ValueError):
        make_session(tmp_path, fork_session=parent)


def test_record_usage_reads_openrouter_cost(tmp_path):
    session = make_session(tmp_path)
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=150),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=40),
        cost=0.0042,
    )
    context = session._record_usage("deepseek/deepseek-chat", usage)
    assert context == 1000
    assert abs(session.total_token_usage.total_cost - 0.0042) < 1e-9
    u = session.total_token_usage.usage["deepseek/deepseek-chat"]
    assert (u.cached_tokens, u.reasoning_tokens) == (150, 40)


def test_reset_clears_state(tmp_path):
    session = make_session(tmp_path)
    session.conversation.append({"role": "user", "content": "hi"})
    session._session_id = "abc"
    session.reset()
    assert session.conversation == []
    assert session.session_id is None
