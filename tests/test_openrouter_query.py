import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionChunk

from conftest import FakeStream, NullFormatter, chunk, text_turn
from wake_ai.core.openrouter import OpenRouterResponse, OpenRouterSession
from wake_ai.core.session_abc import FunctionTool


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def tool_call_turn(name: str, arguments: str, cost: float) -> list[ChatCompletionChunk]:
    return [
        chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": name, "arguments": arguments}}]},
            "finish_reason": "tool_calls"}]}),
        chunk({"choices": [],
               "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                         "total_tokens": 110, "cost": cost}}),
    ]


def install_fake_streams(monkeypatch, session, turns):
    turns = list(turns)

    async def fake_create_stream(model, tools):
        return FakeStream(turns.pop(0))

    monkeypatch.setattr(session, "_create_stream", fake_create_stream)


async def collect(session, prompt, model, max_cost=None):
    return [r async for r in session.query(prompt, model, max_cost, NullFormatter())]


async def test_simple_text_turn(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    install_fake_streams(monkeypatch, session, [text_turn("Hello!", cost=0.01)])
    responses = await collect(session, "hi", "deepseek/deepseek-chat")
    assert [r.status for r in responses] == ["running", "succeeded"]
    assert responses[-1].final_message == "Hello!"
    assert abs(responses[-1].cost - 0.01) < 1e-9
    assert responses[0].context_tokens == 100
    assert session.conversation == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert session.session_id is not None


async def test_function_tool_loop(tmp_path, monkeypatch):
    calls = []

    async def echo(value: str) -> str:
        calls.append(value)
        return f"echo:{value}"

    session = OpenRouterSession(tmp_path, shell=False, tools=[
        FunctionTool(name="echo",
                     input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                     description="echo", handler=echo)])
    install_fake_streams(monkeypatch, session, [
        tool_call_turn("echo", json.dumps({"value": "ping"}), cost=0.01),
        text_turn("done", cost=0.02),
    ])
    responses = await collect(session, "run echo", "deepseek/deepseek-chat")
    assert calls == ["ping"]
    assert [r.status for r in responses] == ["running", "running", "succeeded"]
    assert abs(responses[-1].cost - 0.03) < 1e-9
    roles = [m["role"] for m in session.conversation]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_message = session.conversation[2]
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == "echo:ping"


async def test_unknown_tool_returns_error_result(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    install_fake_streams(monkeypatch, session, [
        tool_call_turn("nonexistent", "{}", cost=0.01),
        text_turn("ok", cost=0.01),
    ])
    await collect(session, "go", "deepseek/deepseek-chat")
    tool_message = session.conversation[2]
    assert "Unknown tool" in tool_message["content"]


async def test_max_cost_terminates(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    install_fake_streams(monkeypatch, session, [
        tool_call_turn("whatever", "{}", cost=0.02),
        text_turn("never reached", cost=0.02),
    ])
    responses = await collect(session, "go", "deepseek/deepseek-chat", max_cost=0.015)
    assert responses[-1].status == "terminating_on_max_cost"
    assert abs(responses[-1].cost - 0.02) < 1e-9


async def test_max_cost_zero_short_circuits(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    responses = await collect(session, "go", "deepseek/deepseek-chat", max_cost=0.0)
    assert [r.status for r in responses] == ["terminating_on_max_cost"]
    assert session.conversation == []


async def test_retry_then_success(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False, max_retries=2)
    attempts = {"n": 0}
    good_turn = text_turn("recovered", cost=0.01)

    async def flaky_create_stream(model, tools):
        attempts["n"] += 1
        if attempts["n"] == 1:
            import httpx
            raise httpx.RemoteProtocolError("connection lost")
        return FakeStream(good_turn)

    monkeypatch.setattr(session, "_create_stream", flaky_create_stream)
    monkeypatch.setattr("wake_ai.core.openrouter.compute_backoff_time", lambda retry: 0.0)
    responses = await collect(session, "hi", "deepseek/deepseek-chat")
    assert attempts["n"] == 2
    assert responses[-1].status == "succeeded"
    assert responses[-1].final_message == "recovered"
    # the failed attempt must not have committed anything
    assert [m["role"] for m in session.conversation] == ["user", "assistant"]
