import httpx
import pytest
from openai import APIError
from openai.types.chat import ChatCompletion

from conftest import FakeStream, NullFormatter, chunk, text_turn
from wake_ai.core.openrouter import (
    MidStreamError,
    OpenRouterSession,
    _is_context_length_error,
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def context_error() -> APIError:
    return APIError(
        "This endpoint's maximum context length is 163840 tokens. However, you requested about 200000 tokens.",
        httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        body=None,
    )


def completion(text: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        "id": "gen-2", "object": "chat.completion", "created": 1,
        "model": "deepseek/deepseek-chat",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 500, "completion_tokens": 50,
                  "total_tokens": 550, "cost": 0.001},
    })


def test_classifier_matches_context_length_variants():
    assert _is_context_length_error(context_error())
    assert _is_context_length_error(MidStreamError({"code": 400, "message": "input exceeds the context window of this model"}))
    assert not _is_context_length_error(MidStreamError({"code": 500, "message": "provider unavailable"}))
    assert not _is_context_length_error(
        APIError("rate limited", httpx.Request("POST", "https://x"), body=None))


async def test_compact_rebuilds_conversation(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    session.conversation = [
        {"role": "user", "content": "audit this"},
        {"role": "assistant", "content": "long analysis..."},
    ]

    captured = {}

    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return completion("SUMMARY OF WORK")

    monkeypatch.setattr(session.client.chat.completions, "create", fake_create)
    await session._compact("deepseek/deepseek-chat", resume_hint=True)

    assert len(session.conversation) == 1
    only = session.conversation[0]
    assert only["role"] == "user"
    assert "SUMMARY OF WORK" in only["content"]
    assert "continue" in only["content"]
    # the summarization request contained the old conversation + compaction prompt
    assert captured["messages"][-1]["role"] == "user"
    assert "summary" in captured["messages"][-1]["content"].lower()
    # compaction request cost was tracked
    assert session.total_token_usage.total_cost > 0


async def test_compact_without_resume_hint_has_no_continue(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    session.conversation = [{"role": "user", "content": "hi"}]

    async def fake_create(**kwargs):
        return completion("S")

    monkeypatch.setattr(session.client.chat.completions, "create", fake_create)
    await session._compact("deepseek/deepseek-chat", resume_hint=False)
    assert "continue" not in session.conversation[0]["content"]


async def test_compact_truncates_when_summarization_also_overflows(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)
    session.conversation = [{"role": "user", "content": f"msg {i}"} for i in range(8)]

    attempts = {"n": 0}

    async def fake_create(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise context_error()
        captured_len = len(kwargs["messages"])
        return completion(f"S after {captured_len} messages")

    monkeypatch.setattr(session.client.chat.completions, "create", fake_create)
    await session._compact("deepseek/deepseek-chat", resume_hint=False)
    assert attempts["n"] == 2
    assert len(session.conversation) == 1


async def test_context_error_in_stream_triggers_compaction_end_to_end(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)

    calls = {"n": 0}
    recovered = text_turn("recovered after compaction", cost=0.01)

    async def fake_create_stream(model, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            raise context_error()
        return FakeStream(recovered)

    async def fake_create(**kwargs):
        return completion("COMPACTED")

    monkeypatch.setattr(session, "_create_stream", fake_create_stream)
    monkeypatch.setattr(session.client.chat.completions, "create", fake_create)

    responses = [r async for r in session.query("go", "deepseek/deepseek-chat", None, NullFormatter())]
    assert responses[-1].status == "succeeded"
    assert responses[-1].final_message == "recovered after compaction"
    assert responses[-1].compaction_count == 1


async def test_finish_reason_length_compacts_and_continues(tmp_path, monkeypatch):
    session = OpenRouterSession(tmp_path, shell=False)

    truncated = [
        chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"},
                            "finish_reason": "length"}]}),
        chunk({"choices": [],
               "usage": {"prompt_tokens": 100, "completion_tokens": 999,
                         "total_tokens": 1099, "cost": 0.01}}),
    ]
    streams = [truncated, text_turn("finished", cost=0.01)]

    async def fake_create_stream(model, tools):
        return FakeStream(streams.pop(0))

    async def fake_create(**kwargs):
        return completion("COMPACTED")

    monkeypatch.setattr(session, "_create_stream", fake_create_stream)
    monkeypatch.setattr(session.client.chat.completions, "create", fake_create)

    responses = [r async for r in session.query("go", "deepseek/deepseek-chat", None, NullFormatter())]
    assert responses[-1].status == "succeeded"
    assert responses[-1].final_message == "finished"
    assert responses[-1].compaction_count == 1
    # post-compaction conversation: summary user message + final assistant message
    assert [m["role"] for m in session.conversation] == ["user", "assistant"]
