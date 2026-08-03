from openai.types.chat import ChatCompletionChunk

from wake_ai.core.openrouter import ChatTurnAccumulator


def make_chunk(delta: dict, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "deepseek/deepseek-chat",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    })


def test_accumulates_content_across_chunks():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"role": "assistant", "content": "Hel"}))
    acc.add_chunk(make_chunk({"content": "lo"}))
    acc.add_chunk(make_chunk({}, finish_reason="stop"))
    assert acc.content == "Hello"
    assert acc.finish_reason == "stop"
    assert acc.assistant_message() == {"role": "assistant", "content": "Hello"}


def test_accumulates_tool_call_fragments_by_index():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"tool_calls": [
        {"index": 0, "id": "call_a", "type": "function",
         "function": {"name": "shell", "arguments": ""}}]}))
    acc.add_chunk(make_chunk({"tool_calls": [
        {"index": 0, "function": {"arguments": '{"commands": ['}}]}))
    acc.add_chunk(make_chunk({"tool_calls": [
        {"index": 0, "function": {"arguments": '"ls"], "timeout_ms": 1000}'}}]}))
    acc.add_chunk(make_chunk({}, finish_reason="tool_calls"))
    calls = acc.tool_calls()
    assert len(calls) == 1
    assert calls[0]["id"] == "call_a"
    assert calls[0]["function"]["name"] == "shell"
    assert calls[0]["function"]["arguments"] == '{"commands": ["ls"], "timeout_ms": 1000}'
    msg = acc.assistant_message()
    assert msg["tool_calls"] == calls


def test_parallel_tool_calls_keep_index_order():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"tool_calls": [
        {"index": 1, "id": "call_b", "type": "function",
         "function": {"name": "beta", "arguments": "{}"}}]}))
    acc.add_chunk(make_chunk({"tool_calls": [
        {"index": 0, "id": "call_a", "type": "function",
         "function": {"name": "alpha", "arguments": "{}"}}]}))
    calls = acc.tool_calls()
    assert [c["id"] for c in calls] == ["call_a", "call_b"]


def test_reasoning_details_merged_by_index_and_kept_verbatim():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"reasoning_details": [
        {"type": "reasoning.text", "index": 0, "text": "step one, "}]}))
    acc.add_chunk(make_chunk({"reasoning_details": [
        {"type": "reasoning.text", "index": 0, "text": "step two"}]}))
    acc.add_chunk(make_chunk({"reasoning_details": [
        {"type": "reasoning.encrypted", "index": 1, "data": "AAAA"}]}))
    assert acc.reasoning_details == [
        {"type": "reasoning.text", "index": 0, "text": "step one, step two"},
        {"type": "reasoning.encrypted", "index": 1, "data": "AAAA"},
    ]
    assert acc.assistant_message()["reasoning_details"] == acc.reasoning_details


def test_plain_reasoning_text_accumulates():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"reasoning": "thinking "}))
    acc.add_chunk(make_chunk({"reasoning": "hard"}))
    assert acc.reasoning == "thinking hard"
    # plain reasoning text is NOT included in the replayed assistant message
    assert "reasoning" not in acc.assistant_message()


def test_url_citation_annotations_collected():
    acc = ChatTurnAccumulator()
    acc.add_chunk(make_chunk({"annotations": [
        {"type": "url_citation",
         "url_citation": {"url": "https://example.com", "title": "Example"}}]}))
    assert acc.annotations == [
        {"type": "url_citation",
         "url_citation": {"url": "https://example.com", "title": "Example"}}]


def test_empty_choices_chunk_is_ignored():
    acc = ChatTurnAccumulator()
    chunk = ChatCompletionChunk.model_validate({
        "id": "gen-1", "object": "chat.completion.chunk", "created": 1,
        "model": "m", "choices": [],
    })
    acc.add_chunk(chunk)  # must not raise (usage-only final chunks have no choices)
    assert acc.content is None
