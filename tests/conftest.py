from openai.types.chat import ChatCompletionChunk


class NullFormatter:
    """Duck-typed VerboseFormatter double."""
    def print_user_message(self, message): pass
    def print_agent_message(self, message): pass
    def print_thinking(self, message): pass
    def print_system_message(self, message): pass
    def print_error(self, message): pass
    def print_tool_use(self, name, input): pass
    def print_tool_result(self, result, is_error): pass


class FakeStream:
    def __init__(self, chunks):
        self._iterator = self._generate(chunks)

    async def _generate(self, chunks):
        for chunk in chunks:
            yield chunk

    def __aiter__(self):
        return self._iterator

    async def close(self):
        pass


def chunk(payload: dict) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate({
        "id": "gen-1", "object": "chat.completion.chunk", "created": 1,
        "model": "deepseek/deepseek-chat", **payload,
    })


def text_turn(text: str, cost: float, prompt_tokens: int = 100) -> list[ChatCompletionChunk]:
    return [
        chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                            "finish_reason": "stop"}]}),
        chunk({"choices": [],
               "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10,
                         "total_tokens": prompt_tokens + 10, "cost": cost}}),
    ]
