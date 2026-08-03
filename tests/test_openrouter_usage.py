from wake_ai.core.openrouter import OpenRouterResponse, OpenRouterTotalTokenUsage


def test_total_cost_accumulates_across_models():
    usage = OpenRouterTotalTokenUsage()
    usage.update("deepseek/deepseek-chat", prompt_tokens=1000, completion_tokens=200,
                 cached_tokens=100, reasoning_tokens=0, cost=0.0021)
    usage.update("deepseek/deepseek-chat", prompt_tokens=1500, completion_tokens=300,
                 cached_tokens=900, reasoning_tokens=50, cost=0.0030)
    usage.update("qwen/qwen3-coder", prompt_tokens=500, completion_tokens=100,
                 cached_tokens=0, reasoning_tokens=0, cost=0.0010)
    assert abs(usage.total_cost - 0.0061) < 1e-9


def test_format_summary_lists_models_and_total():
    usage = OpenRouterTotalTokenUsage()
    usage.update("deepseek/deepseek-chat", prompt_tokens=1000, completion_tokens=200,
                 cached_tokens=100, reasoning_tokens=25, cost=0.0021)
    summary = usage.format_summary()
    assert "deepseek/deepseek-chat" in summary
    assert "input=1000" in summary
    assert "cached 100" in summary
    assert "reasoning 25" in summary
    assert "$0.0021" in summary
    assert "total:" in summary


def test_response_defaults():
    r = OpenRouterResponse(cost=1.0, status="running")
    assert r.final_message is None
    assert r.context_tokens is None
    assert r.compaction_count == 0
