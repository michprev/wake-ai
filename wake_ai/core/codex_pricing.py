from typing import NamedTuple


class CodexTokenPricing(NamedTuple):
    # costs per 1M tokens
    input_mtoken_cost: float
    cached_input_mtoken_cost: float
    output_mtoken_cost: float


GPT_PRICING = {
    # GPT-5 series
    "gpt-5": CodexTokenPricing(
        input_mtoken_cost=1.25,
        cached_input_mtoken_cost=0.125,
        output_mtoken_cost=10.00,
    ),
    "gpt-5-mini": CodexTokenPricing(
        input_mtoken_cost=0.25,
        cached_input_mtoken_cost=0.025,
        output_mtoken_cost=2.00,
    ),
    "gpt-5-nano": CodexTokenPricing(
        input_mtoken_cost=0.05,
        cached_input_mtoken_cost=0.005,
        output_mtoken_cost=0.40,
    ),
    "gpt-5-chat-latest": CodexTokenPricing(
        input_mtoken_cost=1.25,
        cached_input_mtoken_cost=0.125,
        output_mtoken_cost=10.00,
    ),
    "gpt-5-codex": CodexTokenPricing(
        input_mtoken_cost=1.25,
        cached_input_mtoken_cost=0.125,
        output_mtoken_cost=10.00,
    ),

    # GPT-4.1 series
    "gpt-4.1": CodexTokenPricing(
        input_mtoken_cost=2.00,
        cached_input_mtoken_cost=0.50,
        output_mtoken_cost=8.00,
    ),
    "gpt-4.1-mini": CodexTokenPricing(
        input_mtoken_cost=0.40,
        cached_input_mtoken_cost=0.10,
        output_mtoken_cost=1.60,
    ),
    "gpt-4.1-nano": CodexTokenPricing(
        input_mtoken_cost=0.10,
        cached_input_mtoken_cost=0.025,
        output_mtoken_cost=0.40,
    ),

    # GPT-4o series
    "gpt-4o": CodexTokenPricing(
        input_mtoken_cost=2.50,
        cached_input_mtoken_cost=1.25,
        output_mtoken_cost=10.00,
    ),
    "gpt-4o-2024-05-13": CodexTokenPricing(
        input_mtoken_cost=5.00,
        cached_input_mtoken_cost=5.00,  # No cached pricing available
        output_mtoken_cost=15.00,
    ),
    "gpt-4o-mini": CodexTokenPricing(
        input_mtoken_cost=0.15,
        cached_input_mtoken_cost=0.075,
        output_mtoken_cost=0.60,
    ),

    # O-series models
    "o1": CodexTokenPricing(
        input_mtoken_cost=15.00,
        cached_input_mtoken_cost=7.50,
        output_mtoken_cost=60.00,
    ),
    "o1-pro": CodexTokenPricing(
        input_mtoken_cost=150.00,
        cached_input_mtoken_cost=150.00,  # No cached pricing available
        output_mtoken_cost=600.00,
    ),
    "o1-mini": CodexTokenPricing(
        input_mtoken_cost=1.10,
        cached_input_mtoken_cost=0.55,
        output_mtoken_cost=4.40,
    ),
    "o3": CodexTokenPricing(
        input_mtoken_cost=2.00,
        cached_input_mtoken_cost=0.50,
        output_mtoken_cost=8.00,
    ),
    "o3-pro": CodexTokenPricing(
        input_mtoken_cost=20.00,
        cached_input_mtoken_cost=20.00,  # No cached pricing available
        output_mtoken_cost=80.00,
    ),
    "o3-mini": CodexTokenPricing(
        input_mtoken_cost=1.10,
        cached_input_mtoken_cost=0.55,
        output_mtoken_cost=4.40,
    ),
    "o3-deep-research": CodexTokenPricing(
        input_mtoken_cost=10.00,
        cached_input_mtoken_cost=2.50,
        output_mtoken_cost=40.00,
    ),
    "o4-mini": CodexTokenPricing(
        input_mtoken_cost=1.10,
        cached_input_mtoken_cost=0.275,
        output_mtoken_cost=4.40,
    ),
    "o4-mini-deep-research": CodexTokenPricing(
        input_mtoken_cost=2.00,
        cached_input_mtoken_cost=0.50,
        output_mtoken_cost=8.00,
    ),

    # Specialized models
    "computer-use-preview": CodexTokenPricing(
        input_mtoken_cost=3.00,
        cached_input_mtoken_cost=3.00,  # No cached pricing available
        output_mtoken_cost=12.00,
    ),
    "codex-mini-latest": CodexTokenPricing(
        input_mtoken_cost=1.50,
        cached_input_mtoken_cost=0.375,
        output_mtoken_cost=6.00,
    ),
}
