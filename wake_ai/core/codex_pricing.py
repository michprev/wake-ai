from typing import NamedTuple


class CodexTokenPricing(NamedTuple):
    # costs per 1M tokens
    input_mtoken_cost: float
    cached_input_mtoken_cost: float
    output_mtoken_cost: float
    # Cost per 1M tokens written to the prompt cache. Introduced with the
    # GPT-5.6 family, which bills cache writes at 1.25x the uncached input rate.
    # None for pre-5.6 models, which fold cache writes into the input price and
    # so incur no separate cache-write charge.
    cache_write_mtoken_cost: float | None = None


GPT_PRICING = {
    # GPT-5.6 series (Sol / Terra / Luna). Adds explicit cache-write billing
    # (1.25x the tier's uncached input rate). Long-context (>272K input tokens)
    # requests are surcharged 2x input/cached/cache-write and 1.5x output, the
    # same as the 5.4/5.5 families — see OpenAITokenUsage.update().
    "gpt-5.6-sol": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=15.00,
            cache_write_mtoken_cost=3.125,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=5.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=30.00,
            cache_write_mtoken_cost=6.25,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=10.00,
            cached_input_mtoken_cost=1.00,
            output_mtoken_cost=60.00,
            cache_write_mtoken_cost=12.50,
        ),
    },
    "gpt-5.6-terra": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=7.50,
            cache_write_mtoken_cost=1.5625,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=15.00,
            cache_write_mtoken_cost=3.125,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=5.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=30.00,
            cache_write_mtoken_cost=6.25,
        ),
    },
    "gpt-5.6-luna": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.50,
            cached_input_mtoken_cost=0.05,
            output_mtoken_cost=3.00,
            cache_write_mtoken_cost=0.625,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.00,
            cached_input_mtoken_cost=0.10,
            output_mtoken_cost=6.00,
            cache_write_mtoken_cost=1.25,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.00,
            cached_input_mtoken_cost=0.20,
            output_mtoken_cost=12.00,
            cache_write_mtoken_cost=2.50,
        ),
    },
    "gpt-5.5": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=15.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=5.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=30.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=12.50,
            cached_input_mtoken_cost=1.25,
            output_mtoken_cost=75.00,
        ),
    },
    # GPT-5.4 series
    "gpt-5.4": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.13,
            output_mtoken_cost=7.50,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=15.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=5.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=30.00,
        ),
    },
    "gpt-5.4-pro": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=15.00,
            cached_input_mtoken_cost=15.00,  # No cached pricing available
            output_mtoken_cost=90.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=30.00,
            cached_input_mtoken_cost=30.00,  # No cached pricing available
            output_mtoken_cost=180.00,
        ),
    },
    # GPT-5.2 series
    "gpt-5.2": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.875,
            cached_input_mtoken_cost=0.0875,
            output_mtoken_cost=7.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.75,
            cached_input_mtoken_cost=0.175,
            output_mtoken_cost=14.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5.2-chat-latest": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.875,
            cached_input_mtoken_cost=0.0875,
            output_mtoken_cost=7.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.75,
            cached_input_mtoken_cost=0.175,
            output_mtoken_cost=14.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5.2-pro": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=21.00,
            cached_input_mtoken_cost=21.00,  # No cached pricing available
            output_mtoken_cost=168.00,
        ),
    },
    # GPT-5.1 series
    "gpt-5.1": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.625,
            cached_input_mtoken_cost=0.0625,
            output_mtoken_cost=5.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5.1-chat-latest": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.625,
            cached_input_mtoken_cost=0.0625,
            output_mtoken_cost=5.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5.1-codex": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.625,
            cached_input_mtoken_cost=0.0625,
            output_mtoken_cost=5.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5.1-codex-mini": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.125,
            cached_input_mtoken_cost=0.0125,
            output_mtoken_cost=1.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.25,
            cached_input_mtoken_cost=0.025,
            output_mtoken_cost=2.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=0.45,
            cached_input_mtoken_cost=0.045,
            output_mtoken_cost=3.60,
        ),
    },

    # GPT-5 series
    "gpt-5": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.625,
            cached_input_mtoken_cost=0.0625,
            output_mtoken_cost=5.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=20.00,
        ),
    },
    "gpt-5-mini": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.125,
            cached_input_mtoken_cost=0.0125,
            output_mtoken_cost=1.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.25,
            cached_input_mtoken_cost=0.025,
            output_mtoken_cost=2.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=0.45,
            cached_input_mtoken_cost=0.045,
            output_mtoken_cost=3.60,
        ),
    },
    "gpt-5-nano": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.025,
            cached_input_mtoken_cost=0.0025,
            output_mtoken_cost=0.20,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.05,
            cached_input_mtoken_cost=0.005,
            output_mtoken_cost=0.40,
        ),
    },
    "gpt-5-chat-latest": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
    },
    "gpt-5-codex": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=10.00,
        ),
    },

    # GPT-4.1 series
    "gpt-4.1": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=8.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=3.50,
            cached_input_mtoken_cost=0.875,
            output_mtoken_cost=14.00,
        ),
    },
    "gpt-4.1-mini": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.40,
            cached_input_mtoken_cost=0.10,
            output_mtoken_cost=1.60,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=0.70,
            cached_input_mtoken_cost=0.175,
            output_mtoken_cost=2.80,
        ),
    },
    "gpt-4.1-nano": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.10,
            cached_input_mtoken_cost=0.025,
            output_mtoken_cost=0.40,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=0.20,
            cached_input_mtoken_cost=0.05,
            output_mtoken_cost=0.80,
        ),
    },

    # GPT-4o series
    "gpt-4o": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.50,
            cached_input_mtoken_cost=1.25,
            output_mtoken_cost=10.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=4.25,
            cached_input_mtoken_cost=2.125,
            output_mtoken_cost=17.00,
        ),
    },
    "gpt-4o-2024-05-13": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=5.00,
            cached_input_mtoken_cost=5.00,  # No cached pricing available
            output_mtoken_cost=15.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=8.75,
            cached_input_mtoken_cost=8.75,  # No cached pricing available
            output_mtoken_cost=26.25,
        ),
    },
    "gpt-4o-mini": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=0.15,
            cached_input_mtoken_cost=0.075,
            output_mtoken_cost=0.60,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=0.25,
            cached_input_mtoken_cost=0.125,
            output_mtoken_cost=1.00,
        ),
    },

    # O-series models
    "o1": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=15.00,
            cached_input_mtoken_cost=7.50,
            output_mtoken_cost=60.00,
        ),
    },
    "o1-pro": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=150.00,
            cached_input_mtoken_cost=150.00,  # No cached pricing available
            output_mtoken_cost=600.00,
        ),
    },
    "o1-mini": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.10,
            cached_input_mtoken_cost=0.55,
            output_mtoken_cost=4.40,
        ),
    },
    "o3": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=1.00,
            cached_input_mtoken_cost=0.25,
            output_mtoken_cost=4.00,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=8.00,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=3.50,
            cached_input_mtoken_cost=0.875,
            output_mtoken_cost=14.00,
        ),
    },
    "o3-pro": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=20.00,
            cached_input_mtoken_cost=20.00,  # No cached pricing available
            output_mtoken_cost=80.00,
        ),
    },
    "o3-mini": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.10,
            cached_input_mtoken_cost=0.55,
            output_mtoken_cost=4.40,
        ),
    },
    "o3-deep-research": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=10.00,
            cached_input_mtoken_cost=2.50,
            output_mtoken_cost=40.00,
        ),
    },
    "o4-mini": {
        "flex": CodexTokenPricing(
            input_mtoken_cost=0.55,
            cached_input_mtoken_cost=0.138,
            output_mtoken_cost=2.20,
        ),
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.10,
            cached_input_mtoken_cost=0.275,
            output_mtoken_cost=4.40,
        ),
        "priority": CodexTokenPricing(
            input_mtoken_cost=2.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=8.00,
        ),
    },
    "o4-mini-deep-research": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=2.00,
            cached_input_mtoken_cost=0.50,
            output_mtoken_cost=8.00,
        ),
    },

    # Specialized models
    "computer-use-preview": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=3.00,
            cached_input_mtoken_cost=3.00,  # No cached pricing available
            output_mtoken_cost=12.00,
        ),
    },
    "codex-mini-latest": {
        "standard": CodexTokenPricing(
            input_mtoken_cost=1.50,
            cached_input_mtoken_cost=0.375,
            output_mtoken_cost=6.00,
        ),
    },
}
