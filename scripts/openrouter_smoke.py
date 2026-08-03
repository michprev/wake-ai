"""Manual smoke test for OpenRouterSession. Not run in CI — costs real money.

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/openrouter_smoke.py [model-slug]

Default model is a cheap one; pass any OpenRouter slug to test another.
Exercises: shell tool, web search server tool, cost reporting, max_cost.
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console

from wake_ai.core.openrouter import OpenRouterSession
from wake_ai.core.verbose_formatter import VerboseFormatter


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-chat-v3.1"
    execution_dir = Path.cwd()
    console = Console()
    formatter = VerboseFormatter(console, "smoke", Path("/tmp/openrouter_smoke.log"), 2)

    session = OpenRouterSession(
        execution_dir,
        instructions="You are a concise assistant. Use tools when asked.",
        web_search=True,
        web_search_engine="exa",
        shell=True,
    )
    prompt = (
        "1. Run `uname -s` with the shell tool and report the output. "
        "2. Use web search to find the latest Solidity release version. "
        "Answer in two short lines."
    )
    async for response in session.query(prompt, model, 0.25, formatter):
        print(f"[{response.status}] cost=${response.cost:.4f} context={response.context_tokens}")

    print(f"\nsession_id: {session.session_id}")
    print(session.total_token_usage.format_summary())


if __name__ == "__main__":
    asyncio.run(main())
