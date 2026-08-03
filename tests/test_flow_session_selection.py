import pytest

from wake_ai.core.claude import ClaudeSession
from wake_ai.core.codex import CodexSession
from wake_ai.core.flow import _default_session
from wake_ai.core.openrouter import OpenRouterSession


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_slash_slug_selects_openrouter(tmp_path):
    session = _default_session("deepseek/deepseek-chat", tmp_path, tmp_path)
    assert isinstance(session, OpenRouterSession)


def test_gpt_model_selects_codex(tmp_path):
    session = _default_session("gpt-5.2", tmp_path, tmp_path)
    assert isinstance(session, CodexSession)


def test_other_model_selects_claude(tmp_path):
    session = _default_session("opus-4.5", tmp_path, tmp_path)
    assert isinstance(session, ClaudeSession)


def test_openrouter_exported_from_package():
    import wake_ai
    assert wake_ai.OpenRouterSession is OpenRouterSession
