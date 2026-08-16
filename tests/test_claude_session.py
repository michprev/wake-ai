from pathlib import Path

import pytest

from wake_ai.core import claude
from wake_ai.core.claude import ClaudeSession


class _StopConnect(Exception):
    pass


class _CapturingClient:
    options = None

    def __init__(self, *, options):
        type(self).options = options

    async def connect(self):
        raise _StopConnect


class _NullFormatter:
    def print_user_message(self, message):
        pass

    def print_agent_message(self, message):
        pass

    def print_thinking(self, message):
        pass

    def print_system_message(self, message):
        pass

    def print_error(self, message):
        pass

    def print_tool_use(self, name, input):
        pass

    def print_tool_result(self, result, is_error):
        pass


@pytest.mark.parametrize("mode", ["default", "bypassPermissions"])
async def test_permission_mode_is_forwarded(monkeypatch, tmp_path: Path, mode):
    monkeypatch.setattr(claude, "ClaudeSDKClient", _CapturingClient)
    session = ClaudeSession(tmp_path, tmp_path / "working", permission_mode=mode)

    with pytest.raises(_StopConnect):
        async for _ in session.query("test", "claude-opus-5", None, _NullFormatter()):
            pass

    assert session.permission_mode == mode
    assert _CapturingClient.options.permission_mode == mode
