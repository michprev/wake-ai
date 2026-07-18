import asyncio
import json
import sys
from pathlib import Path


async def run_under_landlock(command: str, network_access: bool, writable_roots: list[Path], timeout: float | None, cwd: Path) -> tuple[str, str, int]:
    policy = {
        "type": "workspace-write",
        "writable_roots": [root.resolve().as_posix() for root in writable_roots],
        "network_access": network_access,
    }

    args = ["--sandbox-policy-cwd", (cwd.resolve().as_posix()), "--sandbox-policy", json.dumps(policy), "--", "bash", "-lc", command]

    print(f"Running command: {command}", file=sys.stderr)

    proc = await asyncio.create_subprocess_exec(
        Path(__file__).parent / "codex-linux-sandbox",
        *args,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    # communicate() drains stdout/stderr concurrently while waiting. Using
    # proc.wait() with PIPEd output deadlocks once the child fills the OS pipe
    # buffer (~64 KiB): it blocks on write(), never exits, and we hit `timeout`.
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.terminate()
        # Reap the process (and drain pipes) so it doesn't linger; kill if it
        # ignores SIGTERM.
        try:
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
        raise

    returncode = proc.returncode
    assert returncode is not None  # set once communicate() returns normally

    return stdout.decode("utf-8"), stderr.decode("utf-8"), returncode
