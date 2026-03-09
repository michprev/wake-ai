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

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.terminate()
        raise

    stdout = await proc.stdout.read()
    stderr = await proc.stderr.read()
    returncode = proc.returncode

    return stdout.decode("utf-8"), stderr.decode("utf-8"), returncode
