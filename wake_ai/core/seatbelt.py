import asyncio
import os
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


BASE_POLICY = (Path(__file__).parent / "seatbelt_base_policy.sbpl").read_text()
NETWORK_POLICY = (Path(__file__).parent / "seatbelt_network_policy.sbpl").read_text()
RO_POLICY = "; allow read-only file operations\n(allow file-read*)"


async def run_under_seatbelt(command: str, network_access: bool, writable_roots: list[Path], timeout: float | None, cwd: Path) -> tuple[str, str, int]:
    if Path("/tmp") not in writable_roots:
        writable_roots.append(Path("/tmp"))
    if tmpdir := os.getenv("TMPDIR"):
        writable_roots.append(Path(tmpdir))

    if not writable_roots:
        write_policy = ""
    else:
        policy = "\n".join(
            f"(subpath \"{root.resolve().as_posix()}\")"
            for root in writable_roots
        )
        write_policy = f"(allow file-write*\n{policy}\n)"

    network_policy = NETWORK_POLICY if network_access else ""

    policy = f"{BASE_POLICY}\n{RO_POLICY}\n{write_policy}\n{network_policy}"

    args = ["-p", policy, "--", "bash", "-lc", command]

    proc = await asyncio.create_subprocess_exec(
        "/usr/bin/sandbox-exec",
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
