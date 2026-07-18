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

    params: list[str] = []
    if network_access:
        # _CS_DARWIN_USER_CACHE_DIR — not in os.confstr_names, pass the raw int
        cache_dir = os.confstr(65538)
        params += ["-D", f"DARWIN_USER_CACHE_DIR={cache_dir}"]

    args = ["-p", policy, *params, "--", "bash", "-lc", command]

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
