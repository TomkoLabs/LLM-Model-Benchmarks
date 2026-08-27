from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence


TERMINATE_GRACE_SECONDS = 3.0


class ProcessDeadlineExpired(subprocess.TimeoutExpired):
    """A process exceeded its total-wall or productive-activity deadline."""

    def __init__(
        self,
        cmd: Sequence[str],
        timeout: float,
        *,
        deadline_kind: str,
        last_activity_seconds_ago: float,
        output: str = "",
    ) -> None:
        super().__init__(list(cmd), timeout, output=output)
        self.deadline_kind = deadline_kind
        self.last_activity_seconds_ago = last_activity_seconds_ago


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = TERMINATE_GRACE_SECONDS,
) -> None:
    """Bounded TERM/KILL cleanup for a process and all descendants."""

    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace_seconds

    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def run_process_group(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    inactivity_timeout: float | None = None,
    heartbeat_paths: Sequence[Path] = (),
    heartbeat_callback: Callable[[dict[str, object]], None] | None = None,
    poll_interval: float = 0.25,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
        pass_fds=tuple(pass_fds),
    )

    if inactivity_timeout is not None and inactivity_timeout <= 0:
        terminate_process_group(process)
        raise ValueError("inactivity_timeout must be positive")

    # Keep the historical communicate(timeout=...) path untouched unless the
    # caller explicitly asks for liveness monitoring.  This preserves its
    # well-tested partial-output semantics for evaluators and sandbox helpers.
    if inactivity_timeout is None and not heartbeat_paths:
        try:
            stdout, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_group(process)
            stdout, _ = process.communicate()
            exc.stdout = _as_text(exc.stdout) + _as_text(stdout)
            raise
        except BaseException as exc:
            terminate_process_group(process)
            stdout, _ = process.communicate()
            setattr(exc, "hermesbench_stdout", _as_text(stdout))
            raise

        return subprocess.CompletedProcess(
            list(argv),
            process.returncode,
            stdout=stdout,
        )

    result: dict[str, str] = {"stdout": ""}
    communicate_error: list[BaseException] = []

    def _communicate() -> None:
        try:
            stdout, _ = process.communicate()
            result["stdout"] = _as_text(stdout)
        except BaseException as exc:  # pragma: no cover - defensive transport path
            communicate_error.append(exc)

    thread = threading.Thread(target=_communicate, daemon=True)
    thread.start()
    started = time.monotonic()
    last_activity = started
    snapshots: dict[Path, tuple[int, ...]] = {}

    def _snapshot(path: Path) -> tuple[int, ...] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_dir():
            return (stat.st_mtime_ns, stat.st_size)
        children: list[tuple[int, int]] = []
        try:
            for child in path.iterdir():
                try:
                    child_stat = child.stat()
                except OSError:
                    continue
                children.append((child_stat.st_mtime_ns, child_stat.st_size))
        except OSError:
            return None
        latest = max((item[0] for item in children), default=0)
        total_size = sum(item[1] for item in children)
        return (stat.st_mtime_ns, len(children), latest, total_size)

    try:
        while thread.is_alive():
            now = time.monotonic()
            for heartbeat_path in heartbeat_paths:
                snapshot = _snapshot(heartbeat_path)
                if snapshot is None:
                    continue
                if snapshots.get(heartbeat_path) == snapshot:
                    continue
                snapshots[heartbeat_path] = snapshot
                last_activity = now
                if heartbeat_callback:
                    heartbeat_callback(
                        {
                            "path": str(heartbeat_path),
                            "observed_at_monotonic": now,
                            "snapshot": list(snapshot),
                        }
                    )

            elapsed = now - started
            idle = now - last_activity
            deadline_kind = None
            deadline_value = 0.0
            if timeout is not None and elapsed >= timeout:
                deadline_kind = "total_wall"
                deadline_value = timeout
            elif inactivity_timeout is not None and idle >= inactivity_timeout:
                deadline_kind = "inactivity"
                deadline_value = inactivity_timeout

            if deadline_kind:
                terminate_process_group(process)
                thread.join(TERMINATE_GRACE_SECONDS * 2)
                raise ProcessDeadlineExpired(
                    argv,
                    deadline_value,
                    deadline_kind=deadline_kind,
                    last_activity_seconds_ago=idle,
                    output=result["stdout"],
                )
            thread.join(max(0.02, min(poll_interval, 1.0)))
    except BaseException as exc:
        if process.poll() is None:
            terminate_process_group(process)
        thread.join(TERMINATE_GRACE_SECONDS * 2)
        if isinstance(exc, ProcessDeadlineExpired):
            exc.stdout = result["stdout"] or _as_text(exc.stdout)
        else:
            setattr(exc, "hermesbench_stdout", result["stdout"])
        raise

    if communicate_error:
        raise communicate_error[0]

    stdout = result["stdout"]

    return subprocess.CompletedProcess(
        list(argv),
        process.returncode,
        stdout=stdout,
    )
