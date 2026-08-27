from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from harness.artifacts import atomic_write_text
from harness.processes import run_process_group


BWRAP = Path("/usr/bin/bwrap")

SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"


class SandboxError(RuntimeError):
    pass


def _require_dir(path: Path, label: str) -> Path:
    path = Path(path).absolute()

    if path.is_symlink() or not path.is_dir():
        raise SandboxError(
            f"{label} does not exist or is not a real directory: {path}"
        )

    return path


def _directory_options(path: str) -> list[str]:
    current = Path("/")
    options: list[str] = []

    for part in Path(path).parts[1:]:
        current /= part
        options.extend(["--dir", str(current)])

    return options


def build_isolated_command(
    candidate: Path,
    argv: Sequence[str],
    *,
    candidate_writable: bool,
    read_only_files: Mapping[Path, str] | None = None,
    writable_directories: Mapping[Path, str] | None = None,
    result_fd: int | None = None,
) -> list[str]:
    """Build the minimal, networkless namespace used for untrusted code."""

    candidate = _require_dir(candidate, "candidate")

    if not BWRAP.is_file():
        raise SandboxError(f"bubblewrap not found: {BWRAP}")

    command = [
        str(BWRAP),
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/sbin",
        "/sbin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--dir",
        "/etc",
    ]

    for host_path in (
        "/etc/alternatives",
        "/etc/group",
        "/etc/ld.so.cache",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/localtime",
    ):
        if Path(host_path).exists():
            command.extend(["--ro-bind", host_path, host_path])

    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            *_directory_options(str(candidate.parent)),
            "--bind" if candidate_writable else "--ro-bind",
            str(candidate),
            str(candidate),
        ]
    )

    for source, destination in sorted(
        (read_only_files or {}).items(),
        key=lambda item: item[1],
    ):
        source = Path(source).absolute()

        if source.is_symlink() or not source.is_file():
            raise SandboxError(f"read-only input is unsafe: {source}")

        command.extend(_directory_options(str(Path(destination).parent)))
        command.extend(["--ro-bind", str(source), destination])

    for source, destination in sorted(
        (writable_directories or {}).items(),
        key=lambda item: item[1],
    ):
        source = _require_dir(Path(source), "writable scratch directory")
        command.extend(_directory_options(str(Path(destination).parent)))
        command.extend(["--bind", str(source), destination])

    command.extend(
        [
            "--chdir",
            str(candidate),
            "--setenv",
            "HOME",
            str(candidate),
            "--setenv",
            "PATH",
            SAFE_PATH,
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "GIT_CONFIG_NOSYSTEM",
            "1",
            "--setenv",
            "GIT_CONFIG_GLOBAL",
            "/dev/null",
            "--setenv",
            "HERMES_BENCHMARK",
            "1",
            "--setenv",
            "HERMES_BENCH_CANDIDATE",
            str(candidate),
            "--setenv",
            "HERMES_BENCH_SANDBOX_ACTIVE",
            "1",
        ]
    )

    if result_fd is not None:
        command.extend(
            ["--setenv", "HERMES_BENCH_RESULT_FD", str(result_fd)]
        )

    return [*command, "--", *argv]


def build_command(
    candidate: Path,
    runtime_dir: Path,
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Compatibility wrapper for direct offline sandbox probes.

    Runtime paths and caller environment values are deliberately not exposed
    to candidate code. Live Hermes uses :func:`write_shell_wrapper` instead.
    """

    _require_dir(runtime_dir, "runtime directory")

    if env:
        unsupported = ", ".join(sorted(env))
        raise SandboxError(
            "candidate environment is fixed; unsupported keys: " + unsupported
        )

    return build_isolated_command(
        candidate,
        argv,
        candidate_writable=True,
    )


def write_shell_wrapper(
    runtime_dir: Path,
    candidate: Path,
) -> Path:
    """Create the trusted executable used for every Hermes tool shell.

    Hermes selects this path before any model-controlled command text is
    interpreted. The wrapper enters Bubblewrap and only then starts the real
    shell, so containment does not depend on ``BASH_ENV`` or child-shell
    cooperation.
    """

    runtime_dir = _require_dir(runtime_dir, "runtime directory")
    wrapper_path = runtime_dir / "candidate-shell-wrapper"
    sentinel = "__HERMES_BENCH_SHELL_ARGS__"
    command = build_isolated_command(
        candidate,
        [
            "/usr/bin/bash",
            sentinel,
        ],
        candidate_writable=True,
    )
    prefix = " ".join(shlex.quote(item) for item in command[:-1])
    script = (
        "#!/usr/bin/bash\n"
        "# Harness-owned executable; never mounted into the candidate.\n"
        f"exec {prefix} \"$@\"\n"
    )
    atomic_write_text(wrapper_path, script, root=runtime_dir)
    os.chmod(wrapper_path, 0o700, follow_symlinks=False)
    return wrapper_path


def run(
    candidate: Path,
    runtime_dir: Path,
    argv: Sequence[str],
    *,
    timeout: int | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = build_command(
        candidate,
        runtime_dir,
        argv,
        env=env,
    )
    return run_process_group(command, timeout=timeout)
