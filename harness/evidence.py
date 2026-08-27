from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from harness.artifacts import (
    ArtifactSafetyError,
    atomic_write_bytes,
    ensure_subdirectory,
)


class EvidenceError(RuntimeError):
    pass


def _git(
    git_dir: Path,
    work_tree: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }
    process = subprocess.run(
        [
            "/usr/bin/git",
            f"--git-dir={git_dir}",
            f"--work-tree={work_tree}",
            *args,
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )

    if check and process.returncode:
        raise EvidenceError(
            f"git evidence command failed ({process.returncode}): "
            + process.stdout.decode("utf-8", errors="replace")
        )

    return process


def _validate_relative(value: bytes) -> str:
    try:
        text = value.decode("utf-8", errors="surrogateescape")
    except UnicodeDecodeError as exc:
        raise EvidenceError("candidate path is not decodable") from exc

    path = PurePosixPath(text)

    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise EvidenceError(f"unsafe candidate path: {text!r}")

    return text


def _open_candidate_parent(
    candidate: Path,
    relative: str,
) -> tuple[int, str, os.stat_result]:
    """Resolve a candidate entry without following any path-component link."""

    parts = PurePosixPath(relative).parts
    descriptor = os.open(
        candidate,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor

        metadata = os.stat(
            parts[-1],
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        return descriptor, parts[-1], metadata
    except BaseException:
        os.close(descriptor)
        raise


def _copy_entry(candidate: Path, relative: str, destination_root: Path) -> None:
    parent_descriptor, name, metadata = _open_candidate_parent(
        candidate,
        relative,
    )
    destination = destination_root / relative

    try:
        ensure_subdirectory(destination_root, destination.parent)

        if stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(name, dir_fd=parent_descriptor)
            observed = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )

            if (
                observed.st_dev != metadata.st_dev
                or observed.st_ino != metadata.st_ino
                or not stat.S_ISLNK(observed.st_mode)
            ):
                raise EvidenceError(
                    f"candidate link changed during capture: {relative}"
                )

            destination.symlink_to(target)
            return

        if not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError(
                f"unsupported candidate file type: {relative}"
            )

        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )

        try:
            observed = os.fstat(descriptor)

            if (
                observed.st_dev != metadata.st_dev
                or observed.st_ino != metadata.st_ino
                or not stat.S_ISREG(observed.st_mode)
            ):
                raise EvidenceError(
                    f"candidate file changed during capture: {relative}"
                )

            with destination.open("xb") as output:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

            os.chmod(destination, stat.S_IMODE(metadata.st_mode) & 0o755)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def capture_baseline(candidate: Path, runtime_dir: Path) -> dict[str, Path]:
    baseline_tree = ensure_subdirectory(
        runtime_dir,
        runtime_dir / "baseline-tree",
        exclusive=True,
    )
    baseline_git = runtime_dir / "baseline.git"

    shutil.copytree(
        candidate / ".git",
        baseline_git,
        symlinks=True,
    )

    tracked = _git(
        baseline_git,
        candidate,
        "ls-files",
        "-z",
    ).stdout.split(b"\0")

    for raw in tracked:
        if not raw:
            continue

        relative = _validate_relative(raw)
        _copy_entry(candidate, relative, baseline_tree)

    return {"tree": baseline_tree, "git": baseline_git}


def _listed_paths(
    baseline_git: Path,
    candidate: Path,
) -> list[str]:
    tracked = _git(
        baseline_git,
        candidate,
        "ls-files",
        "-z",
    ).stdout.split(b"\0")
    untracked = _git(
        baseline_git,
        candidate,
        "ls-files",
        "-z",
        "--others",
        "--exclude-per-directory=.gitignore",
    ).stdout.split(b"\0")
    listed = {
        _validate_relative(raw)
        for raw in [*tracked, *untracked]
        if raw
    }
    collapsed: set[str] = set()

    for relative in listed:
        existing = _existing_entry_or_blocker(candidate, relative)

        if existing is not None:
            collapsed.add(existing)

    return sorted(collapsed)


def _existing_entry_or_blocker(candidate: Path, relative: str) -> str | None:
    """Return an existing entry, collapsing through replaced directories."""

    parts = PurePosixPath(relative).parts
    descriptor = os.open(
        candidate,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    try:
        for index, part in enumerate(parts):
            try:
                metadata = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return None

            if index == len(parts) - 1:
                return PurePosixPath(*parts[: index + 1]).as_posix()

            if not stat.S_ISDIR(metadata.st_mode):
                return PurePosixPath(*parts[: index + 1]).as_posix()

            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    finally:
        os.close(descriptor)

    return None


def _tree_metadata(root: Path) -> dict[str, tuple[str, int, str]]:
    result: dict[str, tuple[str, int, str]] = {}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()

        if stat.S_ISDIR(metadata.st_mode):
            continue

        if stat.S_ISLNK(metadata.st_mode):
            result[relative] = (
                "symlink",
                stat.S_IMODE(metadata.st_mode),
                os.readlink(path),
            )
        elif stat.S_ISREG(metadata.st_mode):
            result[relative] = (
                "file",
                stat.S_IMODE(metadata.st_mode),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        else:
            raise EvidenceError(f"unsupported captured file type: {path}")

    return result


def _git_no_index(
    baseline_tree: Path,
    final_tree: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.safecrlf=false",
            "diff",
            "--no-index",
            "--no-ext-diff",
            "--no-textconv",
            *args,
            "--",
            str(baseline_tree),
            str(final_tree),
        ],
        cwd=baseline_tree.parent,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=120,
    )


def capture_final(
    candidate: Path,
    runtime_dir: Path,
    baseline: dict[str, Path],
) -> dict[str, Any]:
    final_tree = ensure_subdirectory(
        runtime_dir,
        runtime_dir / "final-tree",
        exclusive=True,
    )

    for relative in _listed_paths(baseline["git"], candidate):
        try:
            _copy_entry(candidate, relative, final_tree)
        except FileNotFoundError:
            continue

    before = _tree_metadata(baseline["tree"])
    after = _tree_metadata(final_tree)
    changed_files = sorted(
        relative
        for relative in set(before) | set(after)
        if before.get(relative) != after.get(relative)
    )
    diff = _git_no_index(
        baseline["tree"],
        final_tree,
        "--binary",
        "-M",
    )

    if diff.returncode not in {0, 1}:
        raise EvidenceError(f"candidate diff failed: {diff.stdout}")

    diff_check = _git_no_index(
        baseline["tree"],
        final_tree,
        "--check",
    )

    if diff_check.returncode not in {0, 1}:
        raise EvidenceError(f"candidate diff check failed: {diff_check.stdout}")

    normalized_diff_check = 0 if not diff_check.stdout.strip() else 1
    diff_stat = _git_no_index(
        baseline["tree"],
        final_tree,
        "--stat",
    )
    diff_numstat = _git_no_index(
        baseline["tree"],
        final_tree,
        "--numstat",
    )

    for label, process in (
        ("stat", diff_stat),
        ("numstat", diff_numstat),
    ):
        if process.returncode not in {0, 1}:
            raise EvidenceError(
                f"candidate diff {label} failed: {process.stdout}"
            )

    archive_buffer = io.BytesIO()

    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        archive.add(
            final_tree,
            arcname="candidate",
            recursive=True,
        )

    archive_path = runtime_dir / "candidate-final.tar.gz"
    atomic_write_bytes(
        archive_path,
        archive_buffer.getvalue(),
        root=runtime_dir,
    )
    return {
        "changed_files": changed_files,
        "diff": diff.stdout,
        "diff_check_returncode": normalized_diff_check,
        "diff_check_output": diff_check.stdout,
        "diff_stat": diff_stat.stdout,
        "diff_numstat": diff_numstat.stdout,
        "archive": archive_path,
        "final_tree": final_tree,
    }


def protected_input_integrity(
    candidate: Path,
    expected: dict[str, str],
) -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}

    for relative, expected_digest in expected.items():
        parent_descriptor: int | None = None
        descriptor: int | None = None

        try:
            parent_descriptor, name, metadata = _open_candidate_parent(
                candidate,
                relative,
            )

            if not stat.S_ISREG(metadata.st_mode):
                raise ArtifactSafetyError("not a regular file")

            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)

            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise ArtifactSafetyError(
                    "file changed during integrity verification"
                )

            digest = hashlib.sha256()

            while True:
                chunk = os.read(descriptor, 1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

            observed = digest.hexdigest()
            status = "UNCHANGED" if observed == expected_digest else "MODIFIED"
        except (OSError, ArtifactSafetyError, EvidenceError) as exc:
            observed = None
            status = f"UNSAFE_OR_MISSING: {exc}"
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)

        items[relative] = {
            "expected_sha256": expected_digest,
            "observed_sha256": observed,
            "status": status,
        }

    return {
        "pass": all(item["status"] == "UNCHANGED" for item in items.values()),
        "items": items,
    }
