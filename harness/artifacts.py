from __future__ import annotations

import errno
import os
import stat
import uuid
from pathlib import Path, PurePosixPath


class ArtifactSafetyError(RuntimeError):
    """Raised when a harness-owned artifact path is not trustworthy."""


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _absolute(path: Path) -> Path:
    value = Path(os.path.abspath(path))

    if not value.is_absolute():
        raise ArtifactSafetyError(f"artifact path is not absolute: {path}")

    return value


def _parts(path: Path) -> tuple[str, ...]:
    absolute = _absolute(path)
    return tuple(part for part in absolute.parts if part != absolute.anchor)


def _open_absolute_directory(path: Path) -> int:
    descriptor = os.open("/", _DIRECTORY_FLAGS)

    try:
        for part in _parts(path):
            next_descriptor = os.open(
                part,
                _DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ArtifactSafetyError(
            f"unsafe or missing artifact directory: {_absolute(path)}: {exc}"
        ) from exc

    return descriptor


def ensure_root(path: Path, *, mode: int = 0o700) -> Path:
    """Create one root leaf without following any parent symlink."""

    absolute = _absolute(path)
    parent = absolute.parent
    parent_descriptor = _open_absolute_directory(parent)

    try:
        try:
            os.mkdir(absolute.name, mode=mode, dir_fd=parent_descriptor)
        except FileExistsError:
            pass

        descriptor = os.open(
            absolute.name,
            _DIRECTORY_FLAGS,
            dir_fd=parent_descriptor,
        )
        os.close(descriptor)
    except OSError as exc:
        raise ArtifactSafetyError(
            f"unsafe artifact root: {absolute}: {exc}"
        ) from exc
    finally:
        os.close(parent_descriptor)

    return absolute


def _relative_parts(root: Path, path: Path) -> tuple[str, ...]:
    absolute_root = _absolute(root)
    absolute_path = _absolute(path)

    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise ArtifactSafetyError(
            f"artifact path escapes trusted root {absolute_root}: {absolute_path}"
        ) from exc

    parts = PurePosixPath(relative.as_posix()).parts

    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactSafetyError(f"invalid artifact path: {absolute_path}")

    return tuple(parts)


def ensure_subdirectory(
    root: Path,
    path: Path,
    *,
    mode: int = 0o700,
    exclusive: bool = False,
) -> Path:
    """Create a directory below a trusted root using no-follow dirfds."""

    absolute_root = _absolute(root)
    absolute_path = _absolute(path)
    parts = _relative_parts(absolute_root, absolute_path)
    descriptor = _open_absolute_directory(absolute_root)

    try:
        for index, part in enumerate(parts):
            is_leaf = index == len(parts) - 1

            try:
                os.mkdir(part, mode=mode, dir_fd=descriptor)
            except FileExistsError as exc:
                if exclusive and is_leaf:
                    raise ArtifactSafetyError(
                        f"artifact directory already exists: {absolute_path}"
                    ) from exc

            next_descriptor = os.open(
                part,
                _DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        raise ArtifactSafetyError(
            f"unsafe artifact directory: {absolute_path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)

    return absolute_path


def _open_parent(root: Path, path: Path) -> tuple[int, str]:
    parts = _relative_parts(root, path)

    if not parts:
        raise ArtifactSafetyError("artifact file cannot be the trusted root")

    parent = _absolute(root).joinpath(*parts[:-1])
    ensure_subdirectory(root, parent)
    return _open_absolute_directory(parent), parts[-1]


def atomic_write_bytes(path: Path, data: bytes, *, root: Path) -> None:
    """Atomically replace a regular harness file without following symlinks."""

    parent_descriptor, name = _open_parent(root, path)
    temporary = f".{name}.{uuid.uuid4().hex}.tmp"
    temporary_descriptor: int | None = None

    try:
        try:
            existing = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None

        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ArtifactSafetyError(
                f"refusing to replace non-regular artifact: {_absolute(path)}"
            )

        temporary_descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )

        view = memoryview(data)

        while view:
            written = os.write(temporary_descriptor, view)
            view = view[written:]

        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            detail = "symlink encountered"
        else:
            detail = str(exc)

        raise ArtifactSafetyError(
            f"cannot write artifact {_absolute(path)}: {detail}"
        ) from exc
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)

        try:
            os.unlink(temporary, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass

        os.close(parent_descriptor)


def atomic_write_text(path: Path, value: str, *, root: Path) -> None:
    atomic_write_bytes(path, value.encode("utf-8"), root=root)


def read_regular_bytes(path: Path, *, root: Path) -> bytes:
    parent_descriptor, name = _open_parent(root, path)

    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise ArtifactSafetyError(
            f"cannot safely open artifact {_absolute(path)}: {exc}"
        ) from exc

    try:
        metadata = os.fstat(descriptor)

        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactSafetyError(
                f"artifact is not a regular file: {_absolute(path)}"
            )

        chunks: list[bytes] = []

        while True:
            chunk = os.read(descriptor, 1024 * 1024)

            if not chunk:
                break

            chunks.append(chunk)

        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
