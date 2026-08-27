from pathlib import Path

from .pathing import normalize_archive_member
from .policy import ExtractionPolicy


def destination_for(
    root: str | Path,
    member_name: str,
    *,
    policy: ExtractionPolicy | None = None,
) -> Path:
    if policy is None:
        policy = ExtractionPolicy()

    normalized = normalize_archive_member(member_name)
    return Path(root) / normalized


def extract_member(
    root: str | Path,
    member_name: str,
    data: bytes,
    *,
    policy: ExtractionPolicy | None = None,
) -> Path:
    if policy is None:
        policy = ExtractionPolicy()

    destination = destination_for(
        root,
        member_name,
        policy=policy,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not policy.allow_overwrite:
        raise FileExistsError(destination)

    destination.write_bytes(data)
    return destination
