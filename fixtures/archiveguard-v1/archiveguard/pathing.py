from urllib.parse import unquote


def normalize_archive_member(
    raw: str,
    *,
    max_length: int = 255,
) -> str:
    """Normalize an archive-member path before extraction."""

    decoded = unquote(raw)
    decoded = decoded.replace("\\", "/")

    parts = [
        part
        for part in decoded.split("/")
        if part not in ("", ".")
    ]

    if any(part == ".." for part in parts):
        raise ValueError("parent traversal is not allowed")

    return "/".join(parts)
