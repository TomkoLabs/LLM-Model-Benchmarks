#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import re
import unittest
from pathlib import Path
from typing import Any, Callable

from archiveguard import (
    ExtractionPolicy,
    destination_for,
    extract_member,
    normalize_archive_member,
)


CASES: list[tuple[str, str, Callable[[], None]]] = []


def test(name: str, function: Callable[[], None]) -> None:
    case_id = "test_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    CASES.append((case_id, name, function))


def expect_equal(actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(
            f"expected {expected!r}, received {actual!r}"
        )


def expect_raises(
    exception_type: type[BaseException],
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    try:
        function(*args, **kwargs)
    except exception_type:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception_type.__name__}, "
            f"received {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"expected {exception_type.__name__}, "
        "but no exception was raised"
    )


test(
    "basic path",
    lambda: expect_equal(
        normalize_archive_member("a/b.txt"),
        "a/b.txt",
    ),
)

test(
    "collapse separators",
    lambda: expect_equal(
        normalize_archive_member("a///b"),
        "a/b",
    ),
)

test(
    "remove dot components",
    lambda: expect_equal(
        normalize_archive_member("./a/./b"),
        "a/b",
    ),
)

test(
    "backslash normalization",
    lambda: expect_equal(
        normalize_archive_member(r"a\b"),
        "a/b",
    ),
)

test(
    "single percent decode",
    lambda: expect_equal(
        normalize_archive_member("a/%41"),
        "a/A",
    ),
)

test(
    "decode exactly once",
    lambda: expect_equal(
        normalize_archive_member("%252e%252e/file"),
        "%2e%2e/file",
    ),
)

test(
    "preserve case",
    lambda: expect_equal(
        normalize_archive_member("Dir/File.TXT"),
        "Dir/File.TXT",
    ),
)

test(
    "preserve decoded spaces",
    lambda: expect_equal(
        normalize_archive_member("space%20name/file"),
        "space name/file",
    ),
)

test(
    "non-string raw",
    lambda: expect_raises(
        TypeError,
        normalize_archive_member,
        123,
    ),
)

test(
    "zero max length",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a",
        max_length=0,
    ),
)

test(
    "negative max length",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a",
        max_length=-1,
    ),
)

test(
    "boolean max length",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a",
        max_length=True,
    ),
)

test(
    "empty path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "",
    ),
)

test(
    "dot-only path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        ".",
    ),
)

test(
    "POSIX absolute path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "/etc/passwd",
    ),
)

test(
    "encoded absolute path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "%2fetc/passwd",
    ),
)

test(
    "UNC path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        r"\\server\share",
    ),
)

test(
    "encoded UNC path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "%5c%5cserver%5cshare",
    ),
)

test(
    "drive absolute path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        r"C:\temp\x",
    ),
)

test(
    "drive-relative path",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "C:temp/file",
    ),
)

test(
    "literal traversal",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "../x",
    ),
)

test(
    "encoded traversal",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a/%2e%2e/b",
    ),
)

test(
    "literal NUL",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a\0b",
    ),
)

test(
    "encoded NUL",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a%00b",
    ),
)

test(
    "malformed percent at end",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a%",
    ),
)

test(
    "short percent escape",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a%2",
    ),
)

test(
    "non-hex percent escape",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "a%ZZ",
    ),
)

test(
    "normalized length",
    lambda: expect_raises(
        ValueError,
        normalize_archive_member,
        "abcdef",
        max_length=5,
    ),
)


def test_policy_length() -> None:
    with tempfile.TemporaryDirectory() as directory:
        expect_raises(
            ValueError,
            destination_for,
            directory,
            "abcdef",
            policy=ExtractionPolicy(
                max_member_length=5,
            ),
        )


test(
    "policy max length reaches destination_for",
    test_policy_length,
)


def test_destination_inside_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "root"
        root.mkdir()

        destination = destination_for(
            root,
            "dir/file.txt",
        )

        expect_equal(
            destination,
            root / "dir/file.txt",
        )


test(
    "valid destination remains under root",
    test_destination_inside_root,
)


def test_symlink_escape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        root = temporary / "root"
        outside = temporary / "outside"

        root.mkdir()
        outside.mkdir()

        (root / "link").symlink_to(
            outside,
            target_is_directory=True,
        )

        expect_raises(
            ValueError,
            destination_for,
            root,
            "link/escaped.txt",
        )


test(
    "existing symlink cannot escape root",
    test_symlink_escape,
)


def test_extract_valid_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        destination = extract_member(
            root,
            "dir/file.txt",
            b"payload",
        )

        expect_equal(
            destination.read_bytes(),
            b"payload",
        )

        expect_equal(
            destination.resolve(),
            (root / "dir/file.txt").resolve(),
        )


test(
    "valid extraction",
    test_extract_valid_file,
)


def test_overwrite_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        destination = root / "file.txt"
        destination.write_bytes(b"original")

        expect_raises(
            FileExistsError,
            extract_member,
            root,
            "file.txt",
            b"replacement",
            policy=ExtractionPolicy(
                allow_overwrite=False,
            ),
        )

        expect_equal(
            destination.read_bytes(),
            b"original",
        )


test(
    "overwrite rejection",
    test_overwrite_rejected,
)


def test_overwrite_allowed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        destination = root / "file.txt"
        destination.write_bytes(b"original")

        extract_member(
            root,
            "file.txt",
            b"replacement",
            policy=ExtractionPolicy(
                allow_overwrite=True,
            ),
        )

        expect_equal(
            destination.read_bytes(),
            b"replacement",
        )


test(
    "overwrite allowed",
    test_overwrite_allowed,
)


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    case_type = type("ArchiveGuardAcceptance", (unittest.TestCase,), {})

    for case_id, name, function in CASES:
        def run_case(
            self: unittest.TestCase,
            function: Callable[[], None] = function,
        ) -> None:
            function()

        run_case.__name__ = case_id
        run_case.__doc__ = name
        setattr(run_case, "hermesbench_case_id", case_id)
        setattr(case_type, case_id, run_case)

    return loader.loadTestsFromTestCase(case_type)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1).run(load_tests(
        unittest.defaultTestLoader,
        unittest.TestSuite(),
        None,
    ))
    raise SystemExit(0 if result.wasSuccessful() else 1)
