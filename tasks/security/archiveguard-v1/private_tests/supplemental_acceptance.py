#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import re
import unittest
from pathlib import Path
from typing import Callable

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


def expect_equal(actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, received {actual!r}")


def expect_error(
    exception: type[BaseException],
    function: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> None:
    try:
        function(*args, **kwargs)
    except exception:
        return
    except Exception as exc:
        raise AssertionError(
            f"expected {exception.__name__}, "
            f"received {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"expected {exception.__name__}, no exception was raised"
    )


test(
    "one Unicode character fits max_length=1",
    lambda: expect_equal(
        normalize_archive_member("é", max_length=1),
        "é",
    ),
)

test(
    "two Unicode characters exceed max_length=1",
    lambda: expect_error(
        ValueError,
        normalize_archive_member,
        "éé",
        max_length=1,
    ),
)

test(
    "boolean max_length rejected",
    lambda: expect_error(
        ValueError,
        normalize_archive_member,
        "file",
        max_length=True,
    ),
)

test(
    "encoded Windows drive path rejected",
    lambda: expect_error(
        ValueError,
        normalize_archive_member,
        "C%3Atemp/file",
    ),
)

test(
    "double-encoded traversal remains singly decoded",
    lambda: expect_equal(
        normalize_archive_member("%252e%252e/file"),
        "%2e%2e/file",
    ),
)


def final_symlink_destination() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        root = temporary / "root"
        outside = temporary / "outside.txt"

        root.mkdir()
        outside.write_bytes(b"original")
        (root / "file.txt").symlink_to(outside)

        expect_error(
            ValueError,
            destination_for,
            root,
            "file.txt",
        )


test(
    "final-component symlink escape rejected",
    final_symlink_destination,
)


def final_symlink_write() -> None:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        root = temporary / "root"
        outside = temporary / "outside.txt"

        root.mkdir()
        outside.write_bytes(b"original")
        (root / "file.txt").symlink_to(outside)

        expect_error(
            ValueError,
            extract_member,
            root,
            "file.txt",
            b"replacement",
            policy=ExtractionPolicy(
                allow_overwrite=True,
            ),
        )

        expect_equal(outside.read_bytes(), b"original")


test(
    "final-component symlink cannot be overwritten",
    final_symlink_write,
)


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    case_type = type("ArchiveGuardSupplemental", (unittest.TestCase,), {})

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
