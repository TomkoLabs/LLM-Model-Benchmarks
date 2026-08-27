import tempfile
import unittest
from pathlib import Path

from archiveguard import (
    ExtractionPolicy,
    extract_member,
    normalize_archive_member,
)


class PublicTests(unittest.TestCase):
    def test_basic_relative_path(self) -> None:
        self.assertEqual(
            normalize_archive_member("dir/file.txt"),
            "dir/file.txt",
        )

    def test_backslash_normalization(self) -> None:
        self.assertEqual(
            normalize_archive_member(r"dir\file.txt"),
            "dir/file.txt",
        )

    def test_parent_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_archive_member("../secret.txt")

    def test_extract_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = extract_member(
                directory,
                "dir/file.txt",
                b"contents",
            )

            self.assertEqual(
                destination.read_bytes(),
                b"contents",
            )

    def test_existing_file_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "file.txt"
            target.write_bytes(b"original")

            with self.assertRaises(FileExistsError):
                extract_member(
                    root,
                    "file.txt",
                    b"replacement",
                    policy=ExtractionPolicy(
                        allow_overwrite=False,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
