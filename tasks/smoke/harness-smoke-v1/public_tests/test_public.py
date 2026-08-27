from __future__ import annotations

import unittest
from pathlib import Path

import smoke_project


class SmokeAcceptance(unittest.TestCase):
    def test_status_is_ready(self) -> None:
        self.assertEqual(smoke_project.STATUS, "READY")

    def test_notes_file_was_created(self) -> None:
        self.assertEqual(
            Path("SMOKE_NOTES.txt").read_text(encoding="utf-8"),
            "Hermes smoke completed.\n",
        )


if __name__ == "__main__":
    unittest.main()
