#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from harness.workspace import prepare


with tempfile.TemporaryDirectory(
    prefix="hermesbench-workspace-"
) as tmp:
    work = Path(tmp)

    first = prepare(
        "archiveguard-security-v1",
        work,
    )

    second = prepare(
        "archiveguard-security-v1",
        work,
    )

    assert (
        first["content_sha256"]
        == second["content_sha256"]
    )

    assert (
        first["baseline_commit"]
        == second["baseline_commit"]
    )

    for metadata in (first, second):
        candidate = Path(metadata["candidate"])

        assert (candidate / ".git").is_dir()
        assert (
            candidate / "BENCHMARK_TASK.md"
        ).is_file()

        assert (
            candidate / "tests" / "test_public.py"
        ).is_file()

        assert not list(
            candidate.rglob("hidden_acceptance.py")
        )

        assert not list(
            candidate.rglob(
                "supplemental_acceptance.py"
            )
        )

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=candidate,
            text=True,
        )

        assert status == ""

    print("WORKSPACE_VALIDATION=PASS")
    print(
        "content_sha256="
        + first["content_sha256"]
    )
    print(
        "baseline_commit="
        + first["baseline_commit"]
    )
