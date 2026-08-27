from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import yaml

from harness.artifacts import atomic_write_text, ensure_root


ROOT = Path(__file__).resolve().parents[1]


def run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if proc.returncode:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n"
            f"{proc.stdout}"
        )

    return proc.stdout.strip()


def discover_tasks(
    root: Path = ROOT,
) -> list[tuple[dict, Path]]:
    tasks: list[tuple[dict, Path]] = []

    for manifest in sorted(
        root.glob("tasks/*/*/task.yaml")
    ):
        with manifest.open("r", encoding="utf-8") as handle:
            task = yaml.safe_load(handle) or {}

        tasks.append((task, manifest.parent))

    return tasks


def load_task(task_id: str) -> tuple[dict, Path]:
    for task, task_dir in discover_tasks():
        if task.get("id") == task_id:
            return task, task_dir

    raise ValueError(f"unknown task: {task_id}")


def content_digest(path: Path) -> str:
    digest = hashlib.sha256()

    ignored_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
    }

    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part in ignored_parts for part in item.parts)
        and item.suffix != ".pyc"
    )

    for item in files:
        rel = item.relative_to(path).as_posix().encode("utf-8")
        data = item.read_bytes()

        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    return digest.hexdigest()


def new_run_id(task_id: str) -> str:
    return f"{task_id}-{uuid.uuid4().hex[:12]}"


def prepare(
    task_id: str,
    work_root: Path,
    *,
    run_id: str | None = None,
) -> dict:
    task, task_dir = load_task(task_id)

    fixture = ROOT / task["fixture"]

    if not fixture.is_dir():
        raise ValueError(f"fixture does not exist: {fixture}")

    work_root = ensure_root(work_root)
    run_id = run_id or new_run_id(task_id)

    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError(f"invalid candidate run id: {run_id!r}")

    candidate = work_root / run_id

    if candidate.exists():
        raise RuntimeError(
            f"candidate path already exists: {candidate}"
        )

    shutil.copytree(
        fixture,
        candidate,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
        ),
    )

    shutil.copy2(
        task_dir / "TASK.md",
        candidate / "BENCHMARK_TASK.md",
    )

    forbidden_names = {
        "hidden_acceptance.py",
        "supplemental_acceptance.py",
        "private_tests",
    }

    leaked = [
        str(item.relative_to(candidate))
        for item in candidate.rglob("*")
        if any(part in forbidden_names for part in item.parts)
    ]

    if leaked:
        shutil.rmtree(candidate)
        raise RuntimeError(
            "private evaluator leaked into candidate: "
            + ", ".join(leaked)
        )

    digest = content_digest(candidate)

    run(["git", "init", "-b", "main"], cwd=candidate)

    run(
        ["git", "config", "user.name", "HermesBench Baseline"],
        cwd=candidate,
    )
    run(
        ["git", "config", "user.email", "bench@local.invalid"],
        cwd=candidate,
    )
    run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=candidate,
    )

    run(["git", "add", "."], cwd=candidate)

    commit_env = dict(os.environ)
    commit_env.update(
        {
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
        }
    )

    run(
        ["git", "commit", "-m", "Benchmark baseline"],
        cwd=candidate,
        env=commit_env,
    )

    baseline_commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=candidate,
    )

    protected_paths = ["BENCHMARK_TASK.md"]

    for check in task.get("evaluation_checks", []):
        if check.get("visibility") != "public":
            continue

        relative = check.get("candidate_path")

        if not isinstance(relative, str) or not relative:
            shutil.rmtree(candidate)
            raise RuntimeError(
                f"public evaluator input has no candidate path: {check.get('id')}"
            )

        protected_paths.append(relative)

    protected_paths = list(dict.fromkeys(protected_paths))
    protected_inputs: dict[str, str] = {}

    for relative in protected_paths:
        protected = (candidate / relative).absolute()

        if (
            not protected.is_relative_to(candidate.absolute())
            or protected.is_symlink()
            or not protected.is_file()
        ):
            shutil.rmtree(candidate)
            raise RuntimeError(
                f"protected candidate input is missing or unsafe: {relative}"
            )

        protected_inputs[relative] = hashlib.sha256(
            protected.read_bytes()
        ).hexdigest()

    metadata = {
        "run_id": run_id,
        "task_id": task["id"],
        "task_version": task["version"],
        "candidate": str(candidate),
        "content_sha256": digest,
        "baseline_commit": baseline_commit,
        "protected_inputs": protected_inputs,
    }

    metadata_path = (
        work_root / f"{run_id}.metadata.json"
    )

    atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2) + "\n",
        root=work_root,
    )

    metadata["metadata_path"] = str(metadata_path)

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=ROOT / "work",
    )
    parser.add_argument(
        "--json",
        action="store_true",
    )
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    ensure_root(work_root)

    metadata = prepare(args.task, work_root)

    if args.json:
        print(json.dumps(metadata, indent=2))
    else:
        for key, value in metadata.items():
            print(f"{key}={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
