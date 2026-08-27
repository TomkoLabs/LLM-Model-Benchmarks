from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.processes import terminate_process_group
from harness.sandbox import build_isolated_command


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "security" / "archiveguard-v1"
WORKER = ROOT / "harness" / "evaluator_worker.py"
RPC_SERVER = ROOT / "harness" / "candidate_rpc.py"
SYSTEM_PYTHON = "/usr/bin/python3"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_STRUCTURED_RESULT_BYTES = 4 * 1024 * 1024


@dataclass
class Check:
    name: str
    returncode: int
    passed: int | None
    total: int | None
    output: str
    timed_out: bool = False
    infrastructure_error: str | None = None
    cases: list[dict[str, Any]] = field(default_factory=list)
    visibility: str | None = None
    test_sha256: str | None = None

    @property
    def success(self) -> bool:
        if self.timed_out or self.infrastructure_error:
            return False

        return (
            self.returncode == 0
            and self.passed is not None
            and self.total is not None
            and 0 <= self.passed <= self.total
            and self.passed == self.total
        )


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()

    if remaining <= 0:
        raise subprocess.TimeoutExpired("benchmark evaluator", 0)

    return remaining


def _read_pipe(descriptor: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    retained = 0
    oversized = False

    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)

            if not chunk:
                break

            remaining = MAX_STRUCTURED_RESULT_BYTES - retained

            if remaining > 0:
                kept = chunk[:remaining]
                chunks.append(kept)
                retained += len(kept)

            if len(chunk) > remaining:
                oversized = True
    finally:
        os.close(descriptor)

    return b"".join(chunks), oversized


def _validate_worker_payload(
    payload_bytes: bytes,
    *,
    check: dict[str, Any],
    returncode: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not payload_bytes:
        return [], "missing structured evaluator result"

    try:
        text = payload_bytes.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], f"malformed structured evaluator result: {exc}"

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return [], "invalid structured evaluator result envelope"

    if payload.get("worker_error"):
        if set(payload) != {"schema_version", "worker_error", "traceback"}:
            return [], "invalid evaluator worker failure envelope"
        if not isinstance(payload["worker_error"], str) or not isinstance(
            payload["traceback"], str
        ):
            return [], "invalid evaluator worker failure detail"
        return [], "evaluator worker failed: " + str(payload["worker_error"])

    if set(payload) != {
        "schema_version",
        "discovered",
        "successful",
        "cases",
    }:
        return [], "structured evaluator result has missing or unknown fields"

    cases = payload.get("cases")
    discovered = payload.get("discovered")
    successful = payload.get("successful")

    if (
        not isinstance(cases, list)
        or type(discovered) is not int
        or discovered < 0
        or type(successful) is not bool
    ):
        return [], "structured evaluator result has invalid case data"

    expected = check.get("cases")

    if not isinstance(expected, dict) or not expected:
        return [], "evaluator check has no harness-owned case mapping"

    normalized: list[dict[str, Any]] = []
    identifiers: list[str] = []

    for item in cases:
        if not isinstance(item, dict):
            return [], "structured evaluator case is not an object"

        if set(item) not in ({"id", "status"}, {"id", "status", "detail"}):
            return [], "structured evaluator case has missing or unknown fields"

        identifier = item.get("id")
        status = item.get("status")
        detail = item.get("detail")

        if not isinstance(identifier, str) or status not in {
            "PASS",
            "FAIL",
            "ERROR",
        }:
            return [], "structured evaluator case has invalid identity/status"
        if detail is not None and not isinstance(detail, str):
            return [], "structured evaluator case has invalid detail"

        identifiers.append(identifier)
        normalized.append(
            {
                "id": identifier,
                "status": status,
                "detail": item.get("detail"),
                "categories": list(expected[identifier]["categories"])
                if identifier in expected
                else [],
                "requirements": list(expected[identifier]["requirements"])
                if identifier in expected
                else [],
            }
        )

    if len(identifiers) != len(set(identifiers)):
        return [], "duplicate structured evaluator case identity"

    expected_ids = set(expected)

    if set(identifiers) != expected_ids or discovered != len(expected_ids):
        return [], (
            "structured evaluator identities/count do not match the frozen "
            "case mapping"
        )

    all_passed = all(item["status"] == "PASS" for item in normalized)

    if successful != all_passed:
        return [], "structured evaluator success flag conflicts with cases"

    if (returncode == 0) != all_passed or returncode not in {0, 1}:
        return [], "evaluator command status conflicts with structured cases"

    return normalized, None


def _check(
    check: dict[str, Any],
    *,
    candidate: Path,
    task_dir: Path,
    deadline: float,
) -> Check:
    name = check["id"]
    test_path = (task_dir / check["test_file"]).absolute()
    expected_sha256 = check["sha256"]

    try:
        actual_sha256 = hashlib.sha256(test_path.read_bytes()).hexdigest()
    except OSError as exc:
        return Check(
            name,
            70,
            None,
            None,
            "",
            infrastructure_error=f"cannot read immutable test input: {exc}",
        )

    if actual_sha256 != expected_sha256:
        return Check(
            name,
            70,
            None,
            None,
            "",
            infrastructure_error="immutable test input digest mismatch",
            test_sha256=actual_sha256,
        )

    read_descriptor, write_descriptor = os.pipe()
    harness_inputs = {
        path: str(path)
        for path in (
            ROOT / "harness" / "__init__.py",
            ROOT / "harness" / "artifacts.py",
            ROOT / "harness" / "processes.py",
            ROOT / "harness" / "sandbox.py",
            RPC_SERVER,
            WORKER,
        )
    }
    harness_inputs[test_path] = f"/tests/{name}.py"
    command = build_isolated_command(
        candidate,
        [
            SYSTEM_PYTHON,
            str(WORKER),
            f"/tests/{name}.py",
            str(candidate),
            json.dumps(sorted(check["cases"])),
        ],
        candidate_writable=False,
        read_only_files=harness_inputs,
        result_fd=write_descriptor,
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env={},
            start_new_session=True,
            pass_fds=(write_descriptor,),
        )
    except BaseException:
        os.close(read_descriptor)
        os.close(write_descriptor)
        raise

    os.close(write_descriptor)
    pipe_result: dict[str, Any] = {}

    def consume_result() -> None:
        try:
            pipe_result["value"] = _read_pipe(read_descriptor)
        except BaseException as exc:
            pipe_result["error"] = exc

    result_reader = threading.Thread(target=consume_result, daemon=True)
    result_reader.start()
    timed_out = False

    try:
        output, _ = process.communicate(timeout=_remaining(deadline))
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process)
        output, _ = process.communicate()
    except BaseException:
        terminate_process_group(process)
        result_reader.join(timeout=5)
        raise

    result_reader.join(timeout=5)

    if result_reader.is_alive():
        return Check(
            name,
            70,
            None,
            None,
            output,
            infrastructure_error="structured evaluator result pipe remained open",
            visibility=check["visibility"],
            test_sha256=actual_sha256,
        )

    if "error" in pipe_result:
        return Check(
            name,
            70,
            None,
            None,
            output,
            infrastructure_error=(
                "cannot read structured evaluator result: "
                f"{pipe_result['error']}"
            ),
            visibility=check["visibility"],
            test_sha256=actual_sha256,
        )

    payload, oversized = pipe_result.get("value", (b"", False))

    if oversized:
        return Check(
            name,
            70,
            None,
            None,
            output,
            infrastructure_error="structured evaluator result exceeds size limit",
            visibility=check["visibility"],
            test_sha256=actual_sha256,
        )

    if timed_out:
        return Check(
            name,
            124,
            None,
            None,
            output,
            timed_out=True,
            visibility=check["visibility"],
            test_sha256=actual_sha256,
        )

    cases, infrastructure_error = _validate_worker_payload(
        payload,
        check=check,
        returncode=process.returncode,
    )
    passed = sum(item["status"] == "PASS" for item in cases)

    return Check(
        name,
        process.returncode,
        passed if not infrastructure_error else None,
        len(cases) if not infrastructure_error else None,
        output,
        infrastructure_error=infrastructure_error,
        cases=cases,
        visibility=check["visibility"],
        test_sha256=actual_sha256,
    )


def evaluate_task(
    candidate: Path,
    task: dict[str, Any],
    task_dir: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    expected_worker_sha256: str | None = None,
    expected_rpc_sha256: str | None = None,
) -> dict[str, Any]:
    candidate = candidate.absolute()

    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"candidate does not exist or is unsafe: {candidate}")

    if timeout < 1:
        raise ValueError("evaluator timeout must be positive")

    worker_sha256 = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    rpc_sha256 = hashlib.sha256(RPC_SERVER.read_bytes()).hexdigest()

    if (
        expected_worker_sha256 is not None
        and worker_sha256 != expected_worker_sha256
    ):
        raise ValueError("frozen evaluator worker digest mismatch")
    if expected_rpc_sha256 is not None and rpc_sha256 != expected_rpc_sha256:
        raise ValueError("frozen candidate RPC digest mismatch")

    configured_checks = task.get("evaluation_checks")

    if not isinstance(configured_checks, list) or not configured_checks:
        raise ValueError(
            f"task has no structured evaluator checks: {task.get('id', 'unknown')}"
        )

    deadline = time.monotonic() + timeout
    checks: dict[str, Check] = {}

    for configured in configured_checks:
        name = configured["id"]

        if name in checks:
            raise ValueError(f"duplicate evaluator check: {name}")

        try:
            checks[name] = _check(
                configured,
                candidate=candidate,
                task_dir=task_dir,
                deadline=deadline,
            )
        except subprocess.TimeoutExpired:
            checks[name] = Check(
                name,
                124,
                None,
                None,
                "evaluator deadline expired before command launch",
                timed_out=True,
                visibility=configured["visibility"],
            )

    infrastructure_errors = [
        f"{name}: {check.infrastructure_error}"
        for name, check in checks.items()
        if check.infrastructure_error
    ]
    timed_out = any(check.timed_out for check in checks.values())
    hard_failures: list[str] = []
    security_cases = [
        case
        for check in checks.values()
        for case in check.cases
        if "security" in case["categories"]
    ]

    if security_cases and any(
        case["status"] != "PASS" for case in security_cases
    ):
        hard_failures.append("SECURITY_FAILURE")

    result: dict[str, Any] = {
        "candidate": str(candidate),
        "hard_failures": hard_failures,
        "pass": (
            not infrastructure_errors
            and not timed_out
            and all(check.success for check in checks.values())
        ),
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "infrastructure_errors": infrastructure_errors,
        "isolation": {
            "architecture": "networkless-trusted-worker-and-nested-candidate-rpc",
            "candidate_mount": "read-only",
            "test_mounts": "exact-files-read-only",
            "candidate_test_visibility": "none",
            "structured_result_channel": "trusted-worker-only",
            "network": "unshared",
            "environment": "cleared-allowlist",
            "writable": ["/tmp"],
            "worker_sha256": worker_sha256,
            "candidate_rpc_sha256": rpc_sha256,
            "worker_digest_status": (
                "VERIFIED"
                if expected_worker_sha256 is not None
                and expected_rpc_sha256 is not None
                else "UNVERIFIED_LOW_LEVEL"
            ),
        },
    }
    result.update({name: asdict(check) for name, check in checks.items()})
    return result


def evaluate(
    candidate: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    from harness.workspace import load_task

    task, task_dir = load_task("archiveguard-security-v1")
    return evaluate_task(candidate, task, task_dir, timeout=timeout)


def show(result: dict[str, Any]) -> None:
    print(f"candidate={result['candidate']}")

    for name, item in result.items():
        if not isinstance(item, dict) or "returncode" not in item:
            continue

        score = (
            f"{item['passed']}/{item['total']}"
            if item["passed"] is not None and item["total"] is not None
            else "unavailable"
        )
        print(f"{name}={score} returncode={item['returncode']}")

    failures = result["hard_failures"]
    print("hard_failures=" + (",".join(failures) if failures else "NONE"))
    print("overall=" + ("PASS" if result["pass"] else "FAIL"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    result = evaluate(args.candidate)
    show(result)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
