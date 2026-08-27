from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from harness.artifacts import atomic_write_text
from harness.benchmark_model import (
    InfrastructureError,
    _exclusive_endpoint_lock,
    _model_result_metadata,
    _verify_control_plane,
    load_execution_plan,
    preflight_model,
)
from harness.hermes_runner import RUNTIME_ROOT, run_once
from harness.workspace import load_task


SMOKE_TASK = "harness-smoke-v1"


def run_smoke(model_alias: str) -> dict:
    plan = load_execution_plan(model_alias)
    smoke_task, _ = load_task(SMOKE_TASK)

    if smoke_task.get("benchmark_role") != "non-scored-smoke":
        raise InfrastructureError("smoke task is not marked non-scored")

    _verify_control_plane(plan, allowed_artifacts=())

    with _exclusive_endpoint_lock(RUNTIME_ROOT, plan["endpoint"]):
        identity = preflight_model(
            plan["endpoint"],
            plan["model"]["runtime_model"],
            expected_digest=plan["model"].get("runtime_digest"),
        )
        resolved_digest = identity["runtime_model_digest"]
        result = run_once(
            task_id=SMOKE_TASK,
            model=plan["model"]["runtime_model"],
            reasoning=plan["model"]["reasoning_effort"],
            max_turns=smoke_task["limits"]["agent_turns"],
            base_url=plan["endpoint"],
            model_alias=model_alias,
            model_metadata={
                **_model_result_metadata(plan),
                "runtime_digest": resolved_digest,
                "runtime_identity_status": "VERIFIED",
            },
            benchmark_metadata={
                **plan["benchmark"],
                "runtime_identity": identity,
            },
            evaluator_timeout=60,
        )
        _verify_control_plane(plan, allowed_artifacts=())
        identity_after = preflight_model(
            plan["endpoint"],
            plan["model"]["runtime_model"],
            expected_digest=resolved_digest,
        )

    if identity_after["runtime_model_digest"] != resolved_digest:
        raise InfrastructureError("runtime digest changed during smoke run")

    runtime = Path(result["paths"]["runtime"])
    changed = set(result["metrics"]["candidate_git"]["changed_files"])
    required_changes = {"smoke_project.py", "SMOKE_NOTES.txt"}
    checks = {
        "non_scored": result["task"].get("benchmark_role")
        == "non-scored-smoke",
        "outcome_pass": result["outcome"] == "PASS",
        "infrastructure_ok": bool(result["infrastructure_ok"]),
        "evaluation_pass": bool(result["evaluation"].get("pass")),
        "required_changes_captured": required_changes <= changed,
        "runtime_digest_recorded": result["model"].get("runtime_digest")
        == resolved_digest,
        "runtime_digest_stable": identity_after["runtime_model_digest"]
        == resolved_digest,
    }
    summary = {
        "schema_version": 1,
        "benchmark_role": "non-scored-smoke",
        "run_id": result["run_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "runtime_digest": resolved_digest,
        "result": str(runtime / "result.json"),
        "diagnostics": result["artifacts"]["diagnostics"],
        "candidate_archive": result["artifacts"]["candidate_archive"],
    }
    summary_path = runtime / "smoke-qualification.json"
    atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        root=runtime,
    )
    summary["summary"] = str(summary_path)

    if not summary["passed"]:
        raise InfrastructureError(
            "non-scored smoke checks failed: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded, non-scored production-path Hermes smoke."
    )
    parser.add_argument("--model", required=True, help="model alias from models.yaml")
    args = parser.parse_args(argv)

    try:
        summary = run_smoke(args.model)
    except Exception as exc:
        print(f"smoke_error={type(exc).__name__}: {exc}")
        return 3

    print("smoke=PASS")
    print(f"run_id={summary['run_id']}")
    print(f"runtime_digest={summary['runtime_digest']}")
    print(f"result={summary['result']}")
    print(f"diagnostics={summary['diagnostics']}")
    print(f"summary={summary['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
