#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

from harness.hermes_runner import (
    HERMES_TOOLSET,
    RUNTIME_ROOT,
    WORK_ROOT,
    cleanup,
    run_once,
)

WORK = WORK_ROOT
RUNTIME = RUNTIME_ROOT

WORK.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

preexisting_work = {
    item.name for item in WORK.iterdir()
}
preexisting_runtime = {
    item.name for item in RUNTIME.iterdir()
}


result = run_once(
    task_id="archiveguard-security-v1",
    model="dry-run-control",
    reasoning="medium",
    max_turns=150,
    dry_run=True,
    model_alias="dry-run-control",
    model_metadata={
        "context_length": 131072,
    },
)

errors: list[str] = []

if result["sandbox_exit_code"] != 0:
    errors.append(
        "sandbox exit code was nonzero"
    )

if not result["trajectory_present"]:
    errors.append(
        "dry-run trajectory missing"
    )

if not result["infrastructure_ok"]:
    errors.append(
        "dry-run infrastructure unexpectedly failed: "
        + ",".join(result["infra_reasons"])
    )

if result["infra_reasons"]:
    errors.append(
        "dry-run recorded infrastructure failure reasons"
    )

context = result["metrics"].get("context_enforcement") or {}

if context.get("context_length") != 131072:
    errors.append("run-local context limit was not captured")

runtime_config = Path(result["artifacts"]["runtime_hermes_config"])

if not runtime_config.is_file():
    errors.append("run-local context config missing")
elif "context_length: 131072" not in runtime_config.read_text(encoding="utf-8"):
    errors.append("run-local context config did not freeze the context")

if (
    result["baseline_commit"]
    != "0cda29d0492cde328e20667bcae11a6968f2c53d"
):
    errors.append(
        "baseline commit changed"
    )

if result["git"]["changed_files"]:
    errors.append(
        "dry run changed candidate files"
    )

if result["git"]["diff_check_rc"] != 0:
    errors.append(
        "git diff --check failed"
    )

evaluation = result["evaluation"]

if evaluation["pass"]:
    errors.append(
        "pristine baseline unexpectedly passed"
    )

if (
    evaluation["public"]["passed"],
    evaluation["public"]["total"],
) != (5, 5):
    errors.append(
        "unexpected pristine public score"
    )

if (
    evaluation["hidden"]["passed"],
    evaluation["hidden"]["total"],
) != (15, 34):
    errors.append(
        "unexpected pristine hidden score"
    )

if (
    evaluation["supplemental"]["passed"],
    evaluation["supplemental"]["total"],
) != (2, 7):
    errors.append(
        "unexpected pristine supplemental score"
    )

if result["false_completion"]:
    errors.append(
        "dry-run control was classified as false completion"
    )

diagnostics = json.loads(
    Path(result["artifacts"]["diagnostics"]).read_text(
        encoding="utf-8"
    )
)
trajectory = diagnostics.get("trajectory") or {}

if trajectory.get("api_calls") != 0:
    errors.append(
        "dry-run unexpectedly recorded API calls"
    )

toolsets = set(
    trajectory.get("toolsets_used", [])
)

if toolsets != {HERMES_TOOLSET}:
    errors.append(
        f"unexpected toolsets: {sorted(toolsets)}"
    )

runtime = Path(
    result["paths"]["runtime"]
)

launch = runtime / "launch.json"

if not launch.is_file():
    errors.append(
        "launch.json missing"
    )

result_path = runtime / "result.json"

if not result_path.is_file():
    errors.append(
        "result.json missing"
    )

agent_log = runtime / "agent.log"

if not agent_log.is_file():
    errors.append(
        "agent.log missing"
    )

archive = Path(result["artifacts"]["candidate_archive"])

if not archive.is_file():
    errors.append("candidate final archive missing")

if not result["metrics"]["input_integrity"]["pass"]:
    errors.append("protected candidate inputs changed during dry run")

if result["outcome"] != "FAIL":
    errors.append("pristine baseline should be a scored model failure")

if errors:
    print("RUNNER_VALIDATION=FAIL")

    for error in errors:
        print(f"  - {error}")

    raise SystemExit(1)

print("RUNNER_VALIDATION=PASS")
print(
    "run_id="
    + result["run_id"]
)
print(
    "baseline_commit="
    + result["baseline_commit"]
)
print(
    "content_sha256="
    + result["content_sha256"]
)
print(
    "sandbox_exit_code="
    + str(result["sandbox_exit_code"])
)
print(
    "trajectory_present="
    + str(result["trajectory_present"])
)
print(
    "infrastructure_ok="
    + str(result["infrastructure_ok"])
)
print(
    "infra_reasons="
    + (
        ",".join(result["infra_reasons"])
        if result["infra_reasons"]
        else "NONE"
    )
)
print(
    "api_calls="
    + str(trajectory.get("api_calls"))
)
print(
    "toolsets="
    + ",".join(
        sorted(toolsets)
    )
)
print(
    "public="
    f"{evaluation['public']['passed']}/"
    f"{evaluation['public']['total']}"
)
print(
    "hidden="
    f"{evaluation['hidden']['passed']}/"
    f"{evaluation['hidden']['total']}"
)
print(
    "supplemental="
    f"{evaluation['supplemental']['passed']}/"
    f"{evaluation['supplemental']['total']}"
)
print(
    "evaluation=FAIL_EXPECTED"
)
print(
    "false_completion="
    + str(result["false_completion"])
)

cleanup(result)

if {item.name for item in WORK.iterdir()} != preexisting_work:
    raise SystemExit(
        "work root changed outside the validator-owned run"
    )

if {item.name for item in RUNTIME.iterdir()} != preexisting_runtime:
    raise SystemExit(
        "runtime root changed outside the validator-owned run"
    )

print("cleanup=PASS")
