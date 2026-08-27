from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from harness.artifacts import atomic_write_text, ensure_root, ensure_subdirectory
from harness.benchmark_model import preflight_model
from harness.hermes_runner import (
    HERMES_DISTRIBUTION,
    HERMES_NO_TOOLS_DISTRIBUTION,
    RUNTIME_ROOT,
    WORK_ROOT,
    _hermes_environment,
    batch_command,
    detect_fallback_attempts,
    final_response,
    load_trajectory,
    utc_now,
    write_runtime_hermes_config,
    write_runtime_http_policy,
)
from harness.processes import ProcessDeadlineExpired, run_process_group
from harness.sandbox import write_shell_wrapper


PROMPTS = {
    "basic": (
        "No tools are available. Reply with exactly GX10_HERMES_BASIC_OK and no other text."
    ),
    "tool": (
        "Use the terminal exactly once to run `printf GX10_HERMES_TOOL_OUTPUT`. "
        "After the tool result, reply with exactly GX10_HERMES_TOOL_OK."
    ),
}


def assess(stage: str, trajectory: dict[str, Any] | None) -> dict[str, Any]:
    response = final_response(trajectory)
    visible_response = re.sub(
        r"\A\s*<think>.*?</think>\s*",
        "",
        response,
        count=1,
        flags=re.DOTALL,
    )
    if not trajectory:
        return {"status": "FAIL", "reason": "trajectory_missing", "final_response": ""}
    toolsets = trajectory.get("toolsets_used")
    tool_stats = trajectory.get("tool_stats")
    api_calls = trajectory.get("api_calls")
    if type(api_calls) is not int or api_calls < 1:
        return {"status": "FAIL", "reason": "no_model_api_activity", "final_response": response}
    if stage == "basic":
        passed = toolsets == [] and visible_response.strip() == "GX10_HERMES_BASIC_OK"
        reason = None if passed else "basic_response_or_no-tools_contract_failed"
    else:
        terminal = tool_stats.get("terminal", {}) if isinstance(tool_stats, dict) else {}
        passed = (
            toolsets == ["hermesbench_terminal_foreground"]
            and terminal.get("count") == 1
            and terminal.get("failure") == 0
            and visible_response.strip() == "GX10_HERMES_TOOL_OK"
        )
        reason = None if passed else "single_terminal_tool_contract_failed"
    return {
        "status": "PASS" if passed else "FAIL",
        "reason": reason,
        "final_response": response,
        "visible_final_response": visible_response,
        "api_calls": api_calls,
        "toolsets_used": toolsets,
        "tool_stats": tool_stats,
    }


def run(
    *,
    stage: str,
    model: str,
    endpoint: str,
    context_length: int,
    reasoning: str,
    expected_digest: str,
    total_timeout: int = 600,
    inactivity_timeout: int = 480,
) -> dict[str, Any]:
    if stage not in PROMPTS:
        raise ValueError(f"unknown Hermes diagnostic stage: {stage}")
    identity = preflight_model(endpoint, model, expected_digest=expected_digest)
    ensure_root(WORK_ROOT)
    ensure_root(RUNTIME_ROOT)
    run_id = f"hermes-{stage}-v1-{uuid.uuid4().hex[:12]}"
    candidate = ensure_subdirectory(WORK_ROOT, WORK_ROOT / run_id, exclusive=True)
    runtime = ensure_subdirectory(RUNTIME_ROOT, RUNTIME_ROOT / run_id, exclusive=True)
    subprocess.run(
        ["/usr/bin/git", "-C", str(candidate), "init", "-q"],
        check=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
    )
    tool_tmp = ensure_subdirectory(candidate, candidate / ".git" / "hermesbench-tmp")
    wrapper = write_shell_wrapper(runtime, candidate)
    policy = write_runtime_http_policy(runtime, shell_wrapper=wrapper, base_url=endpoint)
    runtime_config = write_runtime_hermes_config(
        runtime,
        model=model,
        base_url=endpoint,
        context_length=context_length,
        reasoning=reasoning,
        max_turns=3,
        observed_config={"compression_enabled": False},
    )
    atomic_write_text(
        runtime / "dataset.jsonl",
        json.dumps({"prompt": PROMPTS[stage], "cwd": str(candidate), "task_id": run_id}) + "\n",
        root=runtime,
    )
    distribution = HERMES_NO_TOOLS_DISTRIBUTION if stage == "basic" else HERMES_DISTRIBUTION
    command = batch_command(
        runtime_dir=runtime,
        model=model,
        reasoning=reasoning,
        max_turns=3,
        base_url=endpoint,
        distribution=distribution,
    )
    launch = {
        "run_id": run_id,
        "stage": stage,
        "started_at": utc_now(),
        "model": model,
        "runtime_digest": identity["runtime_model_digest"],
        "endpoint": endpoint,
        "distribution": distribution,
        "total_timeout_seconds": total_timeout,
        "inactivity_timeout_seconds": inactivity_timeout,
        "command": command,
        "network_policy": policy,
    }
    atomic_write_text(runtime / "launch.json", json.dumps(launch, indent=2) + "\n", root=runtime)
    started = time.monotonic()
    timeout_reason = None
    try:
        process = run_process_group(
            command,
            cwd=runtime,
            env=_hermes_environment(runtime_config, policy, tool_tmp, candidate),
            timeout=total_timeout,
            inactivity_timeout=inactivity_timeout,
            heartbeat_paths=(
                Path(runtime_config["hermes_home"]) / "logs" / "agent.log",
                runtime / "data" / "hermes" / "trajectories.jsonl",
            ),
        )
        output = process.stdout
        runner_exit = process.returncode
    except ProcessDeadlineExpired as exc:
        output = str(exc.stdout or "")
        runner_exit = 124
        timeout_reason = exc.deadline_kind
    atomic_write_text(runtime / "agent.log", output, root=runtime)
    trajectory = load_trajectory(runtime)
    assessment = assess(stage, trajectory)
    internal_log = Path(runtime_config["hermes_home"]) / "logs" / "agent.log"
    search = output
    if internal_log.is_file():
        search += "\n" + internal_log.read_text(encoding="utf-8", errors="replace")
    fallback_attempts = detect_fallback_attempts(search)
    blocked_path = Path(policy["blocked_attempts_path"])
    blocked_attempts = (
        blocked_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if blocked_path.is_file()
        else []
    )
    if timeout_reason:
        assessment = {**assessment, "status": "FAIL", "reason": f"{timeout_reason}_timeout"}
    if runner_exit != 0 and not timeout_reason:
        assessment = {**assessment, "status": "INFRA_ERROR", "reason": f"runner_exit_{runner_exit}"}
    if fallback_attempts or blocked_attempts:
        assessment = {**assessment, "status": "FAIL", "reason": "prohibited_network_or_fallback_attempt"}
    shutil.rmtree(candidate)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "status": assessment["status"],
        "reason": assessment.get("reason"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "runner_exit_code": runner_exit,
        "timeout_reason": timeout_reason,
        "model": identity,
        "assessment": assessment,
        "fallback_attempts": fallback_attempts,
        "blocked_network_attempts": blocked_attempts,
        "candidate_cleanup_complete": not candidate.exists(),
        "runtime": str(runtime),
    }
    atomic_write_text(runtime / "diagnostic-result.json", json.dumps(result, indent=2) + "\n", root=runtime)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded incremental Hermes live diagnostic")
    parser.add_argument("--stage", required=True, choices=("basic", "tool"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--expected-digest", required=True)
    args = parser.parse_args(argv)
    result = run(
        stage=args.stage,
        model=args.model,
        endpoint=args.endpoint,
        context_length=args.context_length,
        reasoning=args.reasoning,
        expected_digest=args.expected_digest,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "INFRA_ERROR" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
