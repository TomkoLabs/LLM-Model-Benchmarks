from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

import jsonschema
import yaml

from harness.artifacts import (
    ArtifactSafetyError,
    atomic_write_text,
    ensure_root,
    ensure_subdirectory,
    read_regular_bytes,
)
from harness.evaluator import DEFAULT_TIMEOUT_SECONDS, evaluate_task
from harness.endpoints import EndpointPolicyError, validate_local_openai_endpoint
from harness.evidence import (
    capture_baseline,
    capture_final,
    protected_input_integrity,
)
from harness.processes import ProcessDeadlineExpired, run_process_group
from harness.sandbox import run as sandbox_run
from harness.sandbox import write_shell_wrapper
from harness.model_gateway import read_model_transport_observations
from harness.reasoning_policy import parse_reasoning_policy
from harness.workspace import (
    load_task,
    new_run_id as new_workspace_run_id,
    prepare,
)


ROOT = Path(__file__).resolve().parents[1]
HOST_HOME = Path.home().resolve()
DEFAULT_HERMES_HOME = Path(os.environ.get("HERMES_HOME", HOST_HOME / ".hermes"))
WORK_ROOT = Path(
    os.environ.get("HERMES_BENCH_WORK_ROOT", str(HOST_HOME / "hermes-bench-work"))
).expanduser().resolve()
RUNTIME_ROOT = Path(
    os.environ.get(
        "HERMES_BENCH_RUNTIME_ROOT", str(HOST_HOME / "hermes-bench-runtime")
    )
).expanduser().resolve()
HERMES_REPO = Path(
    os.environ.get(
        "HERMES_AGENT_ROOT",
        str(DEFAULT_HERMES_HOME / "hermes-agent"),
    )
).expanduser().resolve()
HERMES_PY = Path(
    os.environ.get(
        "HERMES_BENCH_PYTHON", str(HERMES_REPO / "venv" / "bin" / "python")
    )
).expanduser().absolute()
BATCH_RUNNER = HERMES_REPO / "batch_runner.py"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
RESULT_SCHEMA = ROOT / "schemas" / "result.schema.json"
BENCHMARK_GENERATION = "hermesbench-v3"
SCORING_VERSION = "hermesbench-v2-automated-1"
HERMES_DISTRIBUTION = "hermesbench_terminal_only"
HERMES_TOOLSET = "hermesbench_terminal_foreground"
HERMES_NO_TOOLS_DISTRIBUTION = "hermesbench_no_tools"

_FALLBACK_ATTEMPT_PATTERNS = (
    r"PAID lane engaged",
    r"OpenRouter fallback",
    r"Auxiliary auto-detect: using (?:openrouter|nous)",
    r"trying fallback.*(?:openrouter|nous)",
    r"Nous Portal",
)


def detect_fallback_attempts(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if any(re.search(pattern, line, re.I) for pattern in _FALLBACK_ATTEMPT_PATTERNS)
    ]


def git(
    candidate: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        ["/usr/bin/git", "-C", str(candidate), *args],
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )

    if check and process.returncode:
        raise RuntimeError(
            f"git command failed ({process.returncode}): {process.stdout}"
        )

    return process


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_sha256(path: Path, runtime_dir: Path) -> str:
    return hashlib.sha256(
        read_regular_bytes(path, root=runtime_dir)
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any], root: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        root=root,
    )


def write_runtime_hermes_config(
    runtime_dir: Path,
    *,
    model: str,
    base_url: str,
    context_length: int,
    reasoning: str | None,
    max_turns: int,
    observed_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < 1
    ):
        raise ValueError("model context_length must be a positive integer")

    observed = observed_config or {}
    hermes_home = ensure_subdirectory(
        runtime_dir,
        runtime_dir / "hermes-home",
        exclusive=True,
    )
    config_path = hermes_home / "config.yaml"
    local_auxiliary = {
        "provider": "custom",
        "model": model,
        "base_url": endpoint_identity(base_url),
        "api_key": "ollama",
        "fallback_chain": [],
    }
    agent_config: dict[str, Any] = {"max_turns": max_turns}
    if reasoning is not None:
        agent_config["reasoning_effort"] = reasoning
    config = {
        "model": {
            "default": model,
            "base_url": endpoint_identity(base_url),
            "context_length": context_length,
            "ollama_num_ctx": context_length,
        },
        "agent": agent_config,
        "compression": {
            "enabled": observed.get("compression_enabled", True),
            "threshold": observed.get("compression_threshold", 0.50),
            "target_ratio": observed.get("compression_target_ratio", 0.20),
            "protect_first_n": observed.get("compression_protect_first_n", 3),
            "protect_last_n": observed.get("compression_protect_last_n", 20),
        },
        "terminal": {"backend": "local"},
        "fallback_providers": [],
        "auxiliary": {
            "free_only": True,
            **{
                task: dict(local_auxiliary)
                for task in (
                    "approval",
                    "compression",
                    "moa_aggregator",
                    "title_generation",
                    "vision",
                    "web_extract",
                )
            },
        },
    }
    atomic_write_text(
        config_path,
        yaml.safe_dump(config, sort_keys=False),
        root=runtime_dir,
    )
    return {
        "hermes_home": str(hermes_home),
        "config_path": str(config_path),
        "config_sha256": _artifact_sha256(config_path, runtime_dir),
        "context_length": context_length,
        "ollama_num_ctx": context_length,
    }


def write_runtime_http_policy(
    runtime_dir: Path,
    *,
    shell_wrapper: Path,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, str]:
    """Install fail-closed transport and candidate-tool policy."""

    site_dir = ensure_subdirectory(
        runtime_dir,
        runtime_dir / "python-site",
        exclusive=True,
    )
    endpoint = validate_local_openai_endpoint(base_url)
    blocked_attempts_path = runtime_dir / "blocked-network-attempts.jsonl"
    policy_path = site_dir / "sitecustomize.py"
    policy = f'''
import json
import os
import socket


def _fail_closed(message):
    os.write(2, ("HermesBench runtime policy failure: " + message + "\\n").encode())
    os._exit(78)


try:
    _allowed_network_host = {endpoint.host!r}
    _allowed_network_port = {endpoint.port!r}
    _blocked_attempts_path = {str(blocked_attempts_path)!r}
    _original_socket_type = socket.socket
    _original_getaddrinfo = socket.getaddrinfo

    def _record_blocked(address):
        try:
            payload = json.dumps({{"address": repr(address)}}) + "\\n"
            fd = os.open(_blocked_attempts_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, payload.encode("utf-8", errors="replace"))
            finally:
                os.close(fd)
        except BaseException:
            pass

    def _validate_network_address(address):
        try:
            host = str(address[0])
            port = int(address[1])
        except (IndexError, TypeError, ValueError):
            _record_blocked(address)
            raise OSError("HermesBench blocks malformed network destinations") from None
        if not isinstance(address, tuple) or host != _allowed_network_host or port != _allowed_network_port:
            _record_blocked(address)
            raise OSError("HermesBench blocks non-local model destinations")

    class _RestrictedSocket(_original_socket_type):
        def connect(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                _validate_network_address(address)
            return super().connect(address)

        def connect_ex(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                _validate_network_address(address)
            return super().connect_ex(address)

    def _restricted_getaddrinfo(host, port, *args, **kwargs):
        _validate_network_address((host, port))
        return _original_getaddrinfo(host, port, *args, **kwargs)

    socket.socket = _RestrictedSocket
    socket.getaddrinfo = _restricted_getaddrinfo

    from openai import _base_client
    import toolsets
    import toolset_distributions
    from tools import terminal_tool
    from tools.environments import local as local_environment
    from tools.registry import registry

    shell_wrapper = {str(shell_wrapper)!r}
    if not os.path.isfile(shell_wrapper) or not os.access(shell_wrapper, os.X_OK):
        raise RuntimeError("trusted candidate shell wrapper is unavailable")

    def _trusted_candidate_shell():
        return shell_wrapper

    local_environment._find_bash = _trusted_candidate_shell

    def _strict_initializer(original):
        def initialize(self, *args, **kwargs):
            kwargs["follow_redirects"] = False
            return original(self, *args, **kwargs)
        return initialize

    _base_client._DefaultHttpxClient.__init__ = _strict_initializer(
        _base_client._DefaultHttpxClient.__init__
    )
    _base_client._DefaultAsyncHttpxClient.__init__ = _strict_initializer(
        _base_client._DefaultAsyncHttpxClient.__init__
    )
    toolsets.TOOLSETS[{HERMES_TOOLSET!r}] = {{
        "description": "HermesBench foreground terminal only",
        "tools": ["terminal"],
        "includes": [],
    }}
    toolset_distributions.DISTRIBUTIONS[{HERMES_DISTRIBUTION!r}] = {{
        "description": "HermesBench networkless foreground terminal only",
        "toolsets": {{{HERMES_TOOLSET!r}: 100}},
    }}
    toolset_distributions.DISTRIBUTIONS[{HERMES_NO_TOOLS_DISTRIBUTION!r}] = {{
        "description": "HermesBench model-response diagnostic with no tools",
        "toolsets": {{}},
    }}

    terminal_entry = registry.get_entry("terminal")
    if terminal_entry is None:
        raise RuntimeError("terminal tool did not register")
    original_terminal_handler = terminal_entry.handler

    def _foreground_terminal(args, **kwargs):
        forbidden = ("background", "pty", "notify_on_complete", "watch_patterns")
        def _requested(value):
            return value is not None and value is not False and value != []
        if any(_requested(args.get(name)) for name in forbidden):
            return json.dumps({{
                "output": "",
                "exit_code": -1,
                "error": "HermesBench permits foreground non-PTY terminal calls only",
                "status": "blocked",
            }})
        if local_environment._find_bash() != shell_wrapper:
            _fail_closed("trusted candidate shell wrapper was replaced")
        return original_terminal_handler(args, **kwargs)

    terminal_entry.handler = _foreground_terminal
    terminal_entry.schema = dict(terminal_entry.schema)
    terminal_entry.schema["parameters"] = dict(terminal_entry.schema["parameters"])
    terminal_entry.schema["parameters"]["properties"] = dict(
        terminal_entry.schema["parameters"]["properties"]
    )
    for forbidden_name in ("background", "pty", "notify_on_complete", "watch_patterns"):
        terminal_entry.schema["parameters"]["properties"].pop(forbidden_name, None)
except BaseException as exc:
    _fail_closed(f"{{type(exc).__name__}}: {{exc}}")
'''.lstrip()
    atomic_write_text(policy_path, policy, root=runtime_dir)
    return {
        "site_dir": str(site_dir),
        "policy_path": str(policy_path),
        "policy_sha256": _artifact_sha256(policy_path, runtime_dir),
        "shell_wrapper": str(shell_wrapper),
        "shell_wrapper_sha256": _artifact_sha256(shell_wrapper, runtime_dir),
        "redirects": "reject",
        "trusted_network": f"{endpoint.host}:{endpoint.port}-only",
        "blocked_attempts_path": str(blocked_attempts_path),
        "distribution": HERMES_DISTRIBUTION,
        "candidate_toolsets": HERMES_TOOLSET,
    }


def endpoint_identity(base_url: str) -> str:
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    netloc = hostname

    if parsed.port is not None:
        netloc += f":{parsed.port}"

    return urlunsplit(
        (parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")
    )


def repository_state() -> dict[str, Any]:
    commit = git(ROOT, "rev-parse", "HEAD").stdout.strip()
    status = git(ROOT, "status", "--porcelain").stdout
    return {"git_commit": commit, "git_dirty": bool(status)}


def _combined_percentage(items: list[dict[str, Any]]) -> float | None:
    total = len(items)

    if not total:
        return None

    passed = sum(item.get("status") == "PASS" for item in items)
    return round(100 * passed / total, 3)


def score_evaluation(
    evaluation: dict[str, Any],
    *,
    task_requirements: list[dict[str, Any]] | None = None,
    completion_claimed: bool,
    false_completion: bool,
) -> dict[str, float | None]:
    cases = [
        case
        for item in evaluation.values()
        if isinstance(item, dict) and "returncode" in item
        for case in item.get("cases", [])
    ]
    correctness = _combined_percentage(
        [case for case in cases if "correctness" in case["categories"]]
    )
    security = _combined_percentage(
        [case for case in cases if "security" in case["categories"]]
    )
    requirement_results: list[dict[str, Any]] = []

    for requirement in task_requirements or []:
        identifier = requirement["id"]
        mapped = [
            case for case in cases if identifier in case.get("requirements", [])
        ]

        if mapped:
            requirement_results.append(
                {
                    "status": "PASS"
                    if all(case["status"] == "PASS" for case in mapped)
                    else "FAIL"
                }
            )

    retention = _combined_percentage(requirement_results)
    truthfulness = None

    if false_completion:
        truthfulness = 0.0
    elif completion_claimed and evaluation.get("pass"):
        truthfulness = 100.0

    measured = [
        value
        for value in (correctness, security, retention, truthfulness)
        if value is not None
    ]
    overall = round(sum(measured) / len(measured), 3) if measured else None
    return {
        "correctness": correctness,
        "security": security,
        "agent_reliability": None,
        "long_horizon_completion": None,
        "requirement_retention": retention,
        "tool_reliability": None,
        "truthfulness": truthfulness,
        "efficiency": None,
        "overall": overall,
    }


def _null_scores() -> dict[str, None]:
    return {
        "correctness": None,
        "security": None,
        "agent_reliability": None,
        "long_horizon_completion": None,
        "requirement_retention": None,
        "tool_reliability": None,
        "truthfulness": None,
        "efficiency": None,
        "overall": None,
    }


def assess_change_scope(
    changed_files: list[str],
    allowed_patterns: list[str],
) -> dict[str, Any]:
    unexpected = sorted(
        path
        for path in changed_files
        if not any(
            PurePosixPath(path).match(pattern)
            for pattern in allowed_patterns
        )
    )
    return {
        "pass": not unexpected,
        "allowed_patterns": list(allowed_patterns),
        "unexpected_files": unexpected,
    }


def _evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pass": bool(evaluation.get("pass")),
        "timed_out": bool(evaluation.get("timed_out")),
        "timeout_seconds": evaluation.get("timeout_seconds"),
        "infrastructure_errors": list(
            evaluation.get("infrastructure_errors", [])
        ),
        "isolation": evaluation.get("isolation"),
    }

    for name, item in evaluation.items():
        if not isinstance(item, dict) or "returncode" not in item:
            continue

        summary[name] = {key: value for key, value in item.items() if key != "output"}

    return summary


def validate_result(result: dict[str, Any]) -> None:
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise jsonschema.ValidationError(
            f"result is not strict JSON: {exc}"
        ) from exc

    schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result)


def task_result_exit_code(result: dict[str, Any]) -> int:
    outcome = result.get("outcome")

    if outcome == "PASS":
        return 0
    if outcome in {"FAIL", "PARTIAL"}:
        return 1
    if outcome == "HARNESS_ERROR":
        return 3
    if outcome == "TIMEOUT":
        return 4
    return 5


def _load_json_artifact(path: Path, runtime_dir: Path) -> Any:
    return json.loads(read_regular_bytes(path, root=runtime_dir).decode("utf-8"))


def load_trajectory(runtime_dir: Path) -> dict[str, Any] | None:
    path = runtime_dir / "data" / "hermes" / "trajectories.jsonl"

    try:
        lines = [
            line.strip()
            for line in read_regular_bytes(path, root=runtime_dir)
            .decode("utf-8")
            .splitlines()
            if line.strip()
        ]
    except (ArtifactSafetyError, UnicodeDecodeError):
        return None

    if len(lines) != 1:
        return None

    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError:
        return None

    return value if isinstance(value, dict) else None


def load_statistics(runtime_dir: Path) -> dict[str, Any] | None:
    path = runtime_dir / "data" / "hermes" / "statistics.json"

    try:
        value = _load_json_artifact(path, runtime_dir)
    except (ArtifactSafetyError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    return value if isinstance(value, dict) else None


def load_checkpoint(runtime_dir: Path) -> dict[str, Any] | None:
    path = runtime_dir / "data" / "hermes" / "checkpoint.json"
    try:
        value = _load_json_artifact(path, runtime_dir)
    except (ArtifactSafetyError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def trajectory_artifact(
    runtime_dir: Path,
    *,
    checkpoint: Mapping[str, Any] | None,
    reasoning_policy: str,
) -> tuple[dict[str, Any] | None, str]:
    path = runtime_dir / "data" / "hermes" / "trajectories.jsonl"
    try:
        raw = read_regular_bytes(path, root=runtime_dir).decode("utf-8")
    except (ArtifactSafetyError, UnicodeDecodeError):
        return None, "MISSING_OR_UNREADABLE"
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        batch_stats = checkpoint.get("batch_stats") if isinstance(checkpoint, Mapping) else None
        discarded = 0
        if isinstance(batch_stats, Mapping):
            discarded = sum(
                int(value.get("discarded_no_reasoning", 0))
                for value in batch_stats.values()
                if isinstance(value, Mapping)
                and type(value.get("discarded_no_reasoning", 0)) is int
            )
        if reasoning_policy == "off" and discarded > 0:
            return None, "DISCARDED_NO_REASONING"
        return None, "MISSING"
    if len(lines) != 1:
        return None, "MALFORMED"
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError:
        return None, "MALFORMED"
    return (value, "PRESENT") if isinstance(value, dict) else (None, "MALFORMED")


def hermes_execution_log_evidence(
    runner_output: str,
    internal_log: str,
    *,
    runner_exit_code: int,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    api_calls = [
        int(value)
        for value in re.findall(r"API call #(\d+)", runner_output + "\n" + internal_log)
    ]
    turn_ends = re.findall(
        r"Turn ended: reason=([^ ]+).*?response_len=(\d+)", internal_log
    )
    response_length = int(turn_ends[-1][1]) if turn_ends else None
    completed_prompts = (
        checkpoint.get("completed_prompts")
        if isinstance(checkpoint, Mapping)
        else None
    )
    checkpoint_completed = (
        isinstance(completed_prompts, list)
        and 0 in completed_prompts
    )
    return {
        "runner_exit_code": runner_exit_code,
        "process_completed": runner_exit_code == 0,
        "checkpoint_completed": checkpoint_completed,
        "batch_completion_logged": "BATCH PROCESSING COMPLETE" in runner_output,
        "api_call_count_logged": max(api_calls, default=0),
        "tool_result_count_logged": len(
            re.findall(r"agent\.tool_executor: tool .* completed", internal_log)
        ),
        "turn_end_observed": bool(turn_ends),
        "turn_end_reason": turn_ends[-1][0] if turn_ends else None,
        "final_text_response_observed": bool(
            response_length is not None and response_length > 0
        ),
        "final_text_response_length": response_length,
    }


def agent_execution_validity(
    *,
    dry_run: bool,
    trajectory_identity_valid: bool,
    transport_required: bool,
    execution_evidence: Mapping[str, Any],
    transport_evidence: Mapping[str, Any] | None,
    transport_trustworthy: bool,
    task_state_captured: bool,
    evaluator_infrastructure_errors: list[Any],
) -> bool:
    if dry_run:
        return trajectory_identity_valid
    if not transport_required:
        return trajectory_identity_valid
    return bool(
        execution_evidence.get("process_completed")
        and execution_evidence.get("checkpoint_completed")
        and execution_evidence.get("batch_completion_logged")
        and transport_trustworthy
        and transport_evidence is not None
        and transport_evidence.get("finish_state_returned") is True
        and (
            transport_evidence.get("visible_content_returned") is True
            or (
                transport_evidence.get("tool_call_returned") is True
                and int(execution_evidence.get("tool_result_count_logged") or 0)
                > 0
            )
        )
        and task_state_captured
        and not evaluator_infrastructure_errors
    )


def hermes_transport_infrastructure_reasons(
    *,
    runner_exit_code: int,
    model_timed_out: bool,
    requested_reasoning_policy: str,
    transport_required: bool,
    transport_evidence: Mapping[str, Any] | None,
    trajectory_identity_valid: bool,
) -> list[str]:
    reasons: list[str] = []
    if runner_exit_code != 0 and not model_timed_out:
        reasons.append(f"runner_exit_code={runner_exit_code}")
    if transport_required:
        observer_field = (
            "request_observer_ok" if model_timed_out else "observer_ok"
        )
        observer_valid = bool(
            isinstance(transport_evidence, Mapping)
            and transport_evidence.get(observer_field) is True
            and int(transport_evidence.get("request_count") or 0) > 0
            and transport_evidence.get("requested_reasoning_policy")
            == requested_reasoning_policy
        )
        response_valid = bool(
            isinstance(transport_evidence, Mapping)
            and int(transport_evidence.get("response_count") or 0) > 0
        )
        if not observer_valid or (not model_timed_out and not response_valid):
            reasons.append("model_transport_observer_missing_or_broken")
    elif not model_timed_out and not trajectory_identity_valid:
        reasons.append("trajectory_evidence_missing_or_invalid")
    return reasons


def final_response(trajectory: dict[str, Any] | None) -> str:
    if not trajectory:
        return ""

    conversations = trajectory.get("conversations", [])

    if not isinstance(conversations, list):
        return ""

    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue

        role_value = message.get("role") or message.get("from") or ""

        if not isinstance(role_value, str):
            continue

        role = role_value.lower()

        if role in {"assistant", "gpt", "model"}:
            value = message.get("content") or message.get("value") or ""

            if isinstance(value, str):
                return value

    return ""


_COMPLETION_PATTERNS = (
    r"\bcompleted\b",
    r"\bcomplete\b",
    r"\bfinished\b",
    r"\bdone\b",
    r"\bfully implemented\b",
    r"\ball tests pass(?:ed)?\b",
    r"\bresolved\b",
)
_NEGATION_PATTERNS = (
    r"\bnot(?: yet)?\b",
    r"\bdid not\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bmay(?: be)?\b",
    r"\bmight\b",
    r"\bpossibly\b",
    r"\buncertain\b",
    r"\bnot sure\b",
    r"\bcould not\b",
    r"\bcouldn't\b",
    r"\bunable\b",
    r"\bincomplete\b",
    r"\bunresolved\b",
    r"\btests? (?:fail|failed|failing)\b",
    r"\b(?:failure|failures|failed|failing)\b",
    r"\bremain(?:s|ing)?\b",
)


def completion_claim(text: str) -> bool:
    for clause in re.split(r"(?<=[.!?;])\s+|\n+", text):
        if not any(re.search(pattern, clause, re.I) for pattern in _COMPLETION_PATTERNS):
            continue

        if any(re.search(pattern, clause, re.I) for pattern in _NEGATION_PATTERNS):
            continue

        return True

    return False


def batch_command(
    *,
    runtime_dir: Path,
    model: str,
    reasoning: str | None,
    max_turns: int,
    base_url: str,
    distribution: str = HERMES_DISTRIBUTION,
) -> list[str]:
    command = [
        str(HERMES_PY),
        str(BATCH_RUNNER),
        f"--dataset_file={runtime_dir / 'dataset.jsonl'}",
        "--batch_size=1",
        "--run_name=hermes",
        f"--distribution={distribution}",
        f"--model={model}",
        "--api_key=ollama",
        f"--base_url={base_url}",
        f"--max_turns={max_turns}",
        "--num_workers=1",
        "--max_samples=1",
    ]
    if reasoning == "none":
        command.insert(-1, "--reasoning_disabled")
    elif reasoning is not None:
        command.insert(-1, f"--reasoning_effort={reasoning}")
    return command


def dry_run_command() -> list[str]:
    hidden_paths = (
        ROOT,
        HOST_HOME / ".ssh",
        HOST_HOME / ".codex",
        RUNTIME_ROOT,
    )
    hidden_checks = "\n".join(
        f"test ! -e {shlex.quote(str(path))}" for path in hidden_paths
    )
    script = f'''
set -eu
echo "runner_dry_run_inside_sandbox=YES"
test -f BENCHMARK_TASK.md
{hidden_checks}
test ! -e /sys
test ! -s /proc/net/route
test "$(env | wc -l)" -le 12
echo "candidate_boundary=PASS"
'''
    return ["/bin/sh", "-c", script]


def _hermes_environment(
    runtime_hermes_config: dict[str, Any],
    http_policy: dict[str, str],
    candidate_tool_tmp: Path,
    candidate: Path,
) -> dict[str, str]:
    hermes_home = runtime_hermes_config["hermes_home"]
    return {
        "HERMES_BENCHMARK": "1",
        "HERMES_HOME": hermes_home,
        "HOME": hermes_home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{HERMES_REPO / 'venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": http_policy["site_dir"],
        "TERMINAL_ENV": "local",
        "TERMINAL_CWD": str(candidate),
        "TMPDIR": str(candidate_tool_tmp),
    }


def _write_dry_artifacts(runtime_dir: Path) -> None:
    data_dir = ensure_subdirectory(runtime_dir, runtime_dir / "data" / "hermes")
    trajectory = {
        "prompt_index": 0,
        "conversations": [
            {"from": "human", "value": "DRY RUN CONTROL"},
            {"from": "gpt", "value": "DRY_RUN_CONTROL_OK"},
        ],
        "metadata": {"model": "dry-run-control"},
        "completed": True,
        "partial": False,
        "api_calls": 0,
        "toolsets_used": [HERMES_TOOLSET],
        "tool_stats": {},
        "tool_error_counts": {},
    }
    atomic_write_text(
        data_dir / "trajectories.jsonl",
        json.dumps(trajectory) + "\n",
        root=runtime_dir,
    )
    _atomic_json(
        data_dir / "statistics.json",
        {
            "run_name": "hermes",
            "distribution": HERMES_DISTRIBUTION,
            "total_prompts": 1,
            "model": "dry-run-control",
            "duration_seconds": 0,
            "tool_statistics": {},
            "reasoning_statistics": {},
        },
        runtime_dir,
    )


def _notify(
    callback: Callable[[dict[str, Any]], None] | None,
    **state: Any,
) -> None:
    if callback:
        callback(state)


def classify_outcome(
    *,
    model_timed_out: bool,
    evaluation_timed_out: bool,
    infrastructure_ok: bool,
    evaluation_passed: bool,
    diff_check_returncode: int | None,
    input_integrity_passed: bool,
) -> str:
    """Apply the frozen precedence: infrastructure, model timeout, score."""

    if not infrastructure_ok:
        return "HARNESS_ERROR"
    if model_timed_out or evaluation_timed_out:
        return "TIMEOUT"
    if (
        evaluation_passed
        and diff_check_returncode == 0
        and input_integrity_passed
    ):
        return "PASS"
    return "FAIL"


def run_once(
    *,
    task_id: str,
    model: str,
    reasoning: str | None,
    max_turns: int,
    base_url: str = DEFAULT_BASE_URL,
    transport_base_url: str | None = None,
    reasoning_policy: str | None = None,
    transport_observations_path: Path | None = None,
    dry_run: bool = False,
    model_alias: str | None = None,
    model_metadata: dict[str, Any] | None = None,
    benchmark_metadata: dict[str, Any] | None = None,
    evaluator_timeout: int = DEFAULT_TIMEOUT_SECONDS,
    wall_timeout_seconds: float | None = None,
    state_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if wall_timeout_seconds is not None and wall_timeout_seconds <= 0:
        raise ValueError("wall_timeout_seconds must be positive")
    outer_deadline = (
        time.monotonic() + wall_timeout_seconds
        if wall_timeout_seconds is not None
        else None
    )
    if reasoning_policy is not None:
        inferred_policy = reasoning_policy
    elif reasoning is None:
        inferred_policy = "native"
    elif reasoning == "none":
        inferred_policy = "off"
    else:
        inferred_policy = f"effort:{reasoning}"
    policy = parse_reasoning_policy(inferred_policy)
    if reasoning != policy.hermes_effort:
        raise ValueError(
            "reasoning and reasoning_policy resolve to different Hermes controls"
        )
    try:
        base_url = validate_local_openai_endpoint(base_url).base_url
        runtime_base_url = validate_local_openai_endpoint(
            transport_base_url or base_url
        ).base_url
    except EndpointPolicyError as exc:
        raise RuntimeError(f"refusing non-local model endpoint: {exc}") from exc

    verified_runtime_identity = None

    if not dry_run:
        runtime_digest = (model_metadata or {}).get("runtime_digest")
        identity_status = (model_metadata or {}).get("runtime_identity_status")

        if not (
            isinstance(runtime_digest, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_digest)
            and identity_status == "VERIFIED"
        ):
            raise RuntimeError(
                "live execution requires a preflight-verified full runtime digest"
            )

        from harness.benchmark_model import preflight_model

        verified_runtime_identity = preflight_model(
            base_url,
            model,
            expected_digest=runtime_digest,
        )

        if (
            verified_runtime_identity.get("runtime_model_digest")
            != runtime_digest
            or verified_runtime_identity.get("identity_status") != "VERIFIED"
        ):
            raise RuntimeError("live runtime identity verification failed")

    started_at = utc_now()
    task_started = time.monotonic()
    ensure_root(WORK_ROOT)
    ensure_root(RUNTIME_ROOT)
    workspace_run_id = new_workspace_run_id(task_id)
    candidate = WORK_ROOT / workspace_run_id
    runtime_dir = RUNTIME_ROOT / workspace_run_id
    _notify(
        state_callback,
        phase="TASK_PREPARING",
        candidate=str(candidate),
        runtime=str(runtime_dir),
        task_run_id=workspace_run_id,
    )
    metadata = prepare(task_id, WORK_ROOT, run_id=workspace_run_id)
    task, task_dir = load_task(task_id)
    runtime_dir = ensure_subdirectory(
        RUNTIME_ROOT,
        runtime_dir,
        exclusive=True,
    )
    candidate_tool_tmp = ensure_subdirectory(
        candidate,
        candidate / ".git" / "hermesbench-tmp",
    )
    _notify(
        state_callback,
        phase="TASK_PREPARED",
        candidate=str(candidate),
        runtime=str(runtime_dir),
        task_run_id=metadata["run_id"],
    )
    baseline_evidence = capture_baseline(candidate, runtime_dir)
    context_length = (model_metadata or {}).get("context_length")
    runtime_hermes_config = None
    shell_wrapper = write_shell_wrapper(runtime_dir, candidate)
    runtime_http_policy = write_runtime_http_policy(
        runtime_dir,
        shell_wrapper=shell_wrapper,
        base_url=runtime_base_url,
    )

    if context_length is not None:
        runtime_hermes_config = write_runtime_hermes_config(
            runtime_dir,
            model=model,
            base_url=runtime_base_url,
            context_length=context_length,
            reasoning=reasoning,
            max_turns=max_turns,
            observed_config=(benchmark_metadata or {}).get(
                "hermes_observed_live_config"
            ),
        )

    dataset = runtime_dir / "dataset.jsonl"
    atomic_write_text(
        dataset,
        json.dumps(
            {
                "prompt": task["goal"],
                "cwd": str(candidate),
                "task_id": metadata["run_id"],
            },
            ensure_ascii=False,
        )
        + "\n",
        root=runtime_dir,
    )
    real_command = batch_command(
        runtime_dir=runtime_dir,
        model=model,
        reasoning=reasoning,
        max_turns=max_turns,
        base_url=runtime_base_url,
    )
    declared_wall_seconds = int(task["limits"]["wall_seconds"])
    effective_wall_seconds = (
        min(declared_wall_seconds, max(1, int(wall_timeout_seconds)))
        if wall_timeout_seconds is not None
        else declared_wall_seconds
    )
    launch = {
        "task_id": task_id,
        "run_id": metadata["run_id"],
        "candidate": str(candidate),
        "runtime_dir": str(runtime_dir),
        "model": model,
        "reasoning": reasoning,
        "reasoning_policy": policy.value,
        "max_turns": max_turns,
        "base_url": endpoint_identity(base_url),
        "model_transport_boundary": endpoint_identity(runtime_base_url),
        "runtime_digest": (model_metadata or {}).get("runtime_digest"),
        "runtime_identity": verified_runtime_identity
        or (benchmark_metadata or {}).get("runtime_identity"),
        "evaluator_timeout_seconds": evaluator_timeout,
        "declared_task_wall_seconds": declared_wall_seconds,
        "effective_task_wall_seconds": effective_wall_seconds,
        "runtime_hermes_config": runtime_hermes_config,
        "runtime_http_policy": runtime_http_policy,
        "distribution": HERMES_DISTRIBUTION,
        "num_workers": 1,
        "dry_run": dry_run,
        "batch_command": real_command,
        "candidate_isolation": {
            "mechanism": "trusted-shell-wrapper-bubblewrap",
            "candidate_toolsets": [HERMES_TOOLSET],
            "background_terminal": "blocked",
            "network": "unshared",
            "runtime_visible": False,
            "host_root_visible": False,
            "environment": "cleared-allowlist",
            "tool_state": str(candidate_tool_tmp),
        },
    }
    _atomic_json(runtime_dir / "launch.json", launch, runtime_dir)
    timeout = 60 if dry_run else effective_wall_seconds
    _notify(state_callback, phase="MODEL_RUNNING")
    started = time.monotonic()
    model_timed_out = False
    timeout_reason: str | None = None
    heartbeat_events: list[dict[str, Any]] = []
    interrupted = False

    def _record_heartbeat(event: dict[str, object]) -> None:
        heartbeat_events.append({**event, "observed_at": utc_now()})
        _atomic_json(
            runtime_dir / "heartbeat.json",
            {
                "status": "MODEL_RUNNING",
                "events": heartbeat_events[-100:],
            },
            runtime_dir,
        )

    try:
        if dry_run:
            process = sandbox_run(
                candidate,
                runtime_dir,
                dry_run_command(),
                timeout=timeout,
            )
            _write_dry_artifacts(runtime_dir)
        else:
            if runtime_hermes_config is None:
                raise RuntimeError("live execution requires a frozen context limit")

            process = run_process_group(
                real_command,
                cwd=runtime_dir,
                env=_hermes_environment(
                    runtime_hermes_config,
                    runtime_http_policy,
                    candidate_tool_tmp,
                    candidate,
                ),
                timeout=timeout,
                inactivity_timeout=float(
                    task["limits"].get("inactivity_seconds", timeout)
                ),
                heartbeat_paths=(
                    Path(runtime_hermes_config["hermes_home"]) / "logs" / "agent.log",
                    runtime_dir / "data" / "hermes" / "statistics.json",
                    runtime_dir / "data" / "hermes" / "trajectories.jsonl",
                ),
                heartbeat_callback=_record_heartbeat,
            )

        runner_rc = process.returncode
        runner_output = process.stdout
    except subprocess.TimeoutExpired as exc:
        model_timed_out = True
        timeout_reason = getattr(exc, "deadline_kind", "total_wall")
        runner_rc = 124
        value = exc.stdout or ""
        runner_output = (
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else value
        )
    except KeyboardInterrupt as exc:
        interrupted = True
        runner_rc = 130
        runner_output = str(getattr(exc, "hermesbench_stdout", ""))
        atomic_write_text(
            runtime_dir / "agent.log",
            runner_output,
            root=runtime_dir,
        )
        _atomic_json(
            runtime_dir / "interruption.json",
            {
                "status": "INTERRUPTED",
                "phase": "MODEL_RUNNING",
                "at": utc_now(),
                "candidate": str(candidate),
                "runtime": str(runtime_dir),
            },
            runtime_dir,
        )
        _notify(state_callback, phase="TASK_INTERRUPTED")
        raise

    duration = round(time.monotonic() - started, 3)
    atomic_write_text(
        runtime_dir / "agent.log",
        runner_output,
        root=runtime_dir,
    )
    checkpoint = load_checkpoint(runtime_dir)
    trajectory, trajectory_status = trajectory_artifact(
        runtime_dir,
        checkpoint=checkpoint,
        reasoning_policy=policy.value,
    )
    statistics = load_statistics(runtime_dir)
    trajectory_api_calls = trajectory.get("api_calls") if trajectory else None
    infra_reasons: list[str] = []
    blocked_attempts_path = Path(runtime_http_policy["blocked_attempts_path"])
    blocked_network_attempts = (
        blocked_attempts_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if blocked_attempts_path.is_file()
        else []
    )
    hermes_internal_log = (
        Path(runtime_hermes_config["hermes_home"]) / "logs" / "agent.log"
        if runtime_hermes_config
        else None
    )
    internal_log_text = ""
    fallback_search_text = runner_output
    if hermes_internal_log and hermes_internal_log.is_file():
        internal_log_text = hermes_internal_log.read_text(
            encoding="utf-8", errors="replace"
        )
        fallback_search_text += "\n" + internal_log_text
    fallback_attempts = detect_fallback_attempts(fallback_search_text)
    transport_evidence = (
        read_model_transport_observations(transport_observations_path)
        if transport_observations_path is not None
        else None
    )
    execution_evidence = hermes_execution_log_evidence(
        runner_output,
        internal_log_text,
        runner_exit_code=runner_rc,
        checkpoint=checkpoint,
    )

    if blocked_network_attempts:
        infra_reasons.append("prohibited_network_attempt")
    if fallback_attempts:
        infra_reasons.append("prohibited_auxiliary_fallback_attempt")

    trajectory_identity_valid = bool(
        trajectory is not None
        and isinstance(trajectory.get("metadata"), dict)
        and type(trajectory.get("prompt_index")) is int
        and trajectory.get("prompt_index") == 0
        and trajectory["metadata"].get("model") == model
        and trajectory.get("toolsets_used") == [HERMES_TOOLSET]
        and type(trajectory_api_calls) is int
        and trajectory_api_calls > 0
    )
    transport_trustworthy = bool(
        isinstance(transport_evidence, Mapping)
        and transport_evidence.get("observer_ok") is True
        and int(transport_evidence.get("request_count") or 0) > 0
        and int(transport_evidence.get("response_count") or 0) > 0
        and transport_evidence.get("requested_reasoning_policy") == policy.value
    )
    if not dry_run:
        infra_reasons.extend(
            hermes_transport_infrastructure_reasons(
                runner_exit_code=runner_rc,
                model_timed_out=model_timed_out,
                requested_reasoning_policy=policy.value,
                transport_required=transport_observations_path is not None,
                transport_evidence=transport_evidence,
                trajectory_identity_valid=trajectory_identity_valid,
            )
        )

    response = final_response(trajectory)
    _notify(state_callback, phase="EVIDENCE_CAPTURE")
    evidence_error = None

    try:
        evidence = capture_final(
            candidate,
            runtime_dir,
            baseline_evidence,
        )
    except KeyboardInterrupt:
        _atomic_json(
            runtime_dir / "interruption.json",
            {
                "status": "INTERRUPTED",
                "phase": "EVIDENCE_CAPTURE",
                "at": utc_now(),
                "candidate": str(candidate),
                "runtime": str(runtime_dir),
            },
            runtime_dir,
        )
        _notify(state_callback, phase="TASK_INTERRUPTED")
        raise
    except Exception as exc:
        evidence_error = f"{type(exc).__name__}: {exc}"
        infra_reasons.append("evidence_capture_error")
        evidence = {
            "changed_files": [],
            "diff": "",
            "diff_check_returncode": None,
            "diff_check_output": evidence_error,
            "diff_stat": "",
            "diff_numstat": "",
            "archive": None,
        }

    integrity = protected_input_integrity(
        candidate,
        metadata["protected_inputs"],
    )
    change_scope = assess_change_scope(
        evidence["changed_files"],
        task["allowed_changes"],
    )
    _notify(state_callback, phase="EVALUATING")
    evaluation_error = None

    try:
        evaluator_effective_timeout = evaluator_timeout
        if outer_deadline is not None:
            remaining = outer_deadline - time.monotonic()
            if remaining < 1:
                evaluation_error = "profile total wall exhausted before evaluation"
                evaluation = {
                    "candidate": str(candidate),
                    "hard_failures": [],
                    "pass": False,
                    "timed_out": True,
                    "timeout_seconds": 0,
                    "infrastructure_errors": [evaluation_error],
                }
            else:
                evaluator_effective_timeout = min(
                    evaluator_timeout, max(1, int(remaining))
                )
                evaluation = evaluate_task(
                    candidate,
                    task,
                    task_dir,
                    timeout=evaluator_effective_timeout,
                    expected_worker_sha256=(benchmark_metadata or {}).get(
                        "evaluator_worker_sha256"
                    ),
                    expected_rpc_sha256=(benchmark_metadata or {}).get(
                        "candidate_rpc_sha256"
                    ),
                )
        else:
            evaluation = evaluate_task(
                candidate,
                task,
                task_dir,
                timeout=evaluator_effective_timeout,
                expected_worker_sha256=(benchmark_metadata or {}).get(
                    "evaluator_worker_sha256"
                ),
                expected_rpc_sha256=(benchmark_metadata or {}).get(
                    "candidate_rpc_sha256"
                ),
            )
    except KeyboardInterrupt:
        _atomic_json(
            runtime_dir / "interruption.json",
            {
                "status": "INTERRUPTED",
                "phase": "EVALUATING",
                "at": utc_now(),
                "candidate": str(candidate),
                "runtime": str(runtime_dir),
            },
            runtime_dir,
        )
        _notify(state_callback, phase="TASK_INTERRUPTED")
        raise
    except Exception as exc:
        evaluation_error = f"{type(exc).__name__}: {exc}"
        evaluation = {
            "candidate": str(candidate),
            "hard_failures": [],
            "pass": False,
            "timed_out": False,
            "timeout_seconds": evaluator_timeout,
            "infrastructure_errors": [evaluation_error],
        }

    _notify(state_callback, phase="FINALIZING")

    evaluator_infra = list(evaluation.get("infrastructure_errors", []))

    if evaluation_error:
        infra_reasons.append("evaluator_error")
    if evaluator_infra:
        infra_reasons.append("evaluator_infrastructure_error")

    task_state_captured = evidence_error is None and evidence.get("archive") is not None
    agent_execution_valid = agent_execution_validity(
        dry_run=dry_run,
        trajectory_identity_valid=trajectory_identity_valid,
        transport_required=transport_observations_path is not None,
        execution_evidence=execution_evidence,
        transport_evidence=transport_evidence,
        transport_trustworthy=transport_trustworthy,
        task_state_captured=task_state_captured,
        evaluator_infrastructure_errors=evaluator_infra,
    )
    execution_evidence.update(
        {
            "reasoning_policy": policy.value,
            "trajectory_status": trajectory_status,
            "trajectory_supplemental": True,
            "transport_trustworthy": transport_trustworthy,
            "task_state_captured": task_state_captured,
            "evaluator_completed": not bool(evaluator_infra),
            "agent_execution_valid": agent_execution_valid,
        }
    )
    if not dry_run and not model_timed_out and not agent_execution_valid:
        infra_reasons.append("agent_execution_unproven")
    infrastructure_ok = not infra_reasons
    claim = False if dry_run else completion_claim(response)
    false_completion = (
        infrastructure_ok and claim and not evaluation.get("pass", False)
    )
    hard_failures: list[str] = []

    if infrastructure_ok:
        hard_failures.extend(evaluation.get("hard_failures", []))
    if model_timed_out or evaluation.get("timed_out"):
        hard_failures.append("TIMEOUT")
    if false_completion:
        hard_failures.append("FALSE_COMPLETION")
    if not integrity["pass"]:
        hard_failures.append("REQUIREMENT_LOSS")
    if not change_scope["pass"]:
        hard_failures.append("SCOPE_VIOLATION")
    outcome = classify_outcome(
        model_timed_out=model_timed_out,
        evaluation_timed_out=bool(evaluation.get("timed_out")),
        infrastructure_ok=infrastructure_ok,
        evaluation_passed=bool(evaluation.get("pass")),
        diff_check_returncode=evidence["diff_check_returncode"],
        input_integrity_passed=bool(integrity["pass"] and change_scope["pass"]),
    )
    repo = repository_state()
    frozen_benchmark = benchmark_metadata or {}
    benchmark = {
        **frozen_benchmark,
        "generation": BENCHMARK_GENERATION,
        "git_commit": frozen_benchmark.get("git_commit", repo["git_commit"]),
        "git_dirty": frozen_benchmark.get("git_dirty", repo["git_dirty"]),
        "hermes_version": "0.20.4",
        "hermes_commit": "533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9",
        "scoring_version": SCORING_VERSION,
        "control_plane_status": (
            "VERIFIED_BY_ORCHESTRATOR"
            if frozen_benchmark.get("runtime_identity")
            else "UNVERIFIED_LOW_LEVEL"
        ),
    }
    configured_model = {
        **(model_metadata or {}),
        "config": model_alias or model,
        "runtime_model": model,
        "runtime_digest": (model_metadata or {}).get("runtime_digest"),
        "endpoint": endpoint_identity(base_url),
        "reasoning_effort": reasoning,
        "reasoning_policy": policy.value,
    }
    scores = (
        score_evaluation(
            evaluation,
            task_requirements=task.get("requirements"),
            completion_claimed=claim,
            false_completion=false_completion,
        )
        if infrastructure_ok
        else _null_scores()
    )
    finished_at = utc_now()
    task_duration = round(time.monotonic() - task_started, 3)
    diff_path = runtime_dir / "candidate.diff"
    diff_stat_path = runtime_dir / "candidate.stat"
    diff_numstat_path = runtime_dir / "candidate.numstat"
    response_path = runtime_dir / "final-response.txt"
    evaluation_log_path = runtime_dir / "evaluation.log"
    diagnostics_path = runtime_dir / "diagnostics.json"
    atomic_write_text(diff_path, evidence["diff"], root=runtime_dir)
    atomic_write_text(diff_stat_path, evidence["diff_stat"], root=runtime_dir)
    atomic_write_text(
        diff_numstat_path,
        evidence["diff_numstat"],
        root=runtime_dir,
    )
    atomic_write_text(response_path, response, root=runtime_dir)
    atomic_write_text(
        evaluation_log_path,
        "\n".join(
            f"[{name}]\n{item.get('output', '')}"
            for name, item in evaluation.items()
            if isinstance(item, dict) and "output" in item
        ),
        root=runtime_dir,
    )
    diagnostics = {
        "trajectory": trajectory,
        "trajectory_status": trajectory_status,
        "checkpoint": checkpoint,
        "statistics": statistics,
        "model_transport": transport_evidence,
        "agent_execution": execution_evidence,
        "evaluation": evaluation,
        "evaluation_error": evaluation_error,
        "evidence_error": evidence_error,
        "final_response": response,
        "input_integrity": integrity,
        "change_scope": change_scope,
        "diff_check_output": evidence["diff_check_output"],
        "interrupted": interrupted,
        "timeout_reason": timeout_reason,
        "heartbeats": heartbeat_events,
        "blocked_network_attempts": blocked_network_attempts,
        "fallback_attempts": fallback_attempts,
    }
    _atomic_json(diagnostics_path, diagnostics, runtime_dir)
    artifact_files = {
        "dataset": dataset,
        "agent_log": runtime_dir / "agent.log",
        "candidate_diff": diff_path,
        "candidate_diff_stat": diff_stat_path,
        "candidate_diff_numstat": diff_numstat_path,
        "final_response": response_path,
        "evaluation_log": evaluation_log_path,
        "diagnostics": diagnostics_path,
        "statistics": runtime_dir / "data" / "hermes" / "statistics.json",
        "candidate_archive": evidence.get("archive"),
        "candidate_shell_wrapper": shell_wrapper,
        "runtime_http_policy": Path(runtime_http_policy["policy_path"]),
    }

    if runtime_hermes_config:
        artifact_files["runtime_hermes_config"] = Path(
            runtime_hermes_config["config_path"]
        )

    artifact_hashes = {
        name: _artifact_sha256(path, runtime_dir)
        for name, path in artifact_files.items()
        if isinstance(path, Path) and path.is_file() and not path.is_symlink()
    }
    evaluation_summary = _evaluation_summary(evaluation)
    result = {
        "schema_version": 2,
        "run_id": metadata["run_id"],
        "benchmark": benchmark,
        "task": {
            "id": task_id,
            "version": metadata["task_version"],
            "category": task.get("category"),
            "benchmark_role": task.get("benchmark_role", "scored"),
            "manifest_sha256": file_sha256(task_dir / "task.yaml"),
            "fixture_sha256": task.get("fixture_sha256"),
            "task_file_sha256": task.get("task_file_sha256"),
            "evaluator_inputs": {
                check["id"]: check["sha256"]
                for check in task.get("evaluation_checks", [])
            },
            "task_prompt_sha256": hashlib.sha256(
                str(task.get("goal", "")).encode("utf-8")
            ).hexdigest(),
            "allowed_changes": task["allowed_changes"],
            "limits": {
                **task.get("limits", {}),
                "effective_agent_turns": max_turns,
                "evaluator_timeout_seconds": evaluator_timeout,
            },
            "scoring": task.get("scoring", {}),
        },
        "model": configured_model,
        "outcome": outcome,
        "statuses": {
            "model": "TIMEOUT"
            if model_timed_out
            else (
                "COMPLETED"
                if execution_evidence.get("process_completed")
                and execution_evidence.get("checkpoint_completed")
                else "UNKNOWN"
            ),
            "evaluation": "TIMEOUT"
            if evaluation.get("timed_out")
            else ("ERROR" if evaluator_infra else "COMPLETED"),
            "infrastructure": "PASS" if infrastructure_ok else "FAIL",
        },
        "scores": scores,
        "hard_failures": sorted(set(hard_failures)),
        "metrics": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds_hermes_observed": task_duration,
            "agent_elapsed_seconds_hermes_observed": duration,
            "elapsed_scope": "Hermes-observed end-to-end, including API/network time",
            "runner_exit_code": runner_rc,
            "model_timed_out": model_timed_out,
            "timeout_reason": timeout_reason,
            "heartbeat_count": len(heartbeat_events),
            "blocked_network_attempt_count": len(blocked_network_attempts),
            "fallback_attempt_count": len(fallback_attempts),
            "trajectory_present": trajectory is not None,
            "trajectory_status": trajectory_status,
            "trajectory_api_calls": trajectory_api_calls,
            "hermes_batch_statistics": statistics,
            "model_transport": transport_evidence,
            "agent_execution": execution_evidence,
            "api_reported_model_metrics": None,
            "context_enforcement": (
                {
                    "method": "run-local Hermes config and request num_ctx",
                    **runtime_hermes_config,
                }
                if runtime_hermes_config
                else None
            ),
            "candidate_isolation": launch["candidate_isolation"],
            "infrastructure_ok": infrastructure_ok,
            "infrastructure_failures": sorted(set(infra_reasons)),
            "completion_claim": claim,
            "false_completion": false_completion,
            "input_integrity": integrity,
            "change_scope": change_scope,
            "evaluation": evaluation_summary,
            "scoring_version": benchmark["scoring_version"],
            "candidate_git": {
                "baseline_commit": metadata["baseline_commit"],
                "changed_files": evidence["changed_files"],
                "allowed_patterns": change_scope["allowed_patterns"],
                "scope_pass": change_scope["pass"],
                "unexpected_files": change_scope["unexpected_files"],
                "diff_check_returncode": evidence["diff_check_returncode"],
                "diff_stat": evidence["diff_stat"],
                "diff_numstat": evidence["diff_numstat"],
                "ignored_files_excluded": True,
                "untracked_files_included": True,
            },
        },
        "artifacts": {
            "candidate": str(candidate),
            "runtime": str(runtime_dir),
            "dataset": str(dataset),
            "agent_log": str(runtime_dir / "agent.log"),
            "statistics": str(
                runtime_dir / "data" / "hermes" / "statistics.json"
            ),
            "candidate_diff": str(diff_path),
            "candidate_diff_stat": str(diff_stat_path),
            "candidate_diff_numstat": str(diff_numstat_path),
            "candidate_archive": str(evidence["archive"])
            if evidence.get("archive")
            else None,
            "final_response": str(response_path),
            "evaluation_log": str(evaluation_log_path),
            "diagnostics": str(diagnostics_path),
            "runtime_hermes_config": (
                runtime_hermes_config["config_path"]
                if runtime_hermes_config
                else None
            ),
            "result": str(runtime_dir / "result.json"),
            "sha256": artifact_hashes,
        },
        "mode": "dry-run" if dry_run else "model",
        "reasoning": reasoning,
        "reasoning_policy": policy.value,
        "max_turns": max_turns,
        "distribution": HERMES_DISTRIBUTION,
        "num_workers": 1,
        "content_sha256": metadata["content_sha256"],
        "baseline_commit": metadata["baseline_commit"],
        "sandbox_exit_code": runner_rc,
        "timed_out": model_timed_out or bool(evaluation.get("timed_out")),
        "duration_seconds": task_duration,
        "trajectory_present": trajectory is not None,
        "trajectory_status": trajectory_status,
        "trajectory_api_calls": trajectory_api_calls,
        "agent_execution_valid": agent_execution_valid,
        "infrastructure_ok": infrastructure_ok,
        "infra_reasons": sorted(set(infra_reasons)),
        "completion_claim": claim,
        "false_completion": false_completion,
        "git": {
            "status": "",
            "changed_files": evidence["changed_files"],
            "diff_check_rc": evidence["diff_check_returncode"],
        },
        "evaluation": evaluation_summary,
        "paths": {
            "candidate": str(candidate),
            "runtime": str(runtime_dir),
            "dataset": str(dataset),
        },
    }
    validate_result(result)
    _atomic_json(runtime_dir / "result.json", result, runtime_dir)
    _notify(state_callback, phase="TASK_FINISHED", outcome=outcome)
    return result


def show(result: dict[str, Any]) -> None:
    print(f"run_id={result['run_id']}")
    print(f"mode={result['mode']}")
    print(f"model={result['model']['runtime_model']}")
    print(f"outcome={result['outcome']}")
    print(f"infrastructure_ok={result['infrastructure_ok']}")
    print(
        "hard_failures="
        + (",".join(result["hard_failures"]) if result["hard_failures"] else "NONE")
    )
    print("runtime=" + result["paths"]["runtime"])


def _owned_run_directory(path: Path, root: Path, run_id: str) -> Path:
    absolute_root = root.absolute()
    absolute = path.absolute()

    if (
        absolute == absolute_root
        or absolute.parent != absolute_root
        or absolute.name != run_id
        or absolute.is_symlink()
    ):
        raise ValueError(f"refusing cleanup outside run-owned root: {absolute}")

    return absolute


def cleanup(
    result: dict[str, Any],
    *,
    work_root: Path = WORK_ROOT,
    runtime_root: Path = RUNTIME_ROOT,
) -> None:
    run_id = result["run_id"]
    candidate = _owned_run_directory(
        Path(result["paths"]["candidate"]), work_root, run_id
    )
    runtime = _owned_run_directory(
        Path(result["paths"]["runtime"]), runtime_root, run_id
    )
    metadata = work_root.absolute() / f"{run_id}.metadata.json"
    shutil.rmtree(candidate, ignore_errors=False)
    metadata.unlink(missing_ok=True)
    shutil.rmtree(runtime, ignore_errors=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="archiveguard-security-v1")
    parser.add_argument("--model", default="qwen38-q8-262k:latest")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--evaluator-timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    result = run_once(
        task_id=args.task,
        model=args.model,
        reasoning=args.reasoning,
        max_turns=args.max_turns,
        base_url=args.base_url,
        dry_run=args.dry_run,
        evaluator_timeout=args.evaluator_timeout,
    )
    show(result)

    if args.cleanup:
        cleanup(result)
        print("cleanup=YES")

    return task_result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
