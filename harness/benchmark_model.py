from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import jsonschema
import yaml

from harness.artifacts import (
    atomic_write_text,
    ensure_root,
    ensure_subdirectory,
)
from harness.endpoints import EndpointPolicyError, validate_local_openai_endpoint
from harness.hermes_runner import (
    HERMES_DISTRIBUTION,
    HERMES_TOOLSET,
    RUNTIME_ROOT,
    endpoint_identity,
    git,
    repository_state,
    run_once,
    validate_result,
)
from harness.model_identity import extract_runtime_metadata
from harness.reasoning_policy import ReasoningPolicyError, parse_reasoning_policy
from harness.workspace import ROOT, content_digest, discover_tasks


DEFAULT_CONFIG = ROOT / "configs" / "hermes-bench-v3.yaml"
DEFAULT_MODELS = ROOT / "models.yaml"
TASK_SCHEMA = ROOT / "schemas" / "task.schema.json"
AGGREGATE_SCHEMA = ROOT / "schemas" / "benchmark-run.schema.json"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
FAILURE_POLICY = "continue_model_failures_stop_infrastructure"
EXPECTED_ENDPOINT = "http://127.0.0.1:11434/v1"
EXPECTED_GENERATION = "hermesbench-v3"
EXPECTED_SCORING_VERSION = "hermesbench-v2-automated-1"
EXPECTED_CANDIDATE_ISOLATION = (
    "trusted-shell-wrapper-bubblewrap-networkless"
)
EXPECTED_EVALUATOR_ISOLATION = (
    "nested-bubblewrap-networkless-test-blind-rpc"
)
MAX_METADATA_RESPONSE_BYTES = 4 * 1024 * 1024


class ConfigurationError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


class SuiteInterrupted(KeyboardInterrupt):
    def __init__(self, aggregate: dict[str, Any]) -> None:
        super().__init__("benchmark interrupted")
        self.aggregate = aggregate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()

    if not path.is_file():
        raise ConfigurationError(f"{label} does not exist: {path}")

    try:
        value = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"cannot read {label}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must contain a mapping")

    return value


def _validate_model_config(alias: str, model: Mapping[str, Any]) -> None:
    if not isinstance(alias, str) or not isinstance(model, Mapping):
        raise ConfigurationError("invalid model alias entry")

    for field in (
        "display_name",
        "provider",
        "runtime",
        "runtime_model",
        "quantization",
        "context_length",
        "reasoning_effort",
        "reasoning_policy",
        "supported_reasoning_policies",
    ):
        if field not in model:
            raise ConfigurationError(
                f"model {alias!r} is missing {field}"
            )

    for field in (
        "display_name",
        "provider",
        "runtime",
        "runtime_model",
        "quantization",
        "reasoning_effort",
        "reasoning_policy",
    ):
        if not isinstance(model[field], str) or not model[field]:
            raise ConfigurationError(
                f"model {alias!r} has invalid {field}"
            )

    if type(model["context_length"]) is not int or (
        model["context_length"] < 1
    ):
        raise ConfigurationError(
            f"model {alias!r} has invalid context_length"
        )

    supported = model["supported_reasoning_policies"]
    if not (
        isinstance(supported, list)
        and supported
        and all(isinstance(value, str) for value in supported)
    ):
        raise ConfigurationError(
            f"model {alias!r} has invalid supported_reasoning_policies"
        )
    try:
        policy = parse_reasoning_policy(model["reasoning_policy"])
        supported_policies = [parse_reasoning_policy(value) for value in supported]
    except ReasoningPolicyError as exc:
        raise ConfigurationError(f"model {alias!r}: {exc}") from exc
    if policy.value not in {value.value for value in supported_policies}:
        raise ConfigurationError(
            f"model {alias!r} configured reasoning_policy is not supported"
        )

    runtime_digest = model.get("runtime_digest")

    if runtime_digest is not None and not (
        isinstance(runtime_digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_digest)
    ):
        raise ConfigurationError(
            f"model {alias!r} has invalid runtime_digest; "
            "use a full sha256 digest or null"
        )

    runtime = model["runtime"]
    if runtime not in {"ollama", "ds4"}:
        raise ConfigurationError(
            f"model {alias!r} has unsupported runtime {runtime!r}"
        )
    if runtime == "ds4":
        for field in ("runtime_version", "api_mode"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise ConfigurationError(
                    f"DS4 model {alias!r} has invalid {field}"
                )
        if model["api_mode"] != "openai-chat-completions":
            raise ConfigurationError(
                f"DS4 model {alias!r} must use OpenAI chat completions"
            )
        if type(model.get("endpoint_port")) is not int or model["endpoint_port"] < 1:
            raise ConfigurationError(
                f"DS4 model {alias!r} has invalid endpoint_port"
            )
        artifacts = model.get("deployment_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ConfigurationError(
                f"DS4 model {alias!r} has no deployment_artifacts"
            )
        for field in ("base_gguf_sha256", "dspark_drafter_sha256"):
            if not (
                isinstance(artifacts.get(field), str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", artifacts[field])
            ):
                raise ConfigurationError(
                    f"DS4 model {alias!r} has invalid {field}"
                )
        if artifacts.get("dspark_enabled") is not True:
            raise ConfigurationError(
                f"DS4 model {alias!r} must record DSpark enabled state"
            )
        if runtime_digest != artifacts["base_gguf_sha256"]:
            raise ConfigurationError(
                f"DS4 model {alias!r} runtime_digest must equal its base GGUF checksum"
            )
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ConfigurationError(
                f"DS4 model {alias!r} has invalid capabilities"
            )
        for field in ("streaming", "tool_calls"):
            if capabilities.get(field) is not True:
                raise ConfigurationError(
                    f"DS4 model {alias!r} must record {field} capability"
                )


def load_models(path: Path = DEFAULT_MODELS) -> dict[str, dict[str, Any]]:
    document = _load_yaml(path, "model configuration")

    if document.get("schema_version") != 3:
        raise ConfigurationError(
            "models.yaml schema_version must be 3"
        )

    models = document.get("models")

    if not isinstance(models, dict) or not models:
        raise ConfigurationError("models.yaml has no model aliases")

    for alias, model in models.items():
        _validate_model_config(alias, model)

    return models


def _validate_endpoint(value: Any) -> str:
    try:
        return validate_local_openai_endpoint(value).base_url
    except EndpointPolicyError as exc:
        raise ConfigurationError(str(exc)) from exc


def _load_task_catalog(
    task_root: Path,
    task_schema: Path,
) -> dict[str, tuple[dict[str, Any], Path]]:
    schema = json.loads(
        task_schema.read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    catalog: dict[str, tuple[dict[str, Any], Path]] = {}

    for task, task_dir in discover_tasks(task_root):
        try:
            json.dumps(task, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"task manifest is not strict JSON data {task_dir / 'task.yaml'}: "
                f"{exc}"
            ) from exc

        errors = sorted(
            validator.iter_errors(task),
            key=lambda error: list(error.path),
        )

        if errors:
            detail = errors[0].message
            raise ConfigurationError(
                f"invalid task manifest {task_dir / 'task.yaml'}: {detail}"
            )

        task_id = task["id"]

        if task_id in catalog:
            raise ConfigurationError(f"duplicate task id: {task_id}")

        catalog[task_id] = (task, task_dir)

    return catalog


def _validate_task_assets(
    task: dict[str, Any],
    task_dir: Path,
    repository_root: Path,
) -> None:
    repository_root = repository_root.resolve()
    fixture = (repository_root / task["fixture"]).resolve()

    if (
        not fixture.is_relative_to(repository_root)
        or not fixture.is_dir()
    ):
        raise ConfigurationError(
            f"task {task['id']} fixture is missing or outside the repository"
        )

    if content_digest(fixture) != task["fixture_sha256"]:
        raise ConfigurationError(
            f"task {task['id']} fixture digest mismatch"
        )

    task_file = task_dir / "TASK.md"

    if not task_file.is_file():
        raise ConfigurationError(
            f"task {task['id']} is missing TASK.md"
        )

    if file_sha256(task_file) != task["task_file_sha256"]:
        raise ConfigurationError(
            f"task {task['id']} task-file digest mismatch"
        )

    checks = task.get("evaluation_checks")

    if not isinstance(checks, list) or not checks:
        raise ConfigurationError(
            f"task {task['id']} has no structured evaluator checks"
        )

    requirement_list = [
        item["id"] for item in task.get("requirements", [])
    ]
    requirement_ids = set(requirement_list)

    if len(requirement_ids) != len(requirement_list):
        raise ConfigurationError(
            f"task {task['id']} has duplicate requirement ids"
        )

    for pattern in task["allowed_changes"]:
        path = PurePosixPath(pattern)
        if path.is_absolute() or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise ConfigurationError(
                f"task {task['id']} has unsafe allowed-change pattern: {pattern}"
            )

    check_ids: set[str] = set()
    mapped_requirements: set[str] = set()

    for check in checks:
        check_id = check["id"]

        if check_id in check_ids:
            raise ConfigurationError(
                f"task {task['id']} has duplicate evaluator check {check_id}"
            )

        check_ids.add(check_id)
        test_path = (task_dir / check["test_file"]).absolute()

        if (
            not test_path.is_relative_to(task_dir.absolute())
            or test_path.is_symlink()
            or not test_path.is_file()
        ):
            raise ConfigurationError(
                f"task {task['id']} evaluator input is unsafe: {test_path}"
            )

        if file_sha256(test_path) != check["sha256"]:
            raise ConfigurationError(
                f"task {task['id']} evaluator digest mismatch: {check_id}"
            )

        if check["visibility"] == "public":
            candidate_path = (fixture / check["candidate_path"]).absolute()

            if (
                not candidate_path.is_relative_to(fixture.absolute())
                or candidate_path.is_symlink()
                or not candidate_path.is_file()
                or file_sha256(candidate_path) != check["sha256"]
            ):
                raise ConfigurationError(
                    f"task {task['id']} public candidate input is unsafe or "
                    f"does not match evaluator input: {candidate_path}"
                )

        for case_id, mapping in check["cases"].items():
            unknown = set(mapping["requirements"]) - requirement_ids

            if unknown:
                raise ConfigurationError(
                    f"task {task['id']} evaluator case {case_id} maps "
                    "unknown requirements: " + ", ".join(sorted(unknown))
                )

            mapped_requirements.update(mapping["requirements"])

    unmapped = requirement_ids - mapped_requirements

    if unmapped:
        raise ConfigurationError(
            f"task {task['id']} has requirements without evaluator cases: "
            + ", ".join(sorted(unmapped))
        )


def _detected_hermes_commit(install: str) -> str | None:
    install = str(Path(os.path.expandvars(install)).expanduser().resolve())
    try:
        process = subprocess.run(
            ["git", "-C", install, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if process.returncode:
        return None

    return process.stdout.strip() or None


def _detected_hermes_state(install: str) -> dict[str, Any]:
    install = str(Path(os.path.expandvars(install)).expanduser().resolve())
    commit = _detected_hermes_commit(install)
    dirty: bool | None = None
    version: str | None = None

    try:
        status = subprocess.run(
            ["git", "-C", install, "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        detected = subprocess.run(
            [
                str(Path(install) / "venv" / "bin" / "python"),
                "-c",
                "import importlib.metadata; "
                "print(importlib.metadata.version('hermes-agent'))",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
        if detected.returncode == 0:
            version = detected.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass

    return {"commit": commit, "dirty": dirty, "version": version}


def load_execution_plan(
    model_alias: str,
    *,
    model_override: Mapping[str, Any] | None = None,
    config_path: Path = DEFAULT_CONFIG,
    models_path: Path = DEFAULT_MODELS,
    task_root: Path = ROOT,
    task_schema: Path = TASK_SCHEMA,
    requested_tasks: Sequence[str] = (),
) -> dict[str, Any]:
    config_path = config_path.resolve()
    models_path = models_path.resolve()
    for schema_path in (
        ROOT / "schemas" / "result.schema.json",
        AGGREGATE_SCHEMA,
    ):
        jsonschema.Draft202012Validator.check_schema(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )
    config = _load_yaml(config_path, "benchmark configuration")
    models = load_models(models_path)

    if config.get("schema_version") != 2:
        raise ConfigurationError(
            "benchmark configuration schema_version must be 2; "
            "generation-v1 data is incompatible with this runner"
        )

    if model_override is None and model_alias not in models:
        raise ConfigurationError(f"unknown model alias: {model_alias}")

    if model_override is not None:
        _validate_model_config(model_alias, model_override)
        model = dict(model_override)
    else:
        model = models[model_alias]

    generation = config.get("benchmark_generation")
    hermes = config.get("hermes")
    runtime = config.get("baseline_runtime")
    suite = config.get("suite")
    execution = config.get("execution")

    if not isinstance(generation, str) or not generation:
        raise ConfigurationError("benchmark_generation is missing")
    if generation != EXPECTED_GENERATION:
        raise ConfigurationError(
            f"incompatible benchmark generation: {generation!r}; "
            f"expected {EXPECTED_GENERATION!r}"
        )
    if not isinstance(hermes, dict):
        raise ConfigurationError("Hermes configuration is missing")
    if not isinstance(runtime, dict):
        raise ConfigurationError("baseline_runtime is missing")
    if not isinstance(suite, dict) or not isinstance(
        suite.get("tasks"), list
    ):
        raise ConfigurationError("configured benchmark suite is missing")
    if not isinstance(execution, dict):
        raise ConfigurationError("execution policy is missing")

    endpoint = _validate_endpoint(runtime.get("endpoint"))

    if model_override is None and model["runtime"] != runtime.get("runtime"):
        raise ConfigurationError(
            "model runtime does not match the benchmark baseline runtime"
        )
    policy = execution.get("failure_policy")

    if policy != FAILURE_POLICY:
        raise ConfigurationError(
            f"unsupported failure policy: {policy!r}"
        )

    evaluator_timeout = execution.get("evaluator_timeout_seconds")
    evaluator_worker_sha256 = execution.get("evaluator_worker_sha256")
    candidate_rpc_sha256 = execution.get("candidate_rpc_sha256")
    scoring_version = execution.get("scoring_version")
    tool_distribution = execution.get("tool_distribution")
    candidate_toolsets = execution.get("candidate_toolsets")
    candidate_isolation = execution.get("candidate_isolation")
    evaluator_isolation = execution.get("evaluator_isolation")
    endpoint_redirects = execution.get("endpoint_redirects")

    if type(evaluator_timeout) is not int or evaluator_timeout < 1:
        raise ConfigurationError(
            "evaluator_timeout_seconds must be a positive integer"
        )
    if not (
        isinstance(evaluator_worker_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", evaluator_worker_sha256)
    ):
        raise ConfigurationError("evaluator_worker_sha256 must be a full sha256")
    if file_sha256(ROOT / "harness" / "evaluator_worker.py") != (
        evaluator_worker_sha256
    ):
        raise ConfigurationError("frozen evaluator worker digest mismatch")
    if not (
        isinstance(candidate_rpc_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", candidate_rpc_sha256)
    ):
        raise ConfigurationError("candidate_rpc_sha256 must be a full sha256")
    if file_sha256(ROOT / "harness" / "candidate_rpc.py") != (
        candidate_rpc_sha256
    ):
        raise ConfigurationError("frozen candidate RPC digest mismatch")

    if not isinstance(scoring_version, str) or not scoring_version:
        raise ConfigurationError("scoring_version is missing")
    if scoring_version != EXPECTED_SCORING_VERSION:
        raise ConfigurationError(
            f"incompatible scoring version: {scoring_version!r}"
        )
    if tool_distribution != HERMES_DISTRIBUTION:
        raise ConfigurationError(
            f"incompatible candidate tool distribution: {tool_distribution!r}"
        )
    if candidate_toolsets != [HERMES_TOOLSET]:
        raise ConfigurationError(
            "candidate_toolsets must contain only the gated foreground "
            "terminal toolset"
        )
    if candidate_isolation != EXPECTED_CANDIDATE_ISOLATION:
        raise ConfigurationError(
            "candidate_isolation must select the frozen trusted wrapper policy"
        )
    if evaluator_isolation != EXPECTED_EVALUATOR_ISOLATION:
        raise ConfigurationError(
            "evaluator_isolation must select the frozen nested RPC policy"
        )
    if endpoint_redirects != "reject":
        raise ConfigurationError("endpoint_redirects must be reject")

    configured_ids = suite["tasks"]

    if (
        not configured_ids
        or any(not isinstance(item, str) for item in configured_ids)
        or len(configured_ids) != len(set(configured_ids))
    ):
        raise ConfigurationError(
            "suite tasks must be a non-empty unique list of task ids"
        )

    catalog = _load_task_catalog(task_root, task_schema)
    missing = [item for item in configured_ids if item not in catalog]

    if missing:
        raise ConfigurationError(
            "configured task not found: " + ", ".join(missing)
        )

    for task_id in configured_ids:
        task, task_dir = catalog[task_id]
        _validate_task_assets(task, task_dir, task_root)

    requested = set(requested_tasks)

    if len(requested) != len(requested_tasks):
        raise ConfigurationError("duplicate --task selection")

    outside_suite = requested - set(configured_ids)

    if outside_suite:
        raise ConfigurationError(
            "task is not in the configured suite: "
            + ", ".join(sorted(outside_suite))
        )

    selected_ids = [
        task_id
        for task_id in configured_ids
        if not requested or task_id in requested
    ]
    selected = [
        {
            "manifest": task,
            "directory": task_dir,
        }
        for task, task_dir in (catalog[task_id] for task_id in selected_ids)
    ]
    repo = repository_state()
    documented_commit = hermes.get("commit")
    documented_version = hermes.get("version")
    install = hermes.get("install")

    if not all(
        isinstance(value, str) and value
        for value in (
            documented_commit,
            documented_version,
            install,
        )
    ):
        raise ConfigurationError("incomplete Hermes version configuration")

    install_override = os.environ.get("HERMES_AGENT_ROOT")
    if install_override:
        install = install_override
    elif install == "~/.hermes/hermes-agent" and os.environ.get("HERMES_HOME"):
        install = str(Path(os.environ["HERMES_HOME"]) / "hermes-agent")
    install = str(Path(os.path.expandvars(install)).expanduser().resolve())
    detected_hermes = _detected_hermes_state(install)
    detected_commit = detected_hermes["commit"]

    if detected_commit != documented_commit:
        raise ConfigurationError(
            "detected Hermes commit does not match the frozen configuration"
        )
    if detected_hermes["version"] != documented_version:
        raise ConfigurationError(
            "detected Hermes version does not match the frozen configuration"
        )

    configured_runtime_version = (
        model.get("runtime_version")
        if model["runtime"] != "ollama"
        else runtime.get("ollama_version")
    )

    if not isinstance(configured_runtime_version, str) or not configured_runtime_version:
        raise ConfigurationError("baseline Ollama version is missing")

    return {
        "model_alias": model_alias,
        "model": model,
        "endpoint": endpoint,
        "tasks": selected,
        "failure_policy": policy,
        "evaluator_timeout_seconds": evaluator_timeout,
        "evaluator_worker_sha256": evaluator_worker_sha256,
        "candidate_rpc_sha256": candidate_rpc_sha256,
        "scoring_version": scoring_version,
        "tool_distribution": tool_distribution,
        "candidate_toolsets": list(candidate_toolsets),
        "candidate_isolation": candidate_isolation,
        "evaluator_isolation": evaluator_isolation,
        "endpoint_redirects": endpoint_redirects,
        "enforce_control_plane_clean": task_root.resolve() == ROOT.resolve(),
        "benchmark": {
            "generation": generation,
            "git_commit": repo["git_commit"],
            "git_dirty": repo["git_dirty"],
            "hermes_version": documented_version,
            "hermes_commit": documented_commit,
            "hermes_detected_commit": detected_commit,
            "hermes_detected_version": detected_hermes["version"],
            "hermes_detected_dirty": detected_hermes["dirty"],
            "hermes_install": install,
            "runtime_version_configured": configured_runtime_version,
            "runtime_version_observed": None,
            "runtime_version_status": "UNVERIFIED_METADATA_SCOPE",
            "hermes_observed_live_config": config.get(
                "observed_live_config",
                {},
            ),
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "models_path": str(models_path),
            "models_sha256": file_sha256(models_path),
            "evaluator_scoring_git_commit": repo["git_commit"],
            "evaluator_worker_sha256": evaluator_worker_sha256,
            "candidate_rpc_sha256": candidate_rpc_sha256,
            "scoring_version": scoring_version,
        },
    }


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _endpoint_json(
    url: str,
    *,
    timeout: int,
    request_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoded_body = (
        json.dumps(request_body, allow_nan=False).encode("utf-8")
        if request_body is not None
        else None
    )
    headers = {"Authorization": "Bearer ollama"}

    if encoded_body is not None:
        headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=encoded_body,
        headers=headers,
        method="POST" if encoded_body is not None else "GET",
    )
    opener = build_opener(ProxyHandler({}), _RejectRedirects())

    try:
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != url:
                raise InfrastructureError("model endpoint redirected unexpectedly")

            raw = response.read(MAX_METADATA_RESPONSE_BYTES + 1)

            if len(raw) > MAX_METADATA_RESPONSE_BYTES:
                raise InfrastructureError(
                    "model endpoint metadata response exceeds size limit"
                )

            payload = json.loads(raw.decode("utf-8"))
    except InfrastructureError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise InfrastructureError(
            "model endpoint preflight failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise InfrastructureError("model endpoint returned a non-object response")

    return payload


def preflight_model(
    endpoint: str,
    runtime_model: str,
    *,
    expected_digest: str | None,
    timeout: int = 10,
    runtime_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        endpoint = validate_local_openai_endpoint(endpoint).base_url
    except EndpointPolicyError as exc:
        raise InfrastructureError(f"refusing non-local model endpoint: {exc}") from exc

    if expected_digest is not None and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", expected_digest
    ):
        raise InfrastructureError(
            "configured expected runtime digest is malformed"
        )

    models_payload = _endpoint_json(
        endpoint + "/models",
        timeout=timeout,
    )
    available = models_payload.get("data")

    if not isinstance(available, list):
        raise InfrastructureError("model endpoint returned an invalid /models response")

    model_matches = [
        item
        for item in available
        if isinstance(item, dict) and item.get("id") == runtime_model
    ]

    if len(model_matches) != 1:
        raise InfrastructureError(
            f"configured model identity is missing or ambiguous: {runtime_model} "
            "(models are never pulled automatically)"
        )

    runtime_config = (
        runtime_config if isinstance(runtime_config, Mapping) else {}
    )
    runtime = str(runtime_config.get("runtime") or "ollama")
    if runtime == "ds4":
        if expected_digest is None:
            raise InfrastructureError(
                "DS4 requires a configured immutable base GGUF checksum"
            )
        artifacts = runtime_config.get("deployment_artifacts")
        if not isinstance(artifacts, Mapping):
            raise InfrastructureError("DS4 deployment artifact provenance is missing")
        if artifacts.get("base_gguf_sha256") != expected_digest:
            raise InfrastructureError(
                "DS4 base GGUF checksum does not match runtime_digest"
            )
        if not (
            isinstance(artifacts.get("dspark_drafter_sha256"), str)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(artifacts["dspark_drafter_sha256"]),
            )
            and artifacts.get("dspark_enabled") is True
        ):
            raise InfrastructureError("DS4 DSpark provenance is incomplete")
        runtime_version = runtime_config.get("runtime_version")
        if not isinstance(runtime_version, str) or not runtime_version:
            raise InfrastructureError("DS4 runtime version is missing")
        match = model_matches[0]
        runtime_metadata = {
            key: runtime_config[key]
            for key in (
                "canonical_name",
                "source_model",
                "source_version",
                "architecture",
                "parameter_variant",
                "parameter_count",
                "quantization",
                "native_context_length",
            )
            if runtime_config.get(key) is not None
        }
        if "source_version" in runtime_metadata:
            runtime_metadata["version"] = runtime_metadata.pop("source_version")
        runtime_metadata["effective_context_length"] = runtime_config.get(
            "context_length"
        )
        runtime_metadata["metadata_sources"] = {
            "runtime_model": "/v1/models exact identifier match",
            "runtime_digest": "models.yaml operator-verified base GGUF checksum",
            "dspark_drafter": "models.yaml operator-verified DSpark checksum",
            "effective_context_length": "models.yaml configured DS4 context",
        }
        return {
            "endpoint": endpoint,
            "redirect_policy": "reject",
            "runtime": "ds4",
            "runtime_version": runtime_version,
            "runtime_version_source": "models.yaml installed deployment provenance",
            "runtime_model": runtime_model,
            "reported_id": match.get("id"),
            "reported_object": match.get("object"),
            "reported_owner": match.get("owned_by"),
            "runtime_model_digest": expected_digest,
            "expected_runtime_digest": expected_digest,
            "expected_digest_status": "CONFIGURED_ARTIFACT_MATCH",
            "metadata_endpoints": {
                "/v1/models": {"method": "GET", "status": "REACHABLE"}
            },
            "identity_status": "VERIFIED_MODEL_ID_AND_CONFIGURED_ARTIFACTS",
            "endpoint_api_mode": "openai-chat-completions",
            "deployment_artifacts": dict(artifacts),
            "capabilities": dict(runtime_config.get("capabilities") or {}),
            "public_runtime_metadata": runtime_metadata,
        }
    if runtime != "ollama":
        raise InfrastructureError(f"unsupported model runtime: {runtime}")

    parsed = urlsplit(endpoint)
    tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
    tags_payload = _endpoint_json(tags_url, timeout=timeout)
    tags = tags_payload.get("models")

    if not isinstance(tags, list):
        raise InfrastructureError("model endpoint returned an invalid /api/tags response")

    identity_matches: list[dict[str, Any]] = []

    for item in tags:
        if not isinstance(item, dict):
            continue

        identifiers = {
            value
            for value in (item.get("name"), item.get("model"))
            if isinstance(value, str) and value
        }

        if runtime_model not in identifiers:
            continue

        if identifiers != {runtime_model}:
            raise InfrastructureError(
                f"runtime alias/identifier mismatch for {runtime_model}: "
                + ", ".join(sorted(identifiers))
            )

        identity_matches.append(item)

    if len(identity_matches) != 1:
        raise InfrastructureError(
            f"runtime digest is missing or ambiguous for {runtime_model}"
        )

    raw_digest = identity_matches[0].get("digest")

    if not isinstance(raw_digest, str):
        raise InfrastructureError("runtime returned a malformed model digest")

    reported_digest = raw_digest.lower()

    if re.fullmatch(r"[0-9a-f]{64}", reported_digest):
        reported_digest = "sha256:" + reported_digest

    if not re.fullmatch(r"sha256:[0-9a-f]{64}", reported_digest):
        raise InfrastructureError("runtime returned a malformed model digest")

    if expected_digest is not None and reported_digest != expected_digest:
        raise InfrastructureError(
            f"runtime digest mismatch for {runtime_model}: "
            f"expected {expected_digest}, received {reported_digest}"
        )

    show_url = f"{parsed.scheme}://{parsed.netloc}/api/show"
    show_payload = _endpoint_json(
        show_url,
        timeout=timeout,
        request_body={"model": runtime_model, "verbose": False},
    )
    show_identifiers = {
        value
        for value in (show_payload.get("name"), show_payload.get("model"))
        if isinstance(value, str) and value
    }

    if show_identifiers and show_identifiers != {runtime_model}:
        raise InfrastructureError(
            f"/api/show alias/identifier mismatch for {runtime_model}: "
            + ", ".join(sorted(show_identifiers))
        )

    if not any(
        key in show_payload
        for key in (
            "details",
            "model_info",
            "modelfile",
            "parameters",
            "template",
            "capabilities",
        )
    ):
        raise InfrastructureError(
            "model endpoint returned an invalid /api/show response"
        )

    match = model_matches[0]
    runtime_metadata = extract_runtime_metadata(
        identity_matches[0], show_payload
    )
    expected_template = runtime_config.get("template_sha256")
    if (
        isinstance(expected_template, str)
        and runtime_metadata.get("template_sha256") != expected_template
    ):
        raise InfrastructureError(
            f"runtime template checksum mismatch for {runtime_model}"
        )
    expected_capabilities = runtime_config.get("runtime_capabilities")
    if (
        isinstance(expected_capabilities, list)
        and sorted(expected_capabilities)
        != runtime_metadata.get("capabilities")
    ):
        raise InfrastructureError(
            f"runtime capabilities mismatch for {runtime_model}"
        )
    metadata_sources = runtime_metadata.setdefault("metadata_sources", {})
    if isinstance(metadata_sources, dict):
        metadata_sources.update(
            {
                "runtime_model": "/v1/models and /api/tags exact identifier match",
                "runtime_digest": "/api/tags digest",
            }
        )
    return {
        "endpoint": endpoint,
        "redirect_policy": "reject",
        "runtime_model": runtime_model,
        "reported_id": match.get("id"),
        "reported_object": match.get("object"),
        "reported_owner": match.get("owned_by"),
        "runtime_model_digest": reported_digest,
        "expected_runtime_digest": expected_digest,
        "expected_digest_status": (
            "MATCHED" if expected_digest is not None else "NOT_CONFIGURED"
        ),
        "metadata_endpoints": {
            "/v1/models": {"method": "GET", "status": "REACHABLE"},
            "/api/tags": {"method": "GET", "status": "REACHABLE"},
            "/api/show": {"method": "POST", "status": "REACHABLE"},
        },
        "identity_status": "VERIFIED",
        "public_runtime_metadata": runtime_metadata,
    }


def new_run_id(model_alias: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_alias = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_alias).strip("-")
    return f"{timestamp}-{safe_alias}-{uuid.uuid4().hex[:10]}"


@contextmanager
def _blocked_write_signals():
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return

    blocked = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)

    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _atomic_json(path: Path, value: dict[str, Any], *, root: Path) -> None:
    _atomic_text(
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


def _atomic_text(path: Path, value: str, *, root: Path) -> None:
    with _blocked_write_signals():
        atomic_write_text(path, value, root=root)


def _aggregate_score(task_results: Sequence[dict[str, Any]]) -> float | None:
    values = [
        item.get("scores", {}).get("overall")
        for item in task_results
        if item.get("outcome") != "HARNESS_ERROR"
        and item.get("scores", {}).get("overall") is not None
    ]

    if not values:
        return None

    return round(sum(values) / len(values), 3)


def validate_aggregate(aggregate: dict[str, Any]) -> None:
    try:
        json.dumps(aggregate, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise jsonschema.ValidationError(
            f"aggregate is not strict JSON: {exc}"
        ) from exc

    schema = json.loads(
        AGGREGATE_SCHEMA.read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(aggregate)

    for result in aggregate["task_results"]:
        validate_result(result)


def _markdown_report(aggregate: dict[str, Any]) -> str:
    model = aggregate["model"]
    score = aggregate["aggregate_score"]
    score_text = "unavailable" if score is None else f"{score:.3f}"
    lines = [
        f"# Benchmark run {aggregate['run_id']}",
        "",
        f"- Status: **{aggregate['status']}**",
        f"- Model: `{model['config']}` (`{model['runtime_model']}`)",
        f"- Endpoint: `{model['endpoint']}`",
        f"- Aggregate score: **{score_text}**",
        f"- Started: {aggregate['started_at']}",
        f"- Finished: {aggregate['finished_at']}",
        f"- Failure policy: `{aggregate['failure_policy']}`",
        "",
        "## Task outcomes",
        "",
        "| Task | Outcome | Score | Elapsed (Hermes-observed) | Hard failures |",
        "|---|---:|---:|---:|---|",
    ]

    for result in aggregate["task_results"]:
        task_score = result["scores"].get("overall")
        task_score_text = (
            "—" if task_score is None else f"{task_score:.3f}"
        )
        elapsed = result["metrics"].get(
            "elapsed_seconds_hermes_observed"
        )
        failures = ", ".join(result["hard_failures"]) or "None"
        lines.append(
            f"| `{result['task']['id']}` | {result['outcome']} | "
            f"{task_score_text} | {elapsed if elapsed is not None else '—'} | "
            f"{failures} |"
        )

    if aggregate.get("infrastructure_failures"):
        lines.extend(
            [
                "",
                "## Infrastructure failures",
                "",
                *(
                    f"- {failure}"
                    for failure in aggregate["infrastructure_failures"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Client-observed elapsed time includes Hermes orchestration, API,",
            "network, model inference, and tool execution; it is not pure model",
            "compute time.",
            "",
        ]
    )

    return "\n".join(lines)


def _write_outputs(aggregate: dict[str, Any]) -> None:
    with _blocked_write_signals():
        validate_aggregate(aggregate)
        result_path = Path(aggregate["artifacts"]["result"])
        report_path = Path(aggregate["artifacts"]["report"])
        state_path = Path(aggregate["artifacts"]["runtime_state"])
        results_root = ensure_root(result_path.parent)
        reports_root = ensure_root(report_path.parent)
        runtime_root = ensure_root(state_path.parent)
        _atomic_json(result_path, aggregate, root=results_root)
        _atomic_json(state_path, aggregate, root=runtime_root)
        _atomic_text(
            report_path,
            _markdown_report(aggregate),
            root=reports_root,
        )


def _model_result_metadata(plan: dict[str, Any]) -> dict[str, Any]:
    model = plan["model"]

    return {
        key: model.get(key)
        for key in (
            "display_name",
            "provider",
            "runtime",
            "quantization",
            "context_length",
            "runtime_digest",
        )
        if key in model
    }


def _verify_control_plane(
    plan: dict[str, Any],
    *,
    allowed_artifacts: Sequence[Path],
) -> None:
    if not plan.get("enforce_control_plane_clean"):
        return

    benchmark = plan["benchmark"]

    if benchmark.get("git_dirty"):
        raise InfrastructureError(
            "benchmark repository was dirty when the execution plan was loaded"
        )

    current = repository_state()

    if current["git_commit"] != benchmark.get("git_commit"):
        raise InfrastructureError("benchmark repository commit changed during run")

    allowed_status: set[str] = set()

    for path in allowed_artifacts:
        absolute = path.absolute()
        try:
            relative = absolute.relative_to(ROOT.absolute()).as_posix()
        except ValueError as exc:
            raise InfrastructureError(
                f"run artifact is outside the benchmark repository: {absolute}"
            ) from exc
        allowed_status.add(f"?? {relative}")

    status = git(
        ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ).stdout.splitlines()
    unexpected = sorted(set(status) - allowed_status)

    if unexpected:
        raise InfrastructureError(
            "benchmark repository changed during run: " + ", ".join(unexpected)
        )

    hermes_state = _detected_hermes_state(benchmark["hermes_install"])

    if (
        hermes_state["commit"] != benchmark.get("hermes_commit")
        or hermes_state["version"] != benchmark.get("hermes_version")
        or hermes_state["dirty"] is not False
    ):
        raise InfrastructureError(
            "Hermes installation changed or became dirty during run"
        )


@contextmanager
def _exclusive_endpoint_lock(
    runtime_root: Path,
    endpoint: str,
):
    runtime_root = ensure_root(runtime_root)
    lock_root = ensure_subdirectory(
        runtime_root,
        runtime_root / ".locks",
    )
    digest = hashlib.sha256(
        endpoint_identity(endpoint).encode("utf-8")
    ).hexdigest()[:16]
    lock_path = lock_root / f"endpoint-{digest}.lock"

    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise InfrastructureError(f"unsafe endpoint lock path: {exc}") from exc

    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise InfrastructureError(
                "another benchmark is already using this inference endpoint"
            ) from exc

        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def execute_suite(
    plan: dict[str, Any],
    *,
    preflight: Callable[..., dict[str, Any]] = preflight_model,
    task_runner: Callable[..., dict[str, Any]] = run_once,
    results_dir: Path = RESULTS_DIR,
    reports_dir: Path = REPORTS_DIR,
    runtime_root: Path = RUNTIME_ROOT,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    with _termination_as_interrupt():
        with _exclusive_endpoint_lock(runtime_root, plan["endpoint"]):
            return _execute_suite_unlocked(
                plan,
                preflight=preflight,
                task_runner=task_runner,
                results_dir=results_dir,
                reports_dir=reports_dir,
                runtime_root=runtime_root,
                progress=progress,
            )


@contextmanager
def _termination_as_interrupt():
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
    except ValueError:
        previous = None

    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


def _execute_suite_unlocked(
    plan: dict[str, Any],
    *,
    preflight: Callable[..., dict[str, Any]],
    task_runner: Callable[..., dict[str, Any]],
    results_dir: Path,
    reports_dir: Path,
    runtime_root: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    run_started = time.monotonic()
    run_id = new_run_id(plan["model_alias"])
    started_at = utc_now()
    results_dir = ensure_root(results_dir)
    reports_dir = ensure_root(reports_dir)
    runtime_root = ensure_root(runtime_root)
    result_path = results_dir / f"{run_id}.json"
    report_path = reports_dir / f"{run_id}.md"
    state_path = runtime_root / f"{run_id}.aggregate.json"
    planned_ids = [
        item["manifest"]["id"]
        for item in plan["tasks"]
    ]
    aggregate = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "INCOMPLETE",
        "benchmark": plan["benchmark"],
        "model": {
            "config": plan["model_alias"],
            "runtime_model": plan["model"]["runtime_model"],
            "endpoint": endpoint_identity(plan["endpoint"]),
            **_model_result_metadata(plan),
        },
        "failure_policy": plan["failure_policy"],
        "started_at": started_at,
        "finished_at": started_at,
        "elapsed_seconds_hermes_observed": 0.0,
        "elapsed_scope": (
            "Hermes-observed suite time including preflight, task setup, "
            "API/network time, evaluation, scoring, and reporting"
        ),
        "planned_tasks": planned_ids,
        "effective_configuration": {
            "sequential_tasks": True,
            "context_enforcement": (
                "run-local Hermes config and per-request Ollama num_ctx"
            ),
            "evaluator_timeout_seconds": plan[
                "evaluator_timeout_seconds"
            ],
            "evaluator_worker_sha256": plan["evaluator_worker_sha256"],
            "candidate_rpc_sha256": plan["candidate_rpc_sha256"],
            "scoring_version": plan["scoring_version"],
            "tool_distribution": plan["tool_distribution"],
            "candidate_toolsets": plan["candidate_toolsets"],
            "candidate_isolation": plan["candidate_isolation"],
            "evaluator_isolation": plan["evaluator_isolation"],
            "endpoint_redirects": plan["endpoint_redirects"],
            "tasks": [
                {
                    "id": item["manifest"]["id"],
                    "version": item["manifest"]["version"],
                    "manifest_sha256": file_sha256(
                        item["directory"] / "task.yaml"
                    ),
                    "fixture_sha256": item["manifest"]["fixture_sha256"],
                    "task_file_sha256": item["manifest"]["task_file_sha256"],
                    "evaluator_inputs": {
                        check["id"]: check["sha256"]
                        for check in item["manifest"]["evaluation_checks"]
                    },
                    "task_prompt_sha256": hashlib.sha256(
                        str(item["manifest"].get("goal", "")).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "limits": item["manifest"]["limits"],
                    "scoring": item["manifest"]["scoring"],
                }
                for item in plan["tasks"]
            ],
        },
        "preflight": None,
        "runtime_identity_verifications": [],
        "lifecycle": {
            "phase": "INITIALIZED",
            "current_task": None,
            "completed_tasks": [],
            "active_task": None,
            "last_transition_at": started_at,
        },
        "task_results": [],
        "aggregate_score": None,
        "infrastructure_failures": [],
        "artifacts": {
            "result": str(result_path.absolute()),
            "report": str(report_path.absolute()),
            "runtime_state": str(state_path.absolute()),
        },
    }

    def transition(phase: str, **updates: Any) -> None:
        aggregate["lifecycle"]["phase"] = phase
        aggregate["lifecycle"]["last_transition_at"] = utc_now()
        aggregate["lifecycle"].update(updates)
        aggregate["finished_at"] = utc_now()
        aggregate["elapsed_seconds_hermes_observed"] = round(
            time.monotonic() - run_started,
            3,
        )
        try:
            _write_outputs(aggregate)
        except KeyboardInterrupt as exc:
            if phase == "INTERRUPTED":
                raise

            aggregate["status"] = "INTERRUPTED"
            aggregate["lifecycle"]["phase"] = "INTERRUPTED"
            aggregate["lifecycle"]["last_transition_at"] = utc_now()
            aggregate["finished_at"] = utc_now()
            aggregate["elapsed_seconds_hermes_observed"] = round(
                time.monotonic() - run_started,
                3,
            )
            _write_outputs(aggregate)
            raise SuiteInterrupted(aggregate) from exc

    # Durable canonical state exists before endpoint preflight and before the
    # first task can allocate a candidate/runtime directory.
    try:
        _write_outputs(aggregate)
    except KeyboardInterrupt as exc:
        aggregate["status"] = "INTERRUPTED"
        aggregate["lifecycle"]["phase"] = "INTERRUPTED"
        aggregate["lifecycle"]["last_transition_at"] = utc_now()
        aggregate["finished_at"] = utc_now()
        aggregate["elapsed_seconds_hermes_observed"] = round(
            time.monotonic() - run_started,
            3,
        )
        _write_outputs(aggregate)
        raise SuiteInterrupted(aggregate) from exc
    progress(f"run_id={run_id}")
    progress(f"model={plan['model_alias']}")

    try:
        transition("CONTROL_PLANE_VERIFYING")
        _verify_control_plane(
            plan,
            allowed_artifacts=(result_path, report_path),
        )
        transition("CONTROL_PLANE_VERIFIED")
        transition("PREFLIGHT_RUNNING")
        aggregate["preflight"] = preflight(
            plan["endpoint"],
            plan["model"]["runtime_model"],
            expected_digest=plan["model"].get("runtime_digest"),
        )
        aggregate["model"]["runtime_digest"] = aggregate["preflight"][
            "runtime_model_digest"
        ]
        aggregate["model"]["runtime_identity_status"] = "VERIFIED"
        aggregate["runtime_identity_verifications"].append(
            {
                "phase": "PRE_RUN",
                "task": None,
                "runtime_digest": aggregate["preflight"][
                    "runtime_model_digest"
                ],
                "verified_at": utc_now(),
            }
        )
        transition("PREFLIGHT_COMPLETE")
    except KeyboardInterrupt as exc:
        aggregate["status"] = "INTERRUPTED"
        transition("INTERRUPTED")
        raise SuiteInterrupted(aggregate) from exc
    except InfrastructureError as exc:
        aggregate["status"] = "INFRASTRUCTURE_FAILURE"
        aggregate["infrastructure_failures"].append(str(exc))
        transition("PREFLIGHT_FAILED")
        progress(f"status={aggregate['status']}")
        return aggregate
    except Exception as exc:
        aggregate["status"] = "INFRASTRUCTURE_FAILURE"
        aggregate["infrastructure_failures"].append(
            "unexpected model preflight failure: "
            f"{type(exc).__name__}: {exc}"
        )
        transition("PREFLIGHT_FAILED")
        progress(f"status={aggregate['status']}")
        return aggregate

    try:
        for index, item in enumerate(plan["tasks"], start=1):
            task = item["manifest"]
            progress(
                f"task={task['id']} ({index}/{len(plan['tasks'])})"
            )
            started = time.monotonic()
            transition(
                "TASK_STARTING",
                current_task=task["id"],
                active_task={
                    "id": task["id"],
                    "candidate": None,
                    "runtime": None,
                    "task_run_id": None,
                },
            )

            def task_state(state: dict[str, Any]) -> None:
                phase = str(state.get("phase", "TASK_RUNNING"))
                active = dict(aggregate["lifecycle"].get("active_task") or {})
                active.update(
                    {
                        key: value
                        for key, value in state.items()
                        if key != "phase"
                    }
                )
                transition(phase, active_task=active)

            try:
                observed_model_metadata = {
                    **_model_result_metadata(plan),
                    "runtime_digest": aggregate["preflight"][
                        "runtime_model_digest"
                    ],
                    "runtime_identity_status": "VERIFIED",
                }
                result = task_runner(
                    task_id=task["id"],
                    model=plan["model"]["runtime_model"],
                    reasoning=plan["model"]["reasoning_effort"],
                    max_turns=task["limits"]["agent_turns"],
                    base_url=plan["endpoint"],
                    model_alias=plan["model_alias"],
                    model_metadata=observed_model_metadata,
                    benchmark_metadata={
                        **plan["benchmark"],
                        "runtime_identity": aggregate["preflight"],
                    },
                    evaluator_timeout=plan["evaluator_timeout_seconds"],
                    state_callback=task_state,
                )
                validate_result(result)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                aggregate["status"] = "INFRASTRUCTURE_FAILURE"
                aggregate["infrastructure_failures"].append(
                    f"task {task['id']} harness error: "
                    f"{type(exc).__name__}: {exc}"
                )
                transition("TASK_HARNESS_FAILED")
                break

            try:
                transition("TASK_CONTROL_PLANE_VERIFYING")
                _verify_control_plane(
                    plan,
                    allowed_artifacts=(result_path, report_path),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                aggregate["status"] = "INFRASTRUCTURE_FAILURE"
                aggregate["infrastructure_failures"].append(
                    f"task {task['id']} control-plane provenance changed: "
                    f"{type(exc).__name__}: {exc}"
                )
                transition("TASK_CONTROL_PLANE_FAILED")
                break

            try:
                transition("TASK_IDENTITY_VERIFYING")
                identity_after_task = preflight(
                    plan["endpoint"],
                    plan["model"]["runtime_model"],
                    expected_digest=aggregate["preflight"][
                        "runtime_model_digest"
                    ],
                )
                aggregate["runtime_identity_verifications"].append(
                    {
                        "phase": "POST_TASK",
                        "task": task["id"],
                        "runtime_digest": identity_after_task[
                            "runtime_model_digest"
                        ],
                        "verified_at": utc_now(),
                    }
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                aggregate["status"] = "INFRASTRUCTURE_FAILURE"
                aggregate["infrastructure_failures"].append(
                    f"task {task['id']} runtime identity changed or became "
                    f"unverifiable: {type(exc).__name__}: {exc}"
                )
                transition("TASK_IDENTITY_FAILED")
                break

            aggregate["task_results"].append(result)
            elapsed = round(time.monotonic() - started, 3)
            progress(
                f"task_outcome={result['outcome']} elapsed_seconds={elapsed}"
            )
            progress(
                f"task_runtime={result.get('artifacts', {}).get('runtime')}"
            )
            aggregate["aggregate_score"] = _aggregate_score(
                aggregate["task_results"]
            )
            aggregate["lifecycle"]["completed_tasks"].append(task["id"])
            transition("BETWEEN_TASKS", active_task=None)

            if result["outcome"] == "HARNESS_ERROR":
                aggregate["status"] = "INFRASTRUCTURE_FAILURE"
                aggregate["infrastructure_failures"].extend(
                    result["metrics"].get(
                        "infrastructure_failures",
                        ["unspecified harness failure"],
                    )
                )
                break
    except KeyboardInterrupt as exc:
        aggregate["status"] = "INTERRUPTED"
        aggregate["aggregate_score"] = _aggregate_score(
            aggregate["task_results"]
        )
        transition("INTERRUPTED")
        raise SuiteInterrupted(aggregate) from exc

    if aggregate["status"] != "INFRASTRUCTURE_FAILURE":
        if len(aggregate["task_results"]) != len(planned_ids):
            aggregate["status"] = "INCOMPLETE"
        elif all(
            result["outcome"] == "PASS"
            for result in aggregate["task_results"]
        ):
            aggregate["status"] = "PASS"
        else:
            aggregate["status"] = "FAIL"

    aggregate["aggregate_score"] = _aggregate_score(
        aggregate["task_results"]
    )
    transition("FINISHED", current_task=None, active_task=None)
    progress(f"result={aggregate['artifacts']['result']}")
    progress(f"report={aggregate['artifacts']['report']}")
    progress(f"aggregate_score={aggregate['aggregate_score']}")
    progress(f"status={aggregate['status']}")
    return aggregate


def aggregate_exit_code(aggregate: dict[str, Any]) -> int:
    status = aggregate["status"]

    if status == "PASS":
        return 0
    if status == "FAIL":
        if any(
            result.get("outcome") == "TIMEOUT"
            for result in aggregate["task_results"]
        ):
            return 4
        return 1
    if status == "INFRASTRUCTURE_FAILURE":
        return 3
    if status == "INTERRUPTED":
        return 130

    return 5


def show_plan(plan: dict[str, Any]) -> None:
    print("dry_run=YES")
    print(f"model_alias={plan['model_alias']}")
    print(f"runtime_model={plan['model']['runtime_model']}")
    print(
        "runtime_digest_configured="
        + (plan["model"].get("runtime_digest") or "UNRESOLVED")
    )
    print("runtime_identity_status=UNVERIFIED_DRY_RUN")
    print(f"context_length={plan['model']['context_length']}")
    print("context_enforcement=run-local-config-and-request-num_ctx")
    print(f"endpoint={endpoint_identity(plan['endpoint'])}")
    print(f"failure_policy={plan['failure_policy']}")
    print(f"tool_distribution={plan['tool_distribution']}")
    print(f"candidate_toolsets={HERMES_TOOLSET}")
    print("sequential_tasks=YES")

    for index, item in enumerate(plan["tasks"], start=1):
        task = item["manifest"]
        print(
            f"task_{index}={task['id']} version={task['version']} "
            f"wall_seconds={task['limits']['wall_seconds']} "
            f"agent_turns={task['limits']['agent_turns']}"
        )

    print("model_contact=NO")
    print("candidate_execution=NO")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./scripts/benchmark-model",
        description=(
            "Run a configured Hermes model benchmark suite sequentially."
        )
    )
    parser.add_argument("--model", help="model alias from models.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="benchmark generation configuration",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="run only this configured task (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and display the plan without model contact or execution",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list configured model aliases and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        try:
            models = load_models()
        except ConfigurationError as exc:
            print(f"configuration_error={exc}", file=sys.stderr)
            return 2

        for alias, model in models.items():
            print(
                f"{alias}\t{model['runtime_model']}\t{model['display_name']}"
            )

        return 0

    if not args.model:
        parser.error("--model is required unless --list-models is used")

    try:
        plan = load_execution_plan(
            args.model,
            config_path=args.config,
            requested_tasks=args.task,
        )
    except (
        ConfigurationError,
        OSError,
        json.JSONDecodeError,
        jsonschema.ValidationError,
        jsonschema.exceptions.SchemaError,
    ) as exc:
        print(f"configuration_error={exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        show_plan(plan)
        return 0

    try:
        aggregate = execute_suite(plan)
    except SuiteInterrupted as exc:
        aggregate = exc.aggregate
        print("status=INTERRUPTED", file=sys.stderr)
        print(
            f"result={aggregate['artifacts']['result']}",
            file=sys.stderr,
        )
        return 130
    except InfrastructureError as exc:
        print(f"infrastructure_error={exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(
            "infrastructure_error=unexpected orchestrator failure: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    return aggregate_exit_code(aggregate)


if __name__ == "__main__":
    raise SystemExit(main())
