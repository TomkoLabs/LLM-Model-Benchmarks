from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import jsonschema
import yaml

from harness.artifacts import ArtifactSafetyError, atomic_write_text
from harness.comparison import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PUBLIC_OUTPUT_ROOT,
    METADATA_CANDIDATES_FILENAME,
    ComparisonError,
)
from harness.endpoints import EndpointPolicyError, LocalEndpoint, validate_local_openai_endpoint
from harness.model_gateway import (
    ModelGateway,
    combine_model_transport_observations,
    read_model_transport_observations,
)
from harness.model_identity import ModelMetadataError, resolve_public_identity
from harness.reasoning_policy import (
    REASONING_POLICY_CONTRACT,
    SUPPORTED_REASONING_POLICIES,
    ReasoningPolicy,
    ReasoningPolicyError,
    classify_direct_response,
    ollama_reasoning_control,
    reasoning_policy_selection,
    resolve_reasoning_policy,
)
from harness.upstreams import (
    HERMES_PYTHON,
    ROOT,
    SPARK_PYTHON_DEPS,
    UpstreamError,
    build_benchlocal_command,
    build_infermark_command,
    build_spark_command,
    ensure_checkouts,
    execute_upstream,
    load_upstream_lock,
    parse_benchlocal,
    parse_infermark,
    parse_spark,
    safe_slug,
    upstream_environment,
    write_local_network_policy,
    write_chromium_network_wrapper,
)


CONFIG_PATH = ROOT / "configs" / "qualification-v4.yaml"
LOCAL_CONFIG_PATH = ROOT / "config" / "local.yaml"
MODELS_PATH = ROOT / "models.yaml"
RUNS_ROOT = ROOT / "runs"
RESULT_SCHEMA = ROOT / "schemas" / "qualification-run.schema.json"
DIRECT_PROBE_CONTRACT = "gx10-direct-probe-v1"
DIRECT_PROBE_MAX_TOKENS = 1024
_ACTIVE_RUN_CONTEXT: dict[str, Any] | None = None


class QualificationError(RuntimeError):
    pass


class QualificationQuarantine(QualificationError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise QualificationError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must be a mapping")
    return value


def load_configuration() -> dict[str, Any]:
    config = _load_yaml(CONFIG_PATH, "qualification configuration")
    if config.get("schema_version") != 1:
        raise QualificationError("unsupported qualification configuration schema")
    if config.get("generation") != "gx10-qualification-v4":
        raise QualificationError("incompatible qualification generation")
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"smoke", "standard", "overnight"}:
        raise QualificationError("qualification profiles are incomplete")
    weights = config.get("weights")
    if not isinstance(weights, dict) or sum(weights.values()) != 100:
        raise QualificationError("qualification weights must total 100")
    if config.get("direct_probe") != {
        "contract": DIRECT_PROBE_CONTRACT,
        "max_tokens": DIRECT_PROBE_MAX_TOKENS,
    }:
        raise QualificationError(
            "qualification direct_probe must freeze the v1 contract at "
            f"{DIRECT_PROBE_MAX_TOKENS} max_tokens"
        )
    return config


def resolve_endpoint(override: str | None) -> LocalEndpoint:
    value = override or os.environ.get("HERMES_BENCH_ENDPOINT")
    if not value and LOCAL_CONFIG_PATH.is_file():
        local = _load_yaml(LOCAL_CONFIG_PATH, "local configuration")
        value = local.get("endpoint")
    value = value or "http://127.0.0.1:11434/v1"
    try:
        return validate_local_openai_endpoint(str(value))
    except EndpointPolicyError as exc:
        raise QualificationError(str(exc)) from exc


def _prohibited_model(value: str) -> bool:
    normalized = value.lower().replace("_", "-")
    return (
        "deepseek" in normalized
        and "v4" in normalized
        and "flash" in normalized
    )


def _reject_prohibited_model_config(
    alias: str,
    model: Mapping[str, Any],
) -> None:
    identity = " ".join(
        str(value)
        for value in (
            alias,
            model.get("display_name", ""),
            model.get("runtime_model", ""),
            model.get("canonical_name", ""),
            model.get("source_model", ""),
        )
    )
    if _prohibited_model(identity):
        raise QualificationError(
            "DeepSeek V4 Flash is prohibited on the current single-GX10 setup"
        )


def _curated_model(
    requested: str,
    models: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, dict[str, Any]] | None:
    selected: tuple[str, Mapping[str, Any]] | None = None
    if requested in models:
        selected = (requested, models[requested])
    else:
        matches = [
            (alias, model)
            for alias, model in models.items()
            if model["runtime_model"] == requested
        ]
        defaults = [
            (alias, model)
            for alias, model in matches
            if model.get("default_for_runtime") is True
        ]
        if len(defaults) == 1:
            selected = defaults[0]
        elif len(matches) == 1:
            selected = matches[0]
        elif matches:
            raise QualificationError(
                "runtime model maps to multiple configurations; "
                "set default_for_runtime in models.yaml"
            )
    if selected is None:
        return None

    alias, configured = selected
    model = dict(configured)
    model["configuration_source"] = "models.yaml curated override"
    model["metadata_sources"] = {
        "runtime_model": "models.yaml curated override",
        "runtime_digest": "models.yaml expected digest; verified by /api/tags",
        "quantization": "models.yaml expected value; verified by /api/show",
        "effective_context_length": (
            "models.yaml run-local request configuration"
        ),
        "reasoning_effort": "models.yaml curated override",
        "reasoning_policy": "models.yaml curated deployment policy",
        "supported_reasoning_policies": "models.yaml explicit support list",
    }
    _reject_prohibited_model_config(alias, model)
    return alias, str(configured["runtime_model"]), model


def _discovered_model_config(
    requested: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_metadata = preflight.get("public_runtime_metadata")
    runtime_metadata = (
        dict(runtime_metadata)
        if isinstance(runtime_metadata, Mapping)
        else {}
    )
    discrepancies = runtime_metadata.get("metadata_discrepancies")
    discrepancies = (
        [str(value) for value in discrepancies]
        if isinstance(discrepancies, list)
        else []
    )
    context = runtime_metadata.get("effective_context_length")
    if type(context) is not int or context < 1:
        if discrepancies:
            raise QualificationError(
                f"model {requested!r} is installed, but effective context "
                f"is unavailable: {discrepancies[0]}"
            )
        raise QualificationError(
            f"model {requested!r} is installed, but effective context is "
            "unavailable: /api/show did not report num_ctx in parameters "
            "or Modelfile"
        )

    digest = preflight.get("runtime_model_digest")
    if not isinstance(digest, str):
        raise QualificationError(
            f"model {requested!r} has no verified immutable runtime digest"
        )
    quantization = runtime_metadata.get("quantization")
    if not isinstance(quantization, str) or not quantization:
        quantization = "UNREPORTED"
    display_name = runtime_metadata.get("canonical_name")
    if not isinstance(display_name, str) or not display_name:
        display_name = f"Runtime-discovered model {requested}"

    return {
        "display_name": display_name,
        "provider": "gx10",
        "runtime": "ollama",
        "runtime_model": requested,
        "runtime_digest": digest,
        "quantization": quantization,
        "context_length": context,
        "role": "runtime-discovered-profile",
        "configuration_source": "Ollama runtime discovery",
        "canonical_name": runtime_metadata.get("canonical_name"),
        "source_model": runtime_metadata.get("source_model"),
        "source_version": runtime_metadata.get("version"),
        "architecture": runtime_metadata.get("architecture"),
        "parameter_variant": runtime_metadata.get("parameter_variant"),
        "parameter_count": runtime_metadata.get("parameter_count"),
        "native_context_length": runtime_metadata.get(
            "native_context_length"
        ),
        "runtime_metadata": runtime_metadata,
        "metadata_sources": {
            "runtime_model": "/v1/models and /api/tags exact identifier match",
            "runtime_digest": "/api/tags digest",
            "architecture": "/api/show model_info or details.family",
            "parameter_variant": "/api/show details or model_info",
            "quantization": "/api/show details.quantization_level",
            "effective_context_length": " and ".join(
                runtime_metadata.get("effective_context_sources", [])
            ),
        },
    }


def resolve_model(
    requested: str,
    *,
    endpoint: str | None = None,
) -> tuple[str, str, dict[str, Any], dict[str, Any] | None]:
    if not isinstance(requested, str) or not requested.strip():
        raise QualificationError("--model must be non-empty")
    if _prohibited_model(requested):
        raise QualificationError(
            "DeepSeek V4 Flash is prohibited on the current single-GX10 setup"
        )

    from harness.benchmark_model import load_models

    models = load_models(MODELS_PATH)
    curated = _curated_model(requested, models)
    if curated is not None:
        alias, runtime_model, model = curated
        return alias, runtime_model, model, None

    if endpoint is None:
        # Dry runs deliberately contact no endpoint. The exact digest and
        # context are therefore deferred until live preflight.
        return (
            requested,
            requested,
            {
                "runtime_model": requested,
                "configuration_source": "Ollama runtime discovery pending",
            },
            None,
        )

    from harness.benchmark_model import InfrastructureError, preflight_model

    try:
        preflight = preflight_model(
            endpoint,
            requested,
            expected_digest=None,
        )
    except InfrastructureError as exc:
        raise QualificationError(str(exc)) from exc
    model = _discovered_model_config(requested, preflight)
    _reject_prohibited_model_config(requested, model)
    return requested, requested, model, preflight


def _effective_model_config(
    model_config: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    resolved = dict(model_config)
    runtime_metadata = preflight.get("public_runtime_metadata")
    runtime_metadata = (
        dict(runtime_metadata)
        if isinstance(runtime_metadata, Mapping)
        else {}
    )
    discrepancies = runtime_metadata.get("metadata_discrepancies")
    reported_discrepancies = (
        [str(value) for value in discrepancies]
        if isinstance(discrepancies, list)
        else []
    )

    runtime_quantization = runtime_metadata.get("quantization")
    configured_quantization = resolved.get("quantization")
    if (
        isinstance(runtime_quantization, str)
        and runtime_quantization
        and isinstance(configured_quantization, str)
        and runtime_quantization.casefold() != configured_quantization.casefold()
    ):
        reported_discrepancies.append(
            "configured quantization "
            f"{configured_quantization!r} differs from runtime "
            f"{runtime_quantization!r}; runtime value recorded"
        )
        resolved["configured_quantization"] = configured_quantization
        resolved["quantization"] = runtime_quantization

    runtime_default_context = runtime_metadata.get("effective_context_length")
    configured_context = resolved.get("context_length")
    if (
        type(runtime_default_context) is int
        and type(configured_context) is int
        and runtime_default_context != configured_context
    ):
        reported_discrepancies.append(
            "runtime default num_ctx "
            f"{runtime_default_context} differs from curated run-local "
            f"context {configured_context}; run-local request value retained"
        )

    resolved["runtime_digest"] = preflight["runtime_model_digest"]
    resolved["runtime_metadata"] = runtime_metadata
    if type(runtime_metadata.get("native_context_length")) is int:
        resolved["native_context_length"] = runtime_metadata[
            "native_context_length"
        ]
    if type(runtime_default_context) is int:
        resolved["runtime_default_context_length"] = runtime_default_context
    reported_discrepancies = list(dict.fromkeys(reported_discrepancies))
    if reported_discrepancies:
        resolved["metadata_discrepancies"] = reported_discrepancies
    return resolved, reported_discrepancies


def _git(*args: str) -> str:
    process = subprocess.run(
        ["/usr/bin/git", "-C", str(ROOT), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if process.returncode:
        raise QualificationError(f"git {' '.join(args)} failed: {process.stdout.strip()}")
    return process.stdout.strip()


def repository_provenance(*, require_clean: bool) -> dict[str, Any]:
    status = _git("status", "--porcelain")
    if require_clean and status:
        raise QualificationError("control-plane worktree and index must be clean before a live run")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "upstream": _git("rev-parse", "@{upstream}"),
        "dirty": bool(status),
    }


def _atomic_json(path: Path, value: Mapping[str, Any], root: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        root=root,
    )


def _endpoint_json(
    url: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: int = 180,
) -> tuple[dict[str, Any], float]:
    from harness.benchmark_model import _endpoint_json as request

    started = time.monotonic()
    value = request(url, timeout=timeout, request_body=body)
    return value, round(time.monotonic() - started, 3)


def direct_checks(
    endpoint: LocalEndpoint,
    model: str,
    *,
    reasoning_policy: ReasoningPolicy,
    direct_probe: Mapping[str, Any],
) -> dict[str, Any]:
    if direct_probe != {
        "contract": DIRECT_PROBE_CONTRACT,
        "max_tokens": DIRECT_PROBE_MAX_TOKENS,
    }:
        raise QualificationError("invalid direct-probe configuration")
    max_tokens = DIRECT_PROBE_MAX_TOKENS
    chat_url = endpoint.base_url + "/chat/completions"
    response, response_wall = _endpoint_json(
        chat_url,
        body={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly GX10_DIRECT_OK."}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    response_result = classify_direct_response(
        response,
        expected_content="GX10_DIRECT_OK",
        reasoning_allowed=reasoning_policy.mode != "off",
    )

    tool_response, tool_wall = _endpoint_json(
        chat_url,
        body={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Call qualification_probe exactly once with token GX10_TOOL_OK. Do not answer in text.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "qualification_probe",
                        "description": "Return a qualification token.",
                        "parameters": {
                            "type": "object",
                            "properties": {"token": {"type": "string"}},
                            "required": ["token"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "qualification_probe"}},
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    tool_result = classify_direct_response(
        tool_response,
        expected_tool_name="qualification_probe",
        expected_tool_arguments={"token": "GX10_TOOL_OK"},
        reasoning_allowed=reasoning_policy.mode != "off",
    )

    return {
        "contract": DIRECT_PROBE_CONTRACT,
        "max_tokens": max_tokens,
        "status": (
            "PASS"
            if response_result["status"] == "PASS"
            and tool_result["status"] == "PASS"
            else "FAIL"
        ),
        "reasoning_policy": reasoning_policy.value,
        "response": {
            **response_result,
            "wall_seconds": response_wall,
            "usage": response.get("usage"),
        },
        "tool_call": {
            **tool_result,
            "wall_seconds": tool_wall,
            "usage": tool_response.get("usage"),
        },
        "raw": {"response": response, "tool_response": tool_response},
    }


def context_prompt(target_units: int) -> str:
    if type(target_units) is not int or target_units < 32 or target_units > 32768:
        raise QualificationError("Infermark context target must be an integer from 32 to 32768")
    prefix = "Read the following repeated context and return a concise two-sentence summary.\n\n"
    return prefix + ("local benchmark context " * max(1, target_units // 3))


def _performance_score(component: Mapping[str, Any]) -> float | None:
    metrics = component.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("errors"):
        return 0.0 if isinstance(metrics, dict) else None
    throughput = float(metrics.get("tokens_per_second_c1") or 0)
    ttft = metrics.get("ttft_seconds_c1")
    ttft_p50 = float(ttft.get("p50") or 0) if isinstance(ttft, dict) else 0
    throughput_score = min(100.0, throughput * 10.0)
    ttft_score = max(0.0, 100.0 - max(0.0, ttft_p50 - 1.0) * 10.0)
    return round((throughput_score + ttft_score) / 2, 3)


def calculate_decision(
    profile_name: str,
    components: Mapping[str, Any],
    config: Mapping[str, Any],
    reasoning_policy: ReasoningPolicy,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    direct = components.get("direct", {})
    if direct.get("status") != "PASS":
        failures.append("direct endpoint response/tool-call contract failed")
    direct_transport = direct.get("model_transport")
    if not (
        isinstance(direct_transport, Mapping)
        and direct_transport.get("observer_ok") is True
        and direct_transport.get("request_count") == 2
        and direct_transport.get("response_count") == 2
    ):
        failures.append("direct model transport evidence missing or broken")

    upstream = components.get("upstreams", {})
    for name, component in upstream.items():
        if component.get("status") != "PASS":
            failures.append(f"mandatory upstream component failed: {name}")
        control = component.get("model_transport")
        if not (
            isinstance(control, Mapping)
            and control.get("observer_ok") is True
            and int(control.get("request_count") or 0) > 0
            and int(control.get("response_count") or 0) > 0
        ):
            failures.append(f"model transport evidence missing or broken: {name}")
        if (
            reasoning_policy.mode == "off"
            and isinstance(control, Mapping)
            and control.get("reasoning_content_returned") is True
        ):
            failures.append(
                f"reasoning-policy off returned reasoning content: {name}"
            )

    hermes_rows = components.get("hermes", [])
    for row in hermes_rows:
        if row.get("outcome") != "PASS":
            failures.append(f"Hermes task did not pass: {row.get('task_id')}")
        if not row.get("agent_execution_valid"):
            failures.append(
                f"Hermes execution evidence missing or invalid: {row.get('task_id')}"
            )
        transport = row.get("model_transport")
        if not (
            isinstance(transport, Mapping)
            and transport.get("observer_ok") is True
            and int(transport.get("request_count") or 0) > 0
            and int(transport.get("response_count") or 0) > 0
        ):
            failures.append(
                f"Hermes model transport evidence missing or broken: {row.get('task_id')}"
            )
        if row.get("blocked_network_attempt_count") or row.get("fallback_attempt_count"):
            failures.append(f"prohibited Hermes network/fallback attempt: {row.get('task_id')}")
        if not row.get("cleanup_complete"):
            failures.append(f"Hermes cleanup incomplete: {row.get('task_id')}")
        if not row.get("scope_pass", True):
            failures.append(f"Hermes candidate scope violation: {row.get('task_id')}")
        control = row.get("model_transport")
        if (
            reasoning_policy.mode == "off"
            and isinstance(control, Mapping)
            and control.get("reasoning_content_returned") is True
        ):
            failures.append(
                f"reasoning-policy off returned reasoning content: Hermes {row.get('task_id')}"
            )

    network_attempts = components.get("blocked_network_attempts", [])
    if network_attempts:
        failures.append("upstream attempted a non-local network connection")

    spark = upstream.get("spark-bench")
    coding_parts: list[float] = []
    reliability_parts: list[float] = []
    if isinstance(spark, dict):
        coding_parts.append(float(spark.get("metrics", {}).get("coding_quality", spark.get("score", 0))))
        reliability_parts.append(float(spark.get("metrics", {}).get("reliability", 0)))
    bench_scores: list[float] = []
    for name, component in upstream.items():
        if name.startswith("benchlocal-"):
            if name == "benchlocal-coding":
                coding_parts.append(float(component.get("score", 0)))
            else:
                bench_scores.append(float(component.get("score", 0)))
    hermes_score = (
        round(100 * sum(row.get("outcome") == "PASS" for row in hermes_rows) / len(hermes_rows), 3)
        if hermes_rows
        else None
    )
    scores: dict[str, float | None] = {
        "coding": round(sum(coding_parts) / len(coding_parts), 3) if coding_parts else None,
        "hermes": hermes_score,
        "tool_instruction": round(sum(bench_scores) / len(bench_scores), 3) if bench_scores else None,
        "reliability": round(sum(reliability_parts) / len(reliability_parts), 3) if reliability_parts else hermes_score,
        "performance": _performance_score(upstream.get("infermark", {})),
    }

    thresholds = config["thresholds"]
    if profile_name != "smoke":
        for key in ("coding", "hermes", "tool_instruction", "reliability"):
            value = scores[key]
            if value is None or value < float(thresholds[key]):
                failures.append(f"{key} threshold not met")

    weights = config["weights"]
    executed_weight = sum(weights[key] for key, value in scores.items() if value is not None)
    aggregate = (
        round(
            sum(float(scores[key]) * weights[key] for key in scores if scores[key] is not None)
            / executed_weight,
            3,
        )
        if executed_weight
        else None
    )
    if profile_name != "smoke" and (aggregate is None or aggregate < float(thresholds["aggregate"])):
        failures.append("aggregate threshold not met")
    score_document = {
        "deployment_configuration_score": aggregate,
        "components": scores,
        "weights": weights,
        "executed_weight": executed_weight,
        "comparable_full_profile": executed_weight == 100,
    }
    gates = {"passed": not failures, "failures": list(dict.fromkeys(failures))}
    return ("QUALIFIED" if gates["passed"] else "NOT_QUALIFIED"), gates, score_document


def render_report(result: Mapping[str, Any]) -> str:
    scores = result["scores"]
    model = result["model"]
    identity = model.get("public_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    direct = result.get("components", {}).get("direct", {})
    direct = direct if isinstance(direct, Mapping) else {}
    direct_response = direct.get("response")
    direct_response = (
        direct_response if isinstance(direct_response, Mapping) else {}
    )
    direct_tool = direct.get("tool_call")
    direct_tool = direct_tool if isinstance(direct_tool, Mapping) else {}
    response_usage = direct_response.get("usage")
    response_usage = response_usage if isinstance(response_usage, Mapping) else {}
    tool_usage = direct_tool.get("usage")
    tool_usage = tool_usage if isinstance(tool_usage, Mapping) else {}
    display_identity = identity.get("display_name") or (
        "METADATA_INCOMPLETE — "
        + str(model.get("runtime_digest") or "digest unavailable")
    )
    lines = [
        "# LLM Model Benchmarks qualification report",
        "",
        f"- Profile decision: **{result.get('profile_decision', 'NOT_ASSESSED')}**",
        f"- Result validity: **{result.get('result_validity', 'INCOMPLETE')}**",
        f"- Technical outcome: **{result['outcome']}**",
        f"- Run: `{result['run_id']}`",
        f"- Profile: `{result['profile']}`",
        f"- Canonical model identity: `{display_identity}`",
        f"- Runtime alias: `{model.get('runtime_model')}`",
        f"- Immutable digest: `{model.get('runtime_digest')}`",
        f"- Endpoint: `{model['endpoint']}`",
        f"- Reasoning policy: `{model.get('reasoning_policy')}`",
        f"- Reasoning cohort: `{model.get('reasoning_cohort')}`",
        f"- Direct-probe contract: `{direct.get('contract')}`",
        f"- Direct-probe max completion tokens: `{direct.get('max_tokens')}`",
        "- Direct response: `"
        f"{direct_response.get('classification')}` / finish `"
        f"{direct_response.get('finish_reason')}` / completion tokens `"
        f"{response_usage.get('completion_tokens')}`",
        "- Direct tool call: `"
        f"{direct_tool.get('classification')}` / finish `"
        f"{direct_tool.get('finish_reason')}` / completion tokens `"
        f"{tool_usage.get('completion_tokens')}`",
        "- Serialized OpenAI control: `"
        + json.dumps(
            result["provenance"].get("reasoning_policy", {}).get(
                "serialized_endpoint_controls", {}
            ).get("openai_chat_completions", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "`",
        "- Serialized native Ollama control: `"
        + json.dumps(
            result["provenance"].get("reasoning_policy", {}).get(
                "serialized_endpoint_controls", {}
            ).get("ollama_native_chat", {}),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "`",
        "",
        "## Deployment score",
        "",
        f"Aggregate: `{scores.get('deployment_configuration_score')}`",
        "",
        "| Component | Score | Weight |",
        "|---|---:|---:|",
    ]
    for key, value in scores["components"].items():
        lines.append(f"| {key} | {value if value is not None else 'not executed'} | {scores['weights'][key]} |")
    lines.extend(["", "## Hard gates", ""])
    if result["gates"]["failures"]:
        lines.extend(f"- FAIL: {failure}" for failure in result["gates"]["failures"])
    else:
        lines.append("- PASS: all mandatory gates for this profile")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a deployment score for the exact canonical checkpoint, immutable digest, quantization, runtime/template, context, sampling, hardware, upstream commits, and harness configuration recorded in `manifest.json`. The runtime alias is only an operator label; this is not an abstract base-model score.",
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_and_cleanup_hermes(
    result: dict[str, Any],
    destination: Path,
    *,
    transport_observations_path: Path | None = None,
) -> dict[str, Any]:
    from harness.hermes_runner import cleanup

    runtime = Path(result["paths"]["runtime"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(runtime, destination, symlinks=True)
    copied_transport_path = destination / "model-transport-observations.jsonl"
    if (
        transport_observations_path is not None
        and transport_observations_path.is_file()
    ):
        shutil.copy2(transport_observations_path, copied_transport_path)
    cleanup(result)
    cleanup_complete = not runtime.exists() and not Path(result["paths"]["candidate"]).exists()
    metrics = result.get("metrics", {})
    candidate_git = metrics.get("candidate_git", {})
    response_path = destination / "final-response.txt"
    transport = metrics.get("model_transport")
    transport = transport if isinstance(transport, Mapping) else {}
    execution = metrics.get("agent_execution")
    execution = execution if isinstance(execution, Mapping) else {}
    return {
        "task_id": result["task"]["id"],
        "run_id": result["run_id"],
        "outcome": result["outcome"],
        "trajectory_present": bool(result.get("trajectory_present")),
        "trajectory_status": result.get("trajectory_status"),
        "final_response_present": response_path.is_file() and bool(response_path.read_text(encoding="utf-8").strip()),
        "final_text_response_observed": execution.get(
            "final_text_response_observed"
        ),
        "agent_execution_valid": result.get("agent_execution_valid"),
        "agent_execution": dict(execution),
        "evaluator_pass": result.get("metrics", {}).get("evaluation", {}).get("pass"),
        "infrastructure_ok": result.get("infrastructure_ok"),
        "blocked_network_attempt_count": metrics.get("blocked_network_attempt_count", 0),
        "fallback_attempt_count": metrics.get("fallback_attempt_count", 0),
        "timeout_reason": metrics.get("timeout_reason"),
        "scope_pass": candidate_git.get("scope_pass"),
        "unexpected_files": candidate_git.get("unexpected_files", []),
        "duration_seconds": result.get("duration_seconds"),
        "cleanup_complete": cleanup_complete,
        "artifact_directory": str(destination),
        "model_transport": dict(transport),
        "model_transport_artifact": (
            str(copied_transport_path) if copied_transport_path.is_file() else None
        ),
    }


def _dry_plan(
    *,
    profile_name: str,
    profile: Mapping[str, Any],
    endpoint: LocalEndpoint,
    model: str,
    model_resolution: str,
    reasoning_policy_request: str,
    reasoning_policy: ReasoningPolicy,
    reasoning_policy_source: str,
) -> dict[str, Any]:
    configuration = load_configuration()
    lock = load_upstream_lock()
    pseudo = ROOT / "runs" / "DRY-RUN"
    commands: list[list[str]] = []
    if profile.get("spark"):
        commands.append(
            build_spark_command(
                ROOT / lock["spark-bench"]["checkout"],
                endpoint=endpoint.base_url,
                model=model,
                output_dir=pseudo / "artifacts" / "spark-bench",
                label="dry-run",
                profile=profile["spark"],
                reasoning_policy=reasoning_policy,
            )
        )
    for selection in profile.get("benchlocal", []):
        commands.append(
            build_benchlocal_command(
                ROOT / lock["benchlocal-cli"]["checkout"],
                endpoint=endpoint.base_url,
                model=model,
                output_path=pseudo / "artifacts" / f"benchlocal-{selection['id']}.json",
                selection=selection,
                reasoning_policy=reasoning_policy,
            )
        )
    for context_name, target in profile["infermark"]["contexts"].items():
        commands.append(
            build_infermark_command(
                ROOT / lock["infermark"]["checkout"],
                endpoint=endpoint.base_url,
                model=model,
                output_path=pseudo / "artifacts" / f"infermark-{context_name}.json",
                profile=profile["infermark"],
                prompt=f"<generated local context: {target} approximate units>",
            )
        )
    selection_mode, benchmark_track = reasoning_policy_selection(
        reasoning_policy_request
    )
    return {
        "profile": profile_name,
        "qualification_generation": configuration["generation"],
        "direct_probe": dict(configuration["direct_probe"]),
        "reasoning_policy": {
            "contract": REASONING_POLICY_CONTRACT,
            "requested": reasoning_policy_request,
            "effective": reasoning_policy.value,
            "source": reasoning_policy_source,
            "selection_mode": selection_mode,
            "benchmark_track": benchmark_track,
            "cohort": reasoning_policy.cohort,
            "openai_chat_completions": ollama_reasoning_control(
                "openai-chat-completions", reasoning_policy
            ),
            "ollama_native_chat": ollama_reasoning_control(
                "ollama-native-chat", reasoning_policy
            ),
        },
        "endpoint": endpoint.base_url,
        "model": model,
        "model_resolution": model_resolution,
        "expected_duration": profile["expected_duration"],
        "commands": commands,
        "hermes_tasks": profile["hermes_tasks"],
        "contacts_endpoint": False,
    }


def _upstream_python_paths(
    name: str,
    checkout: Path,
    policy_dir: Path,
) -> list[Path]:
    paths = [policy_dir]
    if name == "spark-bench":
        paths.append(SPARK_PYTHON_DEPS)
    elif name.startswith("benchlocal-"):
        paths.append(checkout)
    elif name.startswith("infermark-"):
        paths.append(checkout / "src")
    return paths


def execute(
    *,
    requested_model: str,
    endpoint_override: str | None,
    profile_name: str,
    setup: bool,
    reasoning_policy_request: str,
) -> tuple[int, dict[str, Any]]:
    global _ACTIVE_RUN_CONTEXT
    config = load_configuration()
    if profile_name not in config["profiles"]:
        raise QualificationError(f"unknown profile: {profile_name}")
    profile = config["profiles"][profile_name]
    direct_probe_config = dict(config["direct_probe"])
    endpoint = resolve_endpoint(endpoint_override)
    repository = repository_provenance(require_clean=True)
    alias, runtime_model, model_config, discovered_preflight = resolve_model(
        requested_model,
        endpoint=endpoint.base_url,
    )
    try:
        reasoning_policy, reasoning_policy_source = resolve_reasoning_policy(
            reasoning_policy_request, model_config
        )
    except ReasoningPolicyError as exc:
        raise QualificationError(str(exc)) from exc
    reasoning_policy_selection_mode, benchmark_track = reasoning_policy_selection(
        reasoning_policy_request
    )
    openai_control = ollama_reasoning_control(
        "openai-chat-completions", reasoning_policy
    )
    native_control = ollama_reasoning_control(
        "ollama-native-chat", reasoning_policy
    )
    print(f"effective_reasoning_policy={reasoning_policy.value}", flush=True)
    print(f"reasoning_policy_source={reasoning_policy_source}", flush=True)
    print(
        f"reasoning_policy_selection_mode={reasoning_policy_selection_mode}",
        flush=True,
    )
    print(f"benchmark_track={benchmark_track}", flush=True)
    print(
        "openai_reasoning_control="
        + json.dumps(openai_control, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "native_reasoning_control="
        + json.dumps(native_control, sort_keys=True, separators=(",", ":")),
        flush=True,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{safe_slug(runtime_model)}-{profile_name}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    artifacts = run_dir / "artifacts"
    logs.mkdir()
    artifacts.mkdir()
    started = time.monotonic()
    profile_deadline = started + float(profile["total_wall_seconds"])

    def remaining_profile_seconds() -> float:
        remaining = profile_deadline - time.monotonic()
        if remaining < 1:
            raise QualificationError("qualification profile total wall exhausted")
        return remaining

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": utc_now(),
        "status": "RUNNING",
        "profile": profile_name,
        "qualification_generation": config["generation"],
        "scoring_version": config["generation"],
        "profile_config": profile,
        "direct_probe": direct_probe_config,
        "reasoning_policy": {
            "contract": REASONING_POLICY_CONTRACT,
            "requested": reasoning_policy_request,
            "effective": reasoning_policy.value,
            "source": reasoning_policy_source,
            "selection_mode": reasoning_policy_selection_mode,
            "benchmark_track": benchmark_track,
            "cohort": reasoning_policy.cohort,
            "serialized_endpoint_controls": {
                "openai_chat_completions": openai_control,
                "ollama_native_chat": native_control,
            },
        },
        "model_request": requested_model,
        "model_alias": alias,
        "runtime_model": runtime_model,
        "model_config": model_config,
        "endpoint": endpoint.base_url,
        "repository": repository,
        "upstreams": {},
        "host": {
            "hostname": platform.node(),
            "system": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "invocations": {},
        "configuration_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "models_sha256": hashlib.sha256(MODELS_PATH.read_bytes()).hexdigest(),
        "upstream_lock_sha256": hashlib.sha256((ROOT / "upstreams.lock.json").read_bytes()).hexdigest(),
    }
    _ACTIVE_RUN_CONTEXT = {
        "run_dir": run_dir,
        "run_id": run_id,
        "profile": profile_name,
        "requested_model": requested_model,
        "alias": alias,
        "runtime_model": runtime_model,
        "endpoint": endpoint.base_url,
        "reasoning_policy": reasoning_policy.value,
        "reasoning_cohort": reasoning_policy.cohort,
        "reasoning_policy_selection_mode": reasoning_policy_selection_mode,
        "benchmark_track": benchmark_track,
        "manifest": manifest,
    }
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)
    partial: dict[str, Any] = {"run_id": run_id, "status": "RUNNING", "components": {}}
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)

    upstream_provenance = ensure_checkouts(setup=setup)
    manifest["upstreams"] = upstream_provenance
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)

    from harness.benchmark_model import _endpoint_json as endpoint_request
    from harness.benchmark_model import (
        InfrastructureError,
        load_execution_plan,
        preflight_model,
    )

    try:
        preflight = discovered_preflight or preflight_model(
            endpoint.base_url,
            runtime_model,
            expected_digest=model_config.get("runtime_digest"),
        )
        version = endpoint_request(endpoint.origin + "/api/version", timeout=10)
    except InfrastructureError as exc:
        raise QualificationError(str(exc)) from exc
    preflight["runtime_version"] = version.get("version")
    effective_model_config, metadata_discrepancies = _effective_model_config(
        model_config,
        preflight,
    )
    effective_model_config["configured_reasoning_policy"] = model_config.get(
        "reasoning_policy"
    )
    effective_model_config["configured_reasoning_effort"] = model_config.get(
        "reasoning_effort"
    )
    effective_model_config["reasoning_policy"] = reasoning_policy.value
    effective_model_config["reasoning_effort"] = (
        reasoning_policy.hermes_effort or "native"
    )
    effective_model_config["supported_reasoning_policies"] = list(
        model_config.get("supported_reasoning_policies")
        or [reasoning_policy.value]
    )
    manifest["runtime_model_metadata"] = preflight.get(
        "public_runtime_metadata", {}
    )
    manifest["effective_model_config"] = effective_model_config
    manifest["model_metadata_discrepancies"] = metadata_discrepancies
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)
    partial["components"]["preflight"] = preflight
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)

    direct_observations_path = (
        artifacts / "direct" / "model-transport-observations.jsonl"
    )
    with ModelGateway(
        target_base_url=endpoint.base_url,
        policy=reasoning_policy,
        stage="direct",
        observations_path=direct_observations_path,
        upstream_timeout=remaining_profile_seconds(),
    ) as gateway:
        manifest.setdefault("model_transport_boundaries", {})["direct"] = (
            gateway.metadata
        )
        _atomic_json(run_dir / "manifest.json", manifest, run_dir)
        direct = direct_checks(
            gateway.endpoint,
            runtime_model,
            reasoning_policy=reasoning_policy,
            direct_probe=direct_probe_config,
        )
    direct_transport = read_model_transport_observations(
        direct_observations_path
    )
    direct["model_transport"] = direct_transport
    if not (
        direct_transport["observer_ok"]
        and direct_transport["request_count"] == 2
        and direct_transport["response_count"] == 2
        and direct_transport["requested_reasoning_policy"]
        == reasoning_policy.value
    ):
        raise QualificationError(
            "direct model transport observer is missing or broken"
        )
    partial["components"]["direct"] = direct
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)

    upstream_results: dict[str, Any] = {}
    blocked_attempts: list[dict[str, Any]] = []
    transport_observations: dict[str, dict[str, Any]] = {}
    lock = load_upstream_lock()

    def run_upstream_component(
        name: str,
        command_factory: Callable[[str], Sequence[str]],
        checkout: Path,
        heartbeat_paths: Sequence[Path],
    ) -> None:
        component_dir = artifacts / name
        component_dir.mkdir(parents=True, exist_ok=True)
        observations_path = component_dir / "model-transport-observations.jsonl"
        with ModelGateway(
            target_base_url=endpoint.base_url,
            policy=reasoning_policy,
            stage=name,
            observations_path=observations_path,
            upstream_timeout=remaining_profile_seconds(),
        ) as gateway:
            command = list(command_factory(gateway.endpoint.base_url))
            manifest["invocations"][name] = command
            manifest.setdefault("model_transport_boundaries", {})[name] = (
                gateway.metadata
            )
            _atomic_json(run_dir / "manifest.json", manifest, run_dir)
            policy_dir, attempts_path = write_local_network_policy(
                component_dir / "python-policy", gateway.endpoint
            )
            python_paths = _upstream_python_paths(name, checkout, policy_dir)
            env = upstream_environment(python_paths=python_paths)
            if name == "spark-bench":
                chromium_wrapper = write_chromium_network_wrapper(
                    component_dir / "chromium-policy"
                )
                env["SPARK_BENCH_CHROMIUM"] = str(chromium_wrapper)
                raw_http_dir = component_dir / "raw-http"
                raw_http_dir.mkdir()
                env["SPARK_BENCH_DUMP_DIR"] = str(raw_http_dir)
                manifest.setdefault("runtime_policies", {})[name] = {
                    "chromium_wrapper": str(chromium_wrapper),
                    "browser_network": "dead-proxy-with-loopback-only-bypass",
                    "model_transport_boundary": gateway.endpoint.base_url,
                }
                _atomic_json(run_dir / "manifest.json", manifest, run_dir)
            execute_upstream(
                command,
                cwd=checkout,
                env=env,
                log_path=logs / f"{name}.log",
                total_timeout=remaining_profile_seconds(),
                inactivity_timeout=float(profile["inactivity_seconds"]),
                heartbeat_paths=heartbeat_paths,
            )
        if attempts_path.is_file():
            for line in attempts_path.read_text(encoding="utf-8").splitlines():
                try:
                    blocked_attempts.append({"component": name, **json.loads(line)})
                except json.JSONDecodeError:
                    blocked_attempts.append({"component": name, "raw": line})
        transport_observations[name] = read_model_transport_observations(
            observations_path
        )
        observation = transport_observations[name]
        if not (
            observation["observer_ok"]
            and observation["request_count"]
            and observation["response_count"]
            and observation["requested_reasoning_policy"]
            == reasoning_policy.value
        ):
            raise UpstreamError(
                f"{name} model transport observer is missing or broken"
            )

    spark_profile = profile.get("spark")
    if spark_profile:
        checkout = Path(upstream_provenance["spark-bench"]["checkout"])
        output = artifacts / "spark-bench"
        output.mkdir()
        label = f"gx10-{safe_slug(runtime_model, maximum=40)}-{profile_name}"
        run_upstream_component(
            "spark-bench",
            lambda gateway_endpoint: build_spark_command(
                checkout,
                endpoint=gateway_endpoint,
                model=runtime_model,
                output_dir=output,
                label=label,
                profile=spark_profile,
                reasoning_policy=reasoning_policy,
            ),
            checkout,
            (output / "runs",),
        )
        upstream_results["spark-bench"] = parse_spark(
            output / "spark_bench.csv", model=runtime_model
        )
        upstream_results["spark-bench"]["model_transport"] = (
            transport_observations["spark-bench"]
        )
        partial["components"]["upstreams"] = upstream_results
        _atomic_json(run_dir / "partial-results.json", partial, run_dir)
        if upstream_results["spark-bench"]["status"] == "QUARANTINED":
            reason = upstream_results["spark-bench"]["metrics"]["quarantine"]
            raise QualificationQuarantine(
                f"Spark Bench completed but was quarantined: {reason}"
            )

    for selection in profile.get("benchlocal", []):
        component_name = f"benchlocal-{selection['id']}"
        checkout = Path(upstream_provenance["benchlocal-cli"]["checkout"])
        output = artifacts / component_name / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        run_upstream_component(
            component_name,
            lambda gateway_endpoint, selection=selection: build_benchlocal_command(
                checkout,
                endpoint=gateway_endpoint,
                model=runtime_model,
                output_path=output,
                selection=selection,
                reasoning_policy=reasoning_policy,
            ),
            checkout,
            (Path(str(output) + ".partial.jsonl"), output),
        )
        upstream_results[component_name] = parse_benchlocal(output, model=runtime_model)
        upstream_results[component_name]["model_transport"] = (
            transport_observations[component_name]
        )
        partial["components"]["upstreams"] = upstream_results
        _atomic_json(run_dir / "partial-results.json", partial, run_dir)

    checkout = Path(upstream_provenance["infermark"]["checkout"])
    infer_contexts: dict[str, Any] = {}
    for context_name, target in profile["infermark"]["contexts"].items():
        infer_output = artifacts / f"infermark-{context_name}" / "result.json"
        infer_output.parent.mkdir(parents=True, exist_ok=True)
        run_upstream_component(
            f"infermark-{context_name}",
            lambda gateway_endpoint, target=target, infer_output=infer_output: (
                build_infermark_command(
                    checkout,
                    endpoint=gateway_endpoint,
                    model=runtime_model,
                    output_path=infer_output,
                    profile=profile["infermark"],
                    prompt=context_prompt(int(target)),
                )
            ),
            checkout,
            (infer_output,),
        )
        parsed = parse_infermark(infer_output, model=runtime_model)
        parsed["metrics"]["context_target_units"] = int(target)
        parsed["model_transport"] = transport_observations[
            f"infermark-{context_name}"
        ]
        infer_contexts[context_name] = parsed
    infer_rows = list(infer_contexts.values())
    short_metrics = infer_contexts.get("short", infer_rows[0])["metrics"]
    upstream_results["infermark"] = {
        "status": "PASS" if all(row["status"] == "PASS" for row in infer_rows) else "FAIL",
        "score": round(sum(float(row["score"]) for row in infer_rows) / len(infer_rows), 3),
        "metrics": {
            **short_metrics,
            "errors": sum(int(row["metrics"]["errors"]) for row in infer_rows),
            "contexts": {name: row["metrics"] for name, row in infer_contexts.items()},
        },
        "model_transport": combine_model_transport_observations(
            {
                name: row["model_transport"]
                for name, row in infer_contexts.items()
            }
        ),
    }
    partial["components"]["upstreams"] = upstream_results
    partial["components"]["blocked_network_attempts"] = blocked_attempts
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)

    plan = load_execution_plan(alias, model_override=effective_model_config)
    model_metadata = {
        **effective_model_config,
        "runtime_digest": preflight["runtime_model_digest"],
        "runtime_identity_status": "VERIFIED",
    }
    benchmark_metadata = {
        **plan["benchmark"],
        "runtime_identity": preflight,
        "runtime_version_observed": preflight.get("runtime_version"),
    }
    hermes_results: list[dict[str, Any]] = []
    for task_id in profile["hermes_tasks"]:
        from harness.hermes_runner import run_once

        task = next(
            (
                item["manifest"]
                for item in plan["tasks"]
                if item["manifest"]["id"] == task_id
            ),
            None,
        )
        if task is None:
            # The smoke is deliberately outside the scored suite but still
            # uses the same production runner and isolated evaluator path.
            from harness.workspace import load_task

            task, _ = load_task(task_id)
        hermes_transport_path = (
            artifacts
            / "hermes"
            / f"{task_id}-model-transport-observations.jsonl"
        )
        with ModelGateway(
            target_base_url=endpoint.base_url,
            policy=reasoning_policy,
            stage=f"hermes-{task_id}",
            observations_path=hermes_transport_path,
            upstream_timeout=remaining_profile_seconds(),
        ) as gateway:
            manifest.setdefault("model_transport_boundaries", {})[
                f"hermes-{task_id}"
            ] = gateway.metadata
            _atomic_json(run_dir / "manifest.json", manifest, run_dir)
            result = run_once(
                task_id=task_id,
                model=runtime_model,
                reasoning=reasoning_policy.hermes_effort,
                reasoning_policy=reasoning_policy.value,
                max_turns=int(task["limits"]["agent_turns"]),
                base_url=endpoint.base_url,
                transport_base_url=gateway.endpoint.base_url,
                transport_observations_path=hermes_transport_path,
                model_alias=alias,
                model_metadata=model_metadata,
                benchmark_metadata=benchmark_metadata,
                evaluator_timeout=plan["evaluator_timeout_seconds"],
                wall_timeout_seconds=remaining_profile_seconds(),
            )
        copied_hermes = _copy_and_cleanup_hermes(
            result,
            artifacts / "hermes" / f"{task_id}-{result['run_id']}",
            transport_observations_path=hermes_transport_path,
        )
        hermes_results.append(copied_hermes)
        partial["components"]["hermes"] = hermes_results
        _atomic_json(run_dir / "partial-results.json", partial, run_dir)
        if not copied_hermes.get("infrastructure_ok"):
            reasons = ", ".join(str(value) for value in result.get("infra_reasons", []))
            raise QualificationError(
                f"Hermes {task_id} infrastructure failure: {reasons or 'unknown'}"
            )

    components = {
        "preflight": preflight,
        "direct": direct,
        "upstreams": upstream_results,
        "hermes": hermes_results,
        "blocked_network_attempts": blocked_attempts,
    }
    outcome, gates, scores = calculate_decision(
        profile_name, components, config, reasoning_policy
    )
    profile_decision = (
        "MEETS_PROFILE" if outcome == "QUALIFIED" else "DOES_NOT_MEET_PROFILE"
    )
    manifest["status"] = "COMPLETE"
    manifest["finished_at"] = utc_now()
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest["resource_usage"] = {
        "child_max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "scope": "maximum resident set size observed for one orchestrator child process",
    }
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)
    result_model: dict[str, Any] = {
        "requested": requested_model,
        "config_alias": alias,
        "runtime_model": runtime_model,
        "runtime_digest": preflight["runtime_model_digest"],
        "endpoint": endpoint.base_url,
        "quantization": effective_model_config["quantization"],
        "context_length": effective_model_config["context_length"],
        "reasoning_policy": reasoning_policy.value,
        "reasoning_cohort": reasoning_policy.cohort,
        "reasoning_policy_selection_mode": reasoning_policy_selection_mode,
        "benchmark_track": benchmark_track,
        "reasoning_effort": reasoning_policy.hermes_effort or "native",
        "configured_reasoning_policy": effective_model_config.get(
            "configured_reasoning_policy"
        ),
        "configured_reasoning_effort": effective_model_config.get(
            "configured_reasoning_effort"
        ),
        "configuration_source": effective_model_config.get(
            "configuration_source"
        ),
        "metadata_sources": effective_model_config.get("metadata_sources", {}),
        "runtime_metadata": preflight.get("public_runtime_metadata", {}),
        "native_context_length": effective_model_config.get(
            "native_context_length"
        ),
        "runtime_default_context_length": effective_model_config.get(
            "runtime_default_context_length"
        ),
        "metadata_discrepancies": metadata_discrepancies,
    }
    try:
        result_model["public_identity"] = resolve_public_identity(
            result_model,
            manifest,
            runtime_metadata=preflight.get("public_runtime_metadata"),
        )
    except ModelMetadataError as exc:
        raise QualificationError(str(exc)) from exc
    manifest["public_model_identity"] = result_model["public_identity"]
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "profile": profile_name,
        "outcome": outcome,
        "result_validity": "VALID",
        "profile_decision": profile_decision,
        "model": result_model,
        "provenance": manifest,
        "components": components,
        "gates": gates,
        "scores": scores,
        "artifacts": {
            "run_directory": str(run_dir),
            "manifest": str(run_dir / "manifest.json"),
            "results": str(run_dir / "results.json"),
            "report": str(run_dir / "report.md"),
            "logs": str(logs),
            "artifacts": str(artifacts),
        },
    }
    schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(result)
    _atomic_json(run_dir / "results.json", result, run_dir)
    atomic_write_text(run_dir / "report.md", render_report(result), root=run_dir)
    partial["status"] = "COMPLETE"
    partial["outcome"] = outcome
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)
    _ACTIVE_RUN_CONTEXT = None
    return (0 if outcome == "QUALIFIED" else 1), result


def preserve_infrastructure_failure(exc: BaseException) -> Path | None:
    global _ACTIVE_RUN_CONTEXT
    context = _ACTIVE_RUN_CONTEXT
    if not context:
        return None
    run_dir = Path(context["run_dir"])
    manifest = dict(context["manifest"])
    quarantined = isinstance(exc, QualificationQuarantine)
    result_validity = "QUARANTINED" if quarantined else "INFRA_FAILURE"
    manifest["status"] = result_validity
    manifest["finished_at"] = utc_now()
    manifest["infrastructure_error"] = f"{type(exc).__name__}: {exc}"
    try:
        partial = json.loads((run_dir / "partial-results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        partial = {"components": {}}
    result = {
        "schema_version": 1,
        "run_id": context["run_id"],
        "profile": context["profile"],
        "outcome": "INFRA_ERROR",
        "result_validity": result_validity,
        "profile_decision": "NOT_ASSESSED",
        "model": {
            "requested": context["requested_model"],
            "config_alias": context["alias"],
            "runtime_model": context["runtime_model"],
            "endpoint": context["endpoint"],
            "reasoning_policy": context.get("reasoning_policy"),
            "reasoning_cohort": context.get("reasoning_cohort"),
            "reasoning_policy_selection_mode": context.get(
                "reasoning_policy_selection_mode"
            ),
            "benchmark_track": context.get("benchmark_track"),
        },
        "provenance": manifest,
        "components": partial.get("components", {}),
        "gates": {
            "passed": False,
            "failures": [
                (
                    f"quarantined result: {exc}"
                    if quarantined
                    else f"infrastructure error: {type(exc).__name__}: {exc}"
                )
            ],
        },
        "scores": {
            "deployment_configuration_score": None,
            "components": {},
            "weights": {},
            "executed_weight": 0,
            "comparable_full_profile": False,
        },
        "artifacts": {
            "run_directory": str(run_dir),
            "manifest": str(run_dir / "manifest.json"),
            "results": str(run_dir / "results.json"),
            "report": str(run_dir / "report.md"),
            "logs": str(run_dir / "logs"),
            "artifacts": str(run_dir / "artifacts"),
        },
    }
    _atomic_json(run_dir / "manifest.json", manifest, run_dir)
    _atomic_json(run_dir / "results.json", result, run_dir)
    atomic_write_text(run_dir / "report.md", render_report(result), root=run_dir)
    partial["status"] = "INFRA_ERROR"
    partial["outcome"] = "INFRA_ERROR"
    partial["result_validity"] = result_validity
    partial["profile_decision"] = "NOT_ASSESSED"
    partial["error"] = manifest["infrastructure_error"]
    _atomic_json(run_dir / "partial-results.json", partial, run_dir)
    _ACTIVE_RUN_CONTEXT = None
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark-model",
        description=(
            "Qualify one installed local model through pinned upstream suites "
            "and Hermes."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("compare",),
        help="Compare completed local runs without executing a benchmark",
    )
    parser.add_argument("--model", help="Configured alias or exact local runtime model id")
    parser.add_argument("--endpoint", help="Local OpenAI-compatible /v1 URL")
    parser.add_argument("--profile", choices=("smoke", "standard", "overnight"), default="standard")
    parser.add_argument(
        "--reasoning-policy",
        choices=("configured", *SUPPORTED_REASONING_POLICIES),
        default="configured",
        help=(
            "Exact reasoning policy; 'configured' resolves the curated "
            "models.yaml policy"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without setup or endpoint contact")
    parser.add_argument("--no-setup", action="store_true", help="Require already-present pinned upstream checkouts")
    parser.add_argument("--setup-only", action="store_true", help="Create/verify pinned upstream checkouts and exit")
    parser.add_argument("--list-upstreams", action="store_true", help="Print the pinned upstream manifest and exit")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_ROOT,
        help="Run directory scanned by the compare command",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Generated Markdown and JSON destination for compare",
    )
    parser.add_argument(
        "--public-output-dir",
        type=Path,
        default=DEFAULT_PUBLIC_OUTPUT_ROOT,
        help="Sanitized public Markdown and JSON leaderboard destination",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            from harness.comparison import (
                compare_runs,
                render_public_terminal,
                render_terminal,
            )

            (
                summary,
                markdown_path,
                json_path,
                leaderboard,
                public_markdown_path,
                public_json_path,
            ) = compare_runs(
                runs_root=args.runs_dir,
                output_root=args.output_dir,
                public_output_root=args.public_output_dir,
            )
            print(render_terminal(summary), end="")
            print(f"markdown={markdown_path}")
            print(f"json={json_path}")
            print(f"public_markdown={public_markdown_path}")
            print(f"public_json={public_json_path}")
            print(
                "metadata_candidates="
                f"{markdown_path.parent / METADATA_CANDIDATES_FILENAME}"
            )
            print(render_public_terminal(leaderboard), end="")
            return 0
        if args.list_upstreams:
            print(json.dumps(load_upstream_lock(), indent=2))
            return 0
        if args.setup_only:
            print(json.dumps(ensure_checkouts(setup=not args.no_setup), indent=2))
            return 0
        if not args.model:
            parser.error("--model is required unless --setup-only or --list-upstreams is used")
        config = load_configuration()
        endpoint = resolve_endpoint(args.endpoint)
        _alias, runtime_model, _model, _preflight = resolve_model(args.model)
        profile = config["profiles"][args.profile]
        if args.dry_run:
            try:
                dry_policy, dry_policy_source = resolve_reasoning_policy(
                    args.reasoning_policy, _model
                )
            except ReasoningPolicyError as exc:
                raise QualificationError(str(exc)) from exc
            print(
                json.dumps(
                    _dry_plan(
                        profile_name=args.profile,
                        profile=profile,
                        endpoint=endpoint,
                        model=runtime_model,
                        model_resolution=str(
                            _model.get("configuration_source")
                        ),
                        reasoning_policy_request=args.reasoning_policy,
                        reasoning_policy=dry_policy,
                        reasoning_policy_source=dry_policy_source,
                    ),
                    indent=2,
                )
            )
            return 0
        code, result = execute(
            requested_model=args.model,
            endpoint_override=args.endpoint,
            profile_name=args.profile,
            setup=not args.no_setup,
            reasoning_policy_request=args.reasoning_policy,
        )
        identity = result["model"].get("public_identity", {})
        print(f"outcome={result['outcome']}")
        printed_decision = result.get("profile_decision") or (
            "MEETS_PROFILE"
            if result["outcome"] == "QUALIFIED"
            else "DOES_NOT_MEET_PROFILE"
        )
        print(f"profile_decision={printed_decision}")
        print(f"result_validity={result.get('result_validity', 'VALID')}")
        print(f"technical_outcome={result['outcome']}")
        print(f"run_id={result['run_id']}")
        print(f"model_identity={identity.get('display_name', 'METADATA_INCOMPLETE')}")
        print(f"runtime_alias={result['model'].get('runtime_model')}")
        print(f"score={result['scores']['deployment_configuration_score']}")
        print(f"report={result['artifacts']['report']}")
        try:
            from harness.comparison import compare_runs, render_public_terminal

            (
                _summary,
                markdown_path,
                json_path,
                leaderboard,
                public_markdown_path,
                public_json_path,
            ) = compare_runs(
                runs_root=RUNS_ROOT,
                output_root=args.output_dir,
                public_output_root=args.public_output_dir,
            )
        except (ArtifactSafetyError, ComparisonError, jsonschema.ValidationError) as exc:
            print(
                "benchmark-model: POST_PROCESS_ERROR: completed qualification "
                f"outcome remains {result['outcome']}: {exc}",
                file=sys.stderr,
            )
            return code
        print(f"comparison_markdown={markdown_path}")
        print(f"comparison_json={json_path}")
        print(f"public_markdown={public_markdown_path}")
        print(f"public_json={public_json_path}")
        print(
            "metadata_candidates="
            f"{markdown_path.parent / METADATA_CANDIDATES_FILENAME}"
        )
        print(
            render_public_terminal(
                leaderboard, highlight_run_id=result["run_id"]
            ),
            end="",
        )
        return code
    except KeyboardInterrupt:
        print("benchmark-model: interrupted", file=sys.stderr)
        return 130
    except (
        ArtifactSafetyError,
        ComparisonError,
        QualificationError,
        UpstreamError,
        jsonschema.ValidationError,
    ) as exc:
        print(f"benchmark-model: INFRA_ERROR: {exc}", file=sys.stderr)
        failure_dir = preserve_infrastructure_failure(exc)
        if failure_dir:
            print(f"artifacts={failure_dir}", file=sys.stderr)
            if isinstance(exc, QualificationQuarantine):
                print("profile_decision=NOT_ASSESSED", file=sys.stderr)
                print("result_validity=QUARANTINED", file=sys.stderr)
                print("technical_outcome=INFRA_ERROR", file=sys.stderr)
                print(f"not_assessed_reason={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
