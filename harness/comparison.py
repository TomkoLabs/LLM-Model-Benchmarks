from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import jsonschema
import yaml

from harness.artifacts import atomic_write_text, ensure_root
from harness.model_identity import (
    REGISTRY_PATH,
    ModelMetadataError,
    load_registry,
    resolve_public_identity,
)
from harness.reasoning_policy import (
    CONFIGURED_POLICY_RULE,
    CONTROLLED_POLICY_TRACK,
    PRIMARY_DEPLOYMENT_TRACK,
    SUPPORTED_REASONING_POLICIES,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs"
DEFAULT_OUTPUT_ROOT = ROOT / "generated-results"
DEFAULT_PUBLIC_OUTPUT_ROOT = ROOT / "public-results"
RESULT_SCHEMA = ROOT / "schemas" / "qualification-run.schema.json"
PUBLIC_SCHEMA = ROOT / "schemas" / "public-leaderboard.schema.json"
QUALIFICATION_CONFIG = ROOT / "configs" / "qualification-v4.yaml"
UPSTREAM_LOCK = ROOT / "upstreams.lock.json"
MAX_RESULT_BYTES = 20 * 1024 * 1024
METADATA_CANDIDATES_FILENAME = "model-metadata-candidates.json"
PROJECT_NAME = "LLM Model Benchmarks"
PROJECT_URL = "https://github.com/TomkoLabs/LLM-Model-Benchmarks"
SCORED_OUTCOMES = {"QUALIFIED", "NOT_QUALIFIED"}
SCORE_NAMES = (
    "coding",
    "hermes",
    "tool_instruction",
    "reliability",
    "performance",
)
RANKING_RULE = (
    "overall score descending, then coding, Hermes, tool/instruction, "
    "reliability, and performance scores descending; then timestamp and run ID"
)


class ComparisonError(RuntimeError):
    pass


def _canonical_hash(value: Any, *, length: int = 16) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ComparisonError("result is missing or is not a regular file")
    if path.stat().st_size > MAX_RESULT_BYTES:
        raise ComparisonError("result exceeds the comparison size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError("result is not readable strict JSON") from exc
    if not isinstance(value, dict):
        raise ComparisonError("result root must be an object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 3)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _upstream_pins(provenance: Mapping[str, Any]) -> dict[str, str]:
    upstreams = provenance.get("upstreams")
    if not isinstance(upstreams, Mapping):
        return {}
    pins: dict[str, str] = {}
    for name, details in sorted(upstreams.items()):
        if not isinstance(name, str) or not isinstance(details, Mapping):
            continue
        commit = details.get("commit")
        if isinstance(commit, str) and commit:
            pins[name] = commit
    return pins


def _pin_set_id(pins: Mapping[str, str]) -> str | None:
    return "pins-" + _canonical_hash(pins, length=12) if pins else None


def current_standard_compatibility() -> dict[str, Any]:
    try:
        config = yaml.safe_load(QUALIFICATION_CONFIG.read_text(encoding="utf-8"))
        lock = json.loads(UPSTREAM_LOCK.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as exc:
        raise ComparisonError(
            f"cannot load current comparison configuration: {exc}"
        ) from exc
    if not isinstance(config, Mapping) or not isinstance(lock, Mapping):
        raise ComparisonError("current comparison configuration is malformed")
    profiles = config.get("profiles")
    weights = config.get("weights")
    generation = _text(config.get("generation"))
    benchmark_track = _text(config.get("public_benchmark_track"))
    reasoning_policy_rule = _text(config.get("public_reasoning_policy_rule"))
    direct_probe = config.get("direct_probe")
    direct_probe = direct_probe if isinstance(direct_probe, Mapping) else {}
    direct_probe_contract = _text(direct_probe.get("contract"))
    direct_probe_max_tokens = direct_probe.get("max_tokens")
    upstreams = lock.get("upstreams")
    if (
        not isinstance(profiles, Mapping)
        or not isinstance(profiles.get("standard"), Mapping)
        or not isinstance(weights, Mapping)
        or not generation
        or benchmark_track != PRIMARY_DEPLOYMENT_TRACK
        or reasoning_policy_rule != CONFIGURED_POLICY_RULE
        or direct_probe_contract != "gx10-direct-probe-v1"
        or direct_probe_max_tokens != 1024
        or not isinstance(upstreams, Mapping)
    ):
        raise ComparisonError(
            "current standard comparison configuration is incomplete"
        )
    pins: dict[str, str] = {}
    for name, details in sorted(upstreams.items()):
        if not isinstance(name, str) or not isinstance(details, Mapping):
            continue
        if details.get("integrated") is False or not details.get("checkout"):
            continue
        commit = details.get("commit")
        if isinstance(commit, str) and commit:
            pins[name] = commit
    key = {
        "qualification_generation": generation,
        "scoring_version": generation,
        "profile": "standard",
        "benchmark_track": benchmark_track,
        "reasoning_policy_rule": reasoning_policy_rule,
        "direct_probe_contract": direct_probe_contract,
        "direct_probe_max_tokens": direct_probe_max_tokens,
        "profile_config_sha256": _canonical_hash(
            profiles["standard"], length=64
        ),
        "scoring_weights": dict(sorted(weights.items())),
        "upstream_pins": pins,
    }
    return {
        "group_id": "group-" + _canonical_hash(key, length=12),
        "complete": True,
        "key": key,
        "upstream_pin_set": _pin_set_id(pins),
    }


def _safe_runtime_alias(value: Any) -> str | None:
    alias = _text(value)
    if alias is None:
        return None
    if len(alias) > 200 or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", alias
    ) is None:
        return "REDACTED"
    if alias.startswith("/") or "://" in alias:
        return "REDACTED"
    if re.search(
        r"(?:^|[^0-9])(?:127(?:\.[0-9]{1,3}){3}|10(?:\.[0-9]{1,3}){3}|"
        r"192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])"
        r"(?:\.[0-9]{1,3}){2})(?:$|[^0-9])",
        alias,
    ):
        return "REDACTED"
    return alias


def _safe_run_id(value: Any) -> str:
    run_id = _text(value)
    if run_id and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", run_id):
        return run_id
    return "REDACTED-" + _canonical_hash(str(value), length=12)


def _public_hardware_runtime(result: Mapping[str, Any]) -> str:
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    configured = provenance.get("model_config")
    configured = configured if isinstance(configured, Mapping) else {}
    components = result.get("components")
    components = components if isinstance(components, Mapping) else {}
    preflight = components.get("preflight")
    preflight = preflight if isinstance(preflight, Mapping) else {}
    host = provenance.get("host")
    host = host if isinstance(host, Mapping) else {}

    provider = _text(configured.get("provider"))
    hardware = (
        "NVIDIA GB10/GX10 model server"
        if provider == "gx10"
        else "local model server"
    )
    endpoint = _text(model.get("endpoint")) or _text(provenance.get("endpoint"))
    layout = "deployment layout not recorded"
    if endpoint:
        hostname = urlsplit(endpoint).hostname
        if hostname in {"127.0.0.1", "::1", "localhost"}:
            layout = "co-located harness/runtime"
        elif hostname:
            layout = "split Linux controller/model server"
    machine = _text(host.get("machine"))
    if machine and re.fullmatch(r"[A-Za-z0-9_+.-]+", machine):
        layout += f" ({machine} controller)"
    runtime = _text(configured.get("runtime"))
    runtime_name = (
        "Ollama"
        if runtime == "ollama"
        else "DS4"
        if runtime == "ds4"
        else "local runtime"
    )
    version = _text(preflight.get("runtime_version"))
    if version is None:
        phase1 = configured.get("phase1")
        if isinstance(phase1, Mapping):
            version = _text(phase1.get("ollama_version"))
    if version and re.fullmatch(r"[A-Za-z0-9_+.-]+", version):
        runtime_name += f" {version}"
    reasoning = _text(model.get("reasoning_policy")) or _text(
        model.get("reasoning_effort")
    )
    if reasoning and re.fullmatch(r"[A-Za-z0-9_+.-]+", reasoning):
        runtime_name += f"; reasoning={reasoning}"
    return f"{hardware}; {layout}; {runtime_name}"


def _compatibility(
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    scores = result.get("scores")
    scores = scores if isinstance(scores, Mapping) else {}
    generation = _text(provenance.get("qualification_generation"))
    scoring_version = _text(provenance.get("scoring_version")) or generation
    profile = _text(result.get("profile"))
    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    profile_config = provenance.get("profile_config")
    weights = scores.get("weights")
    upstream_pins = _upstream_pins(provenance)
    direct_probe = provenance.get("direct_probe")
    direct_probe_recorded = isinstance(direct_probe, Mapping)
    direct_probe = direct_probe if direct_probe_recorded else {}
    direct_probe_contract = _text(direct_probe.get("contract"))
    direct_probe_max_tokens = direct_probe.get("max_tokens")
    if not direct_probe_recorded and generation in {
        "gx10-qualification-v2",
        "gx10-qualification-v3",
        "gx10-qualification-v4",
    }:
        direct_probe_contract = "gx10-direct-probe-v1"
        direct_probe_max_tokens = 128
    direct_probe_issues: list[str] = []
    if not (
        type(direct_probe_max_tokens) is int and direct_probe_max_tokens > 0
    ):
        direct_probe_max_tokens = None
    if direct_probe_recorded:
        components = result.get("components")
        components = components if isinstance(components, Mapping) else {}
        direct_component = components.get("direct")
        direct_component = (
            direct_component if isinstance(direct_component, Mapping) else {}
        )
        if (
            _text(direct_component.get("contract")) != direct_probe_contract
            or direct_component.get("max_tokens") != direct_probe_max_tokens
        ):
            direct_probe_issues.append("direct-probe result consistency")
    policy_provenance = provenance.get("reasoning_policy")
    policy_provenance = (
        policy_provenance if isinstance(policy_provenance, Mapping) else {}
    )
    model_policy = _text(model.get("reasoning_policy"))
    provenance_policy = _text(policy_provenance.get("effective"))
    reasoning_policy = model_policy or provenance_policy
    reasoning_policy_selection_mode: str | None = None
    reasoning_policy_source: str | None = None
    benchmark_track: str | None = None
    reasoning_policy_rule: str | None = None
    policy_issues: list[str] = []
    if model_policy and provenance_policy and model_policy != provenance_policy:
        policy_issues.append("resolved reasoning policy mismatch")

    if generation == "gx10-qualification-v4":
        requested = _text(policy_provenance.get("requested"))
        recorded_selection = _text(policy_provenance.get("selection_mode"))
        model_selection = _text(model.get("reasoning_policy_selection_mode"))
        recorded_track = _text(policy_provenance.get("benchmark_track"))
        model_track = _text(model.get("benchmark_track"))
        reasoning_policy_source = _text(policy_provenance.get("source"))
        derived_selection = (
            "configured"
            if requested == "configured"
            else "explicit"
            if requested in SUPPORTED_REASONING_POLICIES
            else None
        )
        derived_track = (
            PRIMARY_DEPLOYMENT_TRACK
            if derived_selection == "configured"
            else CONTROLLED_POLICY_TRACK
            if derived_selection == "explicit"
            else None
        )
        reasoning_policy_selection_mode = (
            recorded_selection or model_selection or derived_selection
        )
        benchmark_track = recorded_track or model_track or derived_track
        if derived_selection is None:
            policy_issues.append("requested reasoning policy selection")
        if any(
            value and value != derived_selection
            for value in (recorded_selection, model_selection)
        ):
            policy_issues.append("reasoning policy selection consistency")
        if any(
            value and value != derived_track
            for value in (recorded_track, model_track)
        ):
            policy_issues.append("benchmark track consistency")
        expected_source = (
            "models.yaml curated policy"
            if derived_selection == "configured"
            else "explicit CLI"
            if derived_selection == "explicit"
            else None
        )
        if reasoning_policy_source != expected_source:
            policy_issues.append("reasoning policy source consistency")
        if derived_selection == "configured":
            configured_model = provenance.get("model_config")
            configured_model = (
                configured_model if isinstance(configured_model, Mapping) else {}
            )
            configured_policy = _text(configured_model.get("reasoning_policy"))
            if configured_policy is None or configured_policy != reasoning_policy:
                policy_issues.append("configured model reasoning policy consistency")
            reasoning_policy_rule = CONFIGURED_POLICY_RULE
        elif derived_selection == "explicit":
            if requested != reasoning_policy:
                policy_issues.append("explicit reasoning policy consistency")
            reasoning_policy_rule = reasoning_policy
    elif generation == "gx10-qualification-v3":
        if isinstance(profile_config, Mapping) and profile_config.get("thinking") == "off":
            reasoning_policy = "off"
        reasoning_policy_selection_mode = "legacy-unspecified"
        reasoning_policy_source = "historical result"
        benchmark_track = "historical"
        reasoning_policy_rule = reasoning_policy
    elif generation == "gx10-qualification-v2":
        reasoning_policy = "legacy-thinking-control-unspecified"
        reasoning_policy_selection_mode = "legacy-unspecified"
        reasoning_policy_source = "historical result"
        benchmark_track = "historical"
        reasoning_policy_rule = reasoning_policy
    missing: list[str] = []
    for label, value in (
        ("qualification generation", generation),
        ("scoring version", scoring_version),
        ("profile", profile),
        ("reasoning policy", reasoning_policy),
        ("reasoning policy selection", reasoning_policy_selection_mode),
        ("benchmark track", benchmark_track),
        ("reasoning policy comparison rule", reasoning_policy_rule),
        ("direct-probe contract", direct_probe_contract),
        ("direct-probe max tokens", direct_probe_max_tokens),
    ):
        if value is None:
            missing.append(label)
    if not isinstance(profile_config, Mapping):
        missing.append("profile configuration")
    if not isinstance(weights, Mapping) or not weights:
        missing.append("scoring weights")
    if not upstream_pins:
        missing.append("upstream pins")
    missing.extend(policy_issues)
    missing.extend(direct_probe_issues)

    key = {
        "qualification_generation": generation,
        "scoring_version": scoring_version,
        "profile": profile,
        "benchmark_track": benchmark_track,
        "reasoning_policy_rule": reasoning_policy_rule,
        "direct_probe_contract": direct_probe_contract,
        "direct_probe_max_tokens": direct_probe_max_tokens,
        "profile_config_sha256": (
            _canonical_hash(profile_config, length=64)
            if isinstance(profile_config, Mapping)
            else None
        ),
        "scoring_weights": dict(sorted(weights.items()))
        if isinstance(weights, Mapping)
        else None,
        "upstream_pins": upstream_pins,
    }
    group_id = None if missing else "group-" + _canonical_hash(key, length=12)
    return {
        "group_id": group_id,
        "complete": not missing,
        "key": key,
        "reasoning_policy": reasoning_policy,
        "reasoning_policy_selection_mode": reasoning_policy_selection_mode,
        "reasoning_policy_source": reasoning_policy_source,
        "benchmark_track": benchmark_track,
    }, missing


def _base_row(run_dir: Path, outcome: str, reason: str) -> dict[str, Any]:
    return {
        "run_timestamp": None,
        "completed_at": None,
        "run_id": run_dir.name,
        "source": f"{run_dir.name}/results.json",
        "validation_status": outcome,
        "model_tag": None,
        "runtime_alias": None,
        "model_digest": None,
        "model_identity": {
            "status": "METADATA_INCOMPLETE",
            "display_name": "METADATA_INCOMPLETE — digest unavailable",
            "canonical_name": None,
            "source_model": None,
            "source_url": None,
            "version": None,
            "parameter_variant": None,
            "architecture": None,
            "quantization": None,
            "context_length": None,
            "immutable_digest": None,
            "runtime_alias": None,
            "runtime_variant": None,
            "source_filename": None,
            "registry_match": False,
            "alias_registered": False,
            "issues": ["full immutable digest is unavailable"],
        },
        "profile": None,
        "qualification_generation": None,
        "scoring_version": None,
        "reasoning_policy": None,
        "reasoning_policy_selection_mode": None,
        "reasoning_policy_source": None,
        "benchmark_track": None,
        "outcome": outcome,
        "result_validity": "INCOMPLETE",
        "profile_decision": "NOT_ASSESSED",
        "overall_score": None,
        "component_scores": {name: None for name in SCORE_NAMES},
        "duration_seconds": None,
        "infrastructure_error": None,
        "public_hardware_runtime": "deployment metadata unavailable",
        "upstream_pin_set": None,
        "eligible_for_model_comparison": False,
        "exclusion_reason": reason,
        "compatibility": {"group_id": None, "complete": False, "key": None},
    }


def _incomplete_row(run_dir: Path) -> dict[str, Any]:
    row = _base_row(run_dir, "INCOMPLETE", "results.json is missing")
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except ComparisonError:
        return row
    row["run_timestamp"] = _text(manifest.get("started_at"))
    row["completed_at"] = _text(manifest.get("finished_at"))
    row["run_id"] = _text(manifest.get("run_id")) or run_dir.name
    alias = _safe_runtime_alias(
        manifest.get("runtime_model") or manifest.get("model_request")
    )
    row["model_tag"] = alias
    row["runtime_alias"] = alias
    row["model_identity"]["runtime_alias"] = alias
    row["profile"] = _text(manifest.get("profile"))
    row["qualification_generation"] = _text(
        manifest.get("qualification_generation")
    )
    row["duration_seconds"] = _number(manifest.get("elapsed_seconds"))
    row["infrastructure_error"] = (
        True if manifest.get("status") == "INFRA_ERROR" else None
    )
    return row


def _result_row(
    run_dir: Path,
    result: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = result.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    model = result.get("model")
    model = model if isinstance(model, Mapping) else {}
    scores = result.get("scores")
    scores = scores if isinstance(scores, Mapping) else {}
    score_components = scores.get("components")
    score_components = (
        score_components if isinstance(score_components, Mapping) else {}
    )
    run_components = result.get("components")
    run_components = (
        run_components if isinstance(run_components, Mapping) else {}
    )
    outcome = _text(result.get("outcome")) or "INVALID"
    explicit_validity = _text(result.get("result_validity"))
    error_detail = str(provenance.get("infrastructure_error") or "")
    result_validity = explicit_validity or (
        "QUARANTINED"
        if outcome == "INFRA_ERROR" and "quarantin" in error_detail.lower()
        else "INFRA_FAILURE"
        if outcome == "INFRA_ERROR"
        else "VALID"
        if outcome in SCORED_OUTCOMES
        else "INCOMPLETE"
    )
    profile_decision = _text(result.get("profile_decision")) or (
        "MEETS_PROFILE"
        if outcome == "QUALIFIED"
        else "DOES_NOT_MEET_PROFILE"
        if outcome == "NOT_QUALIFIED"
        else "NOT_ASSESSED"
    )
    compatibility, missing = _compatibility(result)
    infrastructure_error = bool(
        result_validity == "INFRA_FAILURE"
        or provenance.get("status") == "INFRA_ERROR"
        and result_validity != "QUARANTINED"
    )
    overall = _number(scores.get("deployment_configuration_score"))
    completed = provenance.get("status") == "COMPLETE"
    runtime_metadata = run_components.get("preflight")
    if isinstance(runtime_metadata, Mapping):
        runtime_metadata = runtime_metadata.get("public_runtime_metadata")
    runtime_metadata = (
        runtime_metadata if isinstance(runtime_metadata, Mapping) else {}
    )
    identity = resolve_public_identity(
        model,
        provenance,
        registry=registry,
        runtime_metadata=runtime_metadata,
    )
    alias = _safe_runtime_alias(
        model.get("runtime_model") or model.get("requested")
    )
    eligible = (
        outcome in SCORED_OUTCOMES
        and result_validity == "VALID"
        and profile_decision in {"MEETS_PROFILE", "DOES_NOT_MEET_PROFILE"}
        and completed
        and overall is not None
        and compatibility["complete"]
    )
    reasons: list[str] = []
    if result_validity == "QUARANTINED":
        reasons.append("completed upstream evaluation was quarantined and cannot be ranked")
    elif result_validity == "INFRA_FAILURE":
        reasons.append("infrastructure error")
    elif outcome not in SCORED_OUTCOMES:
        reasons.append(f"non-scored outcome: {outcome}")
    if not completed:
        reasons.append("run is not marked COMPLETE")
    if overall is None:
        reasons.append("overall score is unavailable")
    if missing:
        reasons.append("missing compatibility metadata: " + ", ".join(missing))

    return {
        "run_timestamp": _text(provenance.get("started_at")),
        "completed_at": _text(provenance.get("finished_at")),
        "run_id": _text(result.get("run_id")) or run_dir.name,
        "source": f"{run_dir.name}/results.json",
        "validation_status": "VALID",
        "model_tag": alias,
        "runtime_alias": alias,
        "model_digest": _text(model.get("runtime_digest")),
        "model_identity": {**identity, "runtime_alias": alias},
        "profile": _text(result.get("profile")),
        "qualification_generation": _text(
            provenance.get("qualification_generation")
        ),
        "scoring_version": compatibility["key"]["scoring_version"],
        "reasoning_policy": compatibility["reasoning_policy"],
        "reasoning_policy_selection_mode": compatibility[
            "reasoning_policy_selection_mode"
        ],
        "reasoning_policy_source": compatibility["reasoning_policy_source"],
        "benchmark_track": compatibility["benchmark_track"],
        "outcome": outcome,
        "result_validity": result_validity,
        "profile_decision": profile_decision,
        "overall_score": overall,
        "component_scores": {
            name: _number(score_components.get(name)) for name in SCORE_NAMES
        },
        "duration_seconds": _number(provenance.get("elapsed_seconds")),
        "infrastructure_error": infrastructure_error,
        "public_hardware_runtime": _public_hardware_runtime(result),
        "upstream_pin_set": _pin_set_id(
            compatibility["key"]["upstream_pins"]
        ),
        "eligible_for_model_comparison": eligible,
        "exclusion_reason": None if eligible else "; ".join(reasons),
        "compatibility": compatibility,
    }


def collect_runs(
    runs_root: Path = DEFAULT_RUNS_ROOT,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> list[dict[str, Any]]:
    runs_root = runs_root.expanduser().resolve()
    if not runs_root.exists():
        return []
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ComparisonError(f"runs root is not a real directory: {runs_root}")
    schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    try:
        registry = load_registry(registry_path)
    except ModelMetadataError as exc:
        raise ComparisonError(str(exc)) from exc
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir(), key=lambda path: path.name):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        result_path = run_dir / "results.json"
        if not result_path.exists():
            if not (run_dir / "manifest.json").exists() and not (
                run_dir / "partial-results.json"
            ).exists():
                continue
            rows.append(_incomplete_row(run_dir))
            continue
        try:
            result = _read_json(result_path)
            validator.validate(result)
        except ComparisonError as exc:
            rows.append(_base_row(run_dir, "INVALID", str(exc)))
            continue
        except jsonschema.ValidationError:
            rows.append(
                _base_row(
                    run_dir,
                    "INVALID",
                    "results.json failed qualification-run schema validation",
                )
            )
            continue
        rows.append(_result_row(run_dir, result, registry))
    return sorted(
        rows,
        key=lambda row: (
            row.get("run_timestamp") or "",
            row["run_id"],
            row["source"],
        ),
    )


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("eligible_for_model_comparison"):
            continue
        compatibility = row.get("compatibility")
        if not isinstance(compatibility, Mapping):
            continue
        group_id = compatibility.get("group_id")
        if not isinstance(group_id, str):
            continue
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "key": compatibility.get("key"),
                "run_ids": [],
            },
        )
        group["run_ids"].append(row["run_id"])
    return {
        "schema_version": 1,
        "ordering": "run_timestamp_then_run_id; no ranking is applied",
        "comparison_rule": (
            "Compare scores only within one compatibility group. "
            "INFRA_ERROR, incomplete, and invalid runs are excluded."
        ),
        "run_count": len(rows),
        "eligible_run_count": sum(
            bool(row.get("eligible_for_model_comparison")) for row in rows
        ),
        "compatibility_groups": [groups[key] for key in sorted(groups)],
        "runs": list(rows),
    }


def build_metadata_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build review-only registry candidates for incomplete exact digests."""

    required_fields = (
        "canonical_name",
        "source_model",
        "version",
        "parameter_variant",
        "architecture",
        "quantization",
    )
    observations: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = row.get("model_identity")
        if not isinstance(identity, Mapping):
            continue
        if identity.get("status") != "METADATA_INCOMPLETE":
            continue
        digest = _text(identity.get("immutable_digest"))
        if digest is None or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            continue
        candidate = observations.setdefault(
            digest,
            {
                "runtime_aliases": set(),
                "run_ids": set(),
                "context_lengths": set(),
                "issues": set(),
                "fields": {
                    field: set()
                    for field in (*required_fields, "source_filename")
                },
            },
        )
        alias = _safe_runtime_alias(
            identity.get("runtime_alias") or row.get("runtime_alias")
        )
        if alias and alias != "REDACTED":
            candidate["runtime_aliases"].add(alias)
        candidate["run_ids"].add(_safe_run_id(row.get("run_id")))
        context = identity.get("context_length")
        if type(context) is int and context > 0:
            candidate["context_lengths"].add(context)
        issues = identity.get("issues")
        if isinstance(issues, list):
            candidate["issues"].update(str(value) for value in issues)
        for field in candidate["fields"]:
            value = _text(identity.get(field))
            if value:
                candidate["fields"][field].add(value)

    candidates: dict[str, Any] = {}
    for digest in sorted(observations):
        observed = observations[digest]
        suggested: dict[str, Any] = {}
        missing: list[str] = []
        conflicts: list[str] = []
        field_observations: dict[str, list[str]] = {}
        for field, values in observed["fields"].items():
            ordered = sorted(values)
            field_observations[field] = ordered
            suggested[field] = ordered[0] if len(ordered) == 1 else None
            if len(ordered) > 1:
                conflicts.append(field)
            if field in required_fields and not ordered:
                missing.append(field)
        suggested["source_url"] = None
        suggested["aliases"] = sorted(observed["runtime_aliases"])
        suggested["metadata_source"] = (
            "Review runtime metadata and original checkpoint provenance "
            "before copying this entry to model-metadata.yaml"
        )
        candidates[digest] = {
            "review_status": "REQUIRED",
            "run_ids": sorted(observed["run_ids"]),
            "runtime_aliases": sorted(observed["runtime_aliases"]),
            "observed_context_lengths": sorted(observed["context_lengths"]),
            "observed_fields": field_observations,
            "missing_registry_fields": missing,
            "conflicting_observed_fields": conflicts,
            "identity_issues": sorted(observed["issues"]),
            "suggested_registry_entry": suggested,
        }
    return {
        "schema_version": 1,
        "purpose": (
            "Review-only digest-keyed candidates; never applied to "
            "model-metadata.yaml automatically"
        ),
        "candidates": candidates,
    }


def _current_key(
    compatibility: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, str | None]:
    current = compatibility or current_standard_compatibility()
    key_value = current.get("key") if isinstance(current, Mapping) else None
    if isinstance(key_value, Mapping):
        key = dict(key_value)
        group_id = _text(current.get("group_id"))
    elif isinstance(current, Mapping):
        key = dict(current)
        group_id = None
    else:
        raise ComparisonError("current compatibility key is malformed")
    required = {
        "qualification_generation",
        "scoring_version",
        "profile",
        "benchmark_track",
        "reasoning_policy_rule",
        "direct_probe_contract",
        "direct_probe_max_tokens",
        "profile_config_sha256",
        "scoring_weights",
        "upstream_pins",
    }
    if set(key) != required or key.get("profile") != "standard":
        raise ComparisonError("current standard compatibility key is incomplete")
    expected_group = "group-" + _canonical_hash(key, length=12)
    if group_id is not None and group_id != expected_group:
        raise ComparisonError("current compatibility group ID does not match its key")
    pins = key.get("upstream_pins")
    pin_set = _pin_set_id(pins) if isinstance(pins, Mapping) else None
    return key, expected_group, pin_set


def _rank_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    components = row.get("component_scores")
    components = components if isinstance(components, Mapping) else {}

    def descending(value: Any) -> float:
        number = _number(value)
        return -number if number is not None else float("inf")

    return (
        descending(row.get("overall_score")),
        *(descending(components.get(name)) for name in SCORE_NAMES),
        row.get("run_timestamp") or "",
        row.get("run_id") or "",
    )


def _public_scored_row(
    row: Mapping[str, Any], *, rank: int | None
) -> dict[str, Any]:
    identity = row.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    components = row.get("component_scores")
    components = components if isinstance(components, Mapping) else {}
    timestamp = _text(row.get("run_timestamp"))
    return {
        "rank": rank,
        "run_id": _safe_run_id(row["run_id"]),
        "run_count": 1,
        "benchmark_date": timestamp[:10] if timestamp else None,
        "started_at": timestamp,
        "completed_at": _text(row.get("completed_at")),
        "model_identity": {
            key: identity.get(key)
            for key in (
                "status",
                "display_name",
                "canonical_name",
                "source_model",
                "source_url",
                "version",
                "parameter_variant",
                "architecture",
                "quantization",
                "context_length",
                "immutable_digest",
                "runtime_variant",
            )
        },
        "runtime_alias": _safe_runtime_alias(row.get("runtime_alias")),
        "benchmark_track": row.get("benchmark_track"),
        "reasoning_policy_selection_mode": row.get(
            "reasoning_policy_selection_mode"
        ),
        "reasoning_policy": row.get("reasoning_policy"),
        "profile": row.get("profile"),
        "profile_version": row.get("qualification_generation"),
        "scoring_version": row.get("scoring_version"),
        "outcome": row.get("outcome"),
        "result_validity": row.get("result_validity"),
        "profile_decision": row.get("profile_decision"),
        "overall_score": row.get("overall_score"),
        "component_scores": {
            name: components.get(name) for name in SCORE_NAMES
        },
        "duration_seconds": row.get("duration_seconds"),
        "hardware_runtime": row.get("public_hardware_runtime"),
        "upstream_pin_set": row.get("upstream_pin_set"),
    }


def _diagnostic_status(
    row: Mapping[str, Any], current_group_id: str, current_generation: str
) -> tuple[str, str]:
    validation = row.get("validation_status")
    outcome = row.get("outcome")
    profile = row.get("profile")
    identity = row.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    if validation in {"INVALID", "INCOMPLETE"}:
        return str(validation), str(row.get("exclusion_reason") or validation)
    if row.get("result_validity") == "QUARANTINED":
        reason = str(
            row.get("exclusion_reason")
            or "completed upstream evaluation was quarantined"
        )
        if row.get("qualification_generation") == "gx10-qualification-v2":
            reason += "; the historical v2 request also predates corrected thinking control"
        return "QUARANTINED", reason
    if outcome == "INFRA_ERROR" or row.get("infrastructure_error"):
        return "INFRA_ERROR", str(
            row.get("exclusion_reason") or "infrastructure error"
        )
    if identity.get("status") != "COMPLETE":
        issues = identity.get("issues")
        detail = (
            ", ".join(str(value) for value in issues)
            if isinstance(issues, list)
            else "canonical model metadata is incomplete"
        )
        return "METADATA_INCOMPLETE", detail
    if (
        row.get("qualification_generation") == "gx10-qualification-v2"
        and current_generation != "gx10-qualification-v2"
    ):
        return (
            "LEGACY_THINKING_CONTROL_MISMATCH",
            "historical v2 claimed thinking-off without uniformly serializing Ollama's supported control; rerun under v4",
        )
    if profile == "smoke":
        return (
            "SMOKE_ONLY",
            "smoke results are never ranked with standard qualification",
        )
    compatibility = row.get("compatibility")
    group_id = (
        compatibility.get("group_id")
        if isinstance(compatibility, Mapping)
        else None
    )
    if profile != "standard" or group_id != current_group_id:
        return (
            "INCOMPATIBLE",
            "benchmark track, controlled reasoning policy, profile, scoring configuration, or upstream pins differ from the current primary standard",
        )
    return "UNRANKED", str(row.get("exclusion_reason") or "result is not rankable")


def _public_diagnostic_row(
    row: Mapping[str, Any], current_group_id: str, current_generation: str
) -> dict[str, Any]:
    identity = row.get("model_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    status, reason = _diagnostic_status(
        row, current_group_id, current_generation
    )
    timestamp = _text(row.get("run_timestamp"))
    return {
        "run_id": _safe_run_id(row["run_id"]),
        "benchmark_date": timestamp[:10] if timestamp else None,
        "status": status,
        "qualification_outcome": row.get("outcome"),
        "result_validity": row.get("result_validity"),
        "profile_decision": row.get("profile_decision"),
        "profile": row.get("profile"),
        "profile_version": row.get("qualification_generation"),
        "model_identity": identity.get("display_name"),
        "runtime_alias": _safe_runtime_alias(row.get("runtime_alias")),
        "immutable_digest": identity.get("immutable_digest"),
        "benchmark_track": row.get("benchmark_track"),
        "reasoning_policy_selection_mode": row.get(
            "reasoning_policy_selection_mode"
        ),
        "reasoning_policy": row.get("reasoning_policy"),
        "reason": reason,
    }


def build_public_leaderboard(
    rows: Sequence[Mapping[str, Any]],
    *,
    current_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    key, current_group_id, pin_set = _current_key(current_compatibility)
    current_rows: list[Mapping[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        compatibility = row.get("compatibility")
        group_id = (
            compatibility.get("group_id")
            if isinstance(compatibility, Mapping)
            else None
        )
        identity = row.get("model_identity")
        metadata_complete = (
            isinstance(identity, Mapping) and identity.get("status") == "COMPLETE"
        )
        if (
            row.get("eligible_for_model_comparison")
            and row.get("profile") == "standard"
            and group_id == current_group_id
            and metadata_complete
        ):
            current_rows.append(row)
        else:
            diagnostics.append(
                _public_diagnostic_row(
                    row,
                    current_group_id,
                    str(key["qualification_generation"]),
                )
            )

    qualified_source = sorted(
        (row for row in current_rows if row.get("outcome") == "QUALIFIED"),
        key=_rank_sort_key,
    )
    not_qualified_source = sorted(
        (row for row in current_rows if row.get("outcome") == "NOT_QUALIFIED"),
        key=_rank_sort_key,
    )
    qualified = [
        _public_scored_row(row, rank=index)
        for index, row in enumerate(qualified_source, start=1)
    ]
    not_qualified = [
        _public_scored_row(row, rank=None) for row in not_qualified_source
    ]
    diagnostics.sort(
        key=lambda row: (
            row.get("benchmark_date") or "",
            row.get("run_id") or "",
        )
    )
    source_times = [
        value
        for row in rows
        for value in (
            _text(row.get("completed_at")),
            _text(row.get("run_timestamp")),
        )
        if value
    ]
    return {
        "schema_version": 1,
        "project": {"name": PROJECT_NAME, "url": PROJECT_URL},
        "generated_at": max(source_times) if source_times else None,
        "generation_basis": "latest source-run timestamp (deterministic for unchanged inputs)",
        "source_run_selection": "all discovered runs in timestamp/run-ID order; ranking selects only the exact current primary configured-deployment compatibility group",
        "ranking_rule": RANKING_RULE,
        "result_scope": "Each row describes the complete configured deployment, including its exact resolved reasoning policy, digest, quantization, runtime/template, context, hardware, and harness configuration.",
        "current_standard": {
            "profile": "standard",
            "qualification_generation": key["qualification_generation"],
            "scoring_version": key["scoring_version"],
            "benchmark_track": key["benchmark_track"],
            "reasoning_policy_rule": key["reasoning_policy_rule"],
            "direct_probe_contract": key["direct_probe_contract"],
            "direct_probe_max_tokens": key["direct_probe_max_tokens"],
            "compatibility_group": current_group_id,
            "upstream_pin_set": pin_set,
            "upstream_pins": key["upstream_pins"],
        },
        "source_run_ids": [_safe_run_id(row["run_id"]) for row in rows],
        "qualified": qualified,
        "evaluated_not_qualified": not_qualified,
        "diagnostics": diagnostics,
    }


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Local benchmark comparison",
        "",
        "Rows are chronological and are not a global ranking. Compare numeric scores only within the same compatibility group.",
        "",
        "| Timestamp | Run ID | Canonical model identity | Runtime alias | Digest | Profile | Version | Track | Policy selection | Effective policy | Decision | Validity | Technical outcome | Overall | Coding | Hermes | Tool/instruction | Reliability | Performance | Duration (s) | Infra error | Compatibility |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["runs"]:
        components = row["component_scores"]
        group = row["compatibility"]["group_id"]
        compatibility = (
            group
            if row["eligible_for_model_comparison"]
            else f"excluded: {row['exclusion_reason']}"
        )
        values = (
            row["run_timestamp"],
            row["run_id"],
            row["model_identity"]["display_name"],
            row["runtime_alias"],
            row["model_digest"],
            row["profile"],
            row["scoring_version"],
            row["benchmark_track"],
            row["reasoning_policy_selection_mode"],
            row["reasoning_policy"],
            row["profile_decision"],
            row["result_validity"],
            row["outcome"],
            row["overall_score"],
            components["coding"],
            components["hermes"],
            components["tool_instruction"],
            components["reliability"],
            components["performance"],
            row["duration_seconds"],
            row["infrastructure_error"],
            compatibility,
        )
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    if not summary["runs"]:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | no runs found |")
    lines.extend(["", "## Compatibility groups", ""])
    if summary["compatibility_groups"]:
        for group in summary["compatibility_groups"]:
            key = group["key"]
            pins = ", ".join(
                f"{name}@{commit[:12]}"
                for name, commit in key["upstream_pins"].items()
            )
            lines.append(
                f"- `{group['group_id']}`: profile `{key['profile']}`, "
                f"track `{key['benchmark_track']}`, policy rule "
                f"`{key['reasoning_policy_rule']}`, "
                f"direct probes `{key['direct_probe_contract']}` at "
                f"`{key['direct_probe_max_tokens']}` max tokens, "
                f"scoring `{key['scoring_version']}`, upstreams {pins}; "
                f"runs: {', '.join(f'`{run_id}`' for run_id in group['run_ids'])}"
            )
    else:
        lines.append("- No comparable completed runs were found.")
    lines.extend(
        [
            "",
            "`INFRA_ERROR`, incomplete, and invalid rows are retained for auditability but excluded from model-quality comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def render_terminal(summary: Mapping[str, Any]) -> str:
    lines = [
        "Local benchmark runs (chronological; not a global ranking)",
        "timestamp             decision                 validity       technical       overall  profile    canonical model identity | runtime alias | compatibility",
    ]
    for row in summary["runs"]:
        model = _display(row["model_identity"]["display_name"])
        alias = _display(row["runtime_alias"])
        group = row["compatibility"]["group_id"]
        compatibility = (
            group if row["eligible_for_model_comparison"] else "EXCLUDED"
        )
        lines.append(
            f"{_display(row['run_timestamp']):20.20}  "
            f"{_display(row['profile_decision']):24.24} "
            f"{_display(row['result_validity']):14.14} "
            f"{_display(row['outcome']):15.15} "
            f"{_display(row['overall_score']):>7.7}  "
            f"{_display(row['profile']):9.9}  "
            f"{model} | {alias} | {compatibility}"
        )
    if not summary["runs"]:
        lines.append("(no run directories found)")
    lines.append(
        f"eligible={summary['eligible_run_count']} total={summary['run_count']}"
    )
    lines.append("Compare scores only within the same compatibility group.")
    return "\n".join(lines) + "\n"


def _public_table(rows: Sequence[Mapping[str, Any]], *, ranked: bool) -> list[str]:
    lines = [
        "| Rank | Canonical model identity | Version | Parameters / architecture | Quantization | Context | Runtime alias | Digest | Date | Profile / version | Track | Policy selection | Effective policy | Decision | Technical outcome | Overall | Coding | Hermes | Tool / instruction | Reliability | Performance | Duration (s) | Hardware / runtime | Pin set | Run ID | Source |",
        "|---:|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        identity = row["model_identity"]
        components = row["component_scores"]
        source_url = identity.get("source_url")
        source_model = identity.get("source_model")
        source = (
            f"[{source_model}]({source_url})"
            if source_url
            else source_model
        )
        values = (
            row["rank"] if ranked else None,
            identity["display_name"],
            identity["version"],
            f"{identity['parameter_variant']} / {identity['architecture']}",
            identity["quantization"],
            identity["context_length"],
            row["runtime_alias"],
            identity["immutable_digest"],
            row["benchmark_date"],
            f"{row['profile']} / {row['profile_version']}",
            row["benchmark_track"],
            row["reasoning_policy_selection_mode"],
            row["reasoning_policy"],
            row["profile_decision"],
            row["outcome"],
            row["overall_score"],
            components["coding"],
            components["hermes"],
            components["tool_instruction"],
            components["reliability"],
            components["performance"],
            row["duration_seconds"],
            row["hardware_runtime"],
            row["upstream_pin_set"],
            row["run_id"],
            source,
        )
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    if not rows:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    return lines


def render_public_markdown(leaderboard: Mapping[str, Any]) -> str:
    current = leaderboard["current_standard"]
    lines = [
        f"# {PROJECT_NAME} public leaderboard",
        "",
        f"Project: [{PROJECT_URL}]({PROJECT_URL})",
        "",
        f"Generated at `{_display(leaderboard['generated_at'])}` using the deterministic latest-source-run timestamp.",
        "",
        "Results identify the complete deployment—not abstract weights. Every scored row is specific to its immutable digest, checkpoint/version, quantization, context, runtime/template, hardware, profile, harness, and upstream pins. Runtime aliases are shown only as secondary operator labels and never determine public identity.",
        "",
        f"Current official group: profile `{current['profile']}`, track `{current['benchmark_track']}`, reasoning-policy rule `{current['reasoning_policy_rule']}`, direct probes `{current['direct_probe_contract']}` at `{current['direct_probe_max_tokens']}` max completion tokens, qualification/scoring `{current['qualification_generation']}`, compatibility `{current['compatibility_group']}`, pin set `{current['upstream_pin_set']}`.",
        "",
        f"Ranking order: {leaderboard['ranking_rule']}.",
        "",
        "Repeated trials remain separate rows; scores are not averaged. Smoke and incompatible runs are never compared with the current standard profile.",
        "",
        "## Meets profile",
        "",
        "These valid completed deployments meet this exact deployment profile.",
        "",
    ]
    lines.extend(_public_table(leaderboard["qualified"], ranked=True))
    lines.extend(
        [
            "",
            "## Completed with profile limitations",
            "",
            "These valid completed evaluations do not meet every gate of this exact coding/agentic deployment profile; this is not a universal model-quality judgment.",
            "",
        ]
    )
    lines.extend(
        _public_table(leaderboard["evaluated_not_qualified"], ranked=False)
    )
    lines.extend(
        [
            "",
            "## No valid result / quarantined diagnostics",
            "",
            "Quarantined, infrastructure-error, incomplete, invalid, smoke-only, metadata-incomplete, legacy thinking-control, and incompatible results are diagnostic evidence, not ranked model outcomes.",
            "",
            "| Status | Decision | Validity | Technical outcome | Canonical model identity | Runtime alias | Digest | Date | Profile / version | Track | Policy selection | Effective policy | Reason | Run ID |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in leaderboard["diagnostics"]:
        values = (
            row["status"],
            row["profile_decision"],
            row["result_validity"],
            row["qualification_outcome"],
            row["model_identity"],
            row["runtime_alias"],
            row["immutable_digest"],
            row["benchmark_date"],
            f"{_display(row['profile'])} / {_display(row['profile_version'])}",
            row["benchmark_track"],
            row["reasoning_policy_selection_mode"],
            row["reasoning_policy"],
            row["reason"],
            row["run_id"],
        )
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    if not leaderboard["diagnostics"]:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "Public files contain only allowlisted summary fields. Raw prompts, trajectories, responses, paths, endpoints, hostnames, and logs remain local.",
            "",
        ]
    )
    return "\n".join(lines)


def render_public_terminal(
    leaderboard: Mapping[str, Any], *, highlight_run_id: str | None = None
) -> str:
    lines = [
        "Updated current-standard deployment ranking",
        "mark rank profile decision          technical       overall  canonical model identity",
    ]
    displayed = list(leaderboard["qualified"]) + list(
        leaderboard["evaluated_not_qualified"]
    )
    for row in displayed:
        mark = "*" if row["run_id"] == highlight_run_id else " "
        rank = str(row["rank"]) if row["rank"] is not None else "—"
        model = row["model_identity"]["display_name"]
        decision = row.get("profile_decision") or (
            "MEETS_PROFILE"
            if row.get("outcome") == "QUALIFIED"
            else "DOES_NOT_MEET_PROFILE"
        )
        lines.append(
            f" {mark}   {rank:>3.3} {_display(decision):24.24} "
            f"{_display(row['outcome']):15.15} "
            f"{_display(row['overall_score']):>7.7}  {model}"
        )
    if not displayed:
        lines.append("(no rankable current-standard results)")
    if highlight_run_id and not any(
        row["run_id"] == highlight_run_id for row in displayed
    ):
        diagnostic = next(
            (
                row
                for row in leaderboard["diagnostics"]
                if row["run_id"] == highlight_run_id
            ),
            None,
        )
        if diagnostic:
            lines.append(
                "* just-completed run is not assessed: "
                f"{highlight_run_id} ({diagnostic['status']}: {diagnostic['reason']})"
            )
        else:
            lines.append(f"* just-completed run is unranked: {highlight_run_id}")
    lines.append(
        "qualified="
        f"{len(leaderboard['qualified'])} "
        f"not_qualified={len(leaderboard['evaluated_not_qualified'])} "
        f"diagnostic={len(leaderboard['diagnostics'])}"
    )
    return "\n".join(lines) + "\n"


def compare_runs(
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    public_output_root: Path = DEFAULT_PUBLIC_OUTPUT_ROOT,
    registry_path: Path = REGISTRY_PATH,
    current_compatibility: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], Path, Path]:
    rows = collect_runs(runs_root, registry_path=registry_path)
    summary = build_summary(rows)
    metadata_candidates = build_metadata_candidates(rows)
    summary["metadata_candidate_count"] = len(
        metadata_candidates["candidates"]
    )
    leaderboard = build_public_leaderboard(
        rows, current_compatibility=current_compatibility
    )
    try:
        public_schema = json.loads(PUBLIC_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(public_schema)
        jsonschema.Draft202012Validator(public_schema).validate(leaderboard)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot load public leaderboard schema: {exc}") from exc
    except (jsonschema.SchemaError, jsonschema.ValidationError) as exc:
        raise ComparisonError(
            "public leaderboard failed schema validation: " + exc.message
        ) from exc
    output_root = ensure_root(output_root.expanduser().resolve())
    json_path = output_root / "comparison.json"
    markdown_path = output_root / "comparison.md"
    metadata_candidates_path = output_root / METADATA_CANDIDATES_FILENAME
    atomic_write_text(
        json_path,
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        root=output_root,
    )
    atomic_write_text(markdown_path, render_markdown(summary), root=output_root)
    atomic_write_text(
        metadata_candidates_path,
        json.dumps(
            metadata_candidates,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        root=output_root,
    )
    public_output_root = ensure_root(public_output_root.expanduser().resolve())
    public_json_path = public_output_root / "leaderboard.json"
    public_markdown_path = public_output_root / "leaderboard.md"
    atomic_write_text(
        public_json_path,
        json.dumps(leaderboard, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        root=public_output_root,
    )
    atomic_write_text(
        public_markdown_path,
        render_public_markdown(leaderboard),
        root=public_output_root,
    )
    return (
        summary,
        markdown_path,
        json_path,
        leaderboard,
        public_markdown_path,
        public_json_path,
    )
