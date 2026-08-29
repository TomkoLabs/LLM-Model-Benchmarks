from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import jsonschema
import yaml


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "model-metadata.yaml"
REGISTRY_SCHEMA_PATH = ROOT / "schemas" / "model-metadata.schema.json"
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NUM_CTX_PATTERN = re.compile(
    r"^\s*(?:PARAMETER\s+)?num_ctx\s+([0-9]+)\s*(?:#.*)?$",
    re.IGNORECASE,
)
PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s'\"])(?:/home/|/Users/|/root/|[A-Za-z]:[\\/]|file://)",
    re.IGNORECASE,
)
PRIVATE_ADDRESS_PATTERN = re.compile(
    r"(?:^|[^0-9])(?:127(?:\.[0-9]{1,3}){3}|10(?:\.[0-9]{1,3}){3}|"
    r"192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])"
    r"(?:\.[0-9]{1,3}){2})(?:$|[^0-9])"
)


class ModelMetadataError(RuntimeError):
    pass


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _public_text(value: Any) -> str | None:
    text = _text(value)
    if (
        text is None
        or len(text) > 300
        or "\n" in text
        or "\r" in text
        or PRIVATE_PATH_PATTERN.search(text)
        or PRIVATE_ADDRESS_PATTERN.search(text)
        or "://" in text
    ):
        return None
    return text.strip()


def safe_basename(value: Any) -> str | None:
    """Return only a safe leaf name from a runtime-owned source reference."""

    text = _text(value)
    if text is None:
        return None
    leaf = text.strip().strip("'\"").replace("\\", "/").rsplit("/", 1)[-1]
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
        return None
    return leaf


def _modelfile_source_basename(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("FROM "):
            continue
        try:
            fields = shlex.split(line)
        except ValueError:
            return None
        if len(fields) == 2 and fields[0].upper() == "FROM":
            return safe_basename(fields[1])
    return None


def _num_ctx_values(value: Any) -> set[int]:
    if not isinstance(value, str):
        return set()
    values: set[int] = set()
    for line in value.splitlines():
        match = NUM_CTX_PATTERN.fullmatch(line)
        if match:
            context = int(match.group(1))
            if context > 0:
                values.add(context)
    return values


def extract_runtime_metadata(
    tag_record: Mapping[str, Any],
    show_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Allowlist safe identity metadata from Ollama's read-only responses."""

    details = show_payload.get("details")
    if not isinstance(details, Mapping):
        details = tag_record.get("details")
    details = details if isinstance(details, Mapping) else {}
    info = show_payload.get("model_info")
    info = info if isinstance(info, Mapping) else {}
    architecture = _public_text(info.get("general.architecture")) or _public_text(
        details.get("family")
    )
    native_context = details.get("context_length")
    if type(native_context) is not int and architecture:
        native_context = info.get(f"{architecture}.context_length")
    if type(native_context) is not int or native_context < 1:
        native_context = None

    parameter_contexts = _num_ctx_values(show_payload.get("parameters"))
    modelfile_contexts = _num_ctx_values(show_payload.get("modelfile"))
    context_values = parameter_contexts | modelfile_contexts
    discrepancies: list[str] = []
    if len(context_values) > 1:
        discrepancies.append(
            "conflicting num_ctx values in /api/show parameters and Modelfile"
        )
    effective_context = (
        next(iter(context_values)) if len(context_values) == 1 else None
    )
    effective_sources: list[str] = []
    if effective_context in parameter_contexts:
        effective_sources.append("/api/show parameters num_ctx")
    if effective_context in modelfile_contexts:
        effective_sources.append("/api/show Modelfile PARAMETER num_ctx")
    template = show_payload.get("template")
    template_sha256 = (
        "sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()
        if isinstance(template, str) and template
        else None
    )
    capabilities = show_payload.get("capabilities")
    capabilities = (
        sorted(set(capabilities))
        if isinstance(capabilities, list)
        and capabilities
        and all(isinstance(value, str) and value for value in capabilities)
        else None
    )

    metadata = {
        "canonical_name": _public_text(info.get("general.basename")),
        "version": _public_text(info.get("general.version")),
        "architecture": architecture,
        "parameter_variant": _public_text(details.get("parameter_size"))
        or _public_text(info.get("general.size_label")),
        "parameter_count": (
            info.get("general.parameter_count")
            if type(info.get("general.parameter_count")) is int
            else None
        ),
        "quantization": _public_text(details.get("quantization_level")),
        "source_model": _public_text(details.get("parent_model")),
        "format": _public_text(details.get("format")),
        # context_length is retained as a backwards-compatible name for the
        # checkpoint's native maximum. It is not an execution default.
        "context_length": native_context,
        "native_context_length": native_context,
        "effective_context_length": effective_context,
        "effective_context_sources": effective_sources,
        "source_filename": _modelfile_source_basename(
            show_payload.get("modelfile")
        ),
        "template_sha256": template_sha256,
        "capabilities": capabilities,
        "metadata_sources": {
            "canonical_name": "/api/show model_info.general.basename",
            "version": "/api/show model_info.general.version",
            "architecture": "/api/show model_info.general.architecture or details.family",
            "parameter_variant": "/api/show details.parameter_size or model_info.general.size_label",
            "parameter_count": "/api/show model_info.general.parameter_count",
            "quantization": "/api/show details.quantization_level",
            "source_model": "/api/show details.parent_model",
            "native_context_length": "/api/tags details.context_length or /api/show model_info architecture context_length",
            "effective_context_length": " and ".join(effective_sources),
            "source_filename": "/api/show Modelfile FROM basename",
            "template_sha256": "/api/show template",
            "capabilities": "/api/show capabilities",
        },
        "metadata_discrepancies": discrepancies,
    }
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != [] and value != {}
    }


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(REGISTRY_SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    except (
        OSError,
        UnicodeDecodeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as exc:
        raise ModelMetadataError(
            f"cannot read public model metadata registry: {exc}"
        ) from exc
    except (jsonschema.SchemaError, jsonschema.ValidationError) as exc:
        raise ModelMetadataError(
            "public model metadata registry failed schema validation: "
            + exc.message
        ) from exc

    assert isinstance(value, dict)
    models = value["models"]
    for digest, entry in models.items():
        if digest != digest.lower():
            raise ModelMetadataError("model metadata digest keys must be lowercase")
        for field, field_value in entry.items():
            if not isinstance(field_value, str):
                continue
            if PRIVATE_PATH_PATTERN.search(field_value):
                raise ModelMetadataError(
                    f"public model metadata contains a private path in {digest}:{field}"
                )
            if (
                PRIVATE_ADDRESS_PATTERN.search(field_value)
                or "localhost" in field_value.casefold()
            ):
                raise ModelMetadataError(
                    f"public model metadata contains a private address in {digest}:{field}"
                )
            if field == "source_url":
                hostname = urlsplit(field_value).hostname
                if not hostname or hostname.endswith((".local", ".home.arpa")):
                    raise ModelMetadataError(
                        f"public model metadata has a non-public source URL in {digest}"
                    )
                try:
                    address = ipaddress.ip_address(hostname)
                except ValueError:
                    pass
                else:
                    if not address.is_global:
                        raise ModelMetadataError(
                            f"public model metadata has a non-public source URL in {digest}"
                        )
    return value


def _model_context(
    model: Mapping[str, Any], provenance: Mapping[str, Any]
) -> int | None:
    context = model.get("context_length")
    if type(context) is int and context > 0:
        return context
    configured = provenance.get("model_config")
    if isinstance(configured, Mapping):
        context = configured.get("context_length")
        if type(context) is int and context > 0:
            return context
    return None


def resolve_public_identity(
    model: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    *,
    registry: Mapping[str, Any] | None = None,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve public identity by exact digest without mutating source data."""

    provenance = provenance if isinstance(provenance, Mapping) else {}
    registry = registry if registry is not None else load_registry()
    runtime_metadata = (
        runtime_metadata if isinstance(runtime_metadata, Mapping) else {}
    )
    raw_digest = _text(model.get("runtime_digest"))
    digest = raw_digest.lower() if raw_digest else None
    alias = (
        _text(model.get("runtime_model"))
        or _text(model.get("requested"))
        or _text(model.get("config_alias"))
    )
    issues: list[str] = []
    if digest is None or DIGEST_PATTERN.fullmatch(digest) is None:
        issues.append("full immutable digest is unavailable")
        entry: Mapping[str, Any] | None = None
    else:
        entries = registry.get("models") if isinstance(registry, Mapping) else None
        entry_value = entries.get(digest) if isinstance(entries, Mapping) else None
        entry = entry_value if isinstance(entry_value, Mapping) else None

    context = _model_context(model, provenance)
    if context is None:
        issues.append("configured context length is unavailable")

    run_quantization = _text(model.get("quantization"))
    canonical_name = (
        _text(entry.get("canonical_name")) if entry else None
    ) or _text(runtime_metadata.get("canonical_name"))
    source_model = (
        _text(entry.get("source_model")) if entry else None
    ) or _text(runtime_metadata.get("source_model"))
    source_url = _text(entry.get("source_url")) if entry else None
    version = (
        _text(entry.get("version")) if entry else None
    ) or _text(runtime_metadata.get("version"))
    parameter_variant = (
        _text(entry.get("parameter_variant")) if entry else None
    ) or _text(runtime_metadata.get("parameter_variant"))
    architecture = (
        _text(entry.get("architecture")) if entry else None
    ) or _text(runtime_metadata.get("architecture"))
    registered_quantization = _text(entry.get("quantization")) if entry else None
    runtime_quantization = _text(runtime_metadata.get("quantization"))
    quantization = (
        runtime_quantization or run_quantization or registered_quantization
    )
    source_filename = (
        safe_basename(entry.get("source_filename")) if entry else None
    ) or safe_basename(runtime_metadata.get("source_filename"))

    required_values = (
        ("canonical checkpoint name", canonical_name),
        ("source model", source_model),
        ("release/version", version),
        ("parameter variant", parameter_variant),
        ("architecture", architecture),
        ("quantization", quantization),
    )
    for label, value in required_values:
        if value is None:
            issues.append(f"{label} is unavailable")

    if entry:
        observed_pairs = (
            ("architecture", architecture, runtime_metadata.get("architecture")),
            (
                "parameter variant",
                parameter_variant,
                runtime_metadata.get("parameter_variant"),
            ),
            (
                "quantization",
                registered_quantization,
                runtime_quantization or run_quantization,
            ),
        )
        for label, registered, observed in observed_pairs:
            if (
                isinstance(registered, str)
                and isinstance(observed, str)
                and registered.casefold() != observed.casefold()
            ):
                issues.append(f"registered {label} does not match run metadata")
    issues = list(dict.fromkeys(issues))

    complete = not issues
    if complete:
        assert digest and canonical_name and version and parameter_variant
        assert architecture and quantization and source_model
        display_name = (
            f"{canonical_name} {version} [{source_model}] — "
            f"{parameter_variant} ({architecture}) — "
            f"{quantization} — {context}-token context — {digest}"
        )
    else:
        display_name = f"METADATA_INCOMPLETE — {digest or 'digest unavailable'}"

    aliases = entry.get("aliases", []) if entry else []
    reasoning = _text(model.get("reasoning_effort"))
    return {
        "status": "COMPLETE" if complete else "METADATA_INCOMPLETE",
        "display_name": display_name,
        "canonical_name": canonical_name,
        "source_model": source_model,
        "source_url": source_url,
        "version": version,
        "parameter_variant": parameter_variant,
        "architecture": architecture,
        "quantization": quantization,
        "context_length": context,
        "immutable_digest": digest,
        "runtime_alias": alias,
        "runtime_variant": f"reasoning={reasoning}" if reasoning else None,
        "source_filename": source_filename,
        "registry_match": entry is not None,
        "alias_registered": bool(alias and alias in aliases),
        "metadata_source": (
            "model-metadata.yaml exact digest registry"
            if entry is not None
            else "Ollama runtime metadata"
        ),
        "issues": issues,
    }
