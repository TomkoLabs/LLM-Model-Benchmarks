from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


REASONING_POLICY_CONTRACT = "ollama-reasoning-policy-v1"
OPENAI_REASONING_POLICY_CONTRACT = "openai-reasoning-policy-v1"
PRIMARY_DEPLOYMENT_TRACK = "primary-deployment"
CONTROLLED_POLICY_TRACK = "controlled-policy"
CONFIGURED_POLICY_RULE = "configured-per-model"
SUPPORTED_EFFORT_LEVELS = ("low", "medium", "high")
SUPPORTED_REASONING_POLICIES = (
    "off",
    "native",
    *(f"effort:{level}" for level in SUPPORTED_EFFORT_LEVELS),
)


class ReasoningPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ReasoningPolicy:
    value: str
    mode: str
    effort: str | None = None

    @property
    def cohort(self) -> str:
        if self.mode == "off":
            return "controlled-off"
        if self.mode == "native":
            return "native-deployment"
        return f"effort-{self.effort}-deployment"

    @property
    def hermes_effort(self) -> str | None:
        if self.mode == "off":
            return "none"
        return self.effort


def parse_reasoning_policy(value: str) -> ReasoningPolicy:
    normalized = str(value or "").strip().lower()
    if normalized == "off":
        return ReasoningPolicy(value="off", mode="off")
    if normalized == "native":
        return ReasoningPolicy(value="native", mode="native")
    prefix = "effort:"
    if normalized.startswith(prefix):
        effort = normalized[len(prefix) :]
        if effort in SUPPORTED_EFFORT_LEVELS:
            return ReasoningPolicy(
                value=f"effort:{effort}", mode="effort", effort=effort
            )
    supported = ", ".join(SUPPORTED_REASONING_POLICIES)
    raise ReasoningPolicyError(
        f"unsupported reasoning policy {value!r}; supported values: {supported}"
    )


def resolve_reasoning_policy(
    requested: str,
    model: Mapping[str, Any],
) -> tuple[ReasoningPolicy, str]:
    normalized = str(requested or "").strip().lower()
    source = "explicit CLI"
    if normalized == "configured":
        configured = model.get("reasoning_policy")
        if not isinstance(configured, str) or not configured.strip():
            raise ReasoningPolicyError(
                "--reasoning-policy configured requires an explicit "
                "reasoning_policy in the selected models.yaml entry"
            )
        normalized = configured
        source = "models.yaml curated policy"
    policy = parse_reasoning_policy(normalized)
    supported = model.get("supported_reasoning_policies")
    if supported is not None:
        if not (
            isinstance(supported, list)
            and supported
            and all(isinstance(item, str) for item in supported)
        ):
            raise ReasoningPolicyError(
                "supported_reasoning_policies must be a non-empty string list"
            )
        parsed_supported = [parse_reasoning_policy(item).value for item in supported]
        if policy.value not in parsed_supported:
            raise ReasoningPolicyError(
                f"reasoning policy {policy.value!r} is not supported by this "
                f"model configuration; supported values: {', '.join(parsed_supported)}"
            )
    elif policy.mode == "effort":
        raise ReasoningPolicyError(
            f"reasoning policy {policy.value!r} requires a curated model entry "
            "that explicitly declares supported_reasoning_policies"
        )
    return policy, source


def reasoning_policy_selection(requested: str) -> tuple[str, str]:
    normalized = str(requested or "").strip().lower()
    if normalized == "configured":
        return "configured", PRIMARY_DEPLOYMENT_TRACK
    if normalized in SUPPORTED_REASONING_POLICIES:
        return "explicit", CONTROLLED_POLICY_TRACK
    raise ReasoningPolicyError(
        f"unsupported reasoning policy selection {requested!r}"
    )


def endpoint_api_mode(path: str) -> str | None:
    normalized = str(path).split("?", 1)[0].rstrip("/")
    if normalized.endswith("/v1/chat/completions") or normalized.endswith(
        "/chat/completions"
    ):
        return "openai-chat-completions"
    if normalized.endswith("/api/chat"):
        return "ollama-native-chat"
    return None


def ollama_reasoning_control(
    api_mode: str, policy: ReasoningPolicy
) -> dict[str, Any]:
    if api_mode == "openai-chat-completions":
        if policy.mode == "off":
            return {"reasoning_effort": "none"}
        if policy.mode == "effort":
            return {"reasoning_effort": policy.effort}
        if policy.mode == "native":
            return {}
    elif api_mode == "ollama-native-chat":
        if policy.mode == "off":
            return {"think": False}
        if policy.mode == "effort":
            return {"think": policy.effort}
        if policy.mode == "native":
            return {}
    raise ReasoningPolicyError(f"unsupported Ollama API mode: {api_mode}")


def reasoning_policy_contract(runtime: str) -> str:
    if runtime == "ollama":
        return REASONING_POLICY_CONTRACT
    if runtime == "ds4":
        return OPENAI_REASONING_POLICY_CONTRACT
    raise ReasoningPolicyError(f"unsupported model runtime: {runtime}")


def runtime_reasoning_control(
    runtime: str,
    api_mode: str,
    policy: ReasoningPolicy,
) -> dict[str, Any]:
    """Return the exact supported control for one runtime/API boundary."""

    if runtime == "ollama":
        return ollama_reasoning_control(api_mode, policy)
    if runtime == "ds4":
        if api_mode != "openai-chat-completions":
            raise ReasoningPolicyError(
                "DS4 supports only the OpenAI chat-completions API mode"
            )
        if policy.mode == "off":
            return {"reasoning_effort": "none"}
        if policy.mode == "effort":
            return {"reasoning_effort": policy.effort}
        if policy.mode == "native":
            return {}
    raise ReasoningPolicyError(f"unsupported model runtime: {runtime}")


def _remove_nested_controls(
    payload: dict[str, Any], key: str, fields: tuple[str, ...], removed: list[str]
) -> None:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        return
    normalized = dict(value)
    for field in fields:
        if field in normalized:
            removed.append(f"{key}.{field}")
            normalized.pop(field, None)
    if normalized:
        payload[key] = normalized
    else:
        payload.pop(key, None)


def normalize_model_request(
    runtime: str,
    api_mode: str,
    payload: Mapping[str, Any],
    policy: ReasoningPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ReasoningPolicyError("target inference request body must be a JSON object")
    normalized = dict(payload)
    removed: list[str] = []
    for field in ("enable_thinking", "reasoning", "reasoning_effort", "think"):
        if field in normalized:
            removed.append(field)
            normalized.pop(field, None)
    _remove_nested_controls(
        normalized,
        "chat_template_kwargs",
        ("enable_thinking", "reasoning", "reasoning_effort", "think"),
        removed,
    )
    _remove_nested_controls(
        normalized,
        "options",
        ("enable_thinking", "reasoning", "reasoning_effort", "think"),
        removed,
    )
    control = runtime_reasoning_control(runtime, api_mode, policy)
    normalized.update(control)
    field = next(iter(control), None)
    return normalized, {
        "contract": reasoning_policy_contract(runtime),
        "runtime": runtime,
        "requested_reasoning_policy": policy.value,
        "reasoning_cohort": policy.cohort,
        "api_mode": api_mode,
        "serialized_control_field": field,
        "serialized_control_value": control.get(field) if field else None,
        "conflicting_fields_removed": sorted(removed),
        "legacy_enable_thinking_present": "enable_thinking" in normalized,
    }


def normalize_ollama_request(
    api_mode: str,
    payload: Mapping[str, Any],
    policy: ReasoningPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Backward-compatible Ollama request normalization entry point."""

    return normalize_model_request("ollama", api_mode, payload, policy)


def visible_reasoning_tag_present(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    normalized = content.casefold()
    return "<think>" in normalized or "</think>" in normalized


def response_field_presence(response: Mapping[str, Any]) -> dict[str, Any]:
    message: Mapping[str, Any] = {}
    finish_reason: Any = None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get("finish_reason")
        candidate = choices[0].get("message")
        if isinstance(candidate, Mapping):
            message = candidate
    elif isinstance(response.get("message"), Mapping):
        message = response["message"]
        finish_reason = response.get("done_reason")

    content = message.get("content")
    reasoning = next(
        (
            message.get(name)
            for name in ("reasoning", "reasoning_content", "thinking")
            if isinstance(message.get(name), str) and message.get(name)
        ),
        "",
    )
    calls = message.get("tool_calls")
    return {
        "reasoning_content_returned": bool(reasoning),
        "visible_content_returned": isinstance(content, str) and bool(content.strip()),
        "tool_call_returned": isinstance(calls, list) and bool(calls),
        "visible_reasoning_tag_returned": visible_reasoning_tag_present(content),
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
    }


def classify_direct_response(
    response: Mapping[str, Any],
    *,
    expected_content: str | None = None,
    expected_tool_name: str | None = None,
    expected_tool_arguments: Mapping[str, Any] | None = None,
    reasoning_allowed: bool = False,
) -> dict[str, Any]:
    presence = response_field_presence(response)
    choices = response.get("choices")
    message: Mapping[str, Any] = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        candidate = choices[0].get("message")
        if isinstance(candidate, Mapping):
            message = candidate
    content = message.get("content")
    content = content.strip() if isinstance(content, str) else ""
    reasoning = presence["reasoning_content_returned"]
    visible_reasoning_tag = presence["visible_reasoning_tag_returned"]

    if expected_content is not None:
        if visible_reasoning_tag:
            classification = "VISIBLE_REASONING_TAG_CONTAMINATION"
        elif reasoning and presence["finish_reason"] == "length" and not content:
            classification = "TRUNCATED_BEFORE_ANSWER"
        elif reasoning and not content:
            classification = "REASONING_ONLY"
        elif reasoning and not reasoning_allowed:
            classification = "REASONING_CONTAMINATION"
        elif content == expected_content:
            classification = "EXACT_VISIBLE_RESPONSE"
        elif not content:
            classification = "EMPTY_OUTPUT"
        else:
            classification = "INVALID_VISIBLE_RESPONSE"
        valid = classification == "EXACT_VISIBLE_RESPONSE"
        return {
            "status": "PASS" if valid else "FAIL",
            "classification": classification,
            **presence,
        }

    calls = message.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    arguments: Mapping[str, Any] | None = None
    name: Any = None
    structurally_valid = False
    if len(calls) == 1 and isinstance(calls[0], Mapping):
        function = calls[0].get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, Mapping):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    decoded = json.loads(raw_arguments)
                    arguments = decoded if isinstance(decoded, Mapping) else None
                except json.JSONDecodeError:
                    arguments = None
            structurally_valid = (
                name == expected_tool_name
                and arguments == expected_tool_arguments
            )
    if visible_reasoning_tag:
        classification = "VISIBLE_REASONING_TAG_CONTAMINATION"
    elif reasoning and presence["finish_reason"] == "length" and not calls:
        classification = "TRUNCATED_BEFORE_TOOL_CALL"
    elif reasoning and not calls:
        classification = "REASONING_ONLY"
    elif reasoning and not reasoning_allowed:
        classification = "REASONING_CONTAMINATION"
    elif structurally_valid:
        classification = "EXACT_TOOL_CALL"
    elif not calls and not content:
        classification = "EMPTY_OUTPUT"
    else:
        classification = "MALFORMED_TOOL_CALL"
    valid = classification == "EXACT_TOOL_CALL"
    return {
        "status": "PASS" if valid else "FAIL",
        "classification": classification,
        "name": name,
        "arguments": arguments,
        **presence,
    }
