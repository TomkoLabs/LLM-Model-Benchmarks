from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness.model_gateway import ModelGateway, read_model_transport_observations
from harness.reasoning_policy import (
    QWEN_ENABLE_THINKING_PROFILE,
    ReasoningPolicyError,
    classify_direct_response,
    normalize_model_request,
    normalize_ollama_request,
    ollama_reasoning_control,
    parse_reasoning_policy,
    resolve_reasoning_policy,
)
from harness.upstreams import HERMES_PYTHON


def _response(
    *,
    content: str | None = "",
    reasoning: str | None = None,
    tool_calls: list[dict[str, object]] | None = None,
    finish: str = "stop",
) -> dict[str, object]:
    message: dict[str, object] = {"content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": finish}]}


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.payloads.append((self.path, payload))  # type: ignore[attr-defined]
        if payload.get("stream") is False:
            body = json.dumps(
                {
                    "model": "fixture",
                    "message": {
                        "role": "assistant",
                        "content": "normal",
                        **(
                            {"thinking": "unexpected"}
                            if payload.get("fixture_reasoning")
                            else {}
                        ),
                    },
                    "done": True,
                    "done_reason": "stop",
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return

        chunks = [
            b'data: {"id":"x","object":"chat.completion.chunk","created":0,',
            b'"model":"fixture","choices":[{"index":0,"delta":{"role":"assistant",',
            b'"content":"visible"},"finish_reason":null}]}\n',
            b'\ndata: {"id":"x","object":"chat.completion.chunk","created":0,"model":',
            b'"fixture","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,',
            b'"id":"call_1","type":"function","function":{"name":"probe",',
            b'"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DO',
            b'NE]\n\n',
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()
        self.close_connection = True


class _FixtureServer:
    def __enter__(self) -> "_FixtureServer":
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
        self.server.payloads = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def payloads(self) -> list[tuple[str, dict[str, object]]]:
        return self.server.payloads  # type: ignore[attr-defined,no-any-return]

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class ReasoningPolicyTests(unittest.TestCase):
    def test_explicit_modes_and_supported_effort(self) -> None:
        self.assertEqual(parse_reasoning_policy("off").mode, "off")
        self.assertEqual(parse_reasoning_policy("native").mode, "native")
        self.assertEqual(parse_reasoning_policy("effort:medium").effort, "medium")

    def test_configured_policy_and_unsupported_effort_preflight(self) -> None:
        model = {
            "reasoning_policy": "effort:medium",
            "supported_reasoning_policies": ["off", "native", "effort:medium"],
        }
        policy, source = resolve_reasoning_policy("configured", model)
        self.assertEqual((policy.value, source), ("effort:medium", "models.yaml curated policy"))
        with self.assertRaisesRegex(ReasoningPolicyError, "supported values"):
            resolve_reasoning_policy("effort:high", model)

    def test_no_automatic_maximum_or_unconfigured_effort(self) -> None:
        with self.assertRaisesRegex(ReasoningPolicyError, "explicit reasoning_policy"):
            resolve_reasoning_policy("configured", {})
        with self.assertRaisesRegex(ReasoningPolicyError, "curated model entry"):
            resolve_reasoning_policy("effort:high", {})
        with self.assertRaises(ReasoningPolicyError):
            parse_reasoning_policy("maximum")

    def test_openai_native_serialization_and_conflict_removal(self) -> None:
        off = parse_reasoning_policy("off")
        native = parse_reasoning_policy("native")
        effort = parse_reasoning_policy("effort:low")
        high_effort = parse_reasoning_policy("effort:high")
        self.assertEqual(ollama_reasoning_control("openai-chat-completions", off), {"reasoning_effort": "none"})
        self.assertEqual(ollama_reasoning_control("ollama-native-chat", off), {"think": False})
        self.assertEqual(ollama_reasoning_control("openai-chat-completions", native), {})
        self.assertEqual(ollama_reasoning_control("ollama-native-chat", effort), {"think": "low"})
        self.assertEqual(
            ollama_reasoning_control("openai-chat-completions", high_effort),
            {"reasoning_effort": "high"},
        )
        self.assertEqual(
            ollama_reasoning_control("ollama-native-chat", high_effort),
            {"think": "high"},
        )
        payload, metadata = normalize_ollama_request(
            "openai-chat-completions",
            {
                "model": "fixture",
                "think": True,
                "enable_thinking": True,
                "reasoning_effort": "high",
                "options": {"think": True, "temperature": 0},
                "chat_template_kwargs": {"enable_thinking": True},
            },
            off,
        )
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("think", payload)
        self.assertNotIn("enable_thinking", payload)
        self.assertEqual(payload["options"], {"temperature": 0})
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertIn("options.think", metadata["conflicting_fields_removed"])

    def test_vllm_native_policy_honors_exact_request_level_switches(self) -> None:
        native = parse_reasoning_policy("native")
        enabled, enabled_metadata = normalize_model_request(
            "vllm",
            "openai-chat-completions",
            {
                "model": "fixture",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "custom": "preserved",
                },
            },
            native,
            QWEN_ENABLE_THINKING_PROFILE,
        )
        disabled, disabled_metadata = normalize_model_request(
            "vllm",
            "openai-chat-completions",
            {
                "model": "fixture",
                "think": False,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            native,
            QWEN_ENABLE_THINKING_PROFILE,
        )

        self.assertEqual(
            enabled["chat_template_kwargs"],
            {"custom": "preserved"},
        )
        self.assertEqual(
            enabled_metadata["requested_reasoning_policy"], "native"
        )
        self.assertEqual(
            enabled_metadata["request_reasoning_policy_source"],
            "request-control",
        )
        self.assertEqual(
            disabled["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("think", disabled)
        self.assertNotIn("reasoning_effort", disabled)
        self.assertEqual(
            disabled_metadata["requested_reasoning_policy"], "off"
        )
        self.assertEqual(
            disabled_metadata["deployment_reasoning_policy"], "native"
        )

    def test_vllm_whole_run_off_overrides_request_on_and_conflicts_fail(self) -> None:
        disabled, metadata = normalize_model_request(
            "vllm",
            "openai-chat-completions",
            {"model": "fixture", "enable_thinking": True, "think": True},
            parse_reasoning_policy("off"),
            QWEN_ENABLE_THINKING_PROFILE,
        )
        self.assertEqual(
            disabled["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("enable_thinking", disabled)
        self.assertNotIn("think", disabled)
        self.assertEqual(metadata["requested_reasoning_policy"], "off")
        self.assertEqual(
            metadata["request_reasoning_policy_source"], "deployment-policy"
        )

        configured_default, configured_metadata = normalize_model_request(
            "vllm",
            "openai-chat-completions",
            {"model": "fixture", "chat_template_kwargs": {"enable_thinking": True}},
            parse_reasoning_policy("off"),
            QWEN_ENABLE_THINKING_PROFILE,
            honor_request_reasoning_controls=True,
        )
        self.assertNotIn("chat_template_kwargs", configured_default)
        self.assertEqual(
            configured_metadata["requested_reasoning_policy"], "native"
        )
        self.assertEqual(
            configured_metadata["request_reasoning_policy_source"],
            "request-control",
        )
        self.assertTrue(
            configured_metadata["honor_request_reasoning_controls"]
        )

        with self.assertRaisesRegex(ReasoningPolicyError, "conflicting"):
            normalize_model_request(
                "vllm",
                "openai-chat-completions",
                {
                    "enable_thinking": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                parse_reasoning_policy("native"),
                QWEN_ENABLE_THINKING_PROFILE,
            )
        with self.assertRaisesRegex(ReasoningPolicyError, "unsupported effort"):
            normalize_model_request(
                "vllm",
                "openai-chat-completions",
                {"reasoning_effort": "medium"},
                parse_reasoning_policy("native"),
                QWEN_ENABLE_THINKING_PROFILE,
            )

    def test_empty_and_off_contamination_remain_model_failures(self) -> None:
        empty = classify_direct_response(_response(), expected_content="wanted")
        contaminated = classify_direct_response(
            _response(content="wanted", reasoning="private"),
            expected_content="wanted",
        )
        allowed = classify_direct_response(
            _response(content="wanted", reasoning="private"),
            expected_content="wanted",
            reasoning_allowed=True,
        )
        self.assertEqual(empty["classification"], "EMPTY_OUTPUT")
        self.assertEqual(contaminated["classification"], "REASONING_CONTAMINATION")
        self.assertEqual(allowed["classification"], "EXACT_VISIBLE_RESPONSE")


class ModelGatewayTests(unittest.TestCase):
    def test_normal_json_observed_once_and_native_off_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _FixtureServer() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="normal",
                observations_path=observations,
                upstream_timeout=10,
            ) as gateway:
                import urllib.request

                request = urllib.request.Request(
                    gateway.endpoint.origin + "/api/chat",
                    data=json.dumps({"model": "fixture", "stream": False, "think": True}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    json.load(response)
            summary = read_model_transport_observations(observations)
            native_payload = fixture.payloads[0][1]
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["response_count"], 1)
        self.assertTrue(summary["observer_ok"])
        self.assertTrue(summary["visible_content_returned"])
        self.assertTrue(summary["finish_state_returned"])
        self.assertIs(native_payload["think"], False)
        self.assertNotIn("messages", summary["requests"][0])

    def test_actual_hermes_openai_stream_split_sse_and_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _FixtureServer() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="hermes-stream",
                observations_path=observations,
                upstream_timeout=10,
            ) as gateway:
                script = (
                    "from openai import OpenAI\n"
                    f"c=OpenAI(base_url={gateway.endpoint.base_url!r},api_key='ollama')\n"
                    "s=c.chat.completions.create(model='fixture',messages=[{'role':'user','content':'hidden'}],stream=True)\n"
                    "list(s)\n"
                )
                process = subprocess.run(
                    [str(HERMES_PYTHON), "-c", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stdout)
            summary = read_model_transport_observations(observations)
            payload = fixture.payloads[0][1]
        self.assertTrue(summary["observer_ok"], summary)
        self.assertEqual((summary["request_count"], summary["response_count"]), (1, 1))
        self.assertTrue(summary["visible_content_returned"])
        self.assertTrue(summary["tool_call_returned"])
        self.assertTrue(summary["finish_state_returned"])
        self.assertEqual(summary["duplicate_response_ids"], [])
        self.assertEqual(payload["reasoning_effort"], "none")

    def test_off_policy_reasoning_contamination_is_observed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _FixtureServer() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="contamination",
                observations_path=observations,
                upstream_timeout=10,
            ) as gateway:
                import urllib.request

                request = urllib.request.Request(
                    gateway.endpoint.origin + "/api/chat",
                    data=json.dumps(
                        {
                            "model": "fixture",
                            "stream": False,
                            "fixture_reasoning": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    json.load(response)
            summary = read_model_transport_observations(observations)
        self.assertTrue(summary["observer_ok"])
        self.assertTrue(summary["reasoning_content_returned"])

    def test_missing_and_broken_observer_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = read_model_transport_observations(root / "missing.jsonl")
            broken_path = root / "broken.jsonl"
            broken_path.write_bytes(b"\xff")
            broken = read_model_transport_observations(broken_path)
        self.assertTrue(missing["observer_ok"])
        self.assertEqual(missing["request_count"], 0)
        self.assertFalse(broken["observer_ok"])
        self.assertEqual(broken["read_error"], "UnicodeDecodeError")

    def test_invalid_contract_and_mixed_policy_observations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observations = Path(temporary) / "observations.jsonl"
            rows = [
                {
                    "event": "request",
                    "transport_contract": "wrong-contract",
                    "contract": "ollama-reasoning-policy-v1",
                    "endpoint_path": "/v1/chat/completions",
                    "api_mode": "openai-chat-completions",
                    "request_id": "request-1",
                    "requested_reasoning_policy": "off",
                    "reasoning_cohort": "controlled-off",
                    "serialized_control_field": "reasoning_effort",
                    "serialized_control_value": "none",
                    "conflicting_fields_removed": [],
                    "legacy_enable_thinking_present": False,
                },
                {
                    "event": "request",
                    "transport_contract": "ollama-model-transport-v1",
                    "contract": "ollama-reasoning-policy-v1",
                    "endpoint_path": "/v1/chat/completions",
                    "api_mode": "openai-chat-completions",
                    "request_id": "request-2",
                    "requested_reasoning_policy": "native",
                    "reasoning_cohort": "native-deployment",
                    "serialized_control_field": None,
                    "serialized_control_value": None,
                    "conflicting_fields_removed": [],
                    "legacy_enable_thinking_present": False,
                },
            ]
            observations.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = read_model_transport_observations(observations)
        self.assertFalse(summary["observer_ok"])
        self.assertEqual(summary["invalid_contract_row_count"], 1)
        self.assertTrue(summary["policy_inconsistent"])
        self.assertIsNone(summary["requested_reasoning_policy"])

    def test_request_only_observation_is_valid_timeout_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observations = Path(temporary) / "observations.jsonl"
            observations.write_text(
                json.dumps(
                    {
                        "event": "request",
                        "transport_contract": "ollama-model-transport-v1",
                        "contract": "ollama-reasoning-policy-v1",
                        "endpoint_path": "/v1/chat/completions",
                        "api_mode": "openai-chat-completions",
                        "request_id": "request-1",
                        "requested_reasoning_policy": "off",
                        "reasoning_cohort": "controlled-off",
                        "serialized_control_field": "reasoning_effort",
                        "serialized_control_value": "none",
                        "conflicting_fields_removed": [],
                        "legacy_enable_thinking_present": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = read_model_transport_observations(observations)
        self.assertTrue(summary["request_observer_ok"])
        self.assertFalse(summary["observer_ok"])
        self.assertEqual(summary["missing_response_ids"], ["request-1"])


if __name__ == "__main__":
    unittest.main()
