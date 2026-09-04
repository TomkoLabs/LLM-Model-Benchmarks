from __future__ import annotations

import json
import socket
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema

from harness import benchmark_model, qualification
from harness.hermes_runner import write_runtime_hermes_config
from harness.model_gateway import ModelGateway, read_model_transport_observations
from harness.model_gateway import allowed_paths
from harness.model_identity import resolve_public_identity
from harness.reasoning_policy import (
    QWEN_ENABLE_THINKING_PROFILE,
    ReasoningPolicyError,
    classify_direct_response,
    parse_reasoning_policy,
)


DS4_DIGEST = (
    "sha256:ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
)
DSPARK_DIGEST = (
    "sha256:8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686"
)
VLLM_DIGEST = (
    "sha256:56d617b9008714d82b491a73c3fc8f86501a2a4dcd17f0b883359762f1af3e17"
)


class _DS4Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": "deepseek-v4-flash",
                        "object": "model",
                        "owned_by": "ds4",
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        self.server.payloads.append(payload)  # type: ignore[attr-defined]
        if payload.get("stream") is True:
            prompt = payload.get("messages", [{}])[-1].get("content")
            if prompt == "http-error":
                body = json.dumps({"error": "fixture model failure"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return
            if prompt == "malformed-success":
                body = b"this is not an SSE event\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return
            if prompt == "visible-tag":
                chunks = [
                    b'data: {"choices":[{"delta":{"content":"<thi"},"finish_reason":null}]}\n\n',
                    b'data: {"choices":[{"delta":{"content":"nk>leak</think>"},"finish_reason":"stop"}]}\n\n',
                    b"data: [DONE]\n\n",
                ]
            else:
                chunks = [
                    b'data: {"choices":[{"delta":{"content":"STREAM_"},"finish_reason":null}]}\n\n',
                    b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n',
                    b"data: [DONE]\n\n",
                ]
            if prompt == "disconnect":
                chunks = [
                    b'data: {"choices":[{"delta":{"content":"PARTIAL"},"finish_reason":null}]}\n\n',
                    *[
                        b'data: {"choices":[{"delta":{"content":"'
                        + (b"L" * 4096)
                        + b'"},"finish_reason":null}]}\n\n'
                        for _ in range(32)
                    ],
                ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for index, chunk in enumerate(chunks):
                if prompt == "disconnect" and index:
                    time.sleep(0.1)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            self.close_connection = True
            return
        if payload.get("messages", [{}])[-1].get("content") == "contaminate":
            message = {"content": "", "reasoning_content": "hidden"}
            finish = "stop"
        elif payload.get("tools"):
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "qualification_probe",
                            "arguments": json.dumps({"token": "GX10_TOOL_OK"}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            message = {"content": "GX10_DIRECT_OK"}
            finish = "stop"
        body = json.dumps(
            {"choices": [{"message": message, "finish_reason": finish}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _DS4Server:
    def __enter__(self) -> "_DS4Server":
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _DS4Handler)
        self.server.paths = []  # type: ignore[attr-defined]
        self.server.payloads = []  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def paths(self) -> list[str]:
        return self.server.paths  # type: ignore[attr-defined,no-any-return]

    @property
    def payloads(self) -> list[dict[str, object]]:
        return self.server.payloads  # type: ignore[attr-defined,no-any-return]

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _VLLMHandler(_DS4Handler):
    def do_GET(self) -> None:  # noqa: N802
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.server.model_id,  # type: ignore[attr-defined]
                        "object": "model",
                        "owned_by": "vllm",
                        "max_model_len": self.server.max_model_len,  # type: ignore[attr-defined]
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _VLLMServer(_DS4Server):
    def __init__(
        self,
        *,
        model_id: str = "qwen3.8-flash-next",
        max_model_len: int = 262144,
    ) -> None:
        self.model_id = model_id
        self.max_model_len = max_model_len

    def __enter__(self) -> "_VLLMServer":
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _VLLMHandler)
        self.server.paths = []  # type: ignore[attr-defined]
        self.server.payloads = []  # type: ignore[attr-defined]
        self.server.model_id = self.model_id  # type: ignore[attr-defined]
        self.server.max_model_len = self.max_model_len  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self


class RuntimeAdapterTests(unittest.TestCase):
    def test_challenger_curated_identity_and_policy(self) -> None:
        models = benchmark_model.load_models()
        gemma = models["gemma4:31b-it-bf16"]
        self.assertEqual(
            (
                gemma["runtime_digest"],
                gemma["architecture"],
                gemma["parameter_count"],
                gemma["quantization"],
                gemma["context_length"],
                gemma["reasoning_policy"],
            ),
            (
                "sha256:236d76ae08745dbc143c31b9271b0f25750885199aa6039d0fc0113171606e6d",
                "gemma4",
                31273089132,
                "F16",
                262144,
                "native",
            ),
        )
        ornith = models["ornith15-q8:latest"]
        self.assertEqual(
            (
                ornith["runtime_digest"],
                ornith["architecture"],
                ornith["parameter_count"],
                ornith["quantization"],
                ornith["context_length"],
                ornith["reasoning_policy"],
            ),
            (
                "sha256:c7c57f189918400a4b4193530cceec9136f95c019357de02abc61018f8bad485",
                "qwen35moe",
                35505251456,
                "Q8_0",
                262144,
                "off",
            ),
        )
        self.assertEqual(ornith["reasoning_effort"], "none")
        for model in (gemma, ornith):
            self.assertEqual(model["supported_reasoning_policies"], ["off", "native"])
            self.assertEqual(
                model["runtime_capabilities"],
                ["completion", "thinking", "tools", "vision"],
            )
            self.assertRegex(model["template_sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_ds4_provenance_and_default_endpoint(self) -> None:
        model = benchmark_model.load_models()["deepseek-v4-flash"]
        self.assertEqual(model["runtime"], "ds4")
        self.assertEqual(model["runtime_version"], "0.6.5")
        self.assertEqual(model["runtime_digest"], DS4_DIGEST)
        self.assertEqual(model["context_length"], 524288)
        self.assertEqual(model["reasoning_policy"], "off")
        self.assertEqual(
            model["deployment_artifacts"],
            {
                "base_gguf_sha256": DS4_DIGEST,
                "dspark_drafter_sha256": DSPARK_DIGEST,
                "dspark_enabled": True,
            },
        )
        self.assertEqual(qualification.resolve_endpoint(None, "deepseek-v4-flash").port, 8000)

    def test_ds4_model_discovery_uses_only_openai_models(self) -> None:
        model = benchmark_model.load_models()["deepseek-v4-flash"]
        with _DS4Server() as fixture:
            result = benchmark_model.preflight_model(
                fixture.base_url,
                "deepseek-v4-flash",
                expected_digest=DS4_DIGEST,
                runtime_config=model,
            )
        self.assertEqual(fixture.paths, ["/v1/models"])
        self.assertEqual(result["runtime_model_digest"], DS4_DIGEST)
        self.assertEqual(result["runtime_version"], "0.6.5")
        self.assertEqual(
            result["identity_status"],
            "VERIFIED_MODEL_ID_AND_CONFIGURED_ARTIFACTS",
        )
        self.assertEqual(result["deployment_artifacts"]["dspark_drafter_sha256"], DSPARK_DIGEST)
        self.assertEqual(
            set(result["metadata_endpoints"]),
            {"/v1/models"},
        )

    def test_ds4_exact_direct_response_tool_call_and_off_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="ds4-direct",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ds4",
            ) as gateway:
                result = qualification.direct_checks(
                    gateway.endpoint,
                    "deepseek-v4-flash",
                    reasoning_policy=parse_reasoning_policy("off"),
                    direct_probe={
                        "contract": "gx10-direct-probe-v1",
                        "max_tokens": 1024,
                    },
                )
            summary = read_model_transport_observations(observations)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["response"]["classification"], "EXACT_VISIBLE_RESPONSE")
        self.assertEqual(result["tool_call"]["classification"], "EXACT_TOOL_CALL")
        self.assertTrue(summary["observer_ok"], summary)
        self.assertEqual(summary["contract"], "openai-model-transport-v1")
        self.assertEqual(summary["runtime"], "ds4")
        self.assertEqual(fixture.paths, ["/v1/chat/completions", "/v1/chat/completions"])
        self.assertEqual(
            [payload.get("reasoning_effort") for payload in fixture.payloads],
            ["none", "none"],
        )

    def test_ds4_streaming_completion_and_finish_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="ds4-stream",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ds4",
            ) as gateway:
                request = urllib.request.Request(
                    gateway.endpoint.base_url + "/chat/completions",
                    data=json.dumps(
                        {
                            "model": "deepseek-v4-flash",
                            "messages": [{"role": "user", "content": "stream"}],
                            "stream": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
            summary = read_model_transport_observations(observations)
        self.assertTrue(summary["observer_ok"], summary)
        self.assertTrue(summary["visible_content_returned"])
        self.assertTrue(summary["finish_state_returned"])
        self.assertEqual(summary["finish_reasons"], ["stop"])
        self.assertEqual(fixture.payloads[0]["reasoning_effort"], "none")

    def test_v5_spark_ollama_http_error_is_complete_observer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="spark-bench",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ollama",
            ) as gateway:
                request = urllib.request.Request(
                    gateway.endpoint.base_url + "/chat/completions",
                    data=json.dumps(
                        {
                            "model": "coder-uncens:latest",
                            "messages": [
                                {"role": "user", "content": "http-error"}
                            ],
                            "stream": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(request, timeout=10)
            summary = read_model_transport_observations(observations)

        self.assertTrue(summary["observer_ok"], summary)
        self.assertEqual(summary["http_error_response_count"], 1)
        self.assertEqual(summary["response_completed_count"], 0)
        self.assertEqual(summary["parser_errors"], [])
        self.assertEqual(fixture.payloads[0]["reasoning_effort"], "none")

    def test_v5_spark_ds4_client_disconnect_is_terminal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="spark-bench",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ds4",
            ) as gateway:
                request = urllib.request.Request(
                    gateway.endpoint.base_url + "/chat/completions",
                    data=json.dumps(
                        {
                            "model": "deepseek-v4-flash",
                            "messages": [
                                {"role": "user", "content": "disconnect"}
                            ],
                            "stream": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                response = urllib.request.urlopen(request, timeout=10)
                response.readline()
                response.readline()
                response.fp.raw._sock.setsockopt(  # type: ignore[attr-defined]
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
                response.close()
            summary = read_model_transport_observations(observations)

        self.assertTrue(summary["observer_ok"], summary)
        self.assertEqual(summary["request_count"], 1)
        self.assertEqual(summary["response_count"], 1)
        self.assertEqual(summary["downstream_disconnected_count"], 1)
        self.assertEqual(summary["missing_response_ids"], [])
        self.assertEqual(fixture.payloads[0]["reasoning_effort"], "none")

    def test_malformed_successful_stream_still_fails_observer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="spark-bench",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ds4",
            ) as gateway:
                request = urllib.request.Request(
                    gateway.endpoint.base_url + "/chat/completions",
                    data=json.dumps(
                        {
                            "model": "deepseek-v4-flash",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "malformed-success",
                                }
                            ],
                            "stream": True,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
            summary = read_model_transport_observations(observations)

        self.assertFalse(summary["observer_ok"])
        self.assertEqual(summary["parser_errors"], ["malformed SSE response line"])
        self.assertFalse(
            qualification._transport_observer_matches_deployment(
                summary,
                parse_reasoning_policy("off"),
            )
        )

    def test_reasoning_fields_and_visible_tags_are_not_final_content(self) -> None:
        reasoning = classify_direct_response(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning": "hidden"},
                        "finish_reason": "stop",
                    }
                ]
            },
            expected_content="GX10_DIRECT_OK",
        )
        visible_tag = classify_direct_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>hidden</think>GX10_DIRECT_OK"
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            expected_content="GX10_DIRECT_OK",
            reasoning_allowed=True,
        )
        self.assertEqual(reasoning["classification"], "REASONING_ONLY")
        self.assertEqual(
            visible_tag["classification"],
            "VISIBLE_REASONING_TAG_CONTAMINATION",
        )

    def test_ds4_reasoning_and_split_visible_tag_contamination_are_observed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _DS4Server() as fixture:
            observations = Path(temporary) / "observations.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="ds4-contamination",
                observations_path=observations,
                upstream_timeout=10,
                runtime="ds4",
            ) as gateway:
                for prompt, stream in (("contaminate", False), ("visible-tag", True)):
                    request = urllib.request.Request(
                        gateway.endpoint.base_url + "/chat/completions",
                        data=json.dumps(
                            {
                                "model": "deepseek-v4-flash",
                                "messages": [{"role": "user", "content": prompt}],
                                "stream": stream,
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        response.read()
            summary = read_model_transport_observations(observations)
        self.assertTrue(summary["observer_ok"], summary)
        self.assertTrue(summary["reasoning_content_returned"])
        self.assertTrue(summary["visible_reasoning_tag_returned"])

    def test_ds4_hermes_config_omits_ollama_only_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ds4").mkdir()
            (root / "ollama").mkdir()
            ds4 = write_runtime_hermes_config(
                root / "ds4",
                model="deepseek-v4-flash",
                base_url="http://127.0.0.1:8000/v1",
                context_length=524288,
                reasoning="none",
                max_turns=2,
                runtime="ds4",
            )
            ollama = write_runtime_hermes_config(
                root / "ollama",
                model="fixture:latest",
                base_url="http://127.0.0.1:11434/v1",
                context_length=65536,
                reasoning=None,
                max_turns=2,
            )
            ds4_text = Path(ds4["config_path"]).read_text(encoding="utf-8")
            ollama_text = Path(ollama["config_path"]).read_text(encoding="utf-8")
        self.assertNotIn("ollama_num_ctx", ds4_text)
        self.assertIn("ollama_num_ctx: 65536", ollama_text)

    def test_vllm_provenance_policy_and_default_endpoint(self) -> None:
        model = benchmark_model.load_models()["qwen38-flash-next-nvfp4-262k"]
        self.assertEqual(model["runtime"], "vllm")
        self.assertEqual(model["runtime_model"], "qwen3.8-flash-next")
        self.assertEqual(model["runtime_digest"], VLLM_DIGEST)
        self.assertEqual(model["runtime_digest_kind"], "vllm-deployment-provenance")
        self.assertEqual(
            benchmark_model.vllm_provenance_digest(model),
            VLLM_DIGEST,
        )
        self.assertEqual(model["reasoning_policy"], "native")
        self.assertEqual(model["supported_reasoning_policies"], ["off", "native"])
        self.assertEqual(
            model["serving_configuration"],
            {
                "gpu_memory_utilization": 0.75,
                "max_num_seqs": 2,
                "mtp_speculative_tokens": 2,
                "prefix_caching": True,
                "exact_topk": True,
                "prewarm": False,
            },
        )
        self.assertEqual(
            qualification.resolve_endpoint(
                None, "qwen38-flash-next-nvfp4-262k"
            ).port,
            18300,
        )

    def test_vllm_model_discovery_uses_only_openai_models_and_context(self) -> None:
        model = benchmark_model.load_models()["qwen38-flash-next-nvfp4-262k"]
        with _VLLMServer() as fixture:
            result = benchmark_model.preflight_model(
                fixture.base_url,
                "qwen3.8-flash-next",
                expected_digest=VLLM_DIGEST,
                runtime_config=model,
            )
        self.assertEqual(fixture.paths, ["/v1/models"])
        self.assertEqual(result["runtime"], "vllm")
        self.assertEqual(result["runtime_model_digest"], VLLM_DIGEST)
        self.assertEqual(result["reported_max_model_len"], 262144)
        self.assertEqual(
            result["identity_status"],
            "VERIFIED_MODEL_ID_AND_CONFIGURED_ARTIFACTS",
        )
        self.assertEqual(set(result["metadata_endpoints"]), {"/v1/models"})
        self.assertEqual(
            result["deployment_artifacts"]["checkpoint_revision"],
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        )
        self.assertEqual(result["serving_configuration"]["max_num_seqs"], 2)

        with _VLLMServer(model_id="another-model") as fixture:
            with self.assertRaisesRegex(
                benchmark_model.InfrastructureError,
                "missing or ambiguous",
            ):
                benchmark_model.preflight_model(
                    fixture.base_url,
                    "qwen3.8-flash-next",
                    expected_digest=VLLM_DIGEST,
                    runtime_config=model,
                )
        self.assertEqual(fixture.paths, ["/v1/models"])

        with _VLLMServer(max_model_len=131072) as fixture:
            with self.assertRaisesRegex(
                benchmark_model.InfrastructureError,
                "max_model_len",
            ):
                benchmark_model.preflight_model(
                    fixture.base_url,
                    "qwen3.8-flash-next",
                    expected_digest=VLLM_DIGEST,
                    runtime_config=model,
                )
        self.assertEqual(fixture.paths, ["/v1/models"])

    def test_vllm_gateway_off_and_native_controls_are_exact(self) -> None:
        self.assertEqual(
            allowed_paths("vllm"),
            {"/v1/models", "/v1/chat/completions"},
        )
        with tempfile.TemporaryDirectory() as temporary, _VLLMServer() as fixture:
            observations = Path(temporary) / "off.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="vllm-off",
                observations_path=observations,
                upstream_timeout=10,
                runtime="vllm",
                reasoning_control_profile=QWEN_ENABLE_THINKING_PROFILE,
            ) as gateway:
                result = qualification.direct_checks(
                    gateway.endpoint,
                    "qwen3.8-flash-next",
                    reasoning_policy=parse_reasoning_policy("off"),
                    direct_probe={
                        "contract": "gx10-direct-probe-v1",
                        "max_tokens": 1024,
                    },
                )
            summary = read_model_transport_observations(observations)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            fixture.paths,
            ["/v1/chat/completions", "/v1/chat/completions"],
        )
        self.assertTrue(summary["observer_ok"], summary)
        self.assertEqual(summary["runtime"], "vllm")
        self.assertEqual(
            summary["reasoning_policy_contract"],
            "vllm-chat-template-reasoning-policy-v1",
        )
        for payload in fixture.payloads:
            self.assertEqual(
                payload.get("chat_template_kwargs"),
                {"enable_thinking": False},
            )
            self.assertNotIn("reasoning_effort", payload)
            self.assertNotIn("think", payload)

        with tempfile.TemporaryDirectory() as temporary, _VLLMServer() as fixture:
            observations = Path(temporary) / "native.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("native"),
                stage="vllm-native",
                observations_path=observations,
                upstream_timeout=10,
                runtime="vllm",
                reasoning_control_profile=QWEN_ENABLE_THINKING_PROFILE,
            ) as gateway:
                for controls in (
                    {"chat_template_kwargs": {"enable_thinking": True}},
                    {
                        "think": False,
                        "reasoning_effort": "none",
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ):
                    request = urllib.request.Request(
                        gateway.endpoint.base_url + "/chat/completions",
                        data=json.dumps(
                            {
                                "model": "qwen3.8-flash-next",
                                "messages": [
                                    {"role": "user", "content": "short"}
                                ],
                                "max_tokens": 8,
                                **controls,
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        response.read()
            summary = read_model_transport_observations(observations)
        self.assertTrue(summary["observer_ok"], summary)
        self.assertNotIn("chat_template_kwargs", fixture.payloads[0])
        self.assertNotIn("reasoning_effort", fixture.payloads[0])
        self.assertNotIn("think", fixture.payloads[0])
        self.assertEqual(
            fixture.payloads[1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("reasoning_effort", fixture.payloads[1])
        self.assertNotIn("think", fixture.payloads[1])
        self.assertEqual(
            summary["requested_reasoning_policies"], ["native", "off"]
        )
        self.assertEqual(summary["deployment_reasoning_policy"], "native")
        self.assertFalse(summary["policy_inconsistent"])
        self.assertEqual(
            [
                row["requested_reasoning_policy"]
                for row in summary["requests"]
            ],
            ["native", "off"],
        )

        with tempfile.TemporaryDirectory() as temporary, _VLLMServer() as fixture:
            observations = Path(temporary) / "configured-off.jsonl"
            with ModelGateway(
                target_base_url=fixture.base_url,
                policy=parse_reasoning_policy("off"),
                stage="vllm-configured-off",
                observations_path=observations,
                upstream_timeout=10,
                runtime="vllm",
                reasoning_control_profile=QWEN_ENABLE_THINKING_PROFILE,
                honor_request_reasoning_controls=True,
            ) as gateway:
                for enabled in (True, False):
                    request = urllib.request.Request(
                        gateway.endpoint.base_url + "/chat/completions",
                        data=json.dumps(
                            {
                                "model": "qwen3.8-flash-next",
                                "messages": [
                                    {"role": "user", "content": "short"}
                                ],
                                "chat_template_kwargs": {
                                    "enable_thinking": enabled
                                },
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=10) as response:
                        response.read()
            summary = read_model_transport_observations(observations)
        self.assertNotIn("chat_template_kwargs", fixture.payloads[0])
        self.assertEqual(
            fixture.payloads[1]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(summary["deployment_reasoning_policy"], "off")
        self.assertEqual(
            summary["requested_reasoning_policies"], ["native", "off"]
        )
        self.assertTrue(summary["observer_ok"])
        self.assertFalse(summary["policy_inconsistent"])
        self.assertEqual(summary["invalid_contract_row_count"], 0)

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ReasoningPolicyError):
                ModelGateway(
                    target_base_url="http://127.0.0.1:18300/v1",
                    policy=parse_reasoning_policy("effort:low"),
                    stage="vllm-unsupported",
                    observations_path=Path(temporary) / "unsupported.jsonl",
                    upstream_timeout=10,
                    runtime="vllm",
                    reasoning_control_profile=QWEN_ENABLE_THINKING_PROFILE,
                )

    def test_vllm_hermes_config_and_public_identity(self) -> None:
        model = benchmark_model.load_models()["qwen38-flash-next-nvfp4-262k"]
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "vllm"
            runtime.mkdir()
            generated = write_runtime_hermes_config(
                runtime,
                model="qwen3.8-flash-next",
                base_url="http://127.0.0.1:18300/v1",
                context_length=262144,
                reasoning=None,
                max_turns=2,
                runtime="vllm",
            )
            text = Path(generated["config_path"]).read_text(encoding="utf-8")
        self.assertNotIn("ollama_num_ctx", text)
        self.assertIn("context_length: 262144", text)
        self.assertIn("qwen3.8-flash-next", text)

        identity = resolve_public_identity(model)
        self.assertEqual(identity["status"], "COMPLETE")
        self.assertEqual(identity["canonical_name"], "Qwen3.8-Flash-Next")
        self.assertEqual(identity["quantization"], "NVFP4")
        self.assertEqual(identity["immutable_digest"], VLLM_DIGEST)

    def test_qualification_schema_accepts_vllm_provenance(self) -> None:
        model = benchmark_model.load_models()["qwen38-flash-next-nvfp4-262k"]
        result = {
            "schema_version": 1,
            "run_id": "vllm-fixture",
            "profile": "smoke",
            "outcome": "QUALIFIED",
            "result_validity": "VALID",
            "profile_decision": "MEETS_PROFILE",
            "model": {
                "runtime": "vllm",
                "runtime_digest": VLLM_DIGEST,
                "runtime_digest_kind": model["runtime_digest_kind"],
                "runtime_version": model["runtime_version"],
                "context_length": model["context_length"],
                "endpoint_api_mode": model["api_mode"],
                "capabilities": model["capabilities"],
                "deployment_artifacts": model["deployment_artifacts"],
                "serving_configuration": model["serving_configuration"],
                "reasoning_control_profile": model["reasoning_control_profile"],
                "reasoning_policy": "native",
                "reasoning_cohort": "native-deployment",
            },
            "provenance": {
                "qualification_generation": "gx10-qualification-v4",
                "direct_probe": {
                    "contract": "gx10-direct-probe-v1",
                    "max_tokens": 1024,
                },
                "reasoning_policy": {
                    "contract": "vllm-chat-template-reasoning-policy-v1",
                    "requested": "configured",
                    "effective": "native",
                    "source": "models.yaml curated policy",
                    "cohort": "native-deployment",
                    "serialized_endpoint_controls": {
                        "openai_chat_completions": {}
                    },
                },
            },
            "components": {},
            "gates": {"passed": True, "failures": []},
            "scores": {},
            "artifacts": {},
        }
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas"
                / "qualification-run.schema.json"
            ).read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator(schema).validate(result)
        result["provenance"]["qualification_generation"] = (
            "gx10-qualification-v5"
        )
        jsonschema.Draft202012Validator(schema).validate(result)


if __name__ == "__main__":
    unittest.main()
