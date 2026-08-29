from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harness import benchmark_model, qualification
from harness.hermes_runner import write_runtime_hermes_config
from harness.model_gateway import ModelGateway, read_model_transport_observations
from harness.reasoning_policy import (
    classify_direct_response,
    parse_reasoning_policy,
)


DS4_DIGEST = (
    "sha256:ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
)
DSPARK_DIGEST = (
    "sha256:8fa269560dc76fd73e4233ad9b1938b5f65dd363381fd9b1a5c6183f7d12d686"
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
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
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


if __name__ == "__main__":
    unittest.main()
