from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness import benchmark_model, qualification
from harness.comparison import ComparisonError
from harness.evaluator import evaluate_task
from harness.hermes_runner import detect_fallback_attempts
from harness.hermes_diagnostics import assess
from harness.processes import ProcessDeadlineExpired, run_process_group
from harness.reasoning_policy import parse_reasoning_policy
from harness.upstreams import (
    UpstreamError,
    build_benchlocal_command,
    build_infermark_command,
    build_spark_command,
    parse_benchlocal,
    parse_infermark,
    parse_spark,
    safe_slug,
    validate_spark_runtime_lock,
    write_chromium_network_wrapper,
)
from harness.workspace import load_task


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "upstreams"
MODEL = "qwen38-q8-262k:latest"
DIRECT_PROBE = {
    "contract": qualification.DIRECT_PROBE_CONTRACT,
    "max_tokens": qualification.DIRECT_PROBE_MAX_TOKENS,
}


class UpstreamAdapterTests(unittest.TestCase):
    def test_offline_parser_fixtures(self) -> None:
        spark = parse_spark(FIXTURES / "spark-bench.csv", model=MODEL)
        benchlocal = parse_benchlocal(FIXTURES / "benchlocal.json", model=MODEL)
        infermark = parse_infermark(FIXTURES / "infermark.json", model=MODEL)
        self.assertEqual(spark["metrics"]["coding_quality"], 75.0)
        self.assertEqual(benchlocal["score"], 100.0)
        self.assertEqual(infermark["metrics"]["tokens_per_second_c1"], 8.0)

    def test_benchlocal_thinking_contamination_is_a_failed_model_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchlocal.json"
            value = json.loads(
                (FIXTURES / "benchlocal.json").read_text(encoding="utf-8")
            )
            value["thinking_validity"] = {
                "toolcall-15": {"status": "contaminated", "responses": 2}
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            parsed = parse_benchlocal(path, model=MODEL)
        self.assertEqual(parsed["status"], "FAIL")
        self.assertEqual(parsed["metrics"]["contaminated_packs"], ["toolcall-15"])

    def test_missing_or_incompatible_output_is_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            path.write_text('{"schema_version":"2"}\n', encoding="utf-8")
            with self.assertRaises(UpstreamError):
                parse_benchlocal(path, model=MODEL)
            with self.assertRaises(UpstreamError):
                parse_infermark(path, model=MODEL)

    def test_model_identifier_is_one_argv_element_and_slug_is_separate(self) -> None:
        hostile = "model; touch /tmp/never $(id)"
        command = build_benchlocal_command(
            Path("/checkout"),
            endpoint="http://127.0.0.1:11434/v1",
            model=hostile,
            output_path=Path("/output.json"),
            selection={"id": "x", "scenarios": ["toolcall-15/TC-01"], "repeat": 1},
        )
        self.assertIn(hostile, command)
        self.assertEqual(command.count(hostile), 1)
        self.assertNotIn("/", safe_slug(hostile))
        self.assertIn("--no-thinking", command)
        effort_index = command.index("--reasoning-effort")
        self.assertEqual(command[effort_index + 1], "none")
        infer = build_infermark_command(
            Path("/checkout"),
            endpoint="http://127.0.0.1:11434/v1",
            model=hostile,
            output_path=Path("/output.json"),
            profile={"requests": 1, "concurrency": [1], "max_tokens": 1, "warmup": 0, "timeout": 1},
            prompt="safe prompt",
        )
        self.assertEqual(infer.count(hostile), 1)

    def test_spark_reasoning_policy_maps_to_upstream_cli(self) -> None:
        profile = qualification.load_configuration()["profiles"]["standard"]["spark"]
        command = build_spark_command(
            Path("/checkout"),
            endpoint="http://127.0.0.1:11434/v1",
            model=MODEL,
            output_dir=Path("/output"),
            label="offline",
            profile=profile,
            reasoning_policy=parse_reasoning_policy("effort:medium"),
        )
        index = command.index("--thinking")
        self.assertEqual(command[index + 1], "on")

    def test_active_profiles_do_not_require_docker_only_packs(self) -> None:
        config = qualification.load_configuration()
        self.assertEqual(config["generation"], "gx10-qualification-v4")
        self.assertTrue(
            all("thinking" not in profile for profile in config["profiles"].values())
        )
        for name in ("standard", "overnight"):
            selections = config["profiles"][name]["benchlocal"]
            self.assertTrue(selections)
            self.assertTrue(all(not row.get("sandboxed") for row in selections))

    def test_spark_render_browser_is_constrained_to_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = write_chromium_network_wrapper(Path(temporary))
            source = wrapper.read_text(encoding="utf-8")
        self.assertIn("--proxy-server=http://127.0.0.1:9", source)
        self.assertIn("--proxy-bypass-list=localhost;127.0.0.1;[::1]", source)
        self.assertIn("MAP * ~NOTFOUND", source)
        self.assertIn("--disable-background-networking", source)

    def test_spark_runtime_lock_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(UpstreamError, "dependency lock mismatch"):
            validate_spark_runtime_lock(
                {
                    "python_dependencies": {"playwright": "unexpected"},
                    "system_browser": "/usr/bin/chromium",
                }
            )


class DeadlineTests(unittest.TestCase):
    def test_productive_heartbeat_avoids_inactivity_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            heartbeat = Path(temporary) / "heartbeat"
            process = run_process_group(
                [
                    "/bin/sh",
                    "-c",
                    f"i=0; while [ $i -lt 5 ]; do printf x >> {heartbeat}; i=$((i+1)); sleep 0.08; done",
                ],
                timeout=2,
                inactivity_timeout=0.2,
                heartbeat_paths=(heartbeat,),
                poll_interval=0.02,
            )
        self.assertEqual(process.returncode, 0)

    def test_inactivity_timeout_is_distinct(self) -> None:
        with self.assertRaises(ProcessDeadlineExpired) as caught:
            run_process_group(
                ["/bin/sh", "-c", "sleep 5"],
                timeout=2,
                inactivity_timeout=0.15,
                poll_interval=0.02,
            )
        self.assertEqual(caught.exception.deadline_kind, "inactivity")

    def test_total_wall_wins_during_productive_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            heartbeat = Path(temporary) / "heartbeat"
            with self.assertRaises(ProcessDeadlineExpired) as caught:
                run_process_group(
                    [
                        "/bin/sh",
                        "-c",
                        f"while :; do printf x >> {heartbeat}; sleep 0.05; done",
                    ],
                    timeout=0.35,
                    inactivity_timeout=0.2,
                    heartbeat_paths=(heartbeat,),
                    poll_interval=0.02,
                )
        self.assertEqual(caught.exception.deadline_kind, "total_wall")


class QualificationDecisionTests(unittest.TestCase):
    @staticmethod
    def _transport() -> dict:
        return {
            "observer_ok": True,
            "request_count": 1,
            "response_count": 1,
            "reasoning_content_returned": False,
        }

    def _components(self) -> dict:
        return {
            "direct": {
                "status": "PASS",
                "model_transport": {
                    **self._transport(),
                    "request_count": 2,
                    "response_count": 2,
                },
            },
            "upstreams": {
                "spark-bench": {
                    "status": "PASS",
                    "score": 82.5,
                    "metrics": {"coding_quality": 75, "reliability": 90},
                    "model_transport": self._transport(),
                },
                "benchlocal-behavioral": {
                    "status": "PASS",
                    "score": 80,
                    "model_transport": self._transport(),
                },
                "benchlocal-coding": {
                    "status": "PASS",
                    "score": 70,
                    "model_transport": self._transport(),
                },
                "infermark": {
                    "status": "PASS",
                    "score": 100,
                    "metrics": {
                        "errors": 0,
                        "tokens_per_second_c1": 8,
                        "ttft_seconds_c1": {"p50": 0.8},
                    },
                    "model_transport": self._transport(),
                },
            },
            "hermes": [
                {
                    "task_id": "task",
                    "outcome": "PASS",
                    "trajectory_present": True,
                    "final_response_present": True,
                    "agent_execution_valid": True,
                    "model_transport": self._transport(),
                    "blocked_network_attempt_count": 0,
                    "fallback_attempt_count": 0,
                    "cleanup_complete": True,
                }
            ],
            "blocked_network_attempts": [],
        }

    def test_score_normalization_and_gates(self) -> None:
        outcome, gates, scores = qualification.calculate_decision(
            "standard",
            self._components(),
            qualification.load_configuration(),
            parse_reasoning_policy("off"),
        )
        self.assertEqual(outcome, "QUALIFIED")
        self.assertTrue(gates["passed"])
        self.assertEqual(scores["components"]["coding"], 72.5)
        self.assertEqual(scores["components"]["tool_instruction"], 80.0)
        self.assertEqual(scores["executed_weight"], 100)

    def test_valid_execution_without_trajectory_or_text_final_can_pass(self) -> None:
        components = self._components()
        components["hermes"][0]["final_response_present"] = False
        components["hermes"][0]["trajectory_present"] = False
        components["hermes"][0]["trajectory_status"] = "DISCARDED_NO_REASONING"
        outcome, gates, _scores = qualification.calculate_decision(
            "standard",
            components,
            qualification.load_configuration(),
            parse_reasoning_policy("off"),
        )
        self.assertEqual(outcome, "QUALIFIED")
        self.assertTrue(gates["passed"])

    def test_candidate_scope_violation_is_hard_failure(self) -> None:
        components = self._components()
        components["hermes"][0]["scope_pass"] = False
        outcome, gates, _scores = qualification.calculate_decision(
            "standard",
            components,
            qualification.load_configuration(),
            parse_reasoning_policy("off"),
        )
        self.assertEqual(outcome, "NOT_QUALIFIED")
        self.assertIn("Hermes candidate scope violation: task", gates["failures"])

    def test_visible_reasoning_tags_fail_every_transport_stage(self) -> None:
        components = self._components()
        components["direct"]["model_transport"][
            "visible_reasoning_tag_returned"
        ] = True
        components["upstreams"]["benchlocal-behavioral"]["model_transport"][
            "visible_reasoning_tag_returned"
        ] = True
        components["hermes"][0]["model_transport"][
            "visible_reasoning_tag_returned"
        ] = True
        outcome, gates, _scores = qualification.calculate_decision(
            "standard",
            components,
            qualification.load_configuration(),
            parse_reasoning_policy("native"),
        )
        self.assertEqual(outcome, "NOT_QUALIFIED")
        self.assertIn(
            "direct endpoint returned visible reasoning tags",
            gates["failures"],
        )
        self.assertIn(
            "visible reasoning tags returned: benchlocal-behavioral",
            gates["failures"],
        )
        self.assertIn(
            "visible reasoning tags returned: Hermes task",
            gates["failures"],
        )

    def test_cloud_fallback_attempt_detection(self) -> None:
        found = detect_fallback_attempts(
            "Auxiliary client: PAID lane engaged for auxiliary task\n"
            "Auxiliary auto-detect: using nous (x)\n"
        )
        self.assertEqual(len(found), 2)
        self.assertFalse(detect_fallback_attempts("using local custom endpoint"))

    def test_hermes_basic_and_single_tool_assessment(self) -> None:
        basic = {
            "conversations": [
                {
                    "from": "gpt",
                    "value": "<think>private reasoning</think>\nGX10_HERMES_BASIC_OK",
                }
            ],
            "api_calls": 1,
            "toolsets_used": [],
            "tool_stats": {},
        }
        tool = {
            "conversations": [{"from": "gpt", "value": "GX10_HERMES_TOOL_OK"}],
            "api_calls": 2,
            "toolsets_used": ["hermesbench_terminal_foreground"],
            "tool_stats": {"terminal": {"count": 1, "success": 1, "failure": 0}},
        }
        self.assertEqual(assess("basic", basic)["status"], "PASS")
        self.assertEqual(
            assess("basic", basic)["visible_final_response"],
            "GX10_HERMES_BASIC_OK",
        )
        self.assertEqual(assess("tool", tool)["status"], "PASS")
        tool["tool_stats"]["terminal"]["count"] = 2
        self.assertEqual(assess("tool", tool)["status"], "FAIL")

    def test_one_command_dry_run_contacts_nothing(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = qualification.main(["--model", MODEL, "--profile", "smoke", "--dry-run"])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["contacts_endpoint"])
        self.assertEqual(plan["model"], MODEL)
        self.assertEqual(plan["direct_probe"], DIRECT_PROBE)
        self.assertEqual(plan["reasoning_policy"]["requested"], "configured")
        self.assertEqual(plan["reasoning_policy"]["effective"], "effort:medium")
        self.assertEqual(
            plan["reasoning_policy"]["selection_mode"], "configured"
        )
        self.assertEqual(
            plan["reasoning_policy"]["benchmark_track"], "primary-deployment"
        )

    def test_completed_outcomes_automatically_regenerate_and_end_with_ranking(
        self,
    ) -> None:
        identity = {
            "display_name": "Example v1 — 7B (dense) — Q8_0 — 4096-token context — sha256:"
            + "a" * 64
        }
        leaderboard = {
            "qualified": [],
            "evaluated_not_qualified": [],
            "diagnostics": [],
        }
        paths = tuple(
            Path("/tmp") / name
            for name in (
                "local.md",
                "local.json",
                "public.md",
                "public.json",
            )
        )
        for outcome, expected_code, section in (
            ("QUALIFIED", 0, "qualified"),
            ("NOT_QUALIFIED", 1, "evaluated_not_qualified"),
        ):
            with self.subTest(outcome=outcome):
                result = {
                    "outcome": outcome,
                    "run_id": f"fixture-{outcome.lower()}",
                    "model": {
                        "runtime_model": MODEL,
                        "public_identity": identity,
                    },
                    "scores": {"deployment_configuration_score": 75.0},
                    "artifacts": {"report": "/tmp/report.md"},
                }
                public = {key: list(value) for key, value in leaderboard.items()}
                public[section] = [
                    {
                        "run_id": result["run_id"],
                        "rank": 1 if outcome == "QUALIFIED" else None,
                        "outcome": outcome,
                        "overall_score": 75.0,
                        "model_identity": identity,
                    }
                ]
                output = io.StringIO()
                with (
                    mock.patch.object(
                        qualification,
                        "execute",
                        return_value=(expected_code, result),
                    ),
                    mock.patch(
                        "harness.comparison.compare_runs",
                        return_value=(
                            {},
                            paths[0],
                            paths[1],
                            public,
                            paths[2],
                            paths[3],
                        ),
                    ) as compare,
                    contextlib.redirect_stdout(output),
                ):
                    code = qualification.main(
                        ["--model", MODEL, "--profile", "standard"]
                    )
                self.assertEqual(code, expected_code)
                compare.assert_called_once()
                self.assertIn("*", output.getvalue())
                self.assertIn(identity["display_name"], output.getvalue())
                self.assertTrue(
                    output.getvalue().rstrip().endswith(
                        "qualified="
                        + ("1" if outcome == "QUALIFIED" else "0")
                        + " not_qualified="
                        + ("0" if outcome == "QUALIFIED" else "1")
                        + " diagnostic=0"
                    )
                )

    def test_post_processing_failure_preserves_completed_outcome_and_exit(
        self,
    ) -> None:
        result = {
            "outcome": "NOT_QUALIFIED",
            "run_id": "fixture-not-qualified",
            "model": {
                "runtime_model": MODEL,
                "public_identity": {"display_name": "Fixture identity"},
            },
            "scores": {"deployment_configuration_score": 42.0},
            "artifacts": {"report": "/tmp/report.md"},
        }
        error = io.StringIO()
        output = io.StringIO()
        with (
            mock.patch.object(qualification, "execute", return_value=(1, result)),
            mock.patch(
                "harness.comparison.compare_runs",
                side_effect=ComparisonError("fixture output failure"),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            code = qualification.main(
                ["--model", MODEL, "--profile", "standard"]
            )
        self.assertEqual(code, 1)
        self.assertIn("POST_PROCESS_ERROR", error.getvalue())
        self.assertIn("outcome remains NOT_QUALIFIED", error.getvalue())

    def test_direct_probe_ceiling_is_uniform_for_every_policy(self) -> None:
        calls: list[dict] = []

        def endpoint(_url: str, *, body: dict, timeout: int = 180):
            calls.append(body)
            if "tools" not in body:
                return (
                    {
                        "choices": [
                            {"message": {"content": "GX10_DIRECT_OK"}, "finish_reason": "stop"}
                        ],
                        "usage": {"completion_tokens": 1},
                    },
                    1.0,
                )
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "qualification_probe",
                                            "arguments": '{"token":"GX10_TOOL_OK"}',
                                        }
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"completion_tokens": 4},
                },
                1.0,
            )

        with mock.patch.object(qualification, "_endpoint_json", side_effect=endpoint):
            for policy in (
                "off",
                "native",
                "effort:low",
                "effort:medium",
                "effort:high",
            ):
                result = qualification.direct_checks(
                    qualification.validate_local_openai_endpoint(
                        "http://127.0.0.1:11434/v1"
                    ),
                    MODEL,
                    reasoning_policy=parse_reasoning_policy(policy),
                    direct_probe=DIRECT_PROBE,
                )
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["max_tokens"], 1024)
        self.assertEqual(len(calls), 10)
        self.assertEqual({call["max_tokens"] for call in calls}, {1024})

    def test_reasoning_enabled_direct_outputs_and_usage_are_recorded(self) -> None:
        def endpoint(_url: str, *, body: dict, timeout: int = 180):
            if "tools" not in body:
                return (
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "GX10_DIRECT_OK",
                                    "reasoning": "verified exact response",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"completion_tokens": 47},
                    },
                    1.0,
                )
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "verified exact tool call",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "qualification_probe",
                                            "arguments": '{"token":"GX10_TOOL_OK"}',
                                        }
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"completion_tokens": 63},
                },
                1.0,
            )

        with mock.patch.object(qualification, "_endpoint_json", side_effect=endpoint):
            result = qualification.direct_checks(
                qualification.validate_local_openai_endpoint(
                    "http://127.0.0.1:11434/v1"
                ),
                MODEL,
                reasoning_policy=parse_reasoning_policy("effort:medium"),
                direct_probe=DIRECT_PROBE,
            )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["response"]["classification"], "EXACT_VISIBLE_RESPONSE"
        )
        self.assertEqual(result["tool_call"]["classification"], "EXACT_TOOL_CALL")
        self.assertEqual(result["response"]["usage"]["completion_tokens"], 47)
        self.assertEqual(result["tool_call"]["usage"]["completion_tokens"], 63)

    def test_reasoning_truncation_remains_failure_without_retry(self) -> None:
        calls: list[dict] = []

        def endpoint(_url: str, *, body: dict, timeout: int = 180):
            calls.append(body)
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "still preparing required output",
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": 1024},
                },
                1.0,
            )

        with mock.patch.object(qualification, "_endpoint_json", side_effect=endpoint):
            result = qualification.direct_checks(
                qualification.validate_local_openai_endpoint(
                    "http://127.0.0.1:11434/v1"
                ),
                MODEL,
                reasoning_policy=parse_reasoning_policy("native"),
                direct_probe=DIRECT_PROBE,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["response"]["classification"], "TRUNCATED_BEFORE_ANSWER"
        )
        self.assertEqual(
            result["tool_call"]["classification"],
            "TRUNCATED_BEFORE_TOOL_CALL",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual({call["max_tokens"] for call in calls}, {1024})

    def test_reasoning_never_substitutes_for_direct_output(self) -> None:
        response = qualification.classify_direct_response(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning": "answer ready"},
                        "finish_reason": "stop",
                    }
                ]
            },
            expected_content="GX10_DIRECT_OK",
            reasoning_allowed=True,
        )
        tool = qualification.classify_direct_response(
            {
                "choices": [
                    {
                        "message": {"content": "", "reasoning": "call ready"},
                        "finish_reason": "stop",
                    }
                ]
            },
            expected_tool_name="qualification_probe",
            expected_tool_arguments={"token": "GX10_TOOL_OK"},
            reasoning_allowed=True,
        )
        self.assertEqual(response["classification"], "REASONING_ONLY")
        self.assertEqual(tool["classification"], "REASONING_ONLY")
        self.assertEqual((response["status"], tool["status"]), ("FAIL", "FAIL"))

    def test_direct_probe_budget_is_visible_in_expert_report(self) -> None:
        report = qualification.render_report(
            {
                "run_id": "fixture-run",
                "profile": "smoke",
                "outcome": "QUALIFIED",
                "result_validity": "VALID",
                "profile_decision": "MEETS_PROFILE",
                "model": {
                    "runtime_model": MODEL,
                    "runtime_digest": "sha256:" + "a" * 64,
                    "endpoint": "http://127.0.0.1:11434/v1",
                    "reasoning_policy": "effort:medium",
                    "reasoning_cohort": "reasoning-effort-medium",
                },
                "provenance": {
                    "reasoning_policy": {"serialized_endpoint_controls": {}}
                },
                "components": {
                    "direct": {
                        **DIRECT_PROBE,
                        "response": {
                            "classification": "EXACT_VISIBLE_RESPONSE",
                            "finish_reason": "stop",
                            "usage": {"completion_tokens": 47},
                        },
                        "tool_call": {
                            "classification": "EXACT_TOOL_CALL",
                            "finish_reason": "tool_calls",
                            "usage": {"completion_tokens": 63},
                        },
                    }
                },
                "gates": {"passed": True, "failures": []},
                "scores": {
                    "deployment_configuration_score": 100,
                    "components": {},
                    "weights": {},
                },
            }
        )
        self.assertIn("Direct-probe contract: `gx10-direct-probe-v1`", report)
        self.assertIn("Direct-probe max completion tokens: `1024`", report)
        self.assertIn("completion tokens `47`", report)
        self.assertIn("completion tokens `63`", report)

    def test_off_policy_still_rejects_reasoning_contamination(self) -> None:
        def endpoint(_url: str, *, body: dict, timeout: int = 180):
            message: dict = {
                "content": "GX10_DIRECT_OK",
                "reasoning": "unexpected reasoning",
            }
            finish_reason = "stop"
            if "tools" in body:
                message = {
                    "content": "",
                    "reasoning": "unexpected reasoning",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "qualification_probe",
                                "arguments": '{"token":"GX10_TOOL_OK"}',
                            }
                        }
                    ],
                }
                finish_reason = "tool_calls"
            return (
                {
                    "choices": [
                        {"message": message, "finish_reason": finish_reason}
                    ],
                    "usage": {"completion_tokens": 8},
                },
                1.0,
            )

        with mock.patch.object(qualification, "_endpoint_json", side_effect=endpoint):
            result = qualification.direct_checks(
                qualification.validate_local_openai_endpoint(
                    "http://127.0.0.1:11434/v1"
                ),
                MODEL,
                reasoning_policy=parse_reasoning_policy("off"),
                direct_probe=DIRECT_PROBE,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["response"]["classification"], "REASONING_CONTAMINATION"
        )
        self.assertEqual(
            result["tool_call"]["classification"], "REASONING_CONTAMINATION"
        )

    def test_infermark_profile_component_exposes_pinned_source(self) -> None:
        checkout = Path("/pinned/infermark")
        policy = Path("/run/python-policy")
        self.assertEqual(
            qualification._upstream_python_paths(
                "infermark-short", checkout, policy
            ),
            [policy, checkout / "src"],
        )
        self.assertEqual(
            qualification._upstream_python_paths(
                "spark-bench", checkout, policy
            ),
            [policy, qualification.SPARK_PYTHON_DEPS],
        )

    @staticmethod
    def _runtime_preflight(
        *,
        digest: str = "sha256:" + "d" * 64,
        context: int | None = 65536,
    ) -> dict:
        metadata = {
            "architecture": "examplemoe",
            "parameter_variant": "35.7B",
            "parameter_count": 35_700_000_000,
            "quantization": "Q6_K_L",
            "native_context_length": 262144,
            "metadata_sources": {"quantization": "/api/show details"},
        }
        if context is not None:
            metadata["effective_context_length"] = context
            metadata["effective_context_sources"] = [
                "/api/show Modelfile PARAMETER num_ctx"
            ]
        return {
            "runtime_model": "new-runtime:latest",
            "runtime_model_digest": digest,
            "identity_status": "VERIFIED",
            "public_runtime_metadata": metadata,
        }

    def test_unknown_installed_tag_is_discovered_in_memory(self) -> None:
        expected = self._runtime_preflight()
        with mock.patch.object(
            benchmark_model, "preflight_model", return_value=expected
        ) as preflight:
            alias, runtime, config, observed = qualification.resolve_model(
                "new-runtime:latest",
                endpoint="http://127.0.0.1:11434/v1",
            )
        preflight.assert_called_once_with(
            "http://127.0.0.1:11434/v1",
            "new-runtime:latest",
            expected_digest=None,
        )
        self.assertEqual(alias, "new-runtime:latest")
        self.assertEqual(runtime, "new-runtime:latest")
        self.assertIs(observed, expected)
        self.assertEqual(config["runtime_digest"], "sha256:" + "d" * 64)
        self.assertEqual(config["context_length"], 65536)
        self.assertEqual(config["quantization"], "Q6_K_L")
        self.assertEqual(config["architecture"], "examplemoe")
        self.assertNotIn("reasoning_effort", config)
        self.assertNotIn("reasoning_policy", config)

    def test_unknown_nonexistent_tag_is_rejected_as_infrastructure(self) -> None:
        with mock.patch.object(
            benchmark_model,
            "preflight_model",
            side_effect=benchmark_model.InfrastructureError(
                "configured model identity is missing or ambiguous: absent:latest"
            ),
        ):
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "model identity is missing or ambiguous",
            ):
                qualification.resolve_model(
                    "absent:latest",
                    endpoint="http://127.0.0.1:11434/v1",
                )

    def test_discovery_requires_only_the_missing_effective_context(self) -> None:
        with mock.patch.object(
            benchmark_model,
            "preflight_model",
            return_value=self._runtime_preflight(context=None),
        ):
            with self.assertRaisesRegex(
                qualification.QualificationError,
                "effective context is unavailable: /api/show did not report "
                "num_ctx in parameters or Modelfile$",
            ):
                qualification.resolve_model(
                    "new-runtime:latest",
                    endpoint="http://127.0.0.1:11434/v1",
                )

    def test_curated_override_precedes_runtime_discovery(self) -> None:
        with mock.patch.object(benchmark_model, "preflight_model") as preflight:
            alias, runtime, config, observed = qualification.resolve_model(
                "agent-main:latest",
                endpoint="http://127.0.0.1:11434/v1",
            )
        preflight.assert_not_called()
        self.assertEqual(alias, "agent-main:latest")
        self.assertEqual(runtime, "agent-main:latest")
        self.assertEqual(config["context_length"], 65536)
        self.assertEqual(
            config["configuration_source"], "models.yaml curated override"
        )
        self.assertEqual(config["reasoning_policy"], "effort:medium")
        self.assertIn("off", config["supported_reasoning_policies"])
        self.assertIsNone(observed)

    def test_same_alias_can_resolve_to_a_new_exact_digest(self) -> None:
        first = self._runtime_preflight(digest="sha256:" + "d" * 64)
        second = self._runtime_preflight(digest="sha256:" + "e" * 64)
        with mock.patch.object(
            benchmark_model,
            "preflight_model",
            side_effect=[first, second],
        ):
            first_config = qualification.resolve_model(
                "new-runtime:latest",
                endpoint="http://127.0.0.1:11434/v1",
            )[2]
            second_config = qualification.resolve_model(
                "new-runtime:latest",
                endpoint="http://127.0.0.1:11434/v1",
            )[2]
        self.assertNotEqual(
            first_config["runtime_digest"], second_config["runtime_digest"]
        )

    def test_discovery_never_modifies_tracked_model_configuration(self) -> None:
        tracked = (qualification.MODELS_PATH, ROOT / "model-metadata.yaml")
        before = {path: path.read_bytes() for path in tracked}
        with mock.patch.object(
            benchmark_model,
            "preflight_model",
            return_value=self._runtime_preflight(),
        ):
            qualification.resolve_model(
                "new-runtime:latest",
                endpoint="http://127.0.0.1:11434/v1",
            )
        self.assertEqual(before, {path: path.read_bytes() for path in tracked})

    def test_ds4_curated_model_resolves_without_ollama_discovery(self) -> None:
        with mock.patch.object(benchmark_model, "preflight_model") as preflight:
            alias, runtime, config, observed = qualification.resolve_model(
                "deepseek-v4-flash"
            )
        preflight.assert_not_called()
        self.assertEqual((alias, runtime), ("deepseek-v4-flash", "deepseek-v4-flash"))
        self.assertEqual(config["runtime"], "ds4")
        self.assertEqual(config["reasoning_policy"], "off")
        self.assertIsNone(observed)

    def test_unconfigured_dry_resolution_defers_endpoint_contact(self) -> None:
        with mock.patch.object(benchmark_model, "preflight_model") as preflight:
            alias, runtime, config, observed = qualification.resolve_model(
                "unknown-model:latest"
            )
        preflight.assert_not_called()
        self.assertEqual(alias, runtime)
        self.assertEqual(
            config["configuration_source"],
            "Ollama runtime discovery pending",
        )
        self.assertIsNone(observed)

    def test_infrastructure_failure_preserves_partial_result(self) -> None:
        previous = qualification._ACTIVE_RUN_CONTEXT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                (run_dir / "logs").mkdir()
                (run_dir / "artifacts").mkdir()
                (run_dir / "partial-results.json").write_text(
                    json.dumps(
                        {
                            "status": "RUNNING",
                            "components": {"direct": {"status": "PASS"}},
                        }
                    ),
                    encoding="utf-8",
                )
                qualification._ACTIVE_RUN_CONTEXT = {
                    "run_dir": run_dir,
                    "run_id": "fixture-run",
                    "profile": "smoke",
                    "requested_model": MODEL,
                    "alias": "qwen38-q8-medium-262k",
                    "runtime_model": MODEL,
                    "endpoint": "http://127.0.0.1:11434/v1",
                    "manifest": {"schema_version": 1, "run_id": "fixture-run"},
                }
                preserved = qualification.preserve_infrastructure_failure(
                    UpstreamError("fixture parser failure")
                )
                result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
                partial = json.loads(
                    (run_dir / "partial-results.json").read_text(encoding="utf-8")
                )
                self.assertEqual(preserved, run_dir)
                self.assertEqual(result["outcome"], "INFRA_ERROR")
                self.assertEqual(result["result_validity"], "INFRA_FAILURE")
                self.assertEqual(result["profile_decision"], "NOT_ASSESSED")
                self.assertEqual(result["components"]["direct"]["status"], "PASS")
                self.assertEqual(partial["status"], "INFRA_ERROR")
        finally:
            qualification._ACTIVE_RUN_CONTEXT = previous

    def test_quarantine_preserves_technical_outcome_but_is_not_assessed(self) -> None:
        previous = qualification._ACTIVE_RUN_CONTEXT
        try:
            with tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                (run_dir / "logs").mkdir()
                (run_dir / "artifacts").mkdir()
                (run_dir / "partial-results.json").write_text(
                    json.dumps({"status": "RUNNING", "components": {}}),
                    encoding="utf-8",
                )
                qualification._ACTIVE_RUN_CONTEXT = {
                    "run_dir": run_dir,
                    "run_id": "quarantined-fixture",
                    "profile": "standard",
                    "requested_model": MODEL,
                    "alias": MODEL,
                    "runtime_model": MODEL,
                    "endpoint": "http://127.0.0.1:11434/v1",
                    "manifest": {
                        "schema_version": 1,
                        "run_id": "quarantined-fixture",
                    },
                }
                qualification.preserve_infrastructure_failure(
                    qualification.QualificationQuarantine(
                        "Spark Bench completed but was quarantined: FLAT_DOMAIN:code=0"
                    )
                )
                result = json.loads(
                    (run_dir / "results.json").read_text(encoding="utf-8")
                )
            self.assertEqual(result["outcome"], "INFRA_ERROR")
            self.assertEqual(result["result_validity"], "QUARANTINED")
            self.assertEqual(result["profile_decision"], "NOT_ASSESSED")
            self.assertIn("quarantined result", result["gates"]["failures"][0])
        finally:
            qualification._ACTIVE_RUN_CONTEXT = previous


class PublicConfigurationTests(unittest.TestCase):
    def test_every_curated_model_has_an_explicit_configured_policy(self) -> None:
        policies = {
            alias: model["reasoning_policy"]
            for alias, model in benchmark_model.load_models().items()
        }
        self.assertEqual(
            policies,
            {
                "laguna-apex-128k": "effort:high",
                "agent-main:latest": "effort:medium",
                "coder-max:latest": "effort:medium",
                "coder-max-128k:latest": "effort:medium",
                "reviewer-deep:latest": "effort:medium",
                "coder-uncens:latest": "effort:high",
                "coder-uncens-256k:latest": "effort:high",
                "coder-uncens-qwen:latest": "effort:medium",
                "hermes4-70b:latest": "effort:medium",
                "qwen38-q8-medium-262k": "effort:medium",
                "qwen38-q8-medium-128k": "effort:medium",
                "gemma4:31b-it-bf16": "native",
                "ornith15-q8:latest": "off",
                "deepseek-v4-flash": "off",
            },
        )
        for alias, model in benchmark_model.load_models().items():
            with self.subTest(alias=alias):
                self.assertIn(
                    model["reasoning_policy"],
                    model["supported_reasoning_policies"],
                )

    def test_curated_runtime_inventory_has_exact_observed_identity(self) -> None:
        models = benchmark_model.load_models()
        expected = {
            "agent-main:latest": (
                "sha256:0218f872e86baa9c7610509f27db36a7bc52eea7afee24688f81ca74ffcb6c77",
                "qwen35moe",
                "36.0B",
                "Q8_0",
                65536,
                262144,
            ),
            "coder-max:latest": (
                "sha256:3f68e12b44eea7c1f464501436bbbe67a8234bf2694efc1d7407ab3df73251b6",
                "qwen3next",
                "79.7B",
                "Q8_0",
                65536,
                262144,
            ),
            "coder-max-128k:latest": (
                "sha256:0dbde7ee79ca3ede12b17811e23a4d18e9e61a5dfa261e831cd3f6cdeeaf1803",
                "qwen3next",
                "79.7B",
                "Q8_0",
                131072,
                262144,
            ),
            "reviewer-deep:latest": (
                "sha256:a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9",
                "gptoss",
                "116.8B",
                "MXFP4",
                65536,
                131072,
            ),
            "coder-uncens:latest": (
                "sha256:50197a047af0f7c198929eb80bbd4bb5a44cd43ca878f5241c846ffbf3ee9a7b",
                "laguna",
                "117.6B",
                "Q6_K",
                131072,
                1048576,
            ),
            "coder-uncens-256k:latest": (
                "sha256:ea34631ee5c00f9acd0182cb9503098e96c551e3841bf537bbc69a4576aea84b",
                "laguna",
                "117.6B",
                "Q6_K",
                262144,
                1048576,
            ),
            "coder-uncens-qwen:latest": (
                "sha256:91e5aee4586a37fda420039bfc30293371bab72026a06239e1183339caff2ae0",
                "qwen3next",
                "79.7B",
                "Q8_0",
                131072,
                262144,
            ),
            "hermes4-70b:latest": (
                "sha256:59c3de79a9d1198dd7065fe1431761274b0bd4fe3582c0c8b59ec044be9627ea",
                "llama",
                "70.6B",
                "Q4_K_M",
                65536,
                131072,
            ),
        }
        for alias, values in expected.items():
            with self.subTest(alias=alias):
                model = models[alias]
                self.assertEqual(
                    (
                        model["runtime_digest"],
                        model["architecture"],
                        model["parameter_variant"],
                        model["quantization"],
                        model["context_length"],
                        model["native_context_length"],
                    ),
                    values,
                )
        self.assertNotIn("coder-uncens-laguna", models)
        qwen = models["qwen38-q8-medium-262k"]
        self.assertEqual(
            qwen["runtime_digest"],
            "sha256:4ab95509a27d7a3f23dcc612a660858e9f28c1a5322bd9240f34559bcf888988",
        )
        self.assertEqual(qwen["context_length"], 262144)

    def test_convergence_acceptance_is_frozen_and_evaluable(self) -> None:
        task, task_dir = load_task("convergence-smoke-v1")
        benchmark_model._validate_task_assets(task, task_dir, ROOT)
        profiles = qualification.load_configuration()["profiles"]
        self.assertNotIn("convergence-smoke-v1", profiles["smoke"]["hermes_tasks"])
        self.assertIn("convergence-smoke-v1", profiles["standard"]["hermes_tasks"])
        self.assertIn("convergence-smoke-v1", profiles["overnight"]["hermes_tasks"])

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "candidate"
            shutil.copytree(ROOT / task["fixture"], candidate)
            (candidate / "policy.py").write_text(
                "def normalize_label(value):\n"
                "    if not isinstance(value, str):\n"
                "        raise ValueError('label must be a string')\n"
                "    normalized = value.strip().lower()\n"
                "    if not normalized:\n"
                "        raise ValueError('label must not be empty')\n"
                "    return normalized\n\n"
                "def validate_values(values):\n"
                "    if not isinstance(values, list) or any(type(item) is not int for item in values):\n"
                "        raise ValueError('values must be a list of integers')\n"
                "    return list(values)\n",
                encoding="utf-8",
            )
            (candidate / "convergence_project.py").write_text(
                "from policy import normalize_label, validate_values\n\n"
                "def build_record(label, values):\n"
                "    cleaned = validate_values(values)\n"
                "    return {'label': normalize_label(label), 'total': sum(cleaned), 'count': len(cleaned)}\n",
                encoding="utf-8",
            )
            (candidate / "RUN_NOTES.md").write_text(
                "Recovered from the expected missing-check.sh failure.\n",
                encoding="utf-8",
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertTrue(result["pass"], result)
        self.assertEqual(result["public"]["passed"], 2)
        self.assertEqual(result["hidden"]["passed"], 6)

    def test_public_default_is_generic_and_local_override_is_ignored(self) -> None:
        public = (ROOT / "config" / "local.example.yaml").read_text(encoding="utf-8")
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("127.0.0.1", public)
        self.assertIn("config/local.yaml", ignored)
        tracked = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "config/local.yaml"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertNotEqual(tracked.returncode, 0)


if __name__ == "__main__":
    unittest.main()
