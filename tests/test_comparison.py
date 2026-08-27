from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

import jsonschema
import yaml

from harness import qualification
from harness.artifacts import ArtifactSafetyError
from harness.comparison import (
    collect_runs,
    compare_runs,
    current_standard_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "comparison"
RESULT_SCHEMA = json.loads(
    (ROOT / "schemas" / "qualification-run.schema.json").read_text(
        encoding="utf-8"
    )
)


class ComparisonTests(unittest.TestCase):
    def test_current_v4_standard_excludes_v2_thinking_control_runs(self) -> None:
        current = current_standard_compatibility()
        self.assertEqual(
            current["key"]["qualification_generation"],
            "gx10-qualification-v4",
        )
        self.assertEqual(current["key"]["benchmark_track"], "primary-deployment")
        self.assertEqual(
            current["key"]["reasoning_policy_rule"],
            "configured-per-model",
        )
        self.assertEqual(
            current["key"]["direct_probe_contract"],
            "gx10-direct-probe-v1",
        )
        self.assertEqual(current["key"]["direct_probe_max_tokens"], 1024)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            registry = self._registry(root)
            self._install(runs, "qualified.json", "historical-v2")
            (
                _summary,
                _markdown,
                _json,
                leaderboard,
                _public_markdown,
                _public_json,
            ) = compare_runs(
                runs_root=runs,
                output_root=root / "generated",
                public_output_root=root / "public",
                registry_path=registry,
            )
        self.assertFalse(leaderboard["qualified"])
        self.assertEqual(
            leaderboard["diagnostics"][0]["status"],
            "LEGACY_THINKING_CONTROL_MISMATCH",
        )
        self.assertEqual(
            leaderboard["diagnostics"][0]["profile_decision"],
            "MEETS_PROFILE",
        )

    def test_configured_deployments_share_primary_track_across_policies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            self._install(
                runs,
                "qualified.json",
                "configured-medium",
                mutate=lambda value: self._set_v4_policy(
                    value, requested="configured", effective="effort:medium"
                ),
            )
            self._install(
                runs,
                "qualified.json",
                "configured-high",
                mutate=lambda value: self._set_v4_policy(
                    value, requested="configured", effective="effort:high"
                ),
            )
            self._install(
                runs,
                "qualified.json",
                "explicit-medium",
                mutate=lambda value: self._set_v4_policy(
                    value, requested="effort:medium", effective="effort:medium"
                ),
            )
            self._install(
                runs,
                "qualified.json",
                "configured-off",
                mutate=lambda value: self._set_v4_policy(
                    value, requested="configured", effective="off"
                ),
            )
            self._install(
                runs,
                "qualified.json",
                "explicit-off",
                mutate=lambda value: self._set_v4_policy(
                    value, requested="off", effective="off"
                ),
            )
            rows = collect_runs(runs)
        by_id = {row["run_id"]: row for row in rows}
        primary_group = by_id["configured-medium"]["compatibility"]["group_id"]
        self.assertEqual(
            primary_group,
            by_id["configured-high"]["compatibility"]["group_id"],
        )
        self.assertEqual(
            primary_group,
            by_id["configured-off"]["compatibility"]["group_id"],
        )
        self.assertNotEqual(
            primary_group,
            by_id["explicit-medium"]["compatibility"]["group_id"],
        )
        self.assertNotEqual(
            by_id["configured-off"]["compatibility"]["group_id"],
            by_id["explicit-off"]["compatibility"]["group_id"],
        )
        self.assertNotEqual(
            by_id["explicit-medium"]["compatibility"]["group_id"],
            by_id["explicit-off"]["compatibility"]["group_id"],
        )
        self.assertEqual(by_id["configured-high"]["reasoning_policy"], "effort:high")
        self.assertEqual(by_id["configured-high"]["benchmark_track"], "primary-deployment")
        self.assertEqual(by_id["explicit-medium"]["benchmark_track"], "controlled-policy")

    def test_public_ranking_selects_primary_track_and_exposes_effective_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            registry = self._registry(root)
            for run_id, requested, effective in (
                ("configured-high", "configured", "effort:high"),
                ("configured-medium", "configured", "effort:medium"),
                ("explicit-medium", "effort:medium", "effort:medium"),
            ):
                def policy(
                    value: dict[str, Any],
                    requested: str = requested,
                    effective: str = effective,
                ) -> None:
                    self._set_v4_policy(
                        value, requested=requested, effective=effective
                    )

                self._install(
                    runs,
                    "qualified.json",
                    run_id,
                    mutate=policy,
                )
            rows = collect_runs(runs, registry_path=registry)
            baseline = next(
                row for row in rows if row["run_id"] == "configured-medium"
            )["compatibility"]
            (
                _summary,
                _markdown,
                _json,
                leaderboard,
                _public_markdown,
                _public_json,
            ) = compare_runs(
                runs_root=runs,
                output_root=root / "generated",
                public_output_root=root / "public",
                registry_path=registry,
                current_compatibility=baseline,
            )

        self.assertEqual(
            {row["run_id"] for row in leaderboard["qualified"]},
            {"configured-high", "configured-medium"},
        )
        self.assertEqual(
            {row["reasoning_policy"] for row in leaderboard["qualified"]},
            {"effort:high", "effort:medium"},
        )
        self.assertEqual(
            leaderboard["diagnostics"][0]["run_id"], "explicit-medium"
        )
        self.assertEqual(
            leaderboard["diagnostics"][0]["benchmark_track"],
            "controlled-policy",
        )

    def test_v4_policy_source_mismatch_fails_comparison_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            runs.mkdir()

            def mismatched(value: dict[str, Any]) -> None:
                self._set_v4_policy(
                    value, requested="configured", effective="effort:medium"
                )
                value["provenance"]["reasoning_policy"]["source"] = "explicit CLI"

            self._install(
                runs,
                "qualified.json",
                "mismatched-source",
                mutate=mismatched,
            )
            row = collect_runs(runs)[0]

        self.assertFalse(row["eligible_for_model_comparison"])
        self.assertFalse(row["compatibility"]["complete"])
        self.assertIn(
            "reasoning policy source consistency", row["exclusion_reason"]
        )

    def test_direct_probe_ceiling_is_required_comparison_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            runs.mkdir()

            def legacy(value: dict[str, Any]) -> None:
                self._set_v4_policy(
                    value, requested="configured", effective="effort:medium"
                )

            def corrected(value: dict[str, Any]) -> None:
                legacy(value)
                self._set_direct_probe(value, max_tokens=1024)

            self._install(
                runs, "qualified.json", "legacy-128", mutate=legacy
            )
            self._install(
                runs, "qualified.json", "corrected-1024", mutate=corrected
            )
            rows = {row["run_id"]: row for row in collect_runs(runs)}

        legacy_compatibility = rows["legacy-128"]["compatibility"]
        corrected_compatibility = rows["corrected-1024"]["compatibility"]
        self.assertEqual(
            legacy_compatibility["key"]["direct_probe_max_tokens"], 128
        )
        self.assertEqual(
            corrected_compatibility["key"]["direct_probe_max_tokens"], 1024
        )
        self.assertNotEqual(
            legacy_compatibility["group_id"],
            corrected_compatibility["group_id"],
        )

    def test_direct_probe_result_mismatch_fails_comparison_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            runs.mkdir()

            def mismatched(value: dict[str, Any]) -> None:
                self._set_v4_policy(
                    value, requested="configured", effective="effort:medium"
                )
                self._set_direct_probe(value, max_tokens=1024)
                value["components"]["direct"]["max_tokens"] = 128

            self._install(
                runs,
                "qualified.json",
                "mismatched-direct-probe",
                mutate=mismatched,
            )
            row = collect_runs(runs)[0]

        self.assertFalse(row["eligible_for_model_comparison"])
        self.assertIn("direct-probe result consistency", row["exclusion_reason"])

    def _registry(self, root: Path) -> Path:
        path = root / "model-metadata.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "models": {
                        "sha256:" + "a" * 64: {
                            "canonical_name": "Model A",
                            "source_model": "Example/Model-A-v1",
                            "source_url": "https://example.invalid/Model-A-v1",
                            "version": "v1",
                            "parameter_variant": "32B",
                            "architecture": "dense",
                            "quantization": "Q8_0",
                            "source_filename": "model-a-q8.gguf",
                            "aliases": ["model-a:latest"],
                        },
                        "sha256:" + "b" * 64: {
                            "canonical_name": "Model B",
                            "source_model": "Example/Model-B-v3",
                            "source_url": None,
                            "version": "v3",
                            "parameter_variant": "14B",
                            "architecture": "moe",
                            "quantization": "Q4_K_M",
                            "source_filename": "model-b-q4.gguf",
                            "aliases": ["model-b:latest"],
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _set_v4_policy(
        value: dict[str, Any], *, requested: str, effective: str
    ) -> None:
        configured = requested == "configured"
        selection_mode = "configured" if configured else "explicit"
        benchmark_track = (
            "primary-deployment" if configured else "controlled-policy"
        )
        value["provenance"]["qualification_generation"] = "gx10-qualification-v4"
        value["provenance"]["scoring_version"] = "gx10-qualification-v4"
        value["provenance"]["model_config"]["reasoning_policy"] = effective
        value["provenance"]["reasoning_policy"] = {
            "contract": "ollama-reasoning-policy-v1",
            "requested": requested,
            "effective": effective,
            "source": (
                "models.yaml curated policy" if configured else "explicit CLI"
            ),
            "selection_mode": selection_mode,
            "benchmark_track": benchmark_track,
            "cohort": "fixture-cohort",
            "serialized_endpoint_controls": {
                "openai_chat_completions": {},
                "ollama_native_chat": {},
            },
        }
        value["model"].update(
            {
                "reasoning_policy": effective,
                "reasoning_cohort": "fixture-cohort",
                "reasoning_policy_selection_mode": selection_mode,
                "benchmark_track": benchmark_track,
            }
        )

    @staticmethod
    def _set_direct_probe(value: dict[str, Any], *, max_tokens: int) -> None:
        direct_probe = {
            "contract": "gx10-direct-probe-v1",
            "max_tokens": max_tokens,
        }
        value["provenance"]["direct_probe"] = direct_probe
        value["components"]["direct"] = dict(direct_probe)

    def _install(
        self,
        runs: Path,
        fixture: str,
        run_id: str,
        *,
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        run_dir = runs / run_id
        run_dir.mkdir()
        value = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
        value["run_id"] = run_id
        if mutate is not None:
            mutate(value)
        (run_dir / "results.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        return run_dir

    def test_static_result_fixtures_validate(self) -> None:
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(path=path.name):
                jsonschema.Draft202012Validator(RESULT_SCHEMA).validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )

    def test_valid_failures_repeats_and_incompatibility_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            self._install(runs, "qualified.json", "qualified")
            self._install(runs, "not-qualified.json", "not-qualified")
            self._install(runs, "not-qualified.json", "not-qualified-repeat")
            self._install(runs, "infra-error.json", "infra-error")

            incomplete = runs / "incomplete"
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "incomplete",
                        "started_at": "2026-01-04T00:00:00Z",
                        "status": "RUNNING",
                        "profile": "standard",
                        "qualification_generation": "gx10-qualification-v2",
                    }
                ),
                encoding="utf-8",
            )
            unrelated = runs / "focused-hermes-output"
            unrelated.mkdir()
            (unrelated / "summary.json").write_text("{}\n", encoding="utf-8")

            def smoke(value: dict[str, Any]) -> None:
                value["profile"] = "smoke"
                value["provenance"]["profile_config"] = {
                    "name": "smoke",
                    "total_wall_seconds": 1800,
                }

            self._install(
                runs,
                "qualified.json",
                "incompatible-profile",
                mutate=smoke,
            )

            def different_pins(value: dict[str, Any]) -> None:
                value["provenance"]["upstreams"]["spark-bench"]["commit"] = (
                    "9" * 40
                )

            self._install(
                runs,
                "qualified.json",
                "incompatible-pins",
                mutate=different_pins,
            )

            rows = collect_runs(runs)

        by_id = {row["run_id"]: row for row in rows}
        self.assertEqual(len(rows), 7)
        self.assertEqual(by_id["not-qualified"]["outcome"], "NOT_QUALIFIED")
        self.assertTrue(by_id["not-qualified"]["eligible_for_model_comparison"])
        self.assertTrue(
            by_id["not-qualified-repeat"]["eligible_for_model_comparison"]
        )
        self.assertFalse(by_id["infra-error"]["eligible_for_model_comparison"])
        self.assertTrue(by_id["infra-error"]["infrastructure_error"])
        self.assertEqual(by_id["incomplete"]["outcome"], "INCOMPLETE")
        baseline_group = by_id["qualified"]["compatibility"]["group_id"]
        self.assertEqual(
            baseline_group,
            by_id["not-qualified-repeat"]["compatibility"]["group_id"],
        )
        self.assertNotEqual(
            baseline_group,
            by_id["incompatible-profile"]["compatibility"]["group_id"],
        )
        self.assertNotEqual(
            baseline_group,
            by_id["incompatible-pins"]["compatibility"]["group_id"],
        )

    def test_outputs_are_deterministic_and_cli_writes_both_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            output = root / "generated"
            public = root / "public"
            runs.mkdir()
            registry = self._registry(root)
            self._install(runs, "qualified.json", "first")
            self._install(runs, "infra-error.json", "second")
            baseline = collect_runs(runs, registry_path=registry)[0]
            (
                summary,
                markdown_path,
                json_path,
                leaderboard,
                public_markdown,
                public_json,
            ) = compare_runs(
                runs_root=runs,
                output_root=output,
                public_output_root=public,
                registry_path=registry,
                current_compatibility=baseline["compatibility"],
            )
            first_markdown = markdown_path.read_bytes()
            first_json = json_path.read_bytes()
            first_public_markdown = public_markdown.read_bytes()
            first_public_json = public_json.read_bytes()
            compare_runs(
                runs_root=runs,
                output_root=output,
                public_output_root=public,
                registry_path=registry,
                current_compatibility=baseline["compatibility"],
            )
            self.assertEqual(first_markdown, markdown_path.read_bytes())
            self.assertEqual(first_json, json_path.read_bytes())
            self.assertEqual(first_public_markdown, public_markdown.read_bytes())
            self.assertEqual(first_public_json, public_json.read_bytes())
            self.assertEqual(summary["eligible_run_count"], 1)
            self.assertEqual(len(leaderboard["qualified"]), 1)
            self.assertEqual(
                summary["compatibility_groups"][0]["run_ids"], ["first"]
            )

            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                code = qualification.main(
                    [
                        "compare",
                        "--runs-dir",
                        str(runs),
                        "--output-dir",
                        str(output),
                        "--public-output-dir",
                        str(public),
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIn("Local benchmark runs", terminal.getvalue())
        self.assertIn("Compare scores only within", terminal.getvalue())
        self.assertTrue(
            terminal.getvalue().rstrip().endswith(
                "qualified=0 not_qualified=0 diagnostic=2"
            )
        )

    def test_invalid_result_is_retained_but_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary) / "runs"
            invalid = runs / "invalid"
            invalid.mkdir(parents=True)
            shutil.copy(FIXTURES / "qualified.json", invalid / "results.json")
            value = json.loads((invalid / "results.json").read_text(encoding="utf-8"))
            value["outcome"] = "UNKNOWN"
            (invalid / "results.json").write_text(
                json.dumps(value), encoding="utf-8"
            )
            rows = collect_runs(runs)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "INVALID")
        self.assertFalse(rows[0]["eligible_for_model_comparison"])

    def test_incomplete_runtime_identity_writes_digest_candidate(self) -> None:
        digest = "sha256:" + "c" * 64
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            registry = self._registry(root)

            def runtime_discovered(value: dict[str, Any]) -> None:
                value["model"].update(
                    {
                        "runtime_model": "runtime-only:latest",
                        "runtime_digest": digest,
                        "quantization": "Q6_K_L",
                        "context_length": 65536,
                    }
                )
                value["components"]["preflight"] = {
                    "public_runtime_metadata": {
                        "parameter_variant": "35.7B",
                        "architecture": "examplemoe",
                        "quantization": "Q6_K_L",
                        "source_filename": "runtime-only-q6.gguf",
                    }
                }

            self._install(
                runs,
                "not-qualified.json",
                "runtime-discovered",
                mutate=runtime_discovered,
            )
            baseline = collect_runs(runs, registry_path=registry)[0]
            (
                _summary,
                _markdown,
                _json,
                leaderboard,
                _public_markdown,
                _public_json,
            ) = compare_runs(
                runs_root=runs,
                output_root=root / "generated",
                public_output_root=root / "public",
                registry_path=registry,
                current_compatibility=baseline["compatibility"],
            )
            candidate_document = json.loads(
                (
                    root
                    / "generated"
                    / "model-metadata-candidates.json"
                ).read_text(encoding="utf-8")
            )

        candidate = candidate_document["candidates"][digest]
        self.assertEqual(candidate["runtime_aliases"], ["runtime-only:latest"])
        self.assertEqual(candidate["observed_context_lengths"], [65536])
        self.assertEqual(
            candidate["observed_fields"]["quantization"], ["Q6_K_L"]
        )
        self.assertEqual(
            candidate["missing_registry_fields"],
            ["canonical_name", "source_model", "version"],
        )
        self.assertEqual(
            leaderboard["diagnostics"][0]["status"],
            "METADATA_INCOMPLETE",
        )

    def test_ranked_public_sections_repeats_ties_and_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            runs.mkdir()
            registry = self._registry(root)

            def private_runtime_details(value: dict[str, Any]) -> None:
                value["model"]["endpoint"] = "http://203.0.113.99:11434/v1"
                value["provenance"]["host"] = {
                    "hostname": "controller.example.invalid",
                    "machine": "x86_64",
                }
                value["components"]["private_payload"] = {
                    "path": "/home/private/raw.json",
                    "prompt": "PRIVATE_PROMPT_SENTINEL",
                    "response": "PRIVATE_RESPONSE_SENTINEL",
                }

            self._install(
                runs,
                "qualified.json",
                "qualified-a",
                mutate=private_runtime_details,
            )
            self._install(runs, "qualified.json", "qualified-b")
            self._install(runs, "not-qualified.json", "not-qualified")
            self._install(runs, "not-qualified.json", "not-qualified-repeat")

            def unknown_digest(value: dict[str, Any]) -> None:
                value["model"]["runtime_digest"] = "sha256:" + "c" * 64

            self._install(
                runs,
                "not-qualified.json",
                "metadata-incomplete",
                mutate=unknown_digest,
            )
            self._install(runs, "infra-error.json", "infra-error")

            def smoke(value: dict[str, Any]) -> None:
                value["profile"] = "smoke"
                value["provenance"]["profile_config"] = {
                    "name": "smoke",
                    "total_wall_seconds": 1800,
                }

            self._install(runs, "qualified.json", "smoke", mutate=smoke)

            def incompatible(value: dict[str, Any]) -> None:
                value["provenance"]["upstreams"]["spark-bench"]["commit"] = (
                    "9" * 40
                )

            self._install(
                runs, "qualified.json", "incompatible", mutate=incompatible
            )
            rows = collect_runs(runs, registry_path=registry)
            baseline = next(
                row for row in rows if row["run_id"] == "qualified-a"
            )["compatibility"]
            (
                _summary,
                _local_markdown,
                _local_json,
                leaderboard,
                public_markdown,
                public_json,
            ) = compare_runs(
                runs_root=runs,
                output_root=root / "local",
                public_output_root=root / "public",
                registry_path=registry,
                current_compatibility=baseline,
            )

            self.assertEqual(
                [row["run_id"] for row in leaderboard["qualified"]],
                ["qualified-a", "qualified-b"],
            )
            self.assertEqual(
                [row["rank"] for row in leaderboard["qualified"]], [1, 2]
            )
            self.assertEqual(
                [row["run_id"] for row in leaderboard["evaluated_not_qualified"]],
                ["not-qualified", "not-qualified-repeat"],
            )
            diagnostic_statuses = {
                row["run_id"]: row["status"]
                for row in leaderboard["diagnostics"]
            }
            self.assertEqual(diagnostic_statuses["infra-error"], "INFRA_ERROR")
            self.assertEqual(diagnostic_statuses["smoke"], "SMOKE_ONLY")
            self.assertEqual(diagnostic_statuses["incompatible"], "INCOMPATIBLE")
            self.assertEqual(
                diagnostic_statuses["metadata-incomplete"],
                "METADATA_INCOMPLETE",
            )
            public_text = public_markdown.read_text(encoding="utf-8")
            public_json_text = public_json.read_text(encoding="utf-8")
            for private_value in (
                "203.0.113.99",
                "controller.example.invalid",
                "/home/private",
                "PRIVATE_PROMPT_SENTINEL",
                "PRIVATE_RESPONSE_SENTINEL",
            ):
                self.assertNotIn(private_value, public_text)
                self.assertNotIn(private_value, public_json_text)

    def test_public_output_refuses_non_regular_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            public = root / "public"
            runs.mkdir()
            public.mkdir()
            (public / "leaderboard.json").mkdir()
            registry = self._registry(root)
            self._install(runs, "qualified.json", "qualified")
            baseline = collect_runs(runs, registry_path=registry)[0][
                "compatibility"
            ]
            with self.assertRaises(ArtifactSafetyError):
                compare_runs(
                    runs_root=runs,
                    output_root=root / "local",
                    public_output_root=public,
                    registry_path=registry,
                    current_compatibility=baseline,
                )


if __name__ == "__main__":
    unittest.main()
