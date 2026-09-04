from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml
import jsonschema

from harness import benchmark_model, hermes_runner
from harness.artifacts import ArtifactSafetyError, ensure_root, ensure_subdirectory
from harness.evaluator import (
    Check,
    _validate_worker_payload,
    evaluate_task,
)
from harness.evidence import (
    capture_baseline,
    capture_final,
    protected_input_integrity,
)
from harness.hermes_runner import (
    HERMES_DISTRIBUTION,
    HERMES_PY,
    HERMES_TOOLSET,
    _hermes_environment,
    agent_execution_validity,
    assess_change_scope,
    batch_command,
    classify_outcome,
    cleanup,
    completion_claim,
    final_response,
    hermes_transport_infrastructure_reasons,
    load_trajectory,
    trajectory_artifact,
    score_evaluation,
    task_result_exit_code,
    validate_result,
    write_runtime_hermes_config,
    write_runtime_http_policy,
)
from harness.processes import run_process_group
from harness.sandbox import build_command, run as sandbox_run, write_shell_wrapper
from harness.workspace import ROOT, content_digest, load_task, prepare


DIGEST = "sha256:" + "a" * 64
HOST_HOME = Path.home().resolve()
HOST_SSH = HOST_HOME / ".ssh"
HOST_CODEX = HOST_HOME / ".codex"
HOST_HERMES = HOST_HOME / ".hermes"


class ConfigurationTests(unittest.TestCase):
    def test_hermes_reasoning_off_uses_agent_config_and_disabled_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            written = write_runtime_hermes_config(
                runtime,
                model="fixture",
                base_url=benchmark_model.EXPECTED_ENDPOINT,
                context_length=1024,
                reasoning="none",
                max_turns=2,
            )
            config = yaml.safe_load(
                Path(written["config_path"]).read_text(encoding="utf-8")
            )
            command = batch_command(
                runtime_dir=runtime,
                model="fixture",
                reasoning="none",
                max_turns=2,
                base_url=benchmark_model.EXPECTED_ENDPOINT,
            )
        self.assertEqual(config["agent"]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", config["model"])
        self.assertIn("--reasoning_disabled", command)
        self.assertFalse(any(value.startswith("--reasoning_effort=") for value in command))

    def test_known_model_alias_has_frozen_live_identity(self) -> None:
        plan = benchmark_model.load_execution_plan("qwen38-q8-medium-128k")
        self.assertEqual(plan["model"]["runtime_model"], "qwen38-q8-262k:latest")
        self.assertEqual(
            plan["model"]["runtime_digest"],
            "sha256:4ab95509a27d7a3f23dcc612a660858e9f28c1a5322bd9240f34559bcf888988",
        )
        self.assertEqual(plan["benchmark"]["generation"], "hermesbench-v3")

    def test_unknown_model_alias(self) -> None:
        with self.assertRaisesRegex(
            benchmark_model.ConfigurationError, "unknown model alias"
        ):
            benchmark_model.load_execution_plan("does-not-exist")

    def test_runtime_discovered_model_can_use_in_memory_plan_override(self) -> None:
        override = dict(
            benchmark_model.load_models()["qwen38-q8-medium-128k"]
        )
        override["runtime_model"] = "runtime-only:latest"
        override["runtime_digest"] = DIGEST
        override["context_length"] = 65536
        plan = benchmark_model.load_execution_plan(
            "runtime-only:latest",
            model_override=override,
        )
        self.assertEqual(plan["model_alias"], "runtime-only:latest")
        self.assertEqual(plan["model"]["runtime_model"], "runtime-only:latest")
        self.assertEqual(plan["model"]["runtime_digest"], DIGEST)
        self.assertEqual(plan["model"]["context_length"], 65536)

    def test_generation_v1_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            benchmark_model.ConfigurationError, "schema_version must be 2"
        ):
            benchmark_model.load_execution_plan(
                "qwen38-q8-medium-128k",
                config_path=ROOT / "configs" / "hermes-bench-v1.yaml",
            )

    def test_endpoint_must_be_exact(self) -> None:
        for value in (
            "http://10.23.45.67:11434",
            "http://10.23.45.67:11434/v1/extra",
            "https://10.23.45.67:11434/v1",
            "http://user@10.23.45.67:11434/v1",
            "http://10.23.45.67:11434/v1?redirect=yes",
            "http://10.23.45.67:11434/v1#fragment",
            "http://8.8.8.8:11434/v1",
        ):
            with self.subTest(value=value):
                with self.assertRaises(benchmark_model.ConfigurationError):
                    benchmark_model._validate_endpoint(value)

    def test_candidate_tool_policy_must_be_terminal_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = yaml.safe_load(
                (ROOT / "configs" / "hermes-bench-v3.yaml").read_text(
                    encoding="utf-8"
                )
            )
            config["execution"]["candidate_toolsets"] = ["terminal", "file"]
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                benchmark_model.ConfigurationError, "only the gated foreground"
            ):
                benchmark_model.load_execution_plan(
                    "qwen38-q8-medium-128k",
                    config_path=config_path,
                )

    def test_dry_run_does_not_execute_or_contact_model(self) -> None:
        plan = benchmark_model.load_execution_plan("qwen38-q8-medium-128k")

        with (
            mock.patch.object(
                benchmark_model, "load_execution_plan", return_value=plan
            ),
            mock.patch.object(benchmark_model, "execute_suite") as execute_suite,
            mock.patch.object(benchmark_model, "preflight_model") as preflight,
        ):
            returncode = benchmark_model.main(
                ["--model", "qwen38-q8-medium-128k", "--dry-run"]
            )

        self.assertEqual(returncode, 0)
        execute_suite.assert_not_called()
        preflight.assert_not_called()

    def test_preflight_resolves_digest_when_expected_is_unconfigured(self) -> None:
        responses = [
            {"data": [{"id": "test:latest", "object": "model"}]},
            {"models": [{"name": "test:latest", "digest": "a" * 64}]},
            {"details": {"format": "gguf"}},
        ]

        with mock.patch.object(
            benchmark_model, "_endpoint_json", side_effect=responses
        ):
            result = benchmark_model.preflight_model(
                benchmark_model.EXPECTED_ENDPOINT,
                "test:latest",
                expected_digest=None,
            )

        self.assertEqual(result["runtime_model_digest"], DIGEST)
        self.assertEqual(result["expected_digest_status"], "NOT_CONFIGURED")

    def test_preflight_requires_exact_unique_digest(self) -> None:
        responses = [
            {"data": [{"id": "test:latest", "object": "model"}]},
            {
                "models": [
                    {
                        "name": "test:latest",
                        "digest": "a" * 64,
                    }
                ]
            },
            {"details": {"format": "gguf"}},
        ]

        with mock.patch.object(
            benchmark_model, "_endpoint_json", side_effect=responses
        ):
            result = benchmark_model.preflight_model(
                benchmark_model.EXPECTED_ENDPOINT,
                "test:latest",
                expected_digest=DIGEST,
            )

        self.assertEqual(result["runtime_model_digest"], DIGEST)
        self.assertEqual(result["identity_status"], "VERIFIED")
        self.assertEqual(result["expected_digest_status"], "MATCHED")

    def test_preflight_rejects_missing_ambiguous_and_mismatched_identity(self) -> None:
        cases = (
            ([{"id": "other"}], [], "missing or ambiguous"),
            (
                [{"id": "test:latest"}, {"id": "test:latest"}],
                [],
                "missing or ambiguous",
            ),
            (
                [{"id": "test:latest"}],
                [{"name": "test:latest", "digest": "b" * 64}],
                "digest mismatch",
            ),
        )

        for models, tags, message in cases:
            with self.subTest(message=message):
                with mock.patch.object(
                    benchmark_model,
                    "_endpoint_json",
                    side_effect=[{"data": models}, {"models": tags}],
                ):
                    with self.assertRaisesRegex(
                        benchmark_model.InfrastructureError, message
                    ):
                        benchmark_model.preflight_model(
                            benchmark_model.EXPECTED_ENDPOINT,
                            "test:latest",
                            expected_digest=DIGEST,
                        )

    def test_preflight_rejects_ambiguous_alias_and_invalid_show_metadata(self) -> None:
        cases = (
            (
                [
                    {"name": "test:latest", "digest": "a" * 64},
                    {"model": "test:latest", "digest": "a" * 64},
                ],
                {"details": {}},
                "missing or ambiguous",
            ),
            (
                [
                    {
                        "name": "test:latest",
                        "model": "other:latest",
                        "digest": "a" * 64,
                    }
                ],
                {"details": {}},
                "alias/identifier mismatch",
            ),
            (
                [{"name": "test:latest", "digest": "a" * 64}],
                {},
                "invalid /api/show",
            ),
            (
                [{"name": "test:latest", "digest": "a" * 64}],
                {"model": "other:latest", "details": {}},
                "/api/show alias/identifier mismatch",
            ),
            (
                [{"name": "test:latest", "digest": 7}],
                {"details": {}},
                "malformed model digest",
            ),
            (
                [
                    {"name": "test:latest", "digest": "a" * 64},
                    {"name": "test:latest"},
                ],
                {"details": {}},
                "missing or ambiguous",
            ),
        )

        for tags, shown, message in cases:
            with self.subTest(message=message):
                with mock.patch.object(
                    benchmark_model,
                    "_endpoint_json",
                    side_effect=[
                        {"data": [{"id": "test:latest"}]},
                        {"models": tags},
                        shown,
                    ],
                ):
                    with self.assertRaisesRegex(
                        benchmark_model.InfrastructureError, message
                    ):
                        benchmark_model.preflight_model(
                            benchmark_model.EXPECTED_ENDPOINT,
                            "test:latest",
                            expected_digest=None,
                        )

    def test_redirect_is_rejected(self) -> None:
        response = mock.MagicMock()
        response.geturl.return_value = "http://elsewhere.invalid/v1/models"
        response.__enter__.return_value = response
        opener = mock.MagicMock()
        opener.open.return_value = response

        with mock.patch.object(benchmark_model, "build_opener", return_value=opener):
            with self.assertRaisesRegex(
                benchmark_model.InfrastructureError, "redirected"
            ):
                benchmark_model._endpoint_json(
                    benchmark_model.EXPECTED_ENDPOINT + "/models", timeout=1
                )

    def test_low_level_live_runner_cannot_bypass_verified_identity(self) -> None:
        from harness.hermes_runner import run_once

        with self.assertRaisesRegex(RuntimeError, "preflight-verified"):
            run_once(
                task_id="archiveguard-security-v1",
                model="test:latest",
                reasoning="medium",
                max_turns=1,
                model_metadata={"context_length": 1024},
            )

        for base_url in (
            "http://user@10.23.45.67:11434/v1",
            "http://10.23.45.67:11434/v1?redirect=yes",
            "http://8.8.8.8:11434/v1",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(RuntimeError, "non-local"):
                    run_once(
                        task_id="archiveguard-security-v1",
                        model="test:latest",
                        reasoning="medium",
                        max_turns=1,
                        base_url=base_url,
                        model_metadata={
                            "context_length": 1024,
                            "runtime_digest": DIGEST,
                            "runtime_identity_status": "VERIFIED",
                        },
                    )

    def test_low_level_runner_rechecks_identity_before_workspace_creation(self) -> None:
        from harness.hermes_runner import run_once

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            runtime = root / "runtime"

            with (
                mock.patch.object(hermes_runner, "WORK_ROOT", work),
                mock.patch.object(hermes_runner, "RUNTIME_ROOT", runtime),
                mock.patch.object(
                    benchmark_model,
                    "preflight_model",
                    side_effect=benchmark_model.InfrastructureError(
                        "runtime digest mismatch"
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    benchmark_model.InfrastructureError,
                    "runtime digest mismatch",
                ):
                    run_once(
                        task_id="archiveguard-security-v1",
                        model="test:latest",
                        reasoning="medium",
                        max_turns=1,
                        model_metadata={
                            "context_length": 1024,
                            "runtime_digest": DIGEST,
                            "runtime_identity_status": "VERIFIED",
                        },
                    )

            self.assertFalse(work.exists())
            self.assertFalse(runtime.exists())


class ArtifactSafetyTests(unittest.TestCase):
    def test_atomic_output_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.write_text("original", encoding="utf-8")
            output_root = root / "results"
            output_root.mkdir()
            (output_root / "result.json").symlink_to(outside)

            with self.assertRaisesRegex(
                ArtifactSafetyError, "non-regular artifact"
            ):
                benchmark_model._atomic_text(
                    output_root / "result.json", "replacement", root=output_root
                )

            self.assertEqual(outside.read_text(encoding="utf-8"), "original")

    def test_atomic_output_rejects_symlink_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            linked = root / "linked"
            linked.symlink_to(real, target_is_directory=True)

            with self.assertRaises(ArtifactSafetyError):
                benchmark_model._atomic_text(
                    linked / "result.json", "unsafe", root=linked
                )

            self.assertFalse((real / "result.json").exists())

    def test_aggregate_runtime_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))
            runtime = root / "runtime"
            runtime.mkdir()
            outside = root / "outside"
            outside.write_text("original", encoding="utf-8")
            fixed = "fixed-run"
            (runtime / f"{fixed}.aggregate.json").symlink_to(outside)

            with mock.patch.object(benchmark_model, "new_run_id", return_value=fixed):
                with self.assertRaises(ArtifactSafetyError):
                    benchmark_model.execute_suite(
                        plan,
                        preflight=fake_preflight,
                        task_runner=mock.Mock(),
                        results_dir=root / "results",
                        reports_dir=root / "reports",
                        runtime_root=runtime,
                        progress=lambda _message: None,
                    )

            self.assertEqual(outside.read_text(encoding="utf-8"), "original")


class SandboxTests(unittest.TestCase):
    def test_suite_maps_native_policy_to_no_hermes_effort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))
            plan["model"]["reasoning_policy"] = "native"
            plan["model"]["reasoning_effort"] = "native"
            observed: dict[str, object] = {}

            def runner(**kwargs: object) -> dict:
                observed.update(kwargs)
                return make_result(plan, "task-a", "PASS", 100.0)

            benchmark_model.execute_suite(
                plan,
                preflight=fake_preflight,
                task_runner=runner,
                results_dir=root / "results",
                reports_dir=root / "reports",
                runtime_root=root / "runtime",
                progress=lambda _message: None,
            )

        self.assertIsNone(observed["reasoning"])
        self.assertEqual(observed["reasoning_policy"], "native")

    def test_command_uses_minimal_networkless_filesystem_and_fixed_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            runtime = root / "runtime"
            candidate.mkdir()
            runtime.mkdir()
            command = build_command(candidate, runtime, ["/bin/true"])

        self.assertIn("--unshare-all", command)
        self.assertIn("--clearenv", command)
        self.assertIn("PYTHONDONTWRITEBYTECODE", command)
        self.assertNotIn("/", [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--ro-bind"
        ])
        self.assertNotIn(str(runtime), command)
        self.assertNotIn(str(HOST_HERMES), command)
        self.assertNotIn(str(HOST_SSH), command)
        self.assertNotIn(str(HOST_CODEX), command)

    def test_caller_environment_cannot_be_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            runtime = root / "runtime"
            candidate.mkdir()
            runtime.mkdir()

            with self.assertRaisesRegex(Exception, "unsupported keys"):
                build_command(
                    candidate,
                    runtime,
                    ["/bin/true"],
                    env={"SECRET_TOKEN": "leak"},
                )

    def test_shell_wrapper_is_harness_owned_and_does_not_mount_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            runtime = root / "runtime"
            candidate.mkdir()
            runtime.mkdir()
            wrapper = write_shell_wrapper(runtime, candidate)
            text = wrapper.read_text(encoding="utf-8")
            wrapper_mode = wrapper.stat().st_mode & 0o777

        self.assertIn("--unshare-all", text)
        self.assertIn("--clearenv", text)
        self.assertNotIn(f"--bind {runtime}", text)
        self.assertIn('"$@"', text)
        self.assertNotIn("BASH_ENV", text)
        self.assertEqual(wrapper_mode, 0o700)

    def test_actual_candidate_cannot_see_host_runtime_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            runtime = root / "runtime"
            candidate.mkdir()
            runtime.mkdir()
            script = f'''
set -eu
test ! -e {shlex.quote(str(HOST_SSH))}
test ! -e {shlex.quote(str(HOST_CODEX))}
test ! -e {shlex.quote(str(ROOT))}
test ! -e {shlex.quote(str(hermes_runner.RUNTIME_ROOT))}
test ! -e /sys
test "$(wc -l < /proc/net/route)" -le 1
printf ok > candidate-write
env
'''
            process = sandbox_run(
                candidate,
                runtime,
                ["/bin/sh", "-c", script],
                timeout=10,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            self.assertEqual(
                (candidate / "candidate-write").read_text(encoding="utf-8"),
                "ok",
            )
            self.assertNotIn("SECRET", process.stdout)

    def test_actual_hermes_terminal_shell_is_forced_through_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            runtime = root / "runtime"
            candidate.mkdir()
            runtime.mkdir()
            candidate_tool_tmp = candidate / ".git" / "hermesbench-tmp"
            candidate_tool_tmp.mkdir(parents=True)
            config = write_runtime_hermes_config(
                runtime,
                model="dry-run-control",
                base_url=benchmark_model.EXPECTED_ENDPOINT,
                context_length=1024,
                reasoning="medium",
                max_turns=2,
            )
            wrapper = write_shell_wrapper(runtime, candidate)
            http_policy = write_runtime_http_policy(
                runtime,
                shell_wrapper=wrapper,
            )
            command_text = (
                "set -eu; "
                "test -z \"${BASH_ENV:-}\"; "
                f"test ! -e {shlex.quote(str(HOST_SSH))}; "
                f"test ! -e {shlex.quote(str(HOST_CODEX))}; "
                f"test ! -e {shlex.quote(str(hermes_runner.RUNTIME_ROOT))}; "
                "test \"$(wc -l < /proc/net/route)\" -le 1; "
                "env -u BASH_ENV /bin/sh -c "
                f"'test ! -e {str(HOST_SSH)}'; "
                "/bin/bash --noprofile --norc -c "
                f"'test ! -e {str(HOST_CODEX)}'; "
                "/usr/bin/python3 -c \"import pathlib,subprocess; "
                f"assert not pathlib.Path({str(HOST_SSH)!r}).exists(); "
                "assert subprocess.run(['/bin/sh','-c',"
                f"'test ! -e {str(ROOT)}']).returncode == 0\"; "
                f"setsid /bin/sh -c 'test ! -e {str(HOST_CODEX)}'; "
                "printf wrapped > hermes-wrapper-write"
            )
            program = (
                "import json; "
                "from openai._base_client import _DefaultHttpxClient; "
                "from toolset_distributions import ("
                "get_distribution, sample_toolsets_from_distribution); "
                "from toolsets import resolve_toolset; "
                "from tools.environments.local import LocalEnvironment; "
                "from tools.environments import local as local_environment; "
                "from tools.registry import registry; "
                "import socket, unittest; "
                "unittest.TestCase().assertRaises("
                "OSError, socket.getaddrinfo, 'openrouter.ai', 443); "
                "unittest.TestCase().assertRaises("
                "OSError, lambda: socket.socket().connect("
                "('203.0.113.1', 443))); "
                "client=_DefaultHttpxClient(); "
                "assert client.follow_redirects is False; client.close(); "
                f"assert get_distribution({HERMES_DISTRIBUTION!r})"
                f"['toolsets'] == {{{HERMES_TOOLSET!r}: 100}}; "
                f"assert sample_toolsets_from_distribution("
                f"{HERMES_DISTRIBUTION!r}) == [{HERMES_TOOLSET!r}]; "
                f"assert resolve_toolset({HERMES_TOOLSET!r}) == ['terminal']; "
                f"assert local_environment._find_bash() == {str(wrapper)!r}; "
                "entry=registry.get_entry('terminal'); "
                "assert 'background' not in entry.schema['parameters']['properties']; "
                "assert 'process' not in resolve_toolset("
                f"{HERMES_TOOLSET!r}); "
                "blocked=json.loads(entry.handler("
                "{'command': 'true', 'background': True})); "
                "assert blocked['status'] == 'blocked'; "
                "blocked_pty=json.loads(entry.handler("
                "{'command': 'true', 'pty': True})); "
                "assert blocked_pty['status'] == 'blocked'; "
                f"environment=LocalEnvironment(cwd={str(candidate)!r}, timeout=10); "
                "first=environment.execute('export HERMES_PERSIST_PROBE=yes'); "
                "assert first['returncode'] == 0; "
                "persisted=environment.execute("
                "'test \"$HERMES_PERSIST_PROBE\" = yes'); "
                "assert persisted['returncode'] == 0; "
                "ordinary=environment.execute("
                "\"printf 'before\\n' > tracked.txt; "
                "git init -b main >/dev/null; git add tracked.txt; "
                "git -c user.name=Smoke -c user.email=smoke@invalid "
                "commit -m baseline >/dev/null; "
                "printf 'after\\n' > tracked.txt; printf 'new\\n' > untracked.txt; "
                "git diff --check; git status --porcelain; "
                "python3 -c 'assert open(\\\"tracked.txt\\\").read() == "
                "\\\"after\\\\n\\\"'\"); "
                "assert ordinary['returncode'] == 0; "
                "assert 'tracked.txt' in ordinary['output']; "
                "assert 'untracked.txt' in ordinary['output']; "
                "streams=environment.execute("
                "\"printf stdout-ok; printf stderr-ok >&2; exit 7\"); "
                "assert streams['returncode'] == 7; "
                "assert 'stdout-ok' in streams['output']; "
                "assert 'stderr-ok' in streams['output']; "
                f"print(json.dumps(environment.execute({command_text!r})))"
            )
            process = run_process_group(
                [str(HERMES_PY), "-c", program],
                cwd=runtime,
                env=_hermes_environment(
                    config,
                    http_policy,
                    candidate_tool_tmp,
                    candidate,
                ),
                timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            wrapper_write = (candidate / "hermes-wrapper-write").read_text(
                encoding="utf-8"
            )
            time.sleep(1)

        self.assertIn('"returncode": 0', process.stdout)
        self.assertEqual(wrapper_write, "wrapped")


class EvaluatorTests(unittest.TestCase):
    def _prepared_candidate(self, root: Path) -> tuple[dict, Path, dict, Path]:
        metadata = prepare("archiveguard-security-v1", root / "work")
        candidate = Path(metadata["candidate"])
        task, task_dir = load_task("archiveguard-security-v1")
        return metadata, candidate, task, task_dir

    def test_pristine_baseline_scores_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertEqual((result["public"]["passed"], result["public"]["total"]), (5, 5))
        self.assertEqual((result["hidden"]["passed"], result["hidden"]["total"]), (15, 34))
        self.assertEqual(
            (result["supplemental"]["passed"], result["supplemental"]["total"]),
            (2, 7),
        )
        self.assertFalse(result["pass"])

    def test_forged_stdout_counts_do_not_change_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            init = candidate / "archiveguard" / "__init__.py"
            original = init.read_text(encoding="utf-8")
            init.write_text(
                'print("Acceptance tests: 999/1")\n'
                'print("Acceptance tests: 34/34")\n'
                'print("5 passed")\n'
                + original,
                encoding="utf-8",
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertEqual((result["hidden"]["passed"], result["hidden"]["total"]), (15, 34))
        self.assertNotEqual(result["hidden"]["returncode"], 0)

    def test_candidate_cannot_forge_harness_structured_result_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            init = candidate / "archiveguard" / "__init__.py"
            original = init.read_text(encoding="utf-8")
            init.write_text(
                "import json as _json, os as _os\n"
                "_payload = (_json.dumps({'schema_version': 1, "
                "'discovered': 999, 'successful': True, 'cases': []}) + "
                "'\\n').encode()\n"
                "_fd = _os.environ.get('HERMES_BENCH_RESULT_FD')\n"
                "if _fd:\n"
                "    try: _os.write(int(_fd), _payload)\n"
                "    except OSError: pass\n"
                "try:\n"
                "    _parent_fds = _os.listdir(f'/proc/{_os.getppid()}/fd')\n"
                "except OSError:\n"
                "    _parent_fds = []\n"
                "for _candidate_fd in _parent_fds:\n"
                "    try:\n"
                "        _probe = _os.open(f'/proc/{_os.getppid()}/fd/{_candidate_fd}', "
                "_os.O_WRONLY)\n"
                "        _os.write(_probe, _payload)\n"
                "        _os.close(_probe)\n"
                "    except OSError:\n"
                "        pass\n"
                + original,
                encoding="utf-8",
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertFalse(result["infrastructure_errors"])
        self.assertEqual(result["hidden"]["total"], 34)
        self.assertLess(result["hidden"]["passed"], result["hidden"]["total"])
        self.assertFalse(result["pass"])

    def test_candidate_early_success_exit_cannot_establish_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            (candidate / "archiveguard" / "__init__.py").write_text(
                "import os\nos._exit(0)\n", encoding="utf-8"
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertFalse(result["infrastructure_errors"])
        self.assertFalse(result["pass"])
        for name in ("public", "hidden", "supplemental"):
            self.assertEqual(result[name]["passed"], 0)
            self.assertTrue(
                all(case["status"] == "ERROR" for case in result[name]["cases"])
            )

    def test_candidate_cannot_see_private_tests_or_result_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            init = candidate / "archiveguard" / "__init__.py"
            original = init.read_text(encoding="utf-8")
            init.write_text(
                "import os as _os\n"
                "from pathlib import Path as _Path\n"
                "assert not _Path('/tests').exists()\n"
                f"assert not _Path({str(ROOT)!r}).exists()\n"
                "assert 'HERMES_BENCH_RESULT_FD' not in _os.environ\n"
                + original,
                encoding="utf-8",
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertFalse(result["infrastructure_errors"])
        self.assertEqual((result["hidden"]["passed"], result["hidden"]["total"]), (15, 34))

    def test_frozen_evaluator_worker_digest_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))

            with self.assertRaisesRegex(ValueError, "worker digest mismatch"):
                evaluate_task(
                    candidate,
                    task,
                    task_dir,
                    timeout=30,
                    expected_worker_sha256="0" * 64,
                )

    def test_non_scored_smoke_fixture_uses_production_workspace_and_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata = prepare("harness-smoke-v1", Path(temporary) / "work")
            candidate = Path(metadata["candidate"])
            (candidate / "smoke_project.py").write_text(
                'STATUS = "READY"\n', encoding="utf-8"
            )
            (candidate / "SMOKE_NOTES.txt").write_text(
                "Hermes smoke completed.\n", encoding="utf-8"
            )
            task, task_dir = load_task("harness-smoke-v1")
            self.assertEqual(task["limits"]["wall_seconds"], 900)
            worker_digest = hashlib.sha256(
                (ROOT / "harness" / "evaluator_worker.py").read_bytes()
            ).hexdigest()
            result = evaluate_task(
                candidate,
                task,
                task_dir,
                timeout=30,
                expected_worker_sha256=worker_digest,
                expected_rpc_sha256=hashlib.sha256(
                    (ROOT / "harness" / "candidate_rpc.py").read_bytes()
                ).hexdigest(),
            )

        self.assertTrue(result["pass"])
        self.assertEqual(
            set(metadata["protected_inputs"]),
            {"BENCHMARK_TASK.md", "tests/test_public.py"},
        )
        self.assertEqual(result["isolation"]["worker_digest_status"], "VERIFIED")

    def test_candidate_import_error_is_a_model_failure_not_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            (candidate / "archiveguard" / "__init__.py").write_text(
                "this is not valid Python !!!\n", encoding="utf-8"
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)

        self.assertFalse(result["pass"])
        self.assertFalse(result["infrastructure_errors"])
        for name in ("public", "hidden", "supplemental"):
            self.assertEqual(result[name]["returncode"], 1)
            self.assertEqual(result[name]["passed"], 0)
            self.assertTrue(
                all(case["status"] == "ERROR" for case in result[name]["cases"])
            )

    def test_structured_result_rejects_malformed_duplicate_missing_and_conflict(self) -> None:
        check = {
            "cases": {
                "test_a": {"categories": ["correctness"], "requirements": ["r"]},
                "test_b": {"categories": ["security"], "requirements": ["r"]},
            }
        }
        payloads = (
            (b"not-json", 1, "malformed"),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 2,
                        "successful": True,
                        "cases": [
                            {"id": "test_a", "status": "PASS"},
                            {"id": "test_a", "status": "PASS"},
                        ],
                    }
                ).encode(),
                0,
                "duplicate",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 1,
                        "successful": True,
                        "cases": [{"id": "test_a", "status": "PASS"}],
                    }
                ).encode(),
                0,
                "identities/count",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 2,
                        "successful": True,
                        "cases": [
                            {"id": "test_a", "status": "PASS"},
                            {"id": "test_b", "status": "PASS"},
                        ],
                    }
                ).encode(),
                1,
                "command status conflicts",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 2,
                        "successful": True,
                        "passed": 999,
                        "total": 1,
                        "cases": [
                            {"id": "test_a", "status": "PASS"},
                            {"id": "test_b", "status": "PASS"},
                        ],
                    }
                ).encode(),
                0,
                "unknown fields",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 2,
                        "cases": [
                            {"id": "test_a", "status": "PASS"},
                            {"id": "test_b", "status": "PASS"},
                        ],
                    }
                ).encode(),
                0,
                "missing or unknown fields",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": True,
                        "successful": True,
                        "cases": [
                            {"id": "test_a", "status": "PASS"},
                            {"id": "test_b", "status": "PASS"},
                        ],
                    }
                ).encode(),
                0,
                "invalid case data",
            ),
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "discovered": 2,
                        "successful": True,
                        "cases": [
                            {"id": "test_a", "status": "PASS", "detail": []},
                            {"id": "test_b", "status": "PASS"},
                        ],
                    }
                ).encode(),
                0,
                "invalid detail",
            ),
        )

        for payload, returncode, expected in payloads:
            with self.subTest(expected=expected):
                _, error = _validate_worker_payload(
                    payload, check=check, returncode=returncode
                )
                self.assertIn(expected, error or "")

    def test_public_test_replacement_is_ignored_and_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, candidate, task, task_dir = self._prepared_candidate(
                Path(temporary)
            )
            (candidate / "tests" / "test_public.py").write_text(
                "def test_trivial(): assert True\n", encoding="utf-8"
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)
            integrity = protected_input_integrity(
                candidate, metadata["protected_inputs"]
            )

        self.assertEqual((result["public"]["passed"], result["public"]["total"]), (5, 5))
        self.assertFalse(integrity["pass"])

    def test_public_test_parent_symlink_is_detected_even_with_same_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, candidate, _, _ = self._prepared_candidate(root)
            runtime_root = ensure_root(root / "runtime")
            runtime = ensure_subdirectory(
                runtime_root, runtime_root / "run", exclusive=True
            )
            baseline = capture_baseline(candidate, runtime)
            original_tests = candidate / "tests"
            outside_tests = root / "outside-tests"
            outside_tests.mkdir()
            (outside_tests / "test_public.py").write_bytes(
                (original_tests / "test_public.py").read_bytes()
            )
            original_tests.rename(candidate / "tests-original")
            original_tests.symlink_to(outside_tests, target_is_directory=True)
            integrity = protected_input_integrity(
                candidate, metadata["protected_inputs"]
            )
            evidence = capture_final(candidate, runtime, baseline)

            with tarfile.open(Path(evidence["archive"]), "r:gz") as archive:
                archived_tests = archive.getmember("candidate/tests")

        self.assertFalse(integrity["pass"])
        self.assertIn(
            "UNSAFE_OR_MISSING",
            integrity["items"]["tests/test_public.py"]["status"],
        )
        self.assertIn("tests", evidence["changed_files"])
        self.assertTrue(archived_tests.issym())

    def test_private_input_cannot_be_copied_into_read_only_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, candidate, task, task_dir = self._prepared_candidate(Path(temporary))
            init = candidate / "archiveguard" / "__init__.py"
            original = init.read_text(encoding="utf-8")
            init.write_text(
                "from pathlib import Path\n"
                "try:\n"
                "    Path('stolen-private-test.py').write_text("
                "Path('/tests/hidden.py').read_text())\n"
                "except OSError:\n"
                "    pass\n"
                + original,
                encoding="utf-8",
            )
            result = evaluate_task(candidate, task, task_dir, timeout=30)
            stolen = candidate / "stolen-private-test.py"

        self.assertFalse(stolen.exists())
        self.assertFalse(result["infrastructure_errors"])

    def test_evaluator_host_network_env_and_write_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            task_dir = root / "task"
            candidate.mkdir()
            task_dir.mkdir()
            (candidate / "probe.py").write_text(
                "import os\n"
                "from pathlib import Path\n"
                "def host_hidden():\n"
                f"    return (not Path({str(HOST_SSH)!r}).exists() "
                f"and not Path({str(HOST_CODEX)!r}).exists())\n"
                "def network_empty():\n"
                "    return len(Path('/proc/net/route').read_text().splitlines()) <= 1\n"
                "def candidate_read_only():\n"
                "    try:\n"
                "        Path(__file__).with_name('write').write_text('x')\n"
                "    except OSError:\n"
                "        return True\n"
                "    return False\n",
                encoding="utf-8",
            )
            test_file = task_dir / "boundary.py"
            test_file.write_text(
                "import unittest\n"
                "from probe import candidate_read_only, host_hidden, network_empty\n"
                "class Boundary(unittest.TestCase):\n"
                "    def test_host_hidden(self):\n"
                "        self.assertTrue(host_hidden())\n"
                "    def test_network_empty(self):\n"
                "        self.assertTrue(network_empty())\n"
                "    def test_candidate_read_only(self):\n"
                "        self.assertTrue(candidate_read_only())\n",
                encoding="utf-8",
            )
            cases = {
                identifier: {
                    "categories": ["correctness"],
                    "requirements": ["boundary"],
                }
                for identifier in (
                    "test_host_hidden",
                    "test_network_empty",
                    "test_candidate_read_only",
                )
            }
            task = {
                "id": "boundary",
                "category": "coding",
                "requirements": [{"id": "boundary", "text": "boundary"}],
                "evaluation_checks": [
                    {
                        "id": "boundary",
                        "visibility": "harness-controlled",
                        "test_file": "boundary.py",
                        "sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
                        "cases": cases,
                    }
                ],
            }
            result = evaluate_task(candidate, task, task_dir, timeout=20)

        self.assertTrue(result["pass"], result)

    def test_evaluator_timeout_kills_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            task_dir = root / "task"
            candidate.mkdir()
            task_dir.mkdir()
            (candidate / "probe.py").write_text(
                "import time\n"
                "def hang(): time.sleep(300)\n",
                encoding="utf-8",
            )
            test_file = task_dir / "timeout.py"
            test_file.write_text(
                "import unittest\n"
                "from probe import hang\n"
                "class Timeout(unittest.TestCase):\n"
                "    def test_hang(self):\n"
                "        hang()\n",
                encoding="utf-8",
            )
            task = {
                "id": "timeout",
                "category": "coding",
                "requirements": [{"id": "timeout", "text": "timeout"}],
                "evaluation_checks": [
                    {
                        "id": "timeout",
                        "visibility": "harness-controlled",
                        "test_file": "timeout.py",
                        "sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
                        "cases": {
                            "test_hang": {
                                "categories": ["correctness"],
                                "requirements": ["timeout"],
                            }
                        },
                    }
                ],
            }
            result = evaluate_task(candidate, task, task_dir, timeout=1)

        self.assertTrue(result["timed_out"])
        self.assertFalse(result["pass"])

    def test_large_structured_failure_does_not_deadlock_result_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            task_dir = root / "task"
            candidate.mkdir()
            task_dir.mkdir()
            (candidate / "probe.py").write_text("VALUE = True\n", encoding="utf-8")
            test_file = task_dir / "large_failure.py"
            test_file.write_text(
                "import unittest\n"
                "class LargeFailure(unittest.TestCase):\n"
                + "".join(
                    f"    def test_large_failure_{index}(self):\n"
                    "        self.fail('x' * 200000)\n"
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            expected_cases = {
                f"test_large_failure_{index}": {
                    "categories": ["correctness"],
                    "requirements": ["failure"],
                }
                for index in range(10)
            }
            task = {
                "id": "large-failure",
                "category": "coding",
                "requirements": [
                    {"id": "failure", "text": "failure is reported"}
                ],
                "evaluation_checks": [
                    {
                        "id": "large-failure",
                        "visibility": "harness-controlled",
                        "test_file": "large_failure.py",
                        "sha256": hashlib.sha256(
                            test_file.read_bytes()
                        ).hexdigest(),
                        "cases": expected_cases,
                    }
                ],
            }
            result = evaluate_task(candidate, task, task_dir, timeout=10)

        self.assertFalse(result["timed_out"])
        self.assertFalse(result["infrastructure_errors"])
        self.assertEqual(result["large-failure"]["returncode"], 1)
        self.assertEqual(result["large-failure"]["total"], 10)


class ScoringAndEvidenceTests(unittest.TestCase):
    def test_change_scope_rejects_unrelated_and_generated_files(self) -> None:
        accepted = assess_change_scope(
            ["policy.py", "RUN_NOTES.md"],
            ["policy.py", "RUN_NOTES.md"],
        )
        rejected = assess_change_scope(
            ["archiveguard/policy.py", "tests/__init__.py", "__pycache__/x.pyc"],
            ["archiveguard/*.py", "tests/test_*.py"],
        )
        self.assertTrue(accepted["pass"])
        self.assertFalse(rejected["pass"])
        self.assertEqual(
            rejected["unexpected_files"],
            ["__pycache__/x.pyc", "tests/__init__.py"],
        )

    def test_timeout_capture_normalizes_partial_bytes_and_preserves_output(self) -> None:
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            run_process_group(
                ["/bin/sh", "-c", "printf before-timeout; sleep 5"],
                timeout=0.1,
            )

        self.assertIsInstance(caught.exception.stdout, str)
        self.assertIn("before-timeout", caught.exception.stdout or "")

    def test_nonsecurity_unicode_case_contributes_to_correctness(self) -> None:
        evaluation = {
            "pass": False,
            "check": {
                "returncode": 1,
                "cases": [
                    {
                        "id": "unicode",
                        "status": "FAIL",
                        "categories": ["correctness"],
                        "requirements": ["unicode-length"],
                    },
                    {
                        "id": "traversal",
                        "status": "PASS",
                        "categories": ["correctness", "security"],
                        "requirements": ["traversal"],
                    },
                ],
            },
        }
        scores = score_evaluation(
            evaluation,
            task_requirements=[
                {"id": "unicode-length"},
                {"id": "traversal"},
            ],
            completion_claimed=False,
            false_completion=False,
        )
        self.assertEqual(scores["correctness"], 50.0)
        self.assertEqual(scores["security"], 100.0)
        self.assertEqual(scores["requirement_retention"], 50.0)

    def test_negated_completion_statements_are_not_claims(self) -> None:
        for text in (
            "I could not complete the task; tests fail.",
            "This is not complete.",
            "I didn't complete the task.",
            "The work isn't done.",
            "The task may be complete, but I am not sure.",
            "The implementation might be done.",
            "Work remains unresolved and the implementation is incomplete.",
            "Not done: one test is failing.",
        ):
            with self.subTest(text=text):
                self.assertFalse(completion_claim(text))

        self.assertTrue(completion_claim("Implementation complete. All tests passed."))

    def test_malformed_trajectory_is_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            trajectory = runtime / "data" / "hermes" / "trajectories.jsonl"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text("{not-json}\n", encoding="utf-8")
            loaded = load_trajectory(runtime)
            optional, status = trajectory_artifact(
                runtime,
                checkpoint=None,
                reasoning_policy="off",
            )

        self.assertIsNone(loaded)
        self.assertIsNone(optional)
        self.assertEqual(status, "MALFORMED")
        self.assertEqual(final_response({"conversations": {"bad": "shape"}}), "")

    def test_no_reasoning_training_discard_is_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            trajectory = runtime / "data" / "hermes" / "trajectories.jsonl"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text("", encoding="utf-8")
            loaded, status = trajectory_artifact(
                runtime,
                checkpoint={
                    "batch_stats": {
                        "0": {"discarded_no_reasoning": 1}
                    }
                },
                reasoning_policy="off",
            )
        self.assertIsNone(loaded)
        self.assertEqual(status, "DISCARDED_NO_REASONING")

    def test_agent_execution_evidence_does_not_require_trajectory_or_text_final(self) -> None:
        execution = {
            "process_completed": True,
            "checkpoint_completed": True,
            "batch_completion_logged": True,
            "final_text_response_observed": False,
            "tool_result_count_logged": 1,
        }
        transport = {
            "finish_state_returned": True,
            "visible_content_returned": False,
            "tool_call_returned": True,
        }
        valid = agent_execution_validity(
            dry_run=False,
            trajectory_identity_valid=False,
            transport_required=True,
            execution_evidence=execution,
            transport_evidence=transport,
            transport_trustworthy=True,
            task_state_captured=True,
            evaluator_infrastructure_errors=[],
        )
        self.assertTrue(valid)
        self.assertFalse(
            agent_execution_validity(
                dry_run=False,
                trajectory_identity_valid=False,
                transport_required=True,
                execution_evidence={**execution, "tool_result_count_logged": 0},
                transport_evidence=transport,
                transport_trustworthy=True,
                task_state_captured=True,
                evaluator_infrastructure_errors=[],
            )
        )
        for field in ("process_completed", "checkpoint_completed"):
            with self.subTest(field=field):
                broken = dict(execution)
                broken[field] = False
                self.assertFalse(
                    agent_execution_validity(
                        dry_run=False,
                        trajectory_identity_valid=False,
                        transport_required=True,
                        execution_evidence=broken,
                        transport_evidence=transport,
                        transport_trustworthy=True,
                        task_state_captured=True,
                        evaluator_infrastructure_errors=[],
                    )
                )
        self.assertFalse(
            agent_execution_validity(
                dry_run=False,
                trajectory_identity_valid=False,
                transport_required=True,
                execution_evidence=execution,
                transport_evidence=None,
                transport_trustworthy=False,
                task_state_captured=True,
                evaluator_infrastructure_errors=[],
            )
        )

    def test_runner_failure_and_timeout_transport_classification(self) -> None:
        completed_transport = {
            "observer_ok": True,
            "request_observer_ok": True,
            "request_count": 1,
            "response_count": 1,
            "requested_reasoning_policy": "off",
        }
        runner_failure = hermes_transport_infrastructure_reasons(
            runner_exit_code=1,
            model_timed_out=False,
            requested_reasoning_policy="off",
            transport_required=True,
            transport_evidence=completed_transport,
            trajectory_identity_valid=False,
        )
        timeout = hermes_transport_infrastructure_reasons(
            runner_exit_code=-15,
            model_timed_out=True,
            requested_reasoning_policy="off",
            transport_required=True,
            transport_evidence={
                **completed_transport,
                "observer_ok": False,
                "response_count": 0,
            },
            trajectory_identity_valid=False,
        )
        broken_observer = hermes_transport_infrastructure_reasons(
            runner_exit_code=0,
            model_timed_out=False,
            requested_reasoning_policy="off",
            transport_required=True,
            transport_evidence={**completed_transport, "observer_ok": False},
            trajectory_identity_valid=False,
        )
        self.assertEqual(runner_failure, ["runner_exit_code=1"])
        self.assertEqual(timeout, [])
        self.assertEqual(
            broken_observer,
            ["model_transport_observer_missing_or_broken"],
        )

    def test_status_precedence_keeps_infrastructure_distinct_from_timeout(self) -> None:
        cases = (
            (
                "model timeout plus evaluator infrastructure error",
                True,
                False,
                False,
                "HARNESS_ERROR",
            ),
            ("missing evaluator runtime", False, False, False, "HARNESS_ERROR"),
            ("malformed evaluator output", False, False, False, "HARNESS_ERROR"),
            ("model-only timeout", True, False, True, "TIMEOUT"),
            ("candidate evaluator timeout", False, True, True, "TIMEOUT"),
        )

        for name, model_timeout, evaluation_timeout, infrastructure_ok, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    classify_outcome(
                        model_timed_out=model_timeout,
                        evaluation_timed_out=evaluation_timeout,
                        infrastructure_ok=infrastructure_ok,
                        evaluation_passed=False,
                        diff_check_returncode=0,
                        input_integrity_passed=True,
                    ),
                    expected,
                )

    def test_untracked_binary_deletion_rename_and_ignored_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = prepare("archiveguard-security-v1", root / "work")
            candidate = Path(metadata["candidate"])
            runtime_root = ensure_root(root / "runtime")
            runtime = ensure_subdirectory(
                runtime_root, runtime_root / "run", exclusive=True
            )
            baseline = capture_baseline(candidate, runtime)
            (candidate / "README.md").rename(candidate / "RENAMED.md")
            (candidate / "archiveguard" / "policy.py").unlink()
            (candidate / "binary.dat").write_bytes(b"\x00\xff\x01")
            (candidate / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
            (candidate / "ignored.tmp").write_text("ignore me", encoding="utf-8")
            evidence = capture_final(candidate, runtime, baseline)
            archive_path = Path(evidence["archive"])

            with tarfile.open(archive_path, "r:gz") as archive:
                names = set(archive.getnames())

        changed = set(evidence["changed_files"])
        self.assertIn("README.md", changed)
        self.assertIn("RENAMED.md", changed)
        self.assertIn("archiveguard/policy.py", changed)
        self.assertIn("binary.dat", changed)
        self.assertIn(".gitignore", changed)
        self.assertNotIn("ignored.tmp", changed)
        self.assertTrue(any(name.endswith("binary.dat") for name in names))
        self.assertFalse(any(name.endswith("ignored.tmp") for name in names))
        self.assertIn("GIT binary patch", evidence["diff"])
        self.assertEqual(evidence["diff_check_returncode"], 0)
        self.assertIn("files changed", evidence["diff_stat"])
        self.assertIn("binary.dat", evidence["diff_numstat"])

    def test_score_schema_rejects_out_of_range_values(self) -> None:
        plan = benchmark_model.load_execution_plan("qwen38-q8-medium-128k")
        result = make_result(plan, "archiveguard-security-v1", "PASS", 999.0)

        with self.assertRaises(jsonschema.ValidationError):
            validate_result(result)

        result["scores"] = {"overall": 50.0, "invented_score": 999.0}

        with self.assertRaises(jsonschema.ValidationError):
            validate_result(result)

        result["scores"] = {"overall": float("nan")}

        with self.assertRaises(jsonschema.ValidationError):
            validate_result(result)

        task_schema = json.loads(
            (ROOT / "schemas" / "task.schema.json").read_text(encoding="utf-8")
        )
        task, _ = load_task("archiveguard-security-v1")
        task["scoring"]["correctness"] = 101

        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(task_schema).validate(task)

    def test_result_schema_accepts_ds4_verified_identity_status(self) -> None:
        plan = benchmark_model.load_execution_plan("qwen38-q8-medium-128k")
        ds4 = benchmark_model.load_models()["deepseek-v4-flash"]
        self.assertEqual(ds4["runtime"], "ds4")
        plan["model_alias"] = "deepseek-v4-flash"
        plan["model"] = ds4
        plan["endpoint"] = "http://10.0.0.2:8000/v1"
        result = make_result(plan, "harness-smoke-v1", "PASS", 100.0)
        result["model"]["runtime_identity_status"] = (
            "VERIFIED_MODEL_ID_AND_CONFIGURED_ARTIFACTS"
        )

        validate_result(result)

        result["model"]["runtime_identity_status"] = "VERIFIED_MODEL_ID_ONLY"
        with self.assertRaises(jsonschema.ValidationError):
            validate_result(result)


class SuiteLifecycleTests(unittest.TestCase):
    def test_model_failure_continues_and_between_task_state_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a", "task-b"))
            calls: list[str] = []

            def runner(**kwargs: object) -> dict:
                task_id = str(kwargs["task_id"])
                calls.append(task_id)
                state_file = next((root / "runtime").glob("*.aggregate.json"))
                state = json.loads(state_file.read_text(encoding="utf-8"))
                self.assertEqual(state["lifecycle"]["phase"], "TASK_STARTING")

                if task_id == "task-b":
                    self.assertEqual(
                        state["lifecycle"]["completed_tasks"], ["task-a"]
                    )

                outcome = "FAIL" if task_id == "task-a" else "PASS"
                score = 40.0 if outcome == "FAIL" else 100.0
                return make_result(plan, task_id, outcome, score)

            aggregate = benchmark_model.execute_suite(
                plan,
                preflight=fake_preflight,
                task_runner=runner,
                results_dir=root / "results",
                reports_dir=root / "reports",
                runtime_root=root / "runtime",
                progress=lambda _message: None,
            )

        self.assertEqual(calls, ["task-a", "task-b"])
        self.assertEqual(aggregate["status"], "FAIL")
        self.assertEqual(aggregate["aggregate_score"], 70.0)
        self.assertEqual(aggregate["lifecycle"]["phase"], "FINISHED")

    def test_initial_state_exists_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))

            def preflight(endpoint: str, runtime_model: str, **_kwargs: object) -> dict:
                state_file = next((root / "runtime").glob("*.aggregate.json"))
                state = json.loads(state_file.read_text(encoding="utf-8"))
                self.assertEqual(state["lifecycle"]["phase"], "PREFLIGHT_RUNNING")
                raise KeyboardInterrupt

            with self.assertRaises(benchmark_model.SuiteInterrupted) as caught:
                benchmark_model.execute_suite(
                    plan,
                    preflight=preflight,
                    task_runner=mock.Mock(),
                    results_dir=root / "results",
                    reports_dir=root / "reports",
                    runtime_root=root / "runtime",
                    progress=lambda _message: None,
                )

            aggregate = caught.exception.aggregate
            state = json.loads(
                Path(aggregate["artifacts"]["runtime_state"]).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(state["status"], "INTERRUPTED")
        self.assertEqual(state["lifecycle"]["phase"], "INTERRUPTED")

    def test_task_interruption_preserves_candidate_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))

            def runner(**kwargs: object) -> dict:
                callback = kwargs["state_callback"]
                assert callable(callback)
                callback(
                    {
                        "phase": "MODEL_RUNNING",
                        "candidate": "/candidate/evidence",
                        "runtime": "/runtime/evidence",
                        "task_run_id": "task-a-run",
                    }
                )
                raise KeyboardInterrupt

            with self.assertRaises(benchmark_model.SuiteInterrupted) as caught:
                benchmark_model.execute_suite(
                    plan,
                    preflight=fake_preflight,
                    task_runner=runner,
                    results_dir=root / "results",
                    reports_dir=root / "reports",
                    runtime_root=root / "runtime",
                    progress=lambda _message: None,
                )

            aggregate = caught.exception.aggregate

        self.assertEqual(aggregate["status"], "INTERRUPTED")
        self.assertEqual(
            aggregate["lifecycle"]["active_task"]["candidate"],
            "/candidate/evidence",
        )

    def test_sigterm_becomes_canonical_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))

            def runner(**_kwargs: object) -> dict:
                os.kill(os.getpid(), signal.SIGTERM)
                raise AssertionError("SIGTERM handler did not interrupt")

            with self.assertRaises(benchmark_model.SuiteInterrupted) as caught:
                benchmark_model.execute_suite(
                    plan,
                    preflight=fake_preflight,
                    task_runner=runner,
                    results_dir=root / "results",
                    reports_dir=root / "reports",
                    runtime_root=root / "runtime",
                    progress=lambda _message: None,
                )

            state = json.loads(
                Path(caught.exception.aggregate["artifacts"]["runtime_state"])
                .read_text(encoding="utf-8")
            )

        self.assertEqual(state["status"], "INTERRUPTED")
        self.assertEqual(state["lifecycle"]["phase"], "INTERRUPTED")

    def test_signal_during_initial_atomic_state_write_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))
            original = benchmark_model._atomic_text
            sent = False

            def interrupted_write(path: Path, value: str, *, root: Path) -> None:
                nonlocal sent

                if not sent:
                    sent = True
                    os.kill(os.getpid(), signal.SIGINT)

                original(path, value, root=root)

            with mock.patch.object(
                benchmark_model, "_atomic_text", side_effect=interrupted_write
            ):
                with self.assertRaises(benchmark_model.SuiteInterrupted) as caught:
                    benchmark_model.execute_suite(
                        plan,
                        preflight=fake_preflight,
                        task_runner=mock.Mock(),
                        results_dir=root / "results",
                        reports_dir=root / "reports",
                        runtime_root=root / "runtime",
                        progress=lambda _message: None,
                    )

            state = json.loads(
                Path(caught.exception.aggregate["artifacts"]["runtime_state"])
                .read_text(encoding="utf-8")
            )

        self.assertEqual(state["status"], "INTERRUPTED")

    def test_harness_failure_stops_later_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a", "task-b"))
            calls: list[str] = []

            def runner(**kwargs: object) -> dict:
                task_id = str(kwargs["task_id"])
                calls.append(task_id)
                return make_result(plan, task_id, "HARNESS_ERROR", None)

            aggregate = benchmark_model.execute_suite(
                plan,
                preflight=fake_preflight,
                task_runner=runner,
                results_dir=root / "results",
                reports_dir=root / "reports",
                runtime_root=root / "runtime",
                progress=lambda _message: None,
            )

        self.assertEqual(calls, ["task-a"])
        self.assertEqual(aggregate["status"], "INFRASTRUCTURE_FAILURE")
        self.assertEqual(benchmark_model.aggregate_exit_code(aggregate), 3)

    def test_runtime_digest_change_after_task_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))
            changed = "sha256:" + "b" * 64
            preflight = mock.Mock(
                side_effect=[
                    fake_preflight(
                        plan["endpoint"],
                        plan["model"]["runtime_model"],
                        expected_digest=DIGEST,
                    ),
                    benchmark_model.InfrastructureError(
                        "runtime digest mismatch: expected "
                        f"{DIGEST}, received {changed}"
                    ),
                ]
            )
            aggregate = benchmark_model.execute_suite(
                plan,
                preflight=preflight,
                task_runner=lambda **_kwargs: make_result(
                    plan, "task-a", "PASS", 100.0
                ),
                results_dir=root / "results",
                reports_dir=root / "reports",
                runtime_root=root / "runtime",
                progress=lambda _message: None,
            )

        self.assertEqual(aggregate["status"], "INFRASTRUCTURE_FAILURE")
        self.assertFalse(aggregate["task_results"])
        self.assertIn("runtime identity changed", aggregate["infrastructure_failures"][0])

    def test_timeout_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, ("task-a",))
            aggregate = benchmark_model.execute_suite(
                plan,
                preflight=fake_preflight,
                task_runner=lambda **_kwargs: make_result(
                    plan, "task-a", "TIMEOUT", 25.0
                ),
                results_dir=root / "results",
                reports_dir=root / "reports",
                runtime_root=root / "runtime",
                progress=lambda _message: None,
            )

        self.assertEqual(benchmark_model.aggregate_exit_code(aggregate), 4)

    def test_cleanup_only_removes_exact_run_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            runtime = root / "runtime"
            work.mkdir()
            runtime.mkdir()
            run_id = "task-run-123"
            candidate = work / run_id
            runtime_dir = runtime / run_id
            candidate.mkdir()
            runtime_dir.mkdir()
            (work / f"{run_id}.metadata.json").write_text("{}\n", encoding="utf-8")
            result = {
                "run_id": run_id,
                "paths": {"candidate": str(candidate), "runtime": str(runtime_dir)},
            }
            cleanup(result, work_root=work, runtime_root=runtime)
            self.assertFalse(candidate.exists())
            outside = root / "outside"
            outside.mkdir()
            unsafe = {
                "run_id": run_id,
                "paths": {
                    "candidate": str(outside),
                    "runtime": str(runtime / run_id),
                },
            }

            with self.assertRaisesRegex(ValueError, "refusing cleanup"):
                cleanup(unsafe, work_root=work, runtime_root=runtime)

            self.assertTrue(outside.is_dir())


def fake_preflight(
    endpoint: str,
    runtime_model: str,
    *,
    expected_digest: str | None,
    runtime_config: object | None = None,
) -> dict[str, object]:
    return {
        "endpoint": endpoint,
        "runtime_model": runtime_model,
        "reported_id": runtime_model,
        "runtime_model_digest": expected_digest or DIGEST,
        "identity_status": "VERIFIED",
    }


def make_plan(root: Path, task_ids: tuple[str, ...]) -> dict:
    config = {
        "schema_version": 2,
        "benchmark_generation": "hermesbench-v3",
        "hermes": {
            "version": "0.20.4",
            "commit": "533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9",
            "install": str(hermes_runner.HERMES_REPO),
        },
        "baseline_runtime": {
            "endpoint": benchmark_model.EXPECTED_ENDPOINT,
            "runtime": "ollama",
            "ollama_version": "test-version",
        },
        "suite": {"tasks": list(task_ids)},
        "execution": {
            "failure_policy": benchmark_model.FAILURE_POLICY,
            "evaluator_timeout_seconds": 10,
            "evaluator_worker_sha256": hashlib.sha256(
                (ROOT / "harness" / "evaluator_worker.py").read_bytes()
            ).hexdigest(),
            "candidate_rpc_sha256": hashlib.sha256(
                (ROOT / "harness" / "candidate_rpc.py").read_bytes()
            ).hexdigest(),
            "scoring_version": benchmark_model.EXPECTED_SCORING_VERSION,
            "tool_distribution": "hermesbench_terminal_only",
            "candidate_toolsets": ["hermesbench_terminal_foreground"],
            "candidate_isolation": (
                benchmark_model.EXPECTED_CANDIDATE_ISOLATION
            ),
            "evaluator_isolation": (
                benchmark_model.EXPECTED_EVALUATOR_ISOLATION
            ),
            "endpoint_redirects": "reject",
        },
    }
    models = {
        "schema_version": 3,
        "models": {
            "test-model": {
                "display_name": "Test Model",
                "provider": "gx10",
                "runtime": "ollama",
                "runtime_model": "test:latest",
                "runtime_digest": DIGEST,
                "quantization": "test",
                "context_length": 1024,
                "reasoning_effort": "medium",
                "reasoning_policy": "effort:medium",
                "supported_reasoning_policies": [
                    "off",
                    "native",
                    "effort:medium",
                ],
            }
        },
    }
    config_path = root / "config.yaml"
    models_path = root / "models.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    models_path.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")

    for task_id in task_ids:
        task_dir = root / "tasks" / "coding" / task_id
        task_dir.mkdir(parents=True)
        task_file = task_dir / "TASK.md"
        task_file.write_text("# Test task\n", encoding="utf-8")
        test_file = task_dir / "test_acceptance.py"
        test_file.write_text(
            "import unittest\n"
            "class Acceptance(unittest.TestCase):\n"
            "    def test_acceptance(self): self.assertTrue(True)\n",
            encoding="utf-8",
        )
        fixture = root / "fixtures" / task_id
        fixture.mkdir(parents=True)
        candidate_test = fixture / "tests" / "test_public.py"
        candidate_test.parent.mkdir(parents=True)
        candidate_test.write_bytes(test_file.read_bytes())
        task = {
            "id": task_id,
            "version": 2,
            "category": "coding",
            "fixture": f"fixtures/{task_id}",
            "fixture_sha256": content_digest(fixture),
            "task_file_sha256": hashlib.sha256(
                task_file.read_bytes()
            ).hexdigest(),
            "goal": "Complete the test task.",
            "limits": {"wall_seconds": 60, "agent_turns": 5},
            "allowed_changes": ["implementation.py"],
            "requirements": [
                {"id": "acceptance", "text": "Pass acceptance", "security": False}
            ],
            "evaluation_checks": [
                {
                    "id": "acceptance",
                    "visibility": "public",
                    "test_file": "test_acceptance.py",
                    "candidate_path": "tests/test_public.py",
                    "sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
                    "cases": {
                        "test_acceptance": {
                            "categories": ["correctness"],
                            "requirements": ["acceptance"],
                        }
                    },
                }
            ],
            "scoring": {"correctness": 100},
        }
        (task_dir / "task.yaml").write_text(
            yaml.safe_dump(task, sort_keys=False), encoding="utf-8"
        )

    plan = benchmark_model.load_execution_plan(
        "test-model",
        config_path=config_path,
        models_path=models_path,
        task_root=root,
        requested_tasks=(),
    )
    plan["benchmark"]["git_dirty"] = False
    return plan


def make_result(
    plan: dict,
    task_id: str,
    outcome: str,
    score: float | None,
) -> dict:
    hard_failures = ["TIMEOUT"] if outcome == "TIMEOUT" else []
    return {
        "schema_version": 2,
        "run_id": f"{task_id}-run",
        "benchmark": plan["benchmark"],
        "task": {"id": task_id, "version": 2},
        "model": {
            "config": plan["model_alias"],
            "runtime_model": plan["model"]["runtime_model"],
            "runtime_digest": plan["model"]["runtime_digest"],
            "runtime_identity_status": "VERIFIED",
            "endpoint": plan["endpoint"],
            "reasoning_policy": plan["model"]["reasoning_policy"],
        },
        "outcome": outcome,
        "statuses": {
            "model": "TIMEOUT" if outcome == "TIMEOUT" else "COMPLETED",
            "evaluation": "ERROR" if outcome == "HARNESS_ERROR" else "COMPLETED",
            "infrastructure": "FAIL" if outcome == "HARNESS_ERROR" else "PASS",
        },
        "scores": {"overall": score},
        "hard_failures": hard_failures,
        "metrics": {
            "elapsed_seconds_hermes_observed": 1.0,
            "infrastructure_failures": (
                ["test harness failure"] if outcome == "HARNESS_ERROR" else []
            ),
        },
        "artifacts": {},
        "mode": "model",
    }


if __name__ == "__main__":
    unittest.main()
