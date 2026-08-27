#!/usr/bin/python3
from __future__ import annotations

import base64
import builtins
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import unittest
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sandbox import build_isolated_command  # noqa: E402


RPC_SERVER = ROOT / "harness" / "candidate_rpc.py"
MAX_CASE_DETAIL_CHARS = 16 * 1024
MAX_RPC_RESPONSE_CHARS = 4 * 1024 * 1024


def _bounded_detail(value: str) -> str:
    if len(value) <= MAX_CASE_DETAIL_CHARS:
        return value
    return value[:MAX_CASE_DETAIL_CHARS] + "\n...[detail truncated]"


class StructuredResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.cases: list[dict[str, object]] = []

    @staticmethod
    def _case_id(test: unittest.case.TestCase) -> str:
        return test.id().rsplit(".", 1)[-1]

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        self.cases.append({"id": self._case_id(test), "status": "PASS"})

    def _add_error(
        self,
        test: unittest.case.TestCase,
        error: tuple[type[BaseException], BaseException, object],
        status: str,
    ) -> None:
        self.cases.append(
            {
                "id": self._case_id(test),
                "status": status,
                "detail": _bounded_detail(self._exc_info_to_string(error, test)),
            }
        )

    def addFailure(
        self,
        test: unittest.case.TestCase,
        error: tuple[type[BaseException], BaseException, object],
    ) -> None:
        super().addFailure(test, error)
        self._add_error(test, error, "FAIL")

    def addError(
        self,
        test: unittest.case.TestCase,
        error: tuple[type[BaseException], BaseException, object],
    ) -> None:
        super().addError(test, error)
        self._add_error(test, error, "ERROR")

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        self.cases.append(
            {"id": self._case_id(test), "status": "ERROR", "detail": reason}
        )


def _result_descriptor() -> int:
    try:
        return int(os.environ.get("HERMES_BENCH_RESULT_FD", ""))
    except ValueError as exc:
        raise RuntimeError("missing result descriptor") from exc


def _write_result(descriptor: int, payload: dict[str, object]) -> None:
    view = memoryview(
        (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    )
    while view:
        view = view[os.write(descriptor, view) :]


def _flatten(suite: unittest.TestSuite) -> Iterable[unittest.case.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def _candidate_modules(candidate: Path) -> list[str]:
    names = {
        item.stem
        for item in candidate.iterdir()
        if item.is_file() and item.suffix == ".py" and item.stem != "__init__"
    }
    names.update(
        item.name
        for item in candidate.iterdir()
        if item.is_dir() and (item / "__init__.py").is_file()
    )
    return sorted(name for name in names if name.isidentifier())


class RemoteObject:
    def __init__(self, identifier: int) -> None:
        self.identifier = identifier


class CandidateExecutionError(RuntimeError):
    pass


class CandidateSession:
    def __init__(self, candidate: Path) -> None:
        self.candidate = candidate
        self.modules = _candidate_modules(candidate)
        if not self.modules:
            raise RuntimeError("candidate exposes no importable top-level modules")

        self.scratch = Path(tempfile.mkdtemp(prefix="hermesbench-evaluator-"))
        command = build_isolated_command(
            candidate,
            [
                "/usr/bin/python3",
                "/harness/candidate_rpc.py",
                *self.modules,
            ],
            candidate_writable=False,
            read_only_files={RPC_SERVER: "/harness/candidate_rpc.py"},
            writable_directories={self.scratch: str(self.scratch)},
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={},
            start_new_session=False,
        )
        self.request_id = 0

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        shutil.rmtree(self.scratch)

    def _encode(self, value: Any) -> Any:
        if isinstance(value, RemoteObject):
            return {"__hermesbench_type__": "remote", "id": value.identifier}
        if isinstance(value, bytes):
            return {
                "__hermesbench_type__": "bytes",
                "value": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, Path):
            return {"__hermesbench_type__": "path", "value": str(value)}
        if isinstance(value, tuple):
            return {
                "__hermesbench_type__": "tuple",
                "items": [self._encode(item) for item in value],
            }
        if isinstance(value, list):
            return [self._encode(item) for item in value]
        if isinstance(value, dict):
            return {key: self._encode(item) for key, item in value.items()}
        return value

    def _decode(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        kind = value.get("__hermesbench_type__")
        if kind == "bytes":
            return base64.b64decode(value["value"], validate=True)
        if kind == "path":
            path = Path(value["value"]).absolute()
            resolved = path.resolve(strict=False)
            allowed_roots = (
                self.candidate.absolute().resolve(),
                self.scratch.absolute().resolve(),
            )
            if not any(resolved.is_relative_to(root) for root in allowed_roots):
                raise CandidateExecutionError(
                    "candidate RPC returned a path outside allowed roots"
                )
            return path
        if kind == "tuple":
            return tuple(self._decode(item) for item in value["items"])
        if kind == "remote":
            return RemoteObject(value["id"])
        return {key: self._decode(item) for key, item in value.items()}

    def request(
        self,
        module: str,
        name: str,
        operation: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.request_id += 1
        request = {
            "id": self.request_id,
            "module": module,
            "name": name,
            "operation": operation,
            "args": self._encode(list(args)),
            "kwargs": self._encode(kwargs),
        }
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("candidate RPC streams are unavailable")
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self.process.stdin.flush()
            raw = self.process.stdout.readline(MAX_RPC_RESPONSE_CHARS + 1)
        except (BrokenPipeError, OSError) as exc:
            raise CandidateExecutionError(
                "candidate RPC process terminated"
            ) from exc
        if not raw:
            detail = self.process.stderr.read() if self.process.stderr else ""
            raise CandidateExecutionError(
                "candidate RPC process terminated without a response: "
                + _bounded_detail(detail)
            )
        if len(raw) > MAX_RPC_RESPONSE_CHARS:
            raise CandidateExecutionError(
                "candidate RPC response exceeds size limit"
            )
        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CandidateExecutionError(
                "candidate RPC returned malformed JSON"
            ) from exc
        if response.get("id") != self.request_id or type(response.get("ok")) is not bool:
            raise CandidateExecutionError(
                "candidate RPC returned an invalid response envelope"
            )
        if response["ok"]:
            return self._decode(response.get("value"))

        error = response.get("error", {})
        exception_name = error.get("name")
        message = error.get("message", "candidate call failed")
        exception_type = getattr(builtins, exception_name, RuntimeError)
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, BaseException
        ):
            exception_type = RuntimeError
        raise exception_type(message)

    def proxy_module(self, name: str) -> types.ModuleType:
        module = types.ModuleType(name)

        def remote_attribute(attribute: str) -> Any:
            class RemoteCallable:
                def __call__(_self, *args: Any, **kwargs: Any) -> Any:
                    return self.request(name, attribute, "call", *args, **kwargs)

            value = self.request(name, attribute, "get")
            if isinstance(value, RemoteObject):
                return RemoteCallable()
            return value

        module.__getattr__ = remote_attribute  # type: ignore[attr-defined]
        return module


def _run_one_case(test_path: Path, candidate: Path, case_id: str) -> dict[str, object]:
    session = CandidateSession(candidate)
    original_tempdir = tempfile.tempdir
    original_cwd = Path.cwd()
    previous_modules = {name: sys.modules.get(name) for name in session.modules}
    try:
        tempfile.tempdir = str(session.scratch)
        os.chdir(candidate)
        for name in session.modules:
            sys.modules[name] = session.proxy_module(name)

        specification = importlib.util.spec_from_file_location(
            "hermesbench_evaluation",
            test_path,
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load evaluation module")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        matches = [
            test
            for test in _flatten(suite)
            if test.id().rsplit(".", 1)[-1] == case_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one discovered test for {case_id}, found {len(matches)}"
            )
        result = StructuredResult()
        unittest.TestSuite(matches).run(result)
        if result.testsRun != 1 or len(result.cases) != 1:
            raise RuntimeError("evaluator did not produce exactly one case result")
        return result.cases[0]
    except CandidateExecutionError as exc:
        return {
            "id": case_id,
            "status": "ERROR",
            "detail": _bounded_detail(f"{type(exc).__name__}: {exc}"),
        }
    finally:
        os.chdir(original_cwd)
        tempfile.tempdir = original_tempdir
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        session.close()


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: evaluator_worker.py TEST_FILE CANDIDATE EXPECTED_CASES_JSON"
        )
    test_path = Path(sys.argv[1])
    candidate = Path(sys.argv[2])
    expected_cases = json.loads(sys.argv[3])
    if (
        not isinstance(expected_cases, list)
        or not expected_cases
        or any(not isinstance(item, str) or not item for item in expected_cases)
        or len(expected_cases) != len(set(expected_cases))
    ):
        raise SystemExit("invalid expected evaluator case identities")

    descriptor = _result_descriptor()
    try:
        cases = [
            _run_one_case(test_path, candidate, case_id)
            for case_id in expected_cases
        ]
        successful = all(item["status"] == "PASS" for item in cases)
        _write_result(
            descriptor,
            {
                "schema_version": 1,
                "discovered": len(cases),
                "successful": successful,
                "cases": cases,
            },
        )
        return 0 if successful else 1
    except BaseException as exc:
        _write_result(
            descriptor,
            {
                "schema_version": 1,
                "worker_error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
