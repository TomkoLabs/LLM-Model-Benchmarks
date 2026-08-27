from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness.artifacts import atomic_write_text, ensure_subdirectory
from harness.endpoints import LocalEndpoint
from harness.processes import ProcessDeadlineExpired, run_process_group
from harness.reasoning_policy import ReasoningPolicy, parse_reasoning_policy


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "upstreams.lock.json"
UPSTREAM_ROOT = ROOT / ".state" / "upstreams"
DEFAULT_HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", Path.home() / ".hermes")
)
HERMES_ROOT = Path(
    os.environ.get(
        "HERMES_AGENT_ROOT",
        str(DEFAULT_HERMES_HOME / "hermes-agent"),
    )
).expanduser().resolve()
HERMES_PYTHON = Path(
    os.environ.get(
        "HERMES_BENCH_PYTHON",
        str(HERMES_ROOT / "venv" / "bin" / "python"),
    )
).expanduser().absolute()
SPARK_DEPENDENCY_VERSIONS = {
    "playwright": "1.62.0",
    "pyee": "13.0.0",
    "greenlet": "3.5.5",
    "typing_extensions": "4.16.0",
}
SPARK_PYTHON_DEPS = (
    ROOT
    / ".state"
    / "python-deps"
    / "playwright-1.62.0-pyee-13.0.0-greenlet-3.5.5-typing-extensions-4.16.0"
)
SPARK_CHROMIUM = Path("/usr/bin/chromium")


class UpstreamError(RuntimeError):
    pass


def load_upstream_lock(path: Path = LOCK_PATH) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamError(f"cannot read upstream lock: {exc}") from exc
    if document.get("schema_version") != 1 or not isinstance(document.get("upstreams"), dict):
        raise UpstreamError("unsupported upstream lock schema")
    return document["upstreams"]


def _git(checkout: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(checkout), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
        env={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
    )


def _checkout_path(entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry.get("checkout") or ""))
    checkout = (ROOT / relative).resolve()
    if not checkout.is_relative_to(UPSTREAM_ROOT.resolve()) or checkout == UPSTREAM_ROOT.resolve():
        raise UpstreamError(f"unsafe upstream checkout path: {relative}")
    return checkout


def validate_checkout(name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    checkout = _checkout_path(entry)
    if not (checkout / ".git").is_dir():
        raise UpstreamError(f"missing pinned upstream checkout: {name}")

    head = _git(checkout, "rev-parse", "HEAD")
    status = _git(checkout, "status", "--porcelain")
    remote = _git(checkout, "remote", "get-url", "origin")
    if head.returncode or status.returncode or remote.returncode:
        raise UpstreamError(f"cannot inspect upstream checkout: {name}")
    expected_commit = str(entry.get("commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit) or head.stdout.strip() != expected_commit:
        raise UpstreamError(
            f"upstream checkout commit mismatch for {name}: expected {expected_commit}, got {head.stdout.strip()}"
        )
    if status.stdout:
        raise UpstreamError(f"upstream checkout is dirty: {name}")
    if remote.stdout.strip().rstrip("/") != str(entry.get("url") or "").rstrip("/"):
        raise UpstreamError(f"upstream origin mismatch: {name}")

    license_file = entry.get("license_file")
    if license_file:
        path = checkout / str(license_file)
        if not path.is_file():
            raise UpstreamError(f"upstream license file is missing: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("license_sha256"):
            raise UpstreamError(f"upstream license digest mismatch: {name}")

    return {
        "name": name,
        "url": remote.stdout.strip(),
        "commit": head.stdout.strip(),
        "version": entry.get("version"),
        "license": entry.get("license"),
        "license_disposition": entry.get("license_disposition"),
        "checkout": str(checkout),
        "clean": True,
    }


def _installed_distributions(path: Path) -> dict[str, str]:
    return {
        distribution.metadata["Name"].lower().replace("-", "_"): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(path)])
        if distribution.metadata.get("Name")
    }


def ensure_spark_runtime_dependencies(*, setup: bool) -> dict[str, Any]:
    expected = dict(SPARK_DEPENDENCY_VERSIONS)
    installed = _installed_distributions(SPARK_PYTHON_DEPS)
    missing = {
        name: version
        for name, version in expected.items()
        if installed.get(name) != version
    }
    if missing:
        if not setup:
            raise UpstreamError(
                "missing pinned Spark runtime dependencies: "
                + ", ".join(f"{name}=={version}" for name, version in missing.items())
            )
        SPARK_PYTHON_DEPS.mkdir(parents=True, exist_ok=True)
        command = [
            str(HERMES_PYTHON),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-compile",
            "--no-deps",
            "--only-binary=:all:",
            "--upgrade",
            "--target",
            str(SPARK_PYTHON_DEPS),
            *(f"{name}=={version}" for name, version in expected.items()),
        ]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
        if process.returncode:
            raise UpstreamError(
                "cannot install pinned Spark runtime dependencies: "
                + process.stdout[-2000:]
            )
        installed = _installed_distributions(SPARK_PYTHON_DEPS)
    observed = {name: installed.get(name) for name in expected}
    if observed != expected:
        raise UpstreamError(
            f"Spark runtime dependency mismatch: expected {expected}, got {observed}"
        )
    if not SPARK_CHROMIUM.is_file() or not os.access(SPARK_CHROMIUM, os.X_OK):
        raise UpstreamError(f"required system Chromium is missing: {SPARK_CHROMIUM}")
    browser = subprocess.run(
        [str(SPARK_CHROMIUM), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    if browser.returncode or not browser.stdout.strip():
        raise UpstreamError("cannot resolve the system Chromium version")
    return {
        "python_path": str(SPARK_PYTHON_DEPS),
        "python_distributions": observed,
        "browser_path": str(SPARK_CHROMIUM),
        "browser_version": browser.stdout.strip(),
        "browser_downloaded": False,
    }


def validate_spark_runtime_lock(entry: Mapping[str, Any]) -> None:
    declared = entry.get("python_dependencies")
    if declared != SPARK_DEPENDENCY_VERSIONS:
        raise UpstreamError(
            "Spark runtime dependency lock mismatch: "
            f"expected {SPARK_DEPENDENCY_VERSIONS}, got {declared}"
        )
    if entry.get("system_browser") != str(SPARK_CHROMIUM):
        raise UpstreamError(
            "Spark system browser lock mismatch: "
            f"expected {SPARK_CHROMIUM}, got {entry.get('system_browser')}"
        )


def ensure_checkouts(*, setup: bool = True) -> dict[str, dict[str, Any]]:
    lock = load_upstream_lock()
    integrated = {
        name: entry
        for name, entry in lock.items()
        if isinstance(entry, dict) and entry.get("checkout")
    }
    spark_entry = integrated.get("spark-bench")
    if spark_entry is None:
        raise UpstreamError("Spark Bench is missing from the integrated upstream lock")
    validate_spark_runtime_lock(spark_entry)
    UPSTREAM_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / ".state" / "upstream-home").mkdir(parents=True, exist_ok=True)

    for name, entry in integrated.items():
        checkout = _checkout_path(entry)
        if checkout.exists():
            continue
        if not setup:
            raise UpstreamError(f"missing pinned upstream checkout: {name}")
        clone = subprocess.run(
            [
                "/usr/bin/git",
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                str(entry["url"]),
                str(checkout),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
            check=False,
        )
        if clone.returncode:
            raise UpstreamError(f"cannot clone {name}: {clone.stdout[-1000:]}")
        selected = _git(checkout, "checkout", "--detach", str(entry["commit"]), timeout=120)
        if selected.returncode:
            raise UpstreamError(f"cannot select pinned commit for {name}: {selected.stdout[-1000:]}")

    provenance = {
        name: validate_checkout(name, entry)
        for name, entry in integrated.items()
    }
    provenance["spark-bench"]["runtime_dependencies"] = (
        ensure_spark_runtime_dependencies(setup=setup)
    )
    return provenance


def safe_slug(value: str, *, maximum: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()
    if not slug:
        slug = "model"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:maximum]}-{suffix}"


def build_spark_command(
    checkout: Path,
    *,
    endpoint: str,
    model: str,
    output_dir: Path,
    label: str,
    profile: Mapping[str, Any],
    reasoning_policy: ReasoningPolicy | None = None,
) -> list[str]:
    policy = reasoning_policy or parse_reasoning_policy("off")
    argv = [
        str(HERMES_PYTHON),
        str(checkout / "spark_bench.py"),
        "eval",
        "--label",
        label,
        "--run-kind",
        "benchmark",
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--out-dir",
        str(output_dir),
        "--tier",
        str(profile["tier"]),
        "--repeats",
        str(profile["repeats"]),
        "--thinking",
        (
            "off"
            if policy.mode == "off"
            else "on"
            if policy.mode == "effort"
            else "auto"
        ),
        "--timeout",
        str(profile.get("timeout", 900)),
    ]
    if profile.get("skip_throughput", True):
        argv.append("--skip-throughput")
    return argv


def build_benchlocal_command(
    checkout: Path,
    *,
    endpoint: str,
    model: str,
    output_path: Path,
    selection: Mapping[str, Any],
    reasoning_policy: ReasoningPolicy | None = None,
) -> list[str]:
    policy = reasoning_policy or parse_reasoning_policy("off")
    argv = [
        str(HERMES_PYTHON),
        "-m",
        "benchlocal_cli.cli",
        "run",
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--save-json",
        str(output_path),
        "--output",
        "json",
        "--incremental",
        "--progress",
        "--no-retry",
        "--max-transient-retries",
        "0",
        "--timeout-per-case",
        str(selection.get("timeout_per_case", 300)),
        "--repeat",
        str(selection.get("repeat", 1)),
    ]
    if policy.mode == "off":
        argv.extend(["--no-thinking", "--reasoning-effort", "none"])
    elif policy.mode == "effort":
        argv.extend(
            ["--enable-thinking", "--reasoning-effort", str(policy.effort)]
        )
    if selection.get("mode"):
        argv.append("--" + str(selection["mode"]))
    if selection.get("pack"):
        argv.extend(["--pack", str(selection["pack"])])
    for scenario in selection.get("scenarios", []):
        argv.extend(["--scenario", str(scenario)])
    if selection.get("sandboxed"):
        argv.append("--enable-sandboxed-packs")
    return argv


def build_infermark_command(
    checkout: Path,
    *,
    endpoint: str,
    model: str,
    output_path: Path,
    profile: Mapping[str, Any],
    prompt: str | None = None,
) -> list[str]:
    argv = [
        str(HERMES_PYTHON),
        "-m",
        "infermark.cli",
        "run",
        endpoint,
        "--model",
        model,
        "--requests",
        str(profile["requests"]),
        "--concurrency",
        ",".join(str(value) for value in profile["concurrency"]),
        "--max-tokens",
        str(profile["max_tokens"]),
        "--warmup",
        str(profile["warmup"]),
        "--timeout",
        str(profile["timeout"]),
        "--output",
        str(output_path),
    ]
    if prompt is not None:
        argv.extend(["--prompt", prompt])
    return argv


def write_local_network_policy(
    policy_dir: Path,
    endpoint: LocalEndpoint,
) -> tuple[Path, Path]:
    policy_dir.mkdir(parents=True, exist_ok=True)
    attempts = policy_dir.parent / "blocked-network-attempts.jsonl"
    source = f'''import json
import os
import socket

_host = {endpoint.host!r}
_port = {endpoint.port!r}
_attempts = {str(attempts)!r}
_socket = socket.socket
_getaddrinfo = socket.getaddrinfo

def _block(address):
    try:
        fd = os.open(_attempts, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps({{"address": repr(address)}}) + "\\n").encode())
        finally:
            os.close(fd)
    except BaseException:
        pass
    raise OSError("qualification policy blocks non-local network destinations")

def _check(address):
    try:
        host, port = str(address[0]), int(address[1])
    except (IndexError, TypeError, ValueError):
        _block(address)
    if host != _host or port != _port:
        _block(address)

class RestrictedSocket(_socket):
    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(address)
        return super().connect(address)
    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check(address)
        return super().connect_ex(address)

def restricted_getaddrinfo(host, port, *args, **kwargs):
    _check((host, port))
    return _getaddrinfo(host, port, *args, **kwargs)

socket.socket = RestrictedSocket
socket.getaddrinfo = restricted_getaddrinfo
'''
    atomic_write_text(policy_dir / "sitecustomize.py", source, root=policy_dir)
    return policy_dir, attempts


def write_chromium_network_wrapper(wrapper_dir: Path) -> Path:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrapper = wrapper_dir / "chromium-local-only"
    source = f'''#!/bin/sh
# Harness-owned Spark render policy: browser traffic is loopback-only.
exec {SPARK_CHROMIUM} \\
  "--proxy-server=http://127.0.0.1:9" \\
  "--proxy-bypass-list=localhost;127.0.0.1;[::1]" \\
  "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1" \\
  --disable-background-networking \\
  --disable-component-update \\
  --disable-sync \\
  "$@"
'''
    atomic_write_text(wrapper, source, root=wrapper_dir)
    os.chmod(wrapper, 0o700, follow_symlinks=False)
    return wrapper


def upstream_environment(*, python_paths: Sequence[Path]) -> dict[str, str]:
    return {
        "HOME": str(ROOT / ".state" / "upstream-home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths),
    }


def execute_upstream(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    total_timeout: float,
    inactivity_timeout: float,
    heartbeat_paths: Sequence[Path] = (),
) -> subprocess.CompletedProcess[str]:
    try:
        process = run_process_group(
            argv,
            cwd=cwd,
            env=env,
            timeout=total_timeout,
            inactivity_timeout=inactivity_timeout,
            heartbeat_paths=heartbeat_paths,
        )
        output = process.stdout
    except ProcessDeadlineExpired as exc:
        output = str(exc.stdout or "")
        atomic_write_text(log_path, output, root=log_path.parent)
        raise UpstreamError(f"upstream {exc.deadline_kind} timeout") from exc
    except BaseException as exc:
        output = str(getattr(exc, "hermesbench_stdout", ""))
        atomic_write_text(log_path, output, root=log_path.parent)
        raise
    atomic_write_text(log_path, output, root=log_path.parent)
    if process.returncode:
        raise UpstreamError(f"upstream exited {process.returncode}; see {log_path}")
    return process


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpstreamError(f"missing or incompatible upstream JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UpstreamError(f"upstream JSON must be an object: {path}")
    return value


def parse_benchlocal(path: Path, *, model: str) -> dict[str, Any]:
    value = _strict_json(path)
    if value.get("schema_version") != "1" or value.get("model") != model:
        raise UpstreamError("BenchLocal output schema or model identity mismatch")
    packs = value.get("packs")
    totals = value.get("totals")
    if not isinstance(packs, list) or not packs or not isinstance(totals, dict):
        raise UpstreamError("BenchLocal output is incomplete")
    if any(not isinstance(pack, dict) or pack.get("status") != "ok" for pack in packs):
        raise UpstreamError("BenchLocal reported an incompatible/infrastructure pack status")
    total = totals.get("total")
    passed = totals.get("passed")
    score = totals.get("score")
    if type(total) is not int or total < 1 or type(passed) is not int or not isinstance(score, (int, float)):
        raise UpstreamError("BenchLocal totals are malformed")
    thinking_validity = value.get("thinking_validity")
    contaminated_packs = sorted(
        str(name)
        for name, validity in (
            thinking_validity.items()
            if isinstance(thinking_validity, dict)
            else ()
        )
        if isinstance(validity, dict) and validity.get("status") == "contaminated"
    )
    return {
        "status": "FAIL" if contaminated_packs else "PASS",
        "score": round(float(score) * 100, 3),
        "metrics": {
            "passed": passed,
            "total": total,
            "repeat": value.get("repeat"),
            "packs": [
                {
                    "id": pack.get("pack_id"),
                    "passed": pack.get("passed"),
                    "total": pack.get("total"),
                    "score": pack.get("score"),
                }
                for pack in packs
            ],
            "completion_tokens": sum(
                int(scenario.get("tokens_completion") or 0)
                for pack in packs
                if isinstance(pack, dict)
                for scenario in pack.get("scenarios", [])
                if isinstance(scenario, dict)
            ),
            "thinking_validity": thinking_validity,
            "contaminated_packs": contaminated_packs,
        },
    }


def parse_infermark(path: Path, *, model: str) -> dict[str, Any]:
    value = _strict_json(path)
    if value.get("model") != model or not isinstance(value.get("config"), dict):
        raise UpstreamError("Infermark output schema or model identity mismatch")
    rows = value.get("results")
    if not isinstance(rows, list) or not rows:
        raise UpstreamError("Infermark output has no concurrency results")
    required = {
        "concurrency",
        "n_requests",
        "n_success",
        "n_error",
        "total_duration",
        "requests_per_second",
        "tokens_per_second",
        "latency",
        "ttft",
        "itl",
    }
    if any(not isinstance(row, dict) or not required.issubset(row) for row in rows):
        raise UpstreamError("Infermark output row is incompatible")
    total = sum(int(row["n_requests"]) for row in rows)
    success = sum(int(row["n_success"]) for row in rows)
    errors = sum(int(row["n_error"]) for row in rows)
    if total < 1 or success + errors != total:
        raise UpstreamError("Infermark request counts are inconsistent")
    single = next((row for row in rows if row["concurrency"] == 1), rows[0])
    return {
        "status": "PASS" if errors == 0 else "FAIL",
        "score": round(100 * success / total, 3),
        "metrics": {
            "requests": total,
            "success": success,
            "errors": errors,
            "error_rate": round(errors / total, 6),
            "concurrency": [row["concurrency"] for row in rows],
            "tokens_per_second_c1": single["tokens_per_second"],
            "requests_per_second_c1": single["requests_per_second"],
            "latency_seconds_c1": single["latency"],
            "ttft_seconds_c1": single["ttft"],
            "itl_seconds_c1": single["itl"],
        },
    }


def parse_spark(path: Path, *, model: str) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise UpstreamError(f"missing Spark Bench CSV: {exc}") from exc
    required = {"run_id", "model", "workload", "metric", "value", "unit"}
    if not rows or not required.issubset(rows[0]):
        raise UpstreamError("Spark Bench output schema is incompatible")
    run_id = rows[-1].get("run_id")
    selected = [row for row in rows if row.get("run_id") == run_id]
    if not run_id or any(row.get("model") != model for row in selected):
        raise UpstreamError("Spark Bench model identity mismatch")
    metrics = {(row["workload"], row["metric"]): row["value"] for row in selected}
    for key, expected in (
        (("provenance", "golden_gate"), "PASS"),
        (("provenance", "run_valid"), "PASS"),
    ):
        if metrics.get(key) != expected:
            raise UpstreamError(f"Spark Bench integrity gate failed: {key[1]}")
    try:
        truescore = float(metrics[("overall", "truescore")])
        quality = float(metrics[("overall", "quality")])
        reliability = float(metrics[("overall", "reliability")])
    except (KeyError, TypeError, ValueError) as exc:
        raise UpstreamError("Spark Bench score rows are missing or malformed") from exc
    coding_value = metrics.get(("code", "domain_quality")) or metrics.get(("coding", "domain_quality"))
    coding = float(coding_value) if coding_value is not None else quality
    quarantine = metrics.get(("provenance", "quarantine"))
    if quarantine != "clean":
        return {
            "status": "QUARANTINED",
            "score": None,
            "result_validity": "QUARANTINED",
            "metrics": {
                "run_id": run_id,
                "quarantine": quarantine or "missing quarantine marker",
                "provisional_truescore": truescore,
                "provisional_quality": quality,
                "coding_quality": coding,
                "reliability": reliability,
                "error_rate_percent": float(metrics.get(("provenance", "error_rate"), 0)),
                "repeats": int(float(metrics.get(("provenance", "repeats"), 1))),
                "methodology": metrics.get(("provenance", "methodology")),
                "total_output_tokens": int(float(metrics.get(("overall", "total_output_tokens"), 0))),
            },
        }
    return {
        "status": "PASS",
        "score": round(truescore, 3),
        "metrics": {
            "run_id": run_id,
            "truescore": truescore,
            "quality": quality,
            "coding_quality": coding,
            "reliability": reliability,
            "error_rate_percent": float(metrics.get(("provenance", "error_rate"), 0)),
            "repeats": int(float(metrics.get(("provenance", "repeats"), 1))),
            "methodology": metrics.get(("provenance", "methodology")),
            "total_output_tokens": int(float(metrics.get(("overall", "total_output_tokens"), 0))),
        },
    }
