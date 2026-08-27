#!/usr/bin/env python3

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from harness.hermes_runner import RUNTIME_ROOT, WORK_ROOT, cleanup
from harness.sandbox import run
from harness.workspace import ROOT, prepare


WORK = WORK_ROOT
HOST_HOME = Path.home().resolve()

WORK.mkdir(parents=True, exist_ok=True)
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

metadata = prepare(
    "archiveguard-security-v1",
    WORK,
)

candidate = Path(metadata["candidate"])
runtime_dir = RUNTIME_ROOT / metadata["run_id"]
runtime_dir.mkdir()

(runtime_dir / "probe.txt").write_text(
    "RUNNER_PROBE_7429\n",
    encoding="utf-8",
)

probe = f'''
set -eu

echo "sandbox_inside=YES"
echo "pwd=$(pwd)"
echo "uid=$(id -u)"
echo "user=$(id -un)"

test -f BENCHMARK_TASK.md
echo "candidate_visible=YES"

printf 'sandbox-write-ok\n' > .sandbox-write-test
echo "candidate_writable=YES"

if [ -e {shlex.quote(str(RUNTIME_ROOT))} ]; then
    echo "runtime_tree_hidden=NO"
    exit 19
fi
echo "runtime_tree_hidden=YES"

if [ -e {shlex.quote(str(ROOT / 'README.md'))} ]; then
    echo "benchmark_root_hidden=NO"
    exit 20
fi
echo "benchmark_root_hidden=YES"

if [ -e {shlex.quote(str(ROOT / 'tasks/security/archiveguard-v1/private_tests/hidden_acceptance.py'))} ]; then
    echo "evaluator_tests_hidden=NO"
    exit 21
fi
echo "evaluator_tests_hidden=YES"

test -d "$HERMES_BENCH_CANDIDATE"
test "$(find {shlex.quote(str(WORK))} -mindepth 1 -maxdepth 1 | wc -l)" -eq 1
echo "other_workspaces_hidden=YES"

test ! -e {shlex.quote(str(HOST_HOME / '.hermes'))}
test ! -e {shlex.quote(str(HOST_HOME / '.ssh'))}
test ! -e {shlex.quote(str(HOST_HOME / '.codex'))}
echo "host_home_hidden=YES"

test ! -e /sys
test "$(wc -l < /proc/net/route)" -le 1
echo "network_unshared=YES"

test "$(env | wc -l)" -le 12
echo "environment_allowlist=YES"
'''

result = run(
    candidate,
    runtime_dir,
    [
        "/bin/sh",
        "-c",
        probe,
    ],
    timeout=30,
)

print(result.stdout, end="")

errors: list[str] = []

if result.returncode != 0:
    errors.append(
        f"sandbox process returned {result.returncode}"
    )

write_probe = candidate / ".sandbox-write-test"

if not write_probe.is_file():
    errors.append(
        "write through /workspace did not reach host candidate"
    )
else:
    value = write_probe.read_text(
        encoding="utf-8"
    ).strip()

    if value != "sandbox-write-ok":
        errors.append(
            f"unexpected write probe value: {value!r}"
        )

expected = {
    "benchmark_root_hidden=YES",
    "evaluator_tests_hidden=YES",
    "other_workspaces_hidden=YES",
    "candidate_visible=YES",
    "candidate_writable=YES",
    "runtime_tree_hidden=YES",
    "host_home_hidden=YES",
    "network_unshared=YES",
    "environment_allowlist=YES",
}

for marker in expected:
    if marker not in result.stdout:
        errors.append(
            f"missing sandbox marker: {marker}"
        )

# Candidate mutation was intentional for the write probe.
# Remove it before cleanup.
write_probe.unlink(missing_ok=True)

cleanup(
    {
        "run_id": metadata["run_id"],
        "paths": {
            "candidate": str(candidate),
            "runtime": str(runtime_dir),
        },
    }
)

if errors:
    print()
    print("SANDBOX_VALIDATION=FAIL")

    for error in errors:
        print(f"  - {error}")

    sys.exit(1)

print()
print("SANDBOX_VALIDATION=PASS")
print(
    "candidate_content_sha256="
    + metadata["content_sha256"]
)
print(
    "candidate_baseline_commit="
    + metadata["baseline_commit"]
)
