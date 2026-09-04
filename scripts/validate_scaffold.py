#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY.md",
    "BENCHMARK_SPEC.md",
    "SCORING.md",
    "models.yaml",
    "model-metadata.yaml",
    "configs/hermes-bench-v1.yaml",
    "configs/hermes-bench-v2.yaml",
    "configs/hermes-bench-v3.yaml",
    "configs/qualification-v2.yaml",
    "configs/qualification-v3.yaml",
    "configs/qualification-v4.yaml",
    "configs/qualification-v5.yaml",
    "upstreams.lock.json",
    "upstreams-v5.lock.json",
    "scripts/bootstrap",
    "schemas/task.schema.json",
    "schemas/result.schema.json",
    "schemas/benchmark-run.schema.json",
    "schemas/qualification-run.schema.json",
    "schemas/model-metadata.schema.json",
    "schemas/public-leaderboard.schema.json",
    "public-results/leaderboard.md",
    "public-results/leaderboard.json",
    "tasks/agent/convergence-smoke-v1/private_tests/test_hidden.py",
    "tasks/security/archiveguard-v1/private_tests/hidden_acceptance.py",
    "tasks/security/archiveguard-v1/private_tests/supplemental_acceptance.py",
]

errors: list[str] = []

for rel in REQUIRED:
    if not (ROOT / rel).exists():
        errors.append(f"missing: {rel}")

with (ROOT / "models.yaml").open("r", encoding="utf-8") as f:
    models = yaml.safe_load(f)

if models.get("schema_version") != 3:
    errors.append("active model configuration is not schema version 3")

required_models = {
    "laguna-apex-128k",
    "qwen38-q8-medium-262k",
    "qwen38-q8-medium-128k",
    "qwen38-flash-next-nvfp4-262k",
    "qwen38-flash-next-nvfp4-262k-quality",
}

actual_models = set((models or {}).get("models", {}))
missing_models = required_models - actual_models

if missing_models:
    errors.append(
        "missing model configs: " + ", ".join(sorted(missing_models))
    )

with (ROOT / "configs/hermes-bench-v3.yaml").open(
    "r", encoding="utf-8"
) as f:
    bench = yaml.safe_load(f)

if bench.get("hermes", {}).get("commit") != (
    "533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9"
):
    errors.append("Hermes v3 commit does not match frozen baseline")

if bench.get("schema_version") != 2:
    errors.append("active benchmark config is not schema version 2")

if bench.get("benchmark_generation") != "hermesbench-v3":
    errors.append("active benchmark generation is not hermesbench-v3")

with (ROOT / "configs/qualification-v5.yaml").open(
    "r", encoding="utf-8"
) as f:
    qualification = yaml.safe_load(f)

if qualification.get("generation") != "gx10-qualification-v5":
    errors.append("active qualification generation is not gx10-qualification-v5")

with (ROOT / "configs/qualification-v4.yaml").open(
    "r", encoding="utf-8"
) as f:
    historical_qualification = yaml.safe_load(f)
if historical_qualification.get("generation") != "gx10-qualification-v4":
    errors.append("historical qualification generation v4 is not preserved")

schemas = {}
for schema in sorted((ROOT / "schemas").glob("*.schema.json")):
    with schema.open("r", encoding="utf-8") as f:
        document = json.load(f)
    schemas[schema.name] = document

with (ROOT / "model-metadata.yaml").open("r", encoding="utf-8") as f:
    public_models = yaml.safe_load(f)
if public_models.get("schema_version") != 1:
    errors.append("public model metadata schema version is not 1")

with (ROOT / "public-results/leaderboard.json").open(
    "r", encoding="utf-8"
) as f:
    public_leaderboard = json.load(f)
if public_leaderboard.get("schema_version") != 1:
    errors.append("public leaderboard schema version is not 1")

if errors:
    print("SCAFFOLD VALIDATION: FAIL")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("SCAFFOLD VALIDATION: PASS")
print(f"model_configs={len(actual_models)}")
print(f"schemas={len(schemas)}")
print(f"benchmark_generation={bench.get('benchmark_generation', 'n/a')}")
