#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path

from harness.evaluator import evaluate


OLD = Path(
    os.environ.get(
        "HERMES_BENCH_LEGACY_EVAL_ROOT",
        str(Path.home() / "hermes-model-eval" / "run-20260806-014012"),
    )
).expanduser().resolve()

EXPECTED = {
    "laguna": {
        "hidden": (34, 34),
        "supplemental": (7, 7),
        "overall": True,
    },
    "qwen": {
        "hidden": (33, 34),
        "supplemental": (6, 7),
        "overall": False,
    },
}

errors: list[str] = []

for model, expected in EXPECTED.items():
    result = evaluate(OLD / model)

    print()
    print(f"===== {model.upper()} =====")

    for name in (
        "public",
        "hidden",
        "supplemental",
    ):
        item = result[name]
        print(
            f"{name}="
            f"{item['passed']}/{item['total']}"
        )

    print(
        "hard_failures="
        + (
            ",".join(result["hard_failures"])
            if result["hard_failures"]
            else "NONE"
        )
    )

    print(
        "overall="
        + ("PASS" if result["pass"] else "FAIL")
    )

    for name in ("hidden", "supplemental"):
        actual = (
            result[name]["passed"],
            result[name]["total"],
        )

        if actual != expected[name]:
            errors.append(
                f"{model}.{name}: "
                f"expected {expected[name]}, "
                f"got {actual}"
            )

    if result["pass"] != expected["overall"]:
        errors.append(
            f"{model}.overall mismatch"
        )

if errors:
    print()
    print("EVALUATOR_VALIDATION=FAIL")

    for error in errors:
        print(f"  - {error}")

    sys.exit(1)

print()
print("EVALUATOR_VALIDATION=PASS")
