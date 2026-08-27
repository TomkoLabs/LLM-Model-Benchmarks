# Results

Canonical machine-readable aggregate run results are stored here. Each
embedded task result validates against `schemas/result.schema.json`; the outer
record validates against `schemas/benchmark-run.schema.json`.

The active schemas are generation-v2 only. Scores are bounded to 0-100,
runtime identity includes the verified full digest for every live task, and
model/evaluator/infrastructure statuses remain independent.

Large raw trajectories, full binary diffs/final candidate archives, and task
diagnostics stay in the external runtime root and are referenced by path and
digest from the canonical result record. Repository-local
`artifacts/runs/` remains available for explicitly preserved large artifacts
and is ignored by Git.
