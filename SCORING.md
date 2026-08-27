# Scoring and qualification semantics

The active deployment scoring generation is `gx10-qualification-v4`. It
combines locally executed, pinned upstream results with the existing
`hermesbench-v2-automated-1` task evaluator. Numeric scores are secondary to
hard qualification gates.

## Top-level outcomes

- `QUALIFIED`: every mandatory gate and threshold for the selected profile
  passed.
- `NOT_QUALIFIED`: the run completed with valid evidence, but a model or
  deployment behavior failed.
- `INFRA_ERROR`: setup, identity, isolation, parser, evaluator, or artifact
  integrity prevented a valid model decision.

An infrastructure error is never converted into a zero model score. A partial
component score remains visible but cannot produce `QUALIFIED`.

The backward-compatible technical outcome is accompanied by two clearer axes:

- `result_validity`: `VALID`, `QUARANTINED`, `INFRA_FAILURE`, or `INCOMPLETE`;
- `profile_decision`: `MEETS_PROFILE`, `DOES_NOT_MEET_PROFILE`, or
  `NOT_ASSESSED`.

A valid `QUALIFIED` result meets the exact deployment profile. A valid
`NOT_QUALIFIED` result completed but does not meet that profile; it is not a
claim that the model is universally unusable. Quarantined, infrastructure, and
incomplete results are `NOT_ASSESSED` and have no meaningful rank. For machine
compatibility, a quarantine retains technical outcome `INFRA_ERROR`.

Spark's quarantine marker remains authoritative. A passing golden gate and
run-valid marker do not turn a quarantined provisional score into a valid
score. The harness records the upstream reason and provisional metrics, sets
`result_validity=QUARANTINED`, and stops the run without ranking it.

## Hard gates

The following are mandatory when their component is selected:

- exact runtime model and immutable digest resolution;
- local endpoint and no-redirect policy;
- direct response and structured tool-call protocol success;
- exact, clean upstream checkout and compatible structured output;
- upstream self-integrity/golden gates;
- no upstream-reported sampling contamination;
- no reasoning contamination when the effective policy is `off`;
- valid reasoning-policy serialization and trustworthy model-transport evidence;
- no non-local socket or auxiliary provider/fallback attempt;
- mandatory coding/tool thresholds;
- every selected Hermes task completes within both inactivity and total wall;
- trustworthy Hermes process, checkpoint, model-response/tool, task-state, and evaluator evidence;
- a training trajectory and textual final response are supplemental rather than mandatory;
- isolated evaluator pass, protected-input integrity, and complete evidence;
- no candidate change outside the task manifest's frozen allowed-change patterns;
- run-owned candidate/runtime cleanup after evidence preservation.

Both direct probes use the frozen `gx10-direct-probe-v1` total completion
ceiling of 1,024 tokens for every model and every effective reasoning policy.
This is one uniform allowance, not a per-model allowance or an adaptive retry
budget. Output-limit truncation remains a model/deployment gate failure, while
the exact answer and exact tool-call requirements are unchanged.

## Deployment score

When gates pass, full-profile components use a 0–100 scale:

| Component | Weight | Inputs |
|---|---:|---|
| coding | 40 | Spark's execution-backed code-domain quality; isolated Hermes coding completion is scored separately below |
| Hermes | 30 | percentage of selected local Hermes tasks with full PASS/evidence/cleanup |
| tool/instruction | 15 | mean BenchLocal behavioral/targeted pack score |
| reliability | 10 | Spark repeat reliability; Hermes pass rate when Spark is absent |
| performance | 5 | equal blend of capped concurrency-1 decode throughput and TTFT normalization, with errors scoring zero |

For executed components `c` with weights `w`, the displayed aggregate is:

```text
sum(score[c] * w[c]) / sum(w[c])
```

The report records `executed_weight`. A smoke result with less than 100 executed
weight is a plumbing qualification, not a comparable full deployment score.

Performance normalization is intentionally small and transparent:

```text
throughput = min(100, concurrency_1_tokens_per_second * 10)
ttft       = max(0, 100 - max(0, ttft_p50_seconds - 1) * 10)
performance = (throughput + ttft) / 2
```

Raw TTFT, ITL, latency distributions, requests/second, tokens/second, request
counts, errors, context, and concurrency are always shown. Performance cannot
outvote correctness.

Default full-profile thresholds are aggregate 70, coding 60, Hermes 100,
tool/instruction 70, and reliability 80. Threshold/configuration changes create
a materially different deployment experiment and must be recorded.

## Public ranking order

Only complete `VALID` / `MEETS_PROFILE` results from the primary
configured-deployment track and exact current standard profile, scoring
generation, weights, profile configuration, direct-probe contract and ceiling,
and upstream pin set enter the
deployment ranking. Their resolved per-model reasoning policies may differ and
remain recorded on each row. Explicit controlled-policy runs are grouped by
their exact effective policy and do not enter this primary ranking. Ranked
deployments are ordered by overall score descending, then by
coding, Hermes, tool/instruction, reliability, and performance scores
descending, followed by timestamp and run ID. This tie-break does not alter any
score. Complete `NOT_QUALIFIED` results use the same deterministic ordering in
a separate “Completed with profile limitations” table. Repeated trials remain
separate and are not averaged. V2 thinking-control results are historical
diagnostics and are excluded from the v4 group even when all other pins match.

## Hermes task evaluator

The retained `hermesbench-v2-automated-1` evaluator maps frozen test cases to
correctness, security, and requirement IDs. Correctness/security are case pass
percentages. Requirement retention is the percentage of requirements for which
every mapped case passes. A disproved affirmative completion claim scores
truthfulness zero; a passing claimed completion scores 100. Unmeasured values
remain null.

Task outcomes remain `PASS`, `FAIL`, `TIMEOUT`, or `HARNESS_ERROR`. A missing
or malformed optional training trajectory does not override trustworthy v3
execution evidence. A genuinely empty direct response is still a model failure;
a missing/broken transport observer, failed runner, invalid identity, fallback
activity, or evaluator contamination is infrastructure failure even if
candidate files happen to satisfy deterministic tests.
