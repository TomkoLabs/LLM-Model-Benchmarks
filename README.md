# LLM Model Benchmarks

[Public ranked results](public-results/leaderboard.md) ·
[Canonical repository](https://github.com/TomkoLabs/LLM-Model-Benchmarks)

LLM Model Benchmarks is a hybrid local deployment-qualification harness for
coding models served through curated Ollama, DS4, and vLLM adapters. Spark Bench,
BenchLocal, and Infermark primarily exercise the configured model endpoint
directly. Hermes stages run controlled repository tasks through Hermes Agent.
Together they qualify the complete local deployment stack—not only Hermes and
not only the abstract model weights—with explicit hard gates and reproducible
provenance.

The result qualifies the complete deployment combination: model weights and
immutable digest, quantization, prompt template and runtime, context and
sampling configuration, hardware, Hermes version, benchmark profile, and
harness revision. It is not a claim about an abstract model family.

## Purpose and intended use

Public leaderboards are useful for discovering candidate models. This project
is the narrower local confirmation step: it qualifies finalist deployments on
NVIDIA GB10/GX10-class hardware, with particular attention to coding and
Hermes-style agentic work. Exact quantization, context, prompt template,
reasoning policy, serving runtime, and reliability are part of the deployment
being compared.

The project is not intended to benchmark every model release. Use it to reduce
a shortlist, then evaluate the finalists on representative real repositories.
Real-work acceptance remains the final model-selection step.

## Release status

This is a stable, frozen reference release. It has no active feature roadmap
or promise of frequent upstream-pin updates. The completed configured-policy
GX10 v5 cohort contains seven standard deployments. Qwen3.8-Flash-Next NVFP4
and Laguna-S-2.1-Uncensored at 262144 context met every hard gate; other valid
completed deployments remain visible as `DOES_NOT_MEET_PROFILE` results. V4
is retained as historical, non-comparable evidence.

## Supported platform

The model-serving baseline is one NVIDIA GB10/GX10-class system running Linux,
with one model runtime loaded at a time. Ollama `0.32.15` remains the frozen
baseline runtime. The harness supports both
co-location on that GX10 and the currently exercised split layout: a Linux
controller runs this harness, Hermes Agent `0.20.4`, Bubblewrap, and Chromium
against an explicitly configured numeric private-network GX10 endpoint. The
same endpoint and fail-closed network policy apply in both layouts.

Other Linux systems and private OpenAI-compatible deployments may work, but
their results describe those deployments and should not be compared with the
GX10 baseline unless the profile, scoring generation, upstream pins, and
runtime configuration match. macOS and Windows are not currently supported
because candidate and evaluator isolation depends on Linux namespaces and
Bubblewrap.

## Prerequisites

- A clean Git checkout and an unprivileged Linux user with user namespaces
  enabled.
- An already-installed curated model runtime. The benchmark never pulls,
  removes, renames, loads, switches, or changes a model.
- Its adapter's read-only metadata and OpenAI-compatible inference endpoints.
  Ollama additionally requires `/api/tags`, `/api/show`, and `/api/version`.
- curl, Git, Bubblewrap at `/usr/bin/bwrap`, Chromium at `/usr/bin/chromium`, and
  system Python at `/usr/bin/python3`.
- Hermes Agent `0.20.4` at commit
  `533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9`, including its managed Python
  3.11 environment and `batch_runner.py`.
- Enough storage for ignored upstream checkouts and run artifacts, plus the
  profile's declared wall time.

On Debian or Ubuntu, install the operating-system tools with the distribution
package manager. Package names are normally `curl`, `git`, `bubblewrap`,
`chromium`, and `python3`. Do not grant the benchmark or candidate model sudo
access.

## Installation and bootstrap

Clone this repository, install Hermes Agent using its official instructions,
and freeze that checkout to the required commit. The managed installation is
normally under `~/.hermes/hermes-agent`; set `HERMES_AGENT_ROOT` if yours is
elsewhere.

```sh
git clone https://github.com/TomkoLabs/LLM-Model-Benchmarks
cd LLM-Model-Benchmarks

curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
export HERMES_AGENT_ROOT="${HERMES_AGENT_ROOT:-$HOME/.hermes/hermes-agent}"
git -C "$HERMES_AGENT_ROOT" fetch origin 533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9
git -C "$HERMES_AGENT_ROOT" checkout --detach 533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9
"$HERMES_AGENT_ROOT/venv/bin/python" -m pip install -e "$HERMES_AGENT_ROOT"

./scripts/bootstrap
```

`./scripts/bootstrap` is the reproducible project setup command. It checks the
required executables and Hermes Python modules, creates or verifies the exact
external checkouts under ignored `.state/`, verifies their commits and license
digests, and installs only the pinned Spark Python packages into
`.state/python-deps/`. It is idempotent and does not install system packages,
change Ollama, or download a browser or model.

No separate package manager, container runtime, or Docker daemon is required
for the active profiles.

## Quick start

After bootstrap, point the harness at a local compatible endpoint and inspect
the fully resolved plan without inference:

```sh
export HERMES_BENCH_ENDPOINT=http://localhost:11434/v1
./benchmark-model \
  --model agent-main:latest \
  --profile smoke \
  --reasoning-policy configured \
  --dry-run \
  --no-setup
```

Remove `--dry-run` only when the configured alias, digest, context, endpoint,
and policy are correct. For an unconfigured runtime tag, select an explicit
policy such as `--reasoning-policy off`; the harness never guesses one.
Standard runs can take hours and should be reserved for finalists that have
already passed bootstrap, dry-run review, and smoke qualification.

## Endpoint and model configuration

The default endpoint is Ollama on the same machine:

```sh
cp config/local.example.yaml config/local.yaml
```

`config/local.yaml` is ignored. Set its `endpoint`, pass `--endpoint`, or set
`HERMES_BENCH_ENDPOINT`. The URL must be explicit `http://`, use loopback or a
numeric RFC1918 private address, include a port, and end exactly in `/v1`.
Credentials, public addresses, DNS names, redirects, queries, fragments, and
implicit ports are rejected.

An installed Ollama tag can run directly without an entry in `models.yaml`.
Live preflight matches the exact tag through `/v1/models` and `/api/tags`, reads
`/api/show`, resolves the immutable digest and reported architecture, parameter
size, quantization, and an explicit `num_ctx`, then constructs an in-memory
configuration. If neither `/api/show` parameters nor its Modelfile contains a
trustworthy `num_ctx`, the command stops with that single missing-field error;
it never guesses from the native checkpoint maximum or from the alias name.

Curated deployments may instead select the `ds4` runtime adapter. That adapter
uses only `/v1/models` and `/v1/chat/completions`; it never sends Ollama-native
requests. DS4 entries must freeze the installed runtime version, configured
context, base GGUF and DSpark drafter checksums, DSpark state, API mode,
reasoning control, and streaming/tool-call capabilities. The runner records
that provenance with the endpoint-reported exact model identifier. It does not
start, stop, or switch model engines automatically; operators must establish
the intended runtime before preflight.

The `vllm` adapter is likewise restricted to `/v1/models` and
`/v1/chat/completions`. A curated vLLM entry freezes the exact checkpoint and
revision, serving implementation and revision, quantization, context, API
capabilities, and a named reasoning-control profile. Because vLLM does not
report a weight-file checksum through this API, `runtime_digest` is explicitly
the SHA-256 of the versioned canonical deployment-provenance descriptor—not a
weight digest. Preflight recomputes that digest and verifies the exact model ID
and reported `max_model_len` without probing Ollama endpoints.

`models.yaml` schema v3 remains the higher-precedence home for curated run-local
contexts, explicit reasoning policies, supported policies, and frozen
reproduction profiles. Public model identity lives
separately in `model-metadata.yaml`, keyed by the full immutable digest. The
registry can override or enrich runtime metadata with a reviewed canonical
checkpoint, release, source, and optional verified URL. An alias is only
secondary metadata and may appear under more than one digest; it is never used
to infer a model name.

Commit curated configuration before benchmarking because the runner rejects a
dirty control-plane checkout. Set `runtime_digest` to a full Ollama `sha256:`
digest when reproducibility requires an exact identity, or `null` to resolve
and record the already-installed digest at run time. Runtime-discovered
canonical metadata can form a complete public identity when every required
field is trustworthy. Otherwise the run still completes, is marked
`METADATA_INCOMPLETE` only for public ranking, and writes a review-only
`generated-results/model-metadata-candidates.json`; tracked metadata is never
updated automatically.

Verify the offline plan before contacting the model:

```sh
./benchmark-model --model <configured-alias-or-runtime-tag> --profile smoke --dry-run
```

`gx10-qualification-v5` is the default generation. Historical v4 replay is
explicit and uses its original configuration and upstream lock:

```sh
./benchmark-model --qualification-generation gx10-qualification-v4 --model <alias> --profile standard
```

The default `--reasoning-policy configured` resolves the exact curated policy
from `models.yaml`. An unconfigured tag must use an explicit policy such as
`--reasoning-policy off` or `--reasoning-policy native`; the benchmark never
guesses a reasoning level. Its dry run then reports runtime discovery as
pending and remains fully offline. Exact existence, digest, context, and model
support are verified by live preflight.

Supported CLI policy syntax is `off`, `native`, `effort:low`,
`effort:medium`, and `effort:high`. A curated model may expose only a subset.
`off` is a controlled no-reasoning cohort; it is useful for compatibility and
efficiency measurement but is not silently substituted for the curated
deployment result. The default `configured` selection places each model's
curated effective policy in the primary deployment ranking, so complete
deployments can be compared even when their resolved policies differ. Explicit
policy selections form controlled-policy cohorts keyed by the exact effective
policy. Every result records both the selection source and resolved policy.

Direct endpoint response and tool-call checks use one frozen 1,024-token total
completion ceiling under `gx10-direct-probe-v1`, regardless of model or
reasoning policy. The runner does not adapt that ceiling or retry truncation.
It still requires the exact visible answer and exact structured tool call, and
records both the ceiling and actual completion-token usage in provenance and
the report. Reasoning fields remain separate evidence, while visible `<think>`
tags are classified as contamination under every policy.

## Run one model

Use smoke to verify a new installation:

```sh
./benchmark-model --model <configured-alias-or-runtime-tag> --profile smoke
```

Run the default standard qualification with one command:

```sh
./benchmark-model --model <configured-alias-or-runtime-tag>
```

The explicit equivalent is:

```sh
./benchmark-model --model <configured-alias-or-runtime-tag> --profile standard
```

Do not start standard or overnight until bootstrap, dry-run, and smoke all
behave as expected.

## Profiles

| Profile | Use | Selected work | Expected duration and hard wall |
|---|---|---|---|
| `smoke` | Installation and plumbing | Direct protocol checks, two targeted BenchLocal cases, one short Infermark request, one small Hermes repository task | About 10–20 minutes; 30-minute wall |
| `standard` | Current comparable deployment qualification | SparkBench v6.8.0 full uncapped (76 scenarios, two repeats, temperature 0.3, thinking off, no request timeout), optional separate tier2 performance sweep, BenchLocal quick, Infermark, and two Hermes tasks | About 8–12 hours on the baseline GX10; 12-hour wall |
| `overnight` | Finalist stability | The same v6.8 full uncapped suite with three repeats, optional tier2 sweep, larger BenchLocal/Infermark work, and ArchiveGuard | About 12–18 hours on the baseline GX10; 18-hour wall |

Each subprocess also has an inactivity deadline. Progress artifacts and trusted
heartbeats reset inactivity, but never extend the profile's total wall.

## What runs

The five major stages are:

1. **Direct checks** — validate configuration, resolve the exact model through
   Ollama's OpenAI and native APIs, record its immutable digest, and verify
   plain-response and structured tool-call behavior.
2. **Spark Bench** — broad deterministic capability, coding, tool, and
   reliability scenarios sent primarily to the endpoint through the pinned
   external checkout.
3. **BenchLocal** — deterministic instruction and tool-call packs sent directly
   to the endpoint, including reasoning-contamination checks.
4. **Infermark** — direct endpoint latency, time to first visible content
   chunk, visible-content inter-chunk latency, errors, request throughput, and
   end-to-end visible output-chunk throughput.
5. **Hermes** — repository tasks through the frozen Hermes Agent runtime,
   networkless candidate workspace, and independent evaluator boundary.

Hard gates are evaluated before the weighted aggregate. A high score cannot
override a wrong digest, protocol failure, mandatory upstream failure,
reasoning contamination under `off`, fallback or prohibited network attempt,
missing or broken model-transport/agent-execution evidence, evaluator failure,
timeout, scope violation, evidence failure, or cleanup failure. A Hermes training
trajectory and textual final response are supplemental when trusted execution
and evaluator evidence prove valid task completion. See `SCORING.md` for the
unchanged weights and thresholds and `BENCHMARK_SPEC.md` for the trust boundaries.

## Outcomes and exit statuses

| Outcome | Exit status | Meaning |
|---|---:|---|
| `QUALIFIED` | 0 | All mandatory gates and profile thresholds passed |
| `NOT_QUALIFIED` | 1 | The run completed validly, but this deployment failed a model-quality gate or threshold |
| `INFRA_ERROR` | 2 | Setup, identity, adapter, evaluator, or artifact integrity prevented a valid model judgment |
| interrupted | 130 | The operator interrupted the process; partial diagnostics are preserved |

Completed output also prints `profile_decision` and `result_validity`.
`MEETS_PROFILE` means the exact deployment profile passed;
`DOES_NOT_MEET_PROFILE` means a valid run completed with profile limitations;
`NOT_ASSESSED` means no valid comparable result exists. A Spark quarantine is
reported as `QUARANTINED / NOT_ASSESSED` while retaining technical
`INFRA_ERROR` compatibility.

`INFRA_ERROR` and incomplete runs are never model failures and must not be
ranked as such. A legitimate `NOT_QUALIFIED` result should be accepted rather
than tuned away by changing thresholds or profiles. Batch orchestration should
continue from exit status 1 when a later profile is requested, and stop or skip
only on exit status 2 (or an operator interruption).

## Results and artifacts

Each invocation writes an ignored directory under `runs/<run-id>/`:

```text
manifest.json         resolved configuration, host, model, pins and provenance
partial-results.json  atomic progress and interruption evidence
results.json          schema-validated outcome, gates, component scores and metrics
report.md             concise human-readable operator summary
logs/                 upstream process logs
artifacts/            raw upstream outputs and copied Hermes evidence
```

The manifest and raw artifacts are needed to diagnose and reproduce a run.
The report is for quick reading; `results.json` is authoritative for automated
comparison. Never infer success from the wrapper's historical shell output
alone.

Run artifacts can contain prompts, model reasoning and responses, generated or
modified code, hostnames, endpoint addresses, and diagnostic logs. Review and
sanitize every artifact before publication. Do not publish raw trajectories or
private run directories by default.

## Compare local runs

Generate a deterministic comparison without contacting a model or rerunning a
benchmark:

```sh
./benchmark-model compare
```

The command scans every `runs/*/results.json`, retains repeated model runs, and
writes ignored local detail to `generated-results/comparison.md` and
`generated-results/comparison.json`. Incomplete exact-digest identities are
also collected in the review-only
`generated-results/model-metadata-candidates.json`. It also regenerates the
deliberately sanitized, tracked `public-results/leaderboard.md` and
`public-results/leaderboard.json`. Invalid, incomplete, `INFRA_ERROR`, quarantined, smoke,
metadata-incomplete, and incompatible results remain unranked diagnostics.
Only complete results in the primary configured-deployment track and exact
current standard profile, scoring version, direct-probe contract and ceiling,
and upstream pin set enter the
public decision tables. Each row prominently retains its resolved effective
reasoning policy. Explicit controlled-policy runs remain visible diagnostics
and never silently replace the primary ranking.

The current `gx10-qualification-v5` profile routes every inference request
through a trusted loopback gateway. It applies the effective runtime policy,
removes conflicting legacy fields, and observes JSON/SSE response-field
presence without retaining prompt or response content. Qwen vLLM serializes
`chat_template_kwargs.enable_thinking=false` for `off` and omits that forcing
field for `native`; Ollama and DS4 retain their existing runtime-specific
controls. A configured Qwen deployment may honor explicit methodology request
controls, while an explicit whole-run `--reasoning-policy off` remains
authoritative. V2 through v4 results remain unchanged historical cohorts; v2
thinking-control results remain explicitly labeled
`LEGACY_THINKING_CONTROL_MISMATCH`.

Every completed benchmark command performs this same finalization after its
individual schema-validated result and report are written. It runs after both
`QUALIFIED` and legitimate `NOT_QUALIFIED` outcomes, prints the updated ranking
last, and preserves the benchmark's outcome and exit status if post-processing
fails. It does not run after an infrastructure failure.

The public leaderboard uses full resolved model identities. The runtime alias
is retained in its own secondary column for local operator convenience. The
qualified table is ranked by score with deterministic component/timestamp
tie-breaking; completed `NOT_QUALIFIED` trials remain separate, and repeated
runs are never averaged or discarded. Smoke scores are never compared with
standard scores.

Generation v5 pins SparkBench commit
`125ba161d9a91b705ff0cbb22471ac2914d9dea8` under methodology
`v6.8.0-full-uncapped`. Standard uses tier `all` (76 scenarios), two repeats,
temperature `0.3`, thinking `off`, uncapped responses, and request timeout `0`
(none). Quality explicitly skips the automatic sweep. The harness invokes
tier2 separately with a positive request timeout, so an unavailable or partial
optional measurement cannot invalidate a valid quality result.

SparkBench tier2 **Generation tok/s** is native completion-token usage divided
by streaming decode time after the first reasoning, content, or tool-call
token. The representative summary is the valid single-stream row with the
shortest measured prompt context; every context and concurrency row is retained.
Tier2 also records prefill tok/s (prompt tokens divided by TTFT) and TTFT. A
failed measurement is null with explicit error evidence, never zero.

Pinned Infermark streaming mode increments its legacy `tokens_per_second` counter once
per non-empty OpenAI `delta.content` event and divides by the entire concurrency
level's wall duration. It does not tokenize text, request streaming usage, or
count `delta.reasoning_content`, so that field is reported as **E2E visible
chunk/s**, not generation token/s. Its TTFT is correspondingly time to first
visible content chunk (TTFC), and its ITL samples are intervals between later
visible content chunks.

The **Generation speed** column prefers the representative SparkBench tier2
decode measurement when present. Pinned streaming Infermark does not provide
true token/s, so results without valid tier2 data show an explicitly
approximate visible stream cadence in `est. chunks/s`, calculated as `1 / mean
ITL`. This is the available metric
closest to perceived visible text cadence, but it is not tokenizer decode
speed and, for reasoning models, excludes the preceding reasoning-content
stream. Raw TTFT and ITL distributions and the legacy field remain in the run
result. Infermark cannot provide prefill throughput; SparkBench tier2 can,
using server-reported prompt-token count and separately observed TTFT.

V4 and v5 are distinct public cohorts. V4 rows—including the preserved Qwen
native-thinking/MTP2 result—remain visible as `HISTORICAL_COHORT` diagnostics
and are never ranked against v5. The current Qwen quality alias records
thinking off and MTP off while retaining the same immutable checkpoint/runtime
provenance digest. Reproductions performed outside this harness may inform
operator investigation, but are never imported as official leaderboard rows.

Generated public files are not automatically committed or pushed. The owner
must inspect the sanitized diff before publishing an update. Third-party
submissions must not be mixed into the official GX10 baseline without
equivalent provenance, canonical model identity, immutable digest, profile and
scoring version, upstream pins, and runtime/hardware metadata.

## Security and isolation

- Candidate commands run in a Bubblewrap namespace with no network, a fixed
  environment, a disposable writable candidate tree, and no view of private
  tests, the benchmark control plane, Hermes home, SSH/Codex data, sibling
  workspaces, or runtime evidence.
- Evaluators run in a separate networkless namespace with a read-only candidate
  mount and harness-owned test inputs. Candidates cannot read hidden tests or
  forge the evaluator result channel.
- Hermes receives a disposable configuration with one trusted local endpoint,
  no fallback provider, and a fail-closed socket policy. OpenRouter, Nous, and
  other auxiliary-provider attempts are hard failures.
- Upstream tools execute as the invoking unprivileged user from exact clean
  commits with a cleared credential/proxy environment and a run-local network
  policy. They are third-party code, not a substitute for host-level isolation;
  review their pinned source before use on sensitive machines.
- The benchmark never requires sudo and never modifies the model server,
  system configuration, Hermes source, or model inventory.

Isolation relies on Linux user namespaces, Bubblewrap, the pinned Hermes
layout, and the evaluator's system-Python dependencies. A hostile kernel,
administrator, model server, or modified upstream checkout is outside the
threat model.

### Published evaluators and contamination limitation

Evaluator implementations, contracts, and hashes—including files under
`private_tests/`—are published for reproducibility. Here, `private_tests`
means withheld from and inaccessible to the candidate process during a run;
it does not mean confidential or unavailable to repository readers. The
runtime isolation boundary remains enforced and tested.

Public task and evaluator source may enter future model training data. Results
are therefore deployment qualification, not proof against benchmark
contamination. Operators evaluating future models should retain separate,
unpublished real-work acceptance tasks and make representative repository work
the final selection step.

## Public-result privacy

Tracked public summaries contain only allowlisted deployment identities,
scores, decisions, compatibility metadata, and sanitized diagnostic reasons.
Raw prompts, responses, reasoning, trajectories, local paths, endpoints,
hostnames, and logs remain in ignored local run directories and are not
published by the comparison command. Operators must still review every
generated public diff before publication.

## Troubleshooting

- **Hermes Python not found:** install the pinned Hermes checkout or set
  `HERMES_AGENT_ROOT`/`HERMES_BENCH_PYTHON`, then rerun `./scripts/bootstrap`.
- **Missing Bubblewrap or Chromium:** install the operating-system package and
  confirm `/usr/bin/bwrap` and `/usr/bin/chromium` are executable. Also confirm
  unprivileged user namespaces are enabled.
- **Endpoint rejected or unreachable:** use an explicit loopback or numeric
  private `http://host:port/v1` URL. Confirm Ollama's `/api/version` and
  `/v1/models` respond without redirects or authentication.
- **Missing or ambiguous model:** confirm the exact installed tag appears once
  in Ollama. For an unconfigured tag, also ensure `/api/show` exposes one
  trustworthy `num_ctx`; add a curated `models.yaml` profile only when the
  runtime does not store the intended run-local context or reproduction needs a
  frozen override.
- **Digest mismatch:** the installed Ollama object differs from the frozen
  configuration. Select the intended local model; do not weaken the identity
  check or silently change the recorded digest.
- **Dependency or license mismatch:** remove no evidence and do not patch an
  upstream checkout in place. Inspect the reported `.state/` checkout, pin,
  and license digest; update pins only as a reviewed methodology change.
- **Timeout:** distinguish inactivity from the total profile wall in
  `manifest.json` and logs. Slow but productive work cannot exceed the declared
  wall. Use smoke for plumbing; do not increase scored limits merely to obtain
  a pass.
- **Dirty repository:** preserve or commit intended configuration changes and
  remove unrelated generated files. Runs require a clean control plane.

## Reproduction and attribution

Exact upstream URLs, commits, versions, entrypoints, license-file hashes, and
output contracts are frozen generation-by-generation in `upstreams.lock.json`
(v4) and `upstreams-v5.lock.json` (v5). `THIRD_PARTY.md` records
how each repository is used and which code is, or is not, incorporated. A pin
update requires a fresh license and adapter audit and normally creates a new
comparison group or qualification generation.

Do not compare incompatible profiles, scoring versions, qualification
generations, upstream pin sets, model digests, quantizations, context limits,
runtime/template versions, or materially different hardware as though they
were repeated measurements of one deployment. Use `manifest.json` to establish
reproduction compatibility.

LLM Model Benchmarks is available under Apache-2.0. External
projects retain their own licenses and terms; see `LICENSE` and
`THIRD_PARTY.md`.
