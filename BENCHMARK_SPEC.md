# LLM Model Benchmarks Specification

Canonical repository: <https://github.com/TomkoLabs/LLM-Model-Benchmarks>

## Active qualification generation

The repository-root `./benchmark-model` interface implements
`gx10-qualification-v4`: a thin orchestrator over pinned Spark Bench,
BenchLocal CLI, Infermark, and the retained local Hermes acceptance path.
`upstreams.lock.json`, `configs/qualification-v4.yaml`, `models.yaml`, the
schemas, and task manifests are authoritative. Upstream tools execute only from
clean exact-commit external checkouts, through array-based adapters, and their
structured outputs are copied into ignored run-owned artifact directories.

The detailed Hermes candidate/evaluator rules below remain authoritative for
the local acceptance component. `hermesbench-v3` scores are component evidence,
not the complete deployment decision; `SCORING.md` defines the outer hard gates
and 40/30/15/10/5 aggregate.

Generation v4 retains v3's exclusion of Docker-only BenchLocal packs from the default profiles
because the `llm` account has no Docker daemon access and Aider Polyglot's
upstream container requires a non-isolated network. It also fails the
BenchLocal component when the upstream result reports thinking contamination
and enforces the profile total wall across successive components.

Generation v4 makes reasoning an explicit deployment configuration and
separates policy, model transport, and agent-execution evidence. One trusted
loopback gateway normalizes OpenAI-compatible and native Ollama requests,
removes conflicting legacy controls, forwards only to the configured local
endpoint, and observes normal JSON or streaming response bytes before client
parsers consume them. It records only sanitized field-presence metadata; it
does not record prompts, response text, credentials, or hidden task content.
Pinned tools talk only to that gateway and remain unmodified.

The exact reasoning policy is `off`, `native`, or one curated supported
`effort:<level>` (`low`, `medium`, or `high` in the current serializer).
`configured` is a CLI resolver, never the effective policy: it is recorded only
as the requested selection source and resolves the explicit `reasoning_policy`
in `models.yaml`. Unsupported model/policy combinations fail preflight. No
inference probes or aliases are used to guess a maximum effort.
`off` serializes `reasoning_effort: none` for OpenAI compatibility and
`think: false` for native Ollama; `native` omits a forcing control; effort mode
serializes the exact named level. The requested selection is also immutable
provenance: `configured` places the resolved per-model policy in the primary
deployment track, while an explicit policy places the run in that exact
controlled-policy cohort. Primary-track deployments remain comparable even
when their curated effective policies differ. Generations v1 through v3 remain
frozen for historical runs.

## 1. Objective

Measure practical local-model deployment quality rather than relying primarily
on vendor benchmark scores. Spark Bench, BenchLocal, and Infermark primarily
exercise the configured endpoint directly; Hermes Agent separately exercises
controlled autonomous repository work. The outer qualification combines both.

A successful model must do more than generate correct code. It must maintain
requirements across a long run, use tools correctly, recover after failures,
verify its own changes, and avoid claiming success while known defects remain.

## 2. Benchmark generations

A benchmark generation freezes at minimum:

- Hermes version and exact Git commit
- Hermes toolsets
- benchmark repository Git commit
- task versions
- runner/scorer versions
- runtime and runtime version
- timeout policy
- turn budgets
- system/goal prompts
- compression policy
- filesystem/environment policy

A material Hermes or harness change creates a new benchmark generation.

Results from different generations may be shown historically but must not be
treated as directly interchangeable measurements.

Initial generation:

- Hermes Agent 0.20.4
- Hermes commit `533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9`

Generation `hermesbench-v2` retained that Hermes version/commit and froze
networkless candidate and
evaluator isolation, exact endpoint/digest verification, structured evaluator
results, per-case score mappings, complete candidate evidence capture, and
durable interruption state. Generation-v1 configuration/result data is
rejected by the v2 runner rather than silently compared.

Current generation `hermesbench-v3` retains the same Hermes commit and scoring
algorithm but changes evidence semantics materially. Hermes training
trajectories are supplemental. A no-reasoning conversation filtered by the
upstream training exporter as `discarded_no_reasoning` can still be a valid
agent execution when the trusted model-transport observer, runner/checkpoint
events, captured task state, and isolated evaluator agree. No trajectory or
final answer is fabricated. A broken observer, runner failure, timeout, or lack
of trustworthy agent-execution evidence remains distinct and fails closed.

## 3. Comparison modes

### Fair mode

All models receive the same effective context ceiling.

Initial target:

    131072 tokens

This isolates model quality more effectively.

### Maximum practical mode

Each model may use the deployment context considered practical for it.

Examples:

- Laguna APEX: 131072
- Qwen3.8-27B: 262144

Fair-mode and maximum-practical results are reported separately.

## 4. Model configurations

A configuration identifies:

- model alias
- exact runtime model/hash captured at run time
- quantization
- runtime
- context limit
- exact reasoning policy and serialized endpoint control
- reasoning/visible/tool response-field presence
- KV-cache format where observable
- sampling values where explicitly controlled

Changing reasoning policy, quantization, or context creates a distinct complete
deployment configuration. Curated configurations selected with `configured`
share the primary deployment ranking; explicit policy selections remain
separate controlled experiments keyed by their exact resolved policy.

Public identity is separate from the runtime alias. `model-metadata.yaml` is
keyed by the full immutable digest and supplies reviewed canonical checkpoint
metadata when present. An exact registry entry safely overrides or enriches
runtime metadata; without one, complete trustworthy `/api/show` identity fields
may form the public identity directly. Resolution never guesses from an alias
and never falls back from an unknown digest to another alias's metadata. The
context remains run-specific, so one digest can have separately reported
context deployments.

A completed result with missing canonical fields is marked
`METADATA_INCOMPLETE` for public display and excluded from the official ranking
without changing its original qualification outcome or artifact. The comparison
writer emits a review-only candidate keyed by digest under `generated-results/`;
it never edits the tracked registry. Historical results are enriched in memory
only.

Every completed `QUALIFIED` or legitimate `NOT_QUALIFIED` command regenerates
the ignored local comparison and the allowlisted public Markdown/JSON using the
same comparison implementation. Public ranking selects only the current
primary configured-deployment track and exact standard
profile/scoring/upstream compatibility group. Smoke, incompatible,
invalid, incomplete, and infrastructure-error runs remain unranked; repeated
trials remain separate. Per-file writes use the harness atomic artifact writer,
and no generated output is committed or pushed automatically.

Generation-v2 results are retained as historical diagnostics under the explicit
`LEGACY_THINKING_CONTROL_MISMATCH` label. Because v2 did not uniformly serialize
Ollama's supported thinking control, no v2 score enters the v4 current ranking,
including results from models that happened not to return reasoning.

Direct response and tool probes classify exact visible output, exact tool calls,
reasoning-only output, empty output, output-limit truncation, malformed tool
calls, and unexpected reasoning contamination separately. Reasoning is never
promoted into visible content or interpreted as a tool call. If a model returns
reasoning after a correctly serialized thinking-off request, that is a valid
model/deployment contract failure; construction, translation, transport, or
endpoint failures remain infrastructure failures.

The `gx10-direct-probe-v1` contract gives each direct response probe and each
direct tool-call probe the same fixed total completion allowance of 1,024
tokens. This ceiling is methodology-defined and does not vary by model,
reasoning policy, observed reasoning length, or prior result; the harness does
not retry a truncated probe. Exact visible-answer and exact structured-tool
gates remain unchanged, and actual completion-token usage is recorded so the
reasoning overhead remains measurable. The contract and ceiling are immutable
run provenance and are part of the comparison compatibility key. Historical
v2-v4 results that predate this recorded field retain their original 128-token
direct-probe cohort without artifact rewriting.

## 5. Task classes

### A. Coding correctness

Tasks may include:

- new implementation
- bug repair
- refactor with compatibility requirements
- API change
- multi-file change
- persistence changes
- concurrency edge cases
- validation
- regression repair

Every deterministic coding task should have public acceptance tests plus
harness-controlled acceptance tests not present in the candidate worktree.

### B. Security correctness

Coverage includes controlled software-security cases such as:

- parent traversal
- encoded traversal
- path normalization
- final-component symlinks
- intermediate symlinks
- archive extraction
- Unicode path semantics
- byte-vs-character length semantics
- overwrite policy
- permissions
- filesystem race-like cases where deterministic
- unsafe input
- authorization logic
- command/input escaping

Security scoring distinguishes implementation failure from a false claim that
the security requirement was satisfied.

### C. Agent/tool reliability

Tasks deliberately exercise:

- failed shell commands
- failed builds
- unavailable dependencies
- unexpected command output
- malformed or incorrect tool use
- failed tests
- failed formatting/linting
- incorrect initial diagnosis
- need for additional repository inspection
- need to revise an earlier plan

Measure whether the model notices, diagnoses, changes strategy, retries,
verifies, and continues.

### D. Long-horizon autonomous project

At least one deterministic project must require many meaningful agent/tool
actions and multiple files.

Target maturity:

    30-100+ meaningful actions

The project should include:

- repository discovery
- requirement analysis
- planning
- implementation
- test execution
- deliberate complications
- failed first approaches
- easy-to-miss requirements
- hidden edge cases
- state retention
- final verification

## 6. Candidate isolation

Each run starts from an identical immutable source snapshot.

Candidate changes occur only in a disposable worktree/sandbox. Hermes and its
model client run in the trusted control plane. Generation v3 registers a
run-local Hermes distribution containing only a foreground, non-PTY terminal
tool. In-process file tools and detached process controls are not exposed
because they execute or spawn from the trusted Hermes process. Every
model-selected terminal shell is forced through a harness-owned executable
wrapper into a new Bubblewrap namespace before model-selected shell text is
interpreted. The wrapper path is installed by trusted Hermes initialization;
containment does not depend on `BASH_ENV`, an inherited environment marker, or
candidate shell cooperation. Each namespace has:

- no network namespace route (candidate egress is blocked, including GX10)
- a cleared, fixed environment allowlist
- only the candidate workspace and ephemeral `/tmp` writable
- a minimal read-only system runtime
- no benchmark repository, run-time control directory, sibling workspace,
  user home, SSH, Codex, or Hermes configuration/source visibility

The wrapper and fail-closed tool-distribution policy are stored in the harness-only
runtime directory and are not mounted into the candidate namespace. A retained
trajectory is supplemental and is accepted only when it records exactly the
terminal-only toolset. The trusted model client may use only the run-local
gateway, which has one fixed validated private `/v1` target and disables
redirects.
The run-local policy also denies trusted-process IPv4/IPv6 connections and
name resolution for every destination except that exact private host and port, preventing
Hermes auxiliary-provider discovery or fallback from escaping that endpoint.

The candidate must not receive harness acceptance tests in its worktree.
Evaluation runs after the model turn with a trusted worker in a networkless
Bubblewrap namespace. The worker mounts only the read-only candidate, the exact
frozen test input, and pinned harness code. Candidate imports and calls run in
a second, nested networkless Bubblewrap namespace through a bounded RPC
adapter. That inner namespace sees the read-only candidate and a per-case
scratch directory, but not the test file, evaluator result descriptor,
benchmark repository, runtime artifacts, or host state. The complete process
group is killed on timeout or interruption. Candidate stdout and RPC envelopes
are untrusted and cannot directly establish case status; the outer worker owns
case discovery, assertions, and the structured result channel. Public tests
are also evaluated from the immutable harness copy, and any candidate public
test or task-prompt modification is separately reported as input-integrity and
requirement loss.

Evaluator implementations, case contracts, and hashes are public
reproducibility material, including sources stored under `private_tests/`.
That name describes the runtime boundary: these files are withheld from and
inaccessible to the candidate process during execution. It does not claim that
the source is confidential or unavailable to repository readers. Merely
placing tests in another directory owned by the same Unix user would not
provide the required runtime isolation.

Publication also means evaluator content may enter future model training data.
These results qualify a deployment; they do not prove absence of benchmark
contamination. Operators should retain unpublished real-work tasks and use
representative repository work as the final selection step.

The nested boundary prevents candidate Python from reading or copying evaluator
tests at runtime, accessing the result descriptor, writing the candidate, or using
network egress. A candidate process exit, malformed/forged RPC response, or
import failure becomes a model case error and cannot produce a pass or a
harness-owned result envelope.

Harness-owned result/runtime writes use no-follow directory traversal,
exclusive temporary files, and atomic replacement. Candidate runtime paths are
never exposed. Symlinked parents or destinations fail closed.

## 6.1 Runtime identity and live-run gate

Live execution resolves a full `sha256:` digest for the selected model tag at
run time. `models.yaml` is a curated override registry, not an execution
allowlist. Preflight checks exactly one matching ID through `/v1/models`,
exactly one matching name/digest through `/api/tags`, and a successful read-only
`/api/show` lookup. An unconfigured tag receives an in-memory model configuration
only when `/api/show` parameters or Modelfile metadata supplies one unambiguous
effective `num_ctx`; native maximum context is recorded separately and is never
substituted for that execution value. Runtime-reported architecture, parameter
size, and quantization are recorded with their sources.

Malformed, missing, ambiguous, or inconsistent identifiers fail closed. If a
curated alias config supplies `runtime_digest`, it is an optional expected value
and must equal the resolved digest. Redirects are rejected. A dry run does not
contact GX10 and reports unconfigured-tag discovery as pending. The resolved
digest is written to the aggregate, launch manifest, task result, and smoke
result. Identity is re-resolved after each task; a digest change or loss of
metadata is an infrastructure failure and invalidates the affected run.

No code path pulls, removes, renames, or changes a model.

The configured Ollama version is recorded as a frozen baseline. The authorized
metadata scope does not include a runtime-version endpoint, so the observed
runtime version is explicitly recorded as unverified rather than inferred.

## 7. Required captured metadata

Where reliably observable:

- benchmark generation
- benchmark Git commit
- task ID/version
- fixture Git commit
- model/config name
- runtime model ID/hash
- quantization
- runtime/version
- Hermes version/commit
- context ceiling
- requested reasoning-policy source and primary/controlled benchmark track
- exact resolved reasoning policy and cohort
- serialized OpenAI-compatible and native Ollama control
- KV format
- task prompt version

Correctness:

- public tests
- harness-controlled tests
- security tests
- requirements satisfied/missed
- unresolved defects

Agent behavior:

- turns
- tool calls
- tool-call failures
- malformed tool calls
- commands failed
- recoveries attempted
- recoveries successful
- repetitive loops
- false completion claims
- human intervention

Tokens/context where exposed:

- input tokens
- output tokens
- reasoning tokens
- compression events
- context usage/peak

Performance:

- wall-clock duration
- time to first token if available
- prompt evaluation speed
- decode speed
- memory where practical

Repository evidence:

- final Git diff
- diff statistics
- `git diff --check`
- final public-test output
- final harness-test output

Generation v3 retains v2's capture of tracked changes plus non-ignored untracked text,
binary, deletion, and rename evidence in the changed-file list, binary diff,
diff check, and final candidate archive. Ignored files are excluded.

Aggregate state is written before preflight and before the first task. Every
phase transition records the current task and any resolved candidate/runtime
paths. SIGINT and SIGTERM produce a canonical `INTERRUPTED` aggregate after
bounded descendant cleanup while retaining partial logs and runtime evidence.

Measured values must be distinguishable from estimates and unavailable values.

## 8. Hard-failure flags

At minimum:

- `SECURITY_FAILURE`
- `FALSE_COMPLETION`
- `UNRECOVERED_BUILD`
- `REQUIREMENT_LOSS`
- `TOOL_LOOP`
- `TIMEOUT`

Hard failures remain visible even if an aggregate numeric score is high.

## 9. Completion truthfulness

Completion claims are evaluated independently from implementation quality.

Examples:

- claims tests pass when they fail
- claims security requirement is fixed when harness tests disprove it
- omits known unresolved defect
- stops without checking acceptance criteria
- changes tests to legitimize behavior contrary to requirements

These are scored as agent/reliability failures and may trigger
`FALSE_COMPLETION`.

## 10. Repetition and variance

Framework development may use one run.

Major model decisions should eventually use at least three independent runs
per important configuration.

Aggregate reporting should include, where meaningful:

- success rate
- mean
- median
- best
- worst
- variance/spread

Raw individual runs remain inspectable.

## 11. Human intervention

Default benchmark runs prohibit corrective human intervention.

If intervention is required:

- record it explicitly
- preserve the pre-intervention state
- mark the run
- do not compare it to fully autonomous runs as equivalent

## 12. Git discipline

Benchmark framework development uses meaningful Git checkpoints.

Candidate repositories are disposable and may use Git internally, but their
commits are run artifacts rather than framework history.
