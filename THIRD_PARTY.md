# Third-party software and references

The repository's original source and documentation are licensed under
Apache-2.0. That license does not relicense the projects below. LLM Model
Benchmarks invokes integrated tools from separate, pinned checkouts under ignored
`.state/` directories; it does not vendor or redistribute their source.

The public-release audit found no upstream source copied or adapted into this
repository. The small files under `tests/fixtures/upstreams/` are synthetic
parser inputs, not upstream benchmark data or model responses.

## Pinned repositories

| Project | Official repository | Pinned commit | License at the pin | Use in LLM Model Benchmarks |
|---|---|---|---|---|
| Hermes Agent | <https://github.com/NousResearch/hermes-agent> | `533886c8b8eb67ff8b389b7f48e7d5e5d9c575b9` | MIT, Copyright 2025 Nous Research | Required external agent runtime. Its `batch_runner.py` and managed Python 3.11 environment are invoked without source modification or incorporation. |
| Spark Bench (qualification v4 historical) | <https://github.com/Weschera/spark-bench> | `364e6ecf684988b024cdd9b1ae1feb0e44342603` | No license file or explicit redistribution license was present at the audited pin. | Cloned and invoked as an unmodified external process. No Spark Bench code, scenarios, or result data are distributed here. Review its terms before use or redistribution. |
| Spark Bench (qualification v5 current) | <https://github.com/Weschera/spark-bench> | `125ba161d9a91b705ff0cbb22471ac2914d9dea8` | No license file or explicit redistribution license was present at the audited pin. | Separately pinned for v6.8.0 full uncapped quality and tier2 measurement; invoked unmodified with no redistributed source or result data. |
| BenchLocal CLI | <https://github.com/noonghunna/benchlocal-cli> | `47d1d66d678c893e6f2b6ff7aba448fcc6094e2d` | MIT, Copyright 2026 noonghunna. The upstream checkout also contains attribution for its MIT-licensed BenchLocal pack sources. | Cloned and invoked as an unmodified external process. No packs or BenchLocal source are incorporated here. |
| Infermark | <https://github.com/stef41/infermark> | `f7216dfc5dd67e68dd023e94b389a641c7111eb3` | Apache-2.0 | Cloned and invoked as an unmodified external process. No Infermark source is incorporated here. |
| Spark evaluations | <https://github.com/DanTup/spark-evals> | `4138b036a128f0f4a27c268194ace4fa15bcd6d4` | No license could be verified at the recorded historical pin. The pin was not fetchable from the official repository during the public-release audit. | Methodological reference only. It is not cloned, invoked, scored, or incorporated. |
| SWE Bench Rig | <https://github.com/lobanov/swe-bench-rig> | `bda19fdc7dc88c0bf538fd8a60c922d6d67190cc` | No license could be verified at the recorded historical pin. The pin was not fetchable from the official repository during the public-release audit. | Optional heavyweight reference only. It is not cloned, invoked, scored, or incorporated. |

`upstreams.lock.json` (v4) and `upstreams-v5.lock.json` (v5) are the
machine-readable authorities for integrated URLs,
commits, license-file digests, entry points, and output contracts. Updating a
pin requires a fresh source, license, adapter, fixture, and methodology review.

## Pinned Spark runtime packages

The setup command installs these packages into ignored `.state/python-deps/`
without vendoring them in Git:

| Package | Version | Declared license |
|---|---:|---|
| Playwright for Python | 1.62.0 | Apache-2.0 |
| pyee | 13.0.0 | MIT |
| greenlet | 3.5.5 | MIT AND PSF-2.0 |
| typing_extensions | 4.16.0 | PSF-2.0 |

The benchmark also invokes user-installed Ollama, Chromium, Bubblewrap, Git,
the system Python evaluator, and standard operating-system utilities. Those
programs are prerequisites and are not distributed by this repository.
