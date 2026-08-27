# Reports

Human-readable aggregate and compatibility-sensitive reports are stored here.

Local comparison output is generated into ignored `generated-results/` by
default:

```sh
./benchmark-model compare
```

The comparison command also regenerates the tracked, allowlisted public files:

```sh
public-results/leaderboard.md
public-results/leaderboard.json
```

Review their diff before committing or publishing. Do not copy raw run logs,
trajectories, prompts, responses, endpoint details, hostnames, local paths, or
candidate archives into a public report.
