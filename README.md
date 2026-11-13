# hermes-usage

Terminal usage dashboard for Hermes Agent, over every model you run. Reads
`state.db` read-only; per-model rows grouped by billing family (Claude
subscription / Local / provider), with subtotals, a grand total, and the full
token split: input, output, cache read, cache write, thinking, est. $.

Reference shape: ccusage-style tables; built because `hermes insights` has no
per-model cost, no msgs column, and no family grouping.

## Use

```
hermes-usage                 # last 30 days, all sessions
hermes-usage --today | --week | --month | --days N | --all
hermes-usage --user          # drop subagent/cron/tool/oneshot sessions
hermes-usage --tasks         # aux split: what approval/compression/title/
                             # vision/background_review burn under each model
hermes-usage --live          # re-render every 5s (Ctrl-C to quit)
hermes-usage --trend 30      # daily est-cost sparklines from the nightly rollup
```

Install: symlink `hermes_usage.py` onto PATH as `hermes-usage`
(the host: `~/.local/bin/hermes-usage` -> this repo).

## Nightly rollup

`scripts/rollup.py` appends per-model daily totals to
`~/.hermes/cache/usage_daily.jsonl`; idempotent per date. Hermes cronjob
`usage-nightly-rollup` (no_agent, 03:17, script
`~/.hermes/scripts/usage-nightly-rollup.sh`) keeps it current. `--trend`
reads that file and degrades to a one-line note before the first run.

## Known approximations

- Msgs per model come from window attribution over `session_model_usage`
  (task='') row spans; sessions that switch models mid-conversation attribute
  per timestamp, rows outside every window are dropped.
- The Sess column is per model: a model-switching session counts twice
  (the header carries the distinct count; the table prints a note).
- Costs are `estimated_cost_usd`/`actual_cost_usd` as Hermes recorded them;
  unpriced models (cost_source=none) show `—`, priced-zero shows `$0.00`.
- Daily rollup attributes a usage row's whole span to its `first_seen` date.
- Empty model/provider rows surface as `(unknown)` under an `Unknown` family
  and are included in totals.

## Development

Tests: `python3 -m unittest tests.test_usage -v` (real temp sqlite fixtures).
Spec: `docs/specs/2026-11-12-hermes-usage-design.md`.
Plan: `docs/superpowers/plans/2026-11-12-hermes-usage.md`.
