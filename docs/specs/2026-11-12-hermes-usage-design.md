# hermes-usage: terminal usage dashboard

Date: 2026-11-12. Status: pending approval. Size: small (one script, one cron job).

## Intent and why

Dillon wants a ccusage-style terminal dashboard for Hermes: per-model usage across
every model he runs, tracked over time, so subscription burn (Claude Opus/Fable/Haiku)
and local Strata usage are visible side by side without opening `hermes insights`
and reading token blobs. Reference design: his own screenshot of ccusage
(Model, Sessions, Msgs, Out M, Est. $ grouped by provider family) plus the full
token split he asked for: input, output, cached, thinking.

## Goals

- One command, `hermes-usage`, rendering a grouped table from state.db:
  per model: Sessions, Msgs, In, Out, Cache read, Cache write, Thinking, Est. $.
- Group by billing family (Claude subscription, Local/Strata, other) with subtotals
  and a grand total.
- Time windows: `--today`, `--week`, `--month`, `--days N` (default 30), `--all`.
- `--tasks`: per-model aux-task split (approval, compression, title_generation,
  vision, background_review) so harness burn under a subscription model is visible.
- `--live`: re-render every 5s.
- `--trend N`: per-model daily sparklines from a nightly rollup, surviving a 700MB DB.
- `--user`: exclude subagent/cron/tool/oneshot sources. Default includes everything
  because a subagent's Claude spend is real money.
- Model-name normalization: merge bracket context suffixes (`opus-5-5[1m]` joins
  `opus-5-5`); quantized local models stay distinct rows.

## Non-goals

- Web pages, TUI widgets, charts, budget alerts, per-session drill-down, prompt-size.

## Constraints

- Python 3 stdlib only; open state.db read-only (`mode=ro`), honor `$HERMES_HOME`.
- Never write to state.db. Rollup output is append-only JSONL.
- Install: source lives in this repo, symlink at `~/.local/bin/hermes-usage`.
- Nightly rollup follows `~/Projects/hermes-cronjobs/jobs/` convention
  (dir `usage-nightly-rollup`), writes `~/.hermes/cache/usage_daily.jsonl`.

## Approaches considered

1. Terminal dashboard over state.db (chosen): zero services, works over SSH,
   exactly the screenshot's shape. Trades: text only, no charts.
2. Built-in dashboard analytics tab: zero build, but browser-gated, no msgs
   column, no family grouping.
3. TUI widget: live panel, but widgets cannot read SQLite; needs JSON pipeline
   plus an endpoint. Most moving parts, smallest surface.

## Calls made by the builder (Steward), not agreed by Dillon

- Single-file stdlib script at `~/Projects/hermes-usage/hermes_usage.py`,
  symlinked onto PATH, importable and testable as `python3 hermes_usage.py test`.
- Msgs per model: attribute assistant rows to the model window
  (`task=''`, first_seen..last_seen) in `session_model_usage`; overlapping
  windows counted toward the window that covers the row's timestamp. Approximate
  by construction; totals will not exactly equal `sessions.message_count`.
- Cost figures come from `estimated_cost_usd`/`actual_cost_usd` columns; local
  models with no pricing render a blank cost cell, not $0, so real zero-cost
  subscriptions are not conflated with unpriced models.
- Rollup cron runs nightly at 03:17; `--trend` degrades gracefully (text note)
  when no rollup file exists yet.

## Verification plan

Driver-tested against read-only fixtures: SQL aggregation verified against direct
`sqlite3 -readonly` queries on a copy of state.db; window boundaries and model
merging covered as unit tests; render verified by running every flag.
