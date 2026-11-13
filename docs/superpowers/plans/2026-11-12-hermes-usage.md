# hermes-usage Terminal Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `hermes-usage` terminal dashboard rendering per-model usage (sessions, msgs, in/out/cache/thinking tokens, est. $) from state.db, grouped by billing family, with windows, aux-task split, live mode, and a nightly rollup feeding `--trend` sparklines.

**Architecture:** One stdlib-only Python module, `hermes_usage.py`, opened read-only against Hermes's state.db. Data layer (SQL aggregation + window attribution) is pure functions over a connection; rendering is pure string builders; `main` wires argparse to them. A separate one-shot rollup script appends per-day per-model JSONL lines for trend history; a `no_agent` Hermes cronjob runs it nightly.

**Tech Stack:** Python 3 stdlib only (sqlite3, argparse, unittest, datetime, pathlib). No network, no third-party deps.

**Spec:** `~/Projects/hermes-usage/docs/specs/2026-11-12-hermes-usage-design.md`

## Global Constraints

- Python 3 stdlib only; runs on the system python3 (`/usr/bin/env python3`).
- state.db opened read-only only: `sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)`. Never write to state.db. Honor `$HERMES_HOME` (fall back to `~/.hermes`).
- All history writes go to `~/.hermes/cache/usage_daily.jsonl` only, and are append/rewrite of that file.
- Model normalization: strip bracket context suffixes — `claude-opus-5-5[1m]` merges into `claude-opus-5-5`. Quantized local model names stay distinct (no suffix stripping outside brackets).
- Window definitions (local time): `--today` = local midnight; `--week` = last 7×24h; `--month` and default = last 30×24h; `--days N` = last N×24h; `--all` = no cutoff.
- `--user` excludes session sources `subagent`, `cron`, `tool`, `oneshot`. Default includes all sources.
- Cost display: `actual_cost_usd > 0` wins over `estimated_cost_usd`; when both are 0 and `cost_source` is `none` (unpriced local models), the cost cell renders `—`, never `$0.00`.
- Family mapping by exact DB values: `billing_provider` starting `claude-subscription` → `Claude subscription`; `custom` → `Local`; empty string → `Unknown`; anything else → the provider string itself.
- Install layout: source repo `~/Projects/hermes-usage/` (git, local only, no remote), executable symlink `~/.local/bin/hermes-usage` → `~/Projects/hermes-usage/hermes_usage.py`.
- Tests: from repo root, `python3 -m unittest tests.test_usage -v` (tests/ is a package). Fixture DBs are real temp sqlite files, not mocks.
- Commit after each task with conventional messages (`feat:`, `test:`, `chore:`).

## Review Focus

Each line is a failure mode the spec implies but a naive build misses; the named task's tests pin it.

1. **DB live under read** (Task 1): state.db runs in WAL mode and is written while the dashboard reads. A `mode=ro` connection must succeed on a WAL database with a `-wal` file present, and a query during an open writer transaction must return a consistent snapshot, not crash.
2. **Empty/NULL model and provider** (Task 2): rows with `model=''`/NULL and `billing_provider=''` render under an `Unknown` family with an `(unknown)` model label, and their tokens appear in the grand total (they are real spend signals).
3. **Cost precedence and unpriced** (Task 2): a model with `actual_cost_usd=1.20, estimated_cost_usd=3.40` shows `$1.20`; a model with both 0 and `cost_source='none'` shows `—`; subtotals sum dollars, not cell strings.
4. **Live tick during a write lock** (Task 5): a tick that hits `sqlite3.OperationalError: database is locked` prints a one-line skip notice and keeps looping; the loop must not exit, and `--ticks N` must terminate it for testing.
5. **Rollup idempotency** (Task 6): running the rollup twice for the same local date replaces that date's lines (no duplicate day rows in the JSONL); `--trend` on a missing file prints a plain note naming the file path, exits 0, and never raises.

---

### Task 1: Repo init + data layer

**Files:**
- Create: `~/Projects/hermes-usage/hermes_usage.py`
- Create: `~/Projects/hermes-usage/tests/__init__.py` (empty)
- Create: `~/Projects/hermes-usage/tests/test_usage.py`
- Init: `git init` in `~/Projects/hermes-usage` (no remote; commit)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `connect_ro(hermes_home: Path | None = None) -> sqlite3.Connection` (rows as `sqlite3.Row`, `busy_timeout=3000`)
  - `normalize_model(model: str | None) -> str`
  - `window_start(kind: str, days: int, now: float) -> float | None` — kinds: `today`, `week`, `month`, `days`, `all`
  - `UsageRow` dataclass: `model, provider, family, sessions, msgs, in_tok, out_tok, cache_r, cache_w, think, api_calls, est_cost, act_cost, cost_source` (`msgs` starts `None`)
  - `fetch_usage(conn: sqlite3.Connection, cutoff: float | None, exclude_sources: tuple[str, ...] = ()) -> list[UsageRow]` — aggregates `session_model_usage` (SQL `GROUP BY` raw model+provider, Python merge after `normalize_model`), `sessions` joined for source filtering, `first_seen >= cutoff` filter, sorted by `in_tok+out_tok` desc. `family` filled as `""` (Task 2 fills it via `assign_family`).

- [ ] **Step 1: Init repo and write the failing tests**

`tests/test_usage.py` (first tests):

```python
import json, sqlite3, tempfile, time, unittest
from pathlib import Path
import hermes_usage as hu

SESSIONS_COLS = ("id TEXT PRIMARY KEY, source TEXT, model TEXT, started_at REAL, "
                 "input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, "
                 "estimated_cost_usd REAL DEFAULT 0, actual_cost_usd REAL DEFAULT 0, "
                 "cost_status TEXT, message_count INTEGER DEFAULT 0")
USAGE_COLS = ("session_id TEXT, model TEXT, billing_provider TEXT DEFAULT '', "
              "billing_base_url TEXT DEFAULT '', billing_mode TEXT DEFAULT '', "
              "task TEXT DEFAULT '', api_call_count INTEGER DEFAULT 0, "
              "input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, "
              "cache_read_tokens INTEGER DEFAULT 0, cache_write_tokens INTEGER DEFAULT 0, "
              "reasoning_tokens INTEGER DEFAULT 0, estimated_cost_usd REAL DEFAULT 0, "
              "actual_cost_usd REAL DEFAULT 0, cost_source TEXT, first_seen REAL, last_seen REAL")

def make_fixture(tmpdir: str, wal: bool = True) -> Path:
    p = Path(tmpdir) / "state.db"
    conn = sqlite3.connect(p)
    if wal: conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"CREATE TABLE sessions ({SESSIONS_COLS})")
    conn.execute(f"CREATE TABLE session_model_usage ({USAGE_COLS})")
    t = 1_700_000_000.0
    conn.execute("INSERT INTO sessions VALUES ('s1','local','swift-1.5-iq3_s',?,0,0,0,0,'unknown',10)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s2','local','claude-opus-5-5[1m]',?,0,0,0,0,'estimated',20)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s3','subagent','claude-opus-5-5',?,0,0,0,0,'estimated',5)", (t,))
    conn.execute("INSERT INTO session_model_usage VALUES ('s1','swift-1.5-iq3_s','custom','','','',"
                 "10,1000,200,50,20,5,0,0,'none',?,?)", (t, t+100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s2','claude-opus-5-5[1m]','claude-subscription-directsdk-experimental','','',"
                 "'',7,900,300,0,0,100,1.5,0,'',?,?)", (t, t+100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s2','claude-opus-5-5[1m]','claude-subscription-directsdk-experimental','','',"
                 "'approval',2,50,1,0,0,0,0.2,0,'',?,?)", (t, t+100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s3','claude-opus-5-5','claude-subscription-directsdk-experimental','','',"
                 "'',3,400,40,0,0,0,0.5,0,'',?,?)", (t, t+100))
    conn.commit()
    return p, conn
```

Write the fixture helper exactly as above (drop the dead conditional line; keep the second INSERT string), then these tests:

```python
class DataLayer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)
    def tearDown(self):
        self.conn.close(); self._tmp.cleanup()

    def test_normalize_model_strips_brackets_keeps_quant_tags(self):
        self.assertEqual(hu.normalize_model("claude-opus-5-5[1m]"), "claude-opus-5-5")
        self.assertEqual(hu.normalize_model("swift-1.5-iq3_s"), "swift-1.5-iq3_s")
        self.assertEqual(hu.normalize_model(None), "(unknown)")

    def test_fetch_usage_merges_bracket_suffix_and_filters_cutoff(self):
        rows = hu.fetch_usage(self.conn, None)
        merged = {r.model: r for r in rows}
        self.assertNotIn("claude-opus-5-5[1m]", merged)
        self.assertEqual(merged["claude-opus-5-5"].sessions, 2)   # s2 + s3 distinct sessions
        self.assertEqual(merged["claude-opus-5-5"].api_calls, 12) # 7 main + 2 approval + 3 s3
        self.assertEqual(merged["swift-1.5-iq3_s"].in_tok, 1000)
        self.assertEqual(merged["swift-1.5-iq3_s"].think, 5)
        future = hu.fetch_usage(self.conn, 1_800_000_000.0)
        self.assertEqual(future, [])

    def test_fetch_usage_excludes_sources(self):
        all_rows = {r.model: r for r in hu.fetch_usage(self.conn, None)}
        user = {r.model: r for r in hu.fetch_usage(self.conn, None, exclude_sources=("subagent",))}
        self.assertEqual(all_rows["claude-opus-5-5"].sessions, 2)
        self.assertEqual(user["claude-opus-5-5"].sessions, 1)

    def test_wal_reader_snapshot_during_open_writer(self):
        # WAL fixture: a ro connection (fresh connect on the file) reads during an open txn.
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("UPDATE sessions SET message_count = 99 WHERE id='s1'")
        ro = hu.connect_ro(Path(self._tmp.name))
        got = ro.execute("SELECT message_count FROM sessions WHERE id='s1'").fetchone()[0]
        self.assertEqual(got, 10)  # pre-transaction snapshot, no OperationalError
        ro.close(); self.conn.rollback()

    def test_window_start_kinds(self):
        now = time.mktime(time.strptime("2026-11-12 15:30", "%Y-%m-%d %H:%M"))
        midnight = time.mktime(time.strptime("2026-11-12 00:00", "%Y-%m-%d %H:%M"))
        self.assertEqual(hu.window_start("today", 30, now), midnight)
        self.assertAlmostEqual(hu.window_start("week", 30, now), now - 7*86400, places=2)
        self.assertIsNone(hu.window_start("all", 30, now))
```

Note: `connect_ro(dir)` resolves `<dir>/state.db`; fixture path matches.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Projects/hermes-usage && python3 -m unittest tests.test_usage -v`
Expected: FAIL — `AttributeError`/`ModuleNotFoundError` on `hu.connect_ro` etc.

- [ ] **Step 3: Implement**

`hermes_usage.py`: module docstring, imports, `UsageRow` dataclass, the four functions. `connect_ro` mirrors hermes-lab's (as_uri + `?mode=ro`, `row_factory=sqlite3.Row`, `busy_timeout`), defaulting to `$HERMES_HOME` else `~/.hermes`. `normalize_model`: `re.sub(r"\[[^\]]*\]$", "", model).strip()` with None/empty → `(unknown)`. `window_start`: `today` = `datetime.now().astimezone().replace(hour=0,minute=0,second=0,microsecond=0).timestamp()`; `week`/`month`/`days` = `now - days*86400` (caller passes 7/30/N); `all` → None. `fetch_usage`: one SQL `SELECT model, billing_provider, cost_source, COUNT(DISTINCT session_id), SUM(api_call_count), SUM(input_tokens), SUM(output_tokens), SUM(cache_read_tokens), SUM(cache_write_tokens), SUM(reasoning_tokens), SUM(estimated_cost_usd), SUM(actual_cost_usd) FROM session_model_usage` with optional `AND first_seen >= ?` and, when `exclude_sources`, `AND session_id IN (SELECT id FROM sessions WHERE source NOT IN (...))`; `GROUP BY model, billing_provider, cost_source`. Merge in Python by `(normalize_model(model), billing_provider)`, summing fields; `family=""`; sort by `in_tok+out_tok` desc.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Projects/hermes-usage && python3 -m unittest tests.test_usage -v`
Expected: all PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
cd ~/Projects/hermes-usage && git init -b main && git add -A && git commit -m "feat: usage data layer over read-only state.db"
```

---

### Task 2: Family grouping, cost cells, token formats

**Files:**
- Modify: `hermes_usage.py` (add `family_for`, `cost_cell`, `fmt_tokens`, `assign_family`)
- Modify: `tests/test_usage.py` (new TestCase class)

**Interfaces:**
- Consumes: `UsageRow` fields (Task 1).
- Produces:
  - `family_for(provider: str, base_url: str, model: str) -> str`
  - `assign_family(rows: list[UsageRow], conn: sqlite3.Connection) -> None` — fills `row.family` in place; needs the dominant `billing_base_url` per (model, provider) pair from `session_model_usage`.
  - `cost_cell(est: float, act: float, cost_source: str | None) -> str`
  - `fmt_tokens(n: int) -> str` — `>=1_000_000` → `"4.57M"` (2dp); `>=1_000` → `"812K"` (0dp); else `str(n)`.
  - Family order for display: `FAMILY_ORDER = ["Claude subscription", "Local", "Unknown"]`, then other families alphabetically.

- [ ] **Step 1: Write the failing tests**

```python
class Grouping(unittest.TestCase):
    def test_family_exact_provider_values(self):
        self.assertEqual(hu.family_for("claude-subscription-directsdk-experimental", "", "claude-opus-5-5"),
                         "Claude subscription")
        self.assertEqual(hu.family_for("custom", "http://127.0.0.1:8081/v1", "swift-1.5-iq3_s"), "Local")
        self.assertEqual(hu.family_for("custom", "", "swift-1.5-iq3_s"), "Local")
        self.assertEqual(hu.family_for("", "", "gpt-oss:20b"), "Unknown")
        self.assertEqual(hu.family_for("openrouter", "", "x"), "openrouter")

    def test_unknown_model_and_provider_reaches_total(self):
        # fixture row with model='' provider='' -> family Unknown, model "(unknown)"
        ...  # insert a usage row with model '' provider '' 100 tokens; assert the row exists in
             # fetch_usage as model '(unknown)' and assign_family gives it family 'Unknown'

    def test_cost_cell_precedence(self):
        self.assertEqual(hu.cost_cell(3.40, 1.20, ""), "$1.20")
        self.assertEqual(hu.cost_cell(3.40, 0.0, ""), "$3.40")
        self.assertEqual(hu.cost_cell(0.0, 0.0, "none"), "—")
        self.assertEqual(hu.cost_cell(0.0, 0.0, ""), "$0.00")  # priced, genuinely zero

    def test_fmt_tokens(self):
        self.assertEqual(hu.fmt_tokens(4_570_000), "4.57M")
        self.assertEqual(hu.fmt_tokens(812_000), "812K")
        self.assertEqual(hu.fmt_tokens(999), "999")
```

Fill the elided assertion fully (insert then query, mirroring DataLayer style).

- [ ] **Step 2: Run, verify fail** — `python3 -m unittest tests.test_usage -v`; expect failures on the four new names.

- [ ] **Step 3: Implement** — `family_for` as pinned in Global Constraints (note: `custom` → Local regardless of base_url; base_url only breaks ties for empty provider with local models, which stays `Unknown`). `cost_cell`: `act>0` → `$%.2f % act`; elif `est>0` → `$%.2f % est`; elif `cost_source=="none"` → `—`; else `$0.00`. `fmt_tokens` per spec. `assign_family` queries `SELECT model, billing_provider, billing_base_url FROM session_model_usage GROUP BY 1,2,3`, picks the url with the max summed tokens for the pair, sets `family_for(...)` per row.

- [ ] **Step 4: Run, verify pass** — all tests PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat: billing family grouping and cost cells"`

---

### Task 3: Msgs window attribution

**Files:**
- Modify: `hermes_usage.py` (add `fetch_msgs`)
- Modify: `tests/test_usage.py`

**Interfaces:**
- Consumes: fixture `sessions`, `session_model_usage` tables.
- Produces: `fetch_msgs(conn: sqlite3.Connection, cutoff: float | None, exclude_sources: tuple[str, ...] = ()) -> dict[tuple[str, str], int]` — keys `(normalize_model(model), billing_provider)`; the message count for that model+provider pair.

Algorithm (fixed by this plan; attribution is approximate by design):
1. Windows: `SELECT session_id, model, billing_provider, first_seen, last_seen FROM session_model_usage WHERE task='' [AND first_seen>=cutoff] [AND session_id IN (user-source subquery)]`; merge per `(normalize_model, provider)` keeping min(first_seen), max(last_seen) per session.
2. Messages: table `messages(session_id, role, timestamp, active, compacted, display_order, content, tool_calls)` — create it in fixtures. Rows with `role IN ('user','assistant')` and `(active=1 OR compacted=1)`, deduped in Python on `(session_id, display_order, role, content, tool_calls)`.
3. Attribute each message to the latest-starting window satisfying `first_seen <= timestamp <= last_seen` and `window.first_seen == max(first_seen among covering)` (tie: latest first wins). Uncovered rows are dropped from per-model counts.

- [ ] **Step 1: Write the failing tests**

Extend `make_fixture` (add `messages` table + rows): session s2 has a `swift` main window `t..t+10` (model row task='') and an `opus` window `t+10..t+100`, 12 user/assistant msgs spread across it, one rewritten duplicate (`active=0, compacted=0`), one compacted-only msg (`active=0, compacted=1`) inside the swift window, and one msg at `t+500` (no window). Assertions:

```python
def test_window_attribution(self):
    counts = hu.fetch_msgs(self.conn, None)
    swift = counts[("swift-1.5-iq3_s", "custom")]
    opus = counts[("claude-opus-5-5", "claude-subscription-directsdk-experimental")]
    self.assertEqual(swift, 5)    # 4 in-window user/assistant + 1 compacted-only, duplicate excluded
    self.assertEqual(opus, 6)     # in-window msgs only; t+500 msg unattributed
def test_msgs_cutoff_and_sources(self):
    counts = hu.fetch_msgs(self.conn, None, exclude_sources=("subagent",))
    self.assertNotIn(("claude-opus-5-5", "s3-provider"), counts)  # pin to the fixture's s3 provider
```

Set exact message counts in the fixture so the numbers above are true; adjust fixture INSERTs to match, and name the s3 provider value in the fixture so the assertion is exact.

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement `fetch_msgs`** per the algorithm.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Commit** — `git commit -am "feat: per-model msgs via session_model_usage window attribution"`

---

### Task 4: render_table + CLI + install

**Files:**
- Modify: `hermes_usage.py` (`render_table`, `build_report`, `main`, argparse, `__main__` guard)
- Modify: `tests/test_usage.py`
- Create: symlink `~/.local/bin/hermes-usage`, `chmod +x` on the script, shebang line.

**Interfaces:**
- Consumes: `fetch_usage`, `assign_family`, `fetch_msgs`, `family_for`, `cost_cell`, `fmt_tokens`, `window_start`, `connect_ro`.
- Produces:
  - `render_table(rows: list[UsageRow], period_label: str) -> str`
  - `build_report(conn, args, now: float) -> str`
  - `main(argv: list[str]) -> int`

Table: header `Model  Sessions  Msgs  In  Out  Cache R  Cache W  Think  Est. $`, families in `FAMILY_ORDER` (others alphabetical) with a `── <family> subtotal` row and a final `══ TOTAL` row. Columns right-aligned, model column truncated to 34 chars with `…`. Subtotals sum numeric fields (costs summed numerically, then formatted).

CLI: flags `--today --week --month --days N --all --user --tasks --live --trend N` (`--days` default 30; window flags mutually exclusive; `--tasks` and `--trend` render instead of the table; `--user` per constraints). Exit 0.

- [ ] **Step 1: Write the failing tests**

```python
class Report(unittest.TestCase):
    ...  # DataLayer-style fixture; then:
    def test_table_subtotals_and_grand_total(self):
        out = hu.build_report(self.conn, make_args(), NOW)
        lines = out.splitlines()
        self.assertIn("Claude subscription", out)
        self.assertIn("── Claude subscription subtotal", out)
        self.assertIn("══ TOTAL", out)
        total = [l for l in lines if "TOTAL" in l][0]
        self.assertIn("4.57M", total)      # swift in 1000 + opus in 1350 = 2.35K... pin to fixture sums
        self.assertIn("$2.20", total)      # 1.5+0.2+0.5 est, no actual
    def test_unpriced_dash(self):
        self.assertIn("—", hu.build_report(self.conn, make_args(), NOW))
    def test_window_flag_parsing(self):
        a = hu.parse_args(["--week"]); self.assertEqual(a.window, "week")
        a = hu.parse_args(["--days", "3"]); self.assertEqual((a.window, a.days), ("days", 3))
        with self.assertRaises(SystemExit): hu.parse_args(["--week", "--all"])
```

`make_args()` = `hu.parse_args([])`. Fix the two value assertions to the fixture's true sums (compute from Task 1 fixture: swift 1000/200, opus 950/341 in/out → totals `2.35K` in, `541` out; cost `$2.20`). Pin expected strings after computing; do not guess at implementation time.

- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** `render_table`, `build_report`, `parse_args`, `main`; wire `--user` to `exclude_sources=("subagent","cron","tool","oneshot")`; call `fetch_msgs` and fill `msgs` (missing key → 0); header line `Hermes usage — <period label> (N sessions incl. subagents)` / `(user sessions only)`.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Real-DB smoke test** — `python3 hermes_usage.py --all` then `~/.local/bin/hermes-usage --days 7` (after `chmod +x` + `ln -s`). Expected: a full table against real state.db, no exception, totals non-empty. Verify WAL live-read while a session runs (this is production, running now).
- [ ] **Step 6: Commit** — `git commit -am "feat: render_table, CLI flags, PATH install" && git add -A` (include symlink step note in README stub created here: name, one-line use).

---

### Task 5: `--tasks` and `--live`

**Files:**
- Modify: `hermes_usage.py` (`render_tasks`, `run_live`, argparse wiring)
- Modify: `tests/test_usage.py`

**Interfaces:**
- Consumes: fixture rows with `task='approval'` (already in fixture).
- Produces:
  - `render_tasks(conn, cutoff, exclude_sources) -> str` — one section per family, rows per (model, task) with `Calls  In  Out  Est. $`, tasks sorted by calls desc, main-conversation `task=''` labeled `(main)`.
  - `run_live(conn_factory, args, render, sleep=time.sleep, clear=None, ticks: int | None = None) -> int` — loop: clear, render, sleep 5; on `sqlite3.OperationalError` containing "locked", print `usage: tick skipped (database busy)` and continue; `ticks` caps iterations (hidden flag `--ticks`, help suppressed).

- [ ] **Step 1: Write the failing tests** — fixture: `render_tasks` output contains `(main)`, `approval`, calls `2` on the approval row, and `$0.20` for that task. `run_live` test: conn_factory yields DataLayer fixture conn first, then raises `sqlite3.OperationalError("database is locked")`, then fixture again; assert output contains `tick skipped`, `ticks=3` returns 0, sleep stubbed to no-op.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement** — tasks SQL: same source-filter/cutoff pattern, `GROUP BY model, billing_provider, task`.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Real-DB smoke** — `hermes-usage --tasks --days 7` (expect approval/compression rows), `hermes-usage --live --ticks 2`.
- [ ] **Step 6: Commit** — `git commit -am "feat: aux-task split and live mode"`

---

### Task 6: Nightly rollup, `--trend`, cron, QA

**Files:**
- Create: `~/Projects/hermes-usage/scripts/rollup.py`
- Create: `~/.hermes/scripts/usage-nightly-rollup.sh` (one line: `exec python3 ~/Projects/hermes-usage/scripts/rollup.py "$@"`)
- Modify: `hermes_usage.py` (`render_trend`, `sparkline`, `--trend` wiring)
- Modify: `tests/test_usage.py`
- Cron: register Hermes cronjob `usage-nightly-rollup`, `no_agent: true`, script `usage-nightly-rollup.sh`, schedule `17 3 * * *`, deliver `local` (via `cronjob_manage`; mirrors `~/Projects/hermes-cronjobs/jobs/*/job.json` shape — repo copy arrives via the existing Sunday `cronjob-repo-sync`).

**Interfaces:**
- Consumes: `session_model_usage` rows.
- Produces:
  - `rollup_day(conn, date_str: str) -> list[dict]` — rows for local day (first_seen within that local calendar day): keys `date, model, provider, sessions, api_calls, in_tok, out_tok, cache_r, cache_w, think, est_cost, act_cost`.
  - `write_rollup(db_conn, out_path: Path, date_str: str) -> int` — idempotent: read JSONL, drop lines with `date == date_str`, append fresh lines, rewrite file. Returns count written. Default `date_str` = yesterday (local).
  - `sparkline(values: list[float], width: int) -> str` — blocks `▁▂▃▄▅▆▇█`, linear to max; all-zero → all `▁`.
  - `render_trend(path: Path, days: int) -> str` — per model (normalized, merged across providers): daily est_cost sparkline, last-day value, window total $, total out M. Missing file → `no rollup yet: <path>` exit 0.

- [ ] **Step 1: Write the failing tests** — `rollup_day` on fixture returns swift + opus rows with pinned sums; `write_rollup` twice → file has each model line once for that date; `sparkline([0,5,10],3) == "▁▅█"`; `sparkline([0,0],2) == "▁▁"`; `render_trend` on missing path returns string containing the path, `main(["--trend","30"])` exit 0.
- [ ] **Step 2: Run, verify fail**
- [ ] **Step 3: Implement rollup.py + trend functions** — rollup.py `main`: connect_ro, `write_rollup(conn, Path.home()/".hermes"/"cache"/"usage_daily.jsonl", sys.argv[1] if len(sys.argv)>1 else yesterday)`, print count, exit 0.
- [ ] **Step 4: Run, verify pass**
- [ ] **Step 5: Real runs + cron** — `python3 scripts/rollup.py` (expect ~1744-row scale day: tens of lines); `hermes-usage --trend 30` shows rows. Register cron via `cronjob_manage` with the pinned shape; verify it appears in `hermes cron list` and `~/.hermes/cron/jobs.json`.
- [ ] **Step 6: Full-flag QA on real DB** — run `hermes-usage`, `--today`, `--week`, `--month`, `--days 3`, `--all`, `--user`, `--tasks`, `--live --ticks 2`, `--trend 30`. All exit 0, no tracebacks. Spot-check TOTAL est-cost against `sqlite3 -readonly ~/.hermes/state.db "select round(sum(estimated_cost_usd),2) from sessions"` (within the window's scope).
- [ ] **Step 7: README + commit** — README.md: name, why, all flags with one-line examples, the msgs/cost approximations noted, spec+plan paths. `git add -A && git commit -m "feat: nightly rollup, trend sparklines, cron, README"`.

---

## Self-review notes (author, done at write time)

- Spec coverage: grouped table (T4), windows (T4), --tasks (T5), --live (T5), --user (T4), normalization (T1), family+subtotal+total (T2/T4), cost unpriced dash (T2), rollup+cron+trend+degradation (T6), install symlink (T4), verification plan (T4/T6 smoke).
- Type consistency: `UsageRow.msgs` filled by T4 from T3 keys; `family` filled by T2; key shape `(normalized model, provider)` shared T3→T4.
- Fixture number assertions in T4 are pinned from T1 fixture arithmetic; executor must recompute, not guess.
