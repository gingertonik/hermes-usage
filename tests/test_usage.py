"""Tests for hermes_usage. Fixtures are real temp sqlite files (WAL), not mocks.

Oracle notes: every expected value below is a hand calculation from the fixture
INSERTs, cited in comments; none was produced by running the implementation.
"""
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
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
OPUS_PROVIDER = "claude-subscription-directsdk-experimental"


def _usage(conn, sid, model, provider, task, api, i, o, cr, cw, th, est, act, cs, t0, t1):
    conn.execute("INSERT INTO session_model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (sid, model, provider, "", "", task, api, i, o, cr, cw, th, est, act, cs, t0, t1))


def make_fixture(tmpdir: str, wal: bool = True):
    p = Path(tmpdir) / "state.db"
    conn = sqlite3.connect(p)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"CREATE TABLE sessions ({SESSIONS_COLS})")
    conn.execute(f"CREATE TABLE session_model_usage ({USAGE_COLS})")
    conn.execute("CREATE TABLE messages (session_id TEXT, role TEXT, timestamp REAL, "
                 "active INTEGER DEFAULT 1, compacted INTEGER DEFAULT 0, display_order INTEGER, "
                 "content TEXT, tool_calls TEXT)")
    t = 1_700_000_000.0
    conn.execute("INSERT INTO sessions VALUES ('s1','local','swift-1.5-iq3_s',?,0,0,0,0,'unknown',10)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s2','local','claude-opus-5-5[1m]',?,0,0,0,0,'estimated',20)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s3','subagent','claude-opus-5-5',?,0,0,0,0,'estimated',5)", (t,))
    _usage(conn, "s1", "swift-1.5-iq3_s", "custom", "", 10, 1000, 200, 50, 20, 5, 0, 0, "none", t, t + 100)
    # review fix fixture: same session/model/provider, second cost_source group ('' + vision
    # task) — mirrors the live DB (swift rows exist under cost_source '' and 'none' together).
    # swift api total 10+1+1=12; distinct sessions stay 2; est/act stay 0 with a 'none' group.
    _usage(conn, "s1", "swift-1.5-iq3_s", "custom", "vision", 1, 0, 0, 0, 0, 0, 0, 0, "", t, t + 100)
    # s2 switches models mid-session: swift owns [t, t+10], opus takes [t+10, t+100]
    _usage(conn, "s2", "swift-1.5-iq3_s", "custom", "", 1, 0, 0, 0, 0, 0, 0, 0, "none", t, t + 10)
    _usage(conn, "s2", "claude-opus-5-5[1m]", OPUS_PROVIDER, "", 7, 900, 300, 0, 0, 100, 1.5, 0, "", t + 10, t + 100)
    _usage(conn, "s2", "claude-opus-5-5[1m]", OPUS_PROVIDER, "approval", 2, 50, 1, 0, 0, 0, 0.2, 0, "", t + 10, t + 100)
    _usage(conn, "s3", "claude-opus-5-5", OPUS_PROVIDER, "", 3, 400, 40, 0, 0, 0, 0.5, 0, "", t, t + 100)
    _add_messages(conn, t)
    conn.commit()
    return p, conn


def _add_messages(conn, t):
    # s2 span: 4 swift-window msgs (t+1..t+4), 6 opus-window msgs (t+11,t+20,t+30,t+40,t+50,t+60),
    # 1 compacted-only msg at t+5 (swift window), 1 rewritten copy active=0/compacted=0 (excluded
    # by row state), 1 orphan at t+500 (no window). 12 rows in the s2 span + the copy.
    s2_rows = [
        ("assistant", t + 1, 1, 0, 0, "c0", None),
        ("user", t + 2, 1, 0, 1, "c1", None),
        ("assistant", t + 3, 1, 0, 2, "c2", None),
        ("user", t + 4, 1, 0, 3, "c3", None),
        ("assistant", t + 11, 1, 0, 4, "c4", None),
        ("user", t + 20, 1, 0, 5, "c5", None),
        ("assistant", t + 30, 1, 0, 6, "c6", None),
        ("user", t + 40, 1, 0, 7, "c7", None),
        ("assistant", t + 50, 1, 0, 8, "c8", None),
        ("user", t + 60, 1, 0, 9, "c9", None),
        ("user", t + 5, 0, 1, 10, "c10", None),   # compacted-only, inside swift window
        ("user", t + 1, 0, 0, 0, "c0", None),     # rewritten copy -> row-state filter
        ("user", t + 500, 1, 0, 12, "c12", None),  # orphan: outside every window
    ]
    s3_rows = [
        ("user", t + 5, 1, 0, 13, "c13", None),
        ("assistant", t + 50, 1, 0, 14, "c14", None),
    ]
    for sid, rows in (("s2", s2_rows), ("s3", s3_rows)):
        conn.executemany(f"INSERT INTO messages VALUES ('{sid}',?,?,?,?,?,?,?)", rows)


class DataLayer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_normalize_model_strips_brackets_keeps_quant_tags(self):
        self.assertEqual(hu.normalize_model("claude-opus-5-5[1m]"), "claude-opus-5-5")
        self.assertEqual(hu.normalize_model("swift-1.5-iq3_s"), "swift-1.5-iq3_s")
        self.assertEqual(hu.normalize_model(None), "(unknown)")

    def test_fetch_usage_merges_bracket_suffix_and_filters_cutoff(self):
        rows = hu.fetch_usage(self.conn, None)
        merged = {r.model: r for r in rows}
        self.assertNotIn("claude-opus-5-5[1m]", merged)
        # expected 2: s2 + s3 are distinct sessions under the merged model (fixture INSERTs)
        self.assertEqual(merged["claude-opus-5-5"].sessions, 2)
        # expected 12: 7 (s2 main) + 2 (s2 approval) + 3 (s3 main); s2 swift row adds 1 to swift
        self.assertEqual(merged["claude-opus-5-5"].api_calls, 12)
        self.assertEqual(merged["swift-1.5-iq3_s"].in_tok, 1000)
        self.assertEqual(merged["swift-1.5-iq3_s"].think, 5)
        future = hu.fetch_usage(self.conn, 1_800_000_000.0)
        self.assertEqual(future, [])

    def test_fetch_usage_excludes_sources(self):
        all_rows = {r.model: r for r in hu.fetch_usage(self.conn, None)}
        user = {r.model: r for r in hu.fetch_usage(self.conn, None, exclude_sources=("subagent",))}
        self.assertEqual(all_rows["claude-opus-5-5"].sessions, 2)
        self.assertEqual(user["claude-opus-5-5"].sessions, 1)  # s3 (source='subagent') dropped

    def test_wal_reader_snapshot_during_open_writer(self):
        # WAL fixture: a ro connection (fresh connect on the file) reads during an open txn.
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("UPDATE sessions SET message_count = 99 WHERE id='s1'")
        ro = hu.connect_ro(Path(self._tmp.name))
        got = ro.execute("SELECT message_count FROM sessions WHERE id='s1'").fetchone()[0]
        self.assertEqual(got, 10)  # pre-transaction snapshot, no OperationalError
        ro.close()
        self.conn.rollback()

    def test_window_start_kinds(self):
        now = time.mktime(time.strptime("2026-11-12 15:30", "%Y-%m-%d %H:%M"))
        midnight = time.mktime(time.strptime("2026-11-12 00:00", "%Y-%m-%d %H:%M"))
        self.assertEqual(hu.window_start("today", 30, now), midnight)
        self.assertAlmostEqual(hu.window_start("week", 30, now), now - 7 * 86400, places=2)
        self.assertIsNone(hu.window_start("all", 30, now))


class Grouping(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_family_exact_provider_values(self):
        self.assertEqual(hu.family_for("claude-subscription-directsdk-experimental", "", "claude-opus-5-5"),
                         "Claude subscription")
        self.assertEqual(hu.family_for("custom", "http://127.0.0.1:8081/v1", "swift-1.5-iq3_s"), "Local")
        self.assertEqual(hu.family_for("custom", "", "swift-1.5-iq3_s"), "Local")
        self.assertEqual(hu.family_for("", "", "gpt-oss:20b"), "Unknown")
        self.assertEqual(hu.family_for("openrouter", "", "x"), "openrouter")

    def test_unknown_model_and_provider_reaches_total(self):
        t = 1_700_000_000.0
        # expected: model '(unknown)' with exactly 100 in_tok (this INSERT), family 'Unknown'
        _usage(self.conn, "s9", "", "", "", 1, 100, 0, 0, 0, 0, 0, 0, "", t, t + 10)
        self.conn.commit()
        rows = {r.model: r for r in hu.fetch_usage(self.conn, None)}
        self.assertIn("(unknown)", rows)
        self.assertEqual(rows["(unknown)"].in_tok, 100)
        hu.assign_family(list(rows.values()), self.conn)
        self.assertEqual(rows["(unknown)"].family, "Unknown")
        self.assertEqual(rows["swift-1.5-iq3_s"].family, "Local")
        self.assertEqual(rows["claude-opus-5-5"].family, "Claude subscription")

    def test_cost_cell_precedence(self):
        self.assertEqual(hu.cost_cell(3.40, 1.20, ""), "$1.20")
        self.assertEqual(hu.cost_cell(3.40, 0.0, ""), "$3.40")
        self.assertEqual(hu.cost_cell(0.0, 0.0, "none"), "—")
        self.assertEqual(hu.cost_cell(0.0, 0.0, ""), "$0.00")  # priced, genuinely zero

    def test_fmt_tokens(self):
        self.assertEqual(hu.fmt_tokens(4_570_000), "4.57M")
        self.assertEqual(hu.fmt_tokens(812_000), "812K")
        self.assertEqual(hu.fmt_tokens(999), "999")
        self.assertEqual(hu.fmt_tokens(2_350), "2350")  # < 100K stays raw: 0dp K would print "2K"


class WindowAttribution(unittest.TestCase):
    SWIFT = ("swift-1.5-iq3_s", "custom")
    OPUS = ("claude-opus-5-5", OPUS_PROVIDER)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_window_attribution(self):
        counts = hu.fetch_msgs(self.conn, None)
        # expected 5: m0..m3 (4 active in [t, t+10]) + m10 (compacted-only, swift window);
        # m11 excluded by row state (active=0, compacted=0), m12 has no window
        self.assertEqual(counts[self.SWIFT], 5)
        # expected 8: 6 s2 msgs in [t+10, t+100] + 2 s3 msgs in s3's opus window
        self.assertEqual(counts[self.OPUS], 8)

    def test_msgs_cutoff_and_sources(self):
        user = hu.fetch_msgs(self.conn, None, exclude_sources=("subagent",))
        self.assertEqual(user[self.OPUS], 6)   # s3's 2 msgs (source='subagent') dropped
        self.assertEqual(user[self.SWIFT], 5)  # s1/s2 untouched
        future = hu.fetch_msgs(self.conn, 1_900_000_000.0)
        self.assertEqual(future, {})  # every window row has first_seen < 1.9e9


class Report(unittest.TestCase):
    NOW = 1_700_000_001.0  # 1s after the fixture t; 30d window covers every row

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_table_families_subtotals_and_total(self):
        # Hand calc from fixture: swift in 1000 + opus in (900+50+400)=1350 -> 2350;
        # out 200 + (300+1+40)=541; est 0+0 + (1.5+0.2+0.5)=2.20 -> $2.20
        out = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        self.assertIn("Claude subscription", out)
        self.assertIn("── Claude subscription subtotal", out)
        self.assertIn("── Local subtotal", out)
        self.assertIn("══ TOTAL", out)
        total = next(l for l in out.splitlines() if "TOTAL" in l)
        self.assertIn("2350", total)
        self.assertIn("541", total)
        self.assertIn("$2.20", total)
        swift = next(l for l in out.splitlines() if "swift-1.5-iq3_s" in l)
        self.assertIn("—", swift)  # unpriced local model: em dash, not $0.00

    def test_table_msgs_from_windows(self):
        out = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        swift = next(l for l in out.splitlines() if "swift-1.5-iq3_s" in l)
        cells = swift.split()
        # swift cells: model sessions msgs in out cacheR cacheW think cost = 9 tokens
        self.assertEqual(len(cells), 9)
        self.assertEqual(cells[2], "5")   # fixture swift window: 5 msgs
        opus = next(l for l in out.splitlines() if l.strip().startswith("claude-opus-5-5 "))
        self.assertEqual(opus.split()[2], "8")  # 6 (s2) + 2 (s3)

    def test_table_footnote_on_session_double_count(self):
        # fixture: swift spans s1+s2, opus spans s2+s3 -> Sess column sums to 4 over 3 distinct
        out = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        self.assertIn("sessions counted per model", out)

    def test_sessions_dedupe_across_cost_source_groups(self):
        # fixture: s1 swift sits in TWO cost_source groups ('none','' after fixture add);
        # distinct sessions for swift = 2 (s1, s2); group-sum would print 3
        out = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        swift = next(l for l in out.splitlines() if "swift-1.5-iq3_s" in l)
        self.assertEqual(swift.split()[1], "2")

    def test_unpriced_dash_survives_cost_source_split(self):
        # s1 swift groups: est/act all 0, one group cost_source='none' -> spec: em dash, not $0.00
        out = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        swift = next(l for l in out.splitlines() if "swift-1.5-iq3_s" in l)
        self.assertTrue(swift.rstrip().endswith("—"))

    def test_header_session_count_respects_user_filter(self):
        # fixture: 3 distinct sessions (s1,s2,s3); s3 is source='subagent' -> user-only = 2
        out_all = hu.build_report(self.conn, hu.parse_args([]), self.NOW)
        self.assertIn("3 sessions", out_all)
        out_user = hu.build_report(self.conn, hu.parse_args(["--user"]), self.NOW)
        self.assertIn("2 sessions · user sessions only", out_user)

    def test_window_flag_parsing(self):
        a = hu.parse_args(["--week"])
        self.assertEqual(a.window, "week")
        a = hu.parse_args(["--days", "3"])
        self.assertEqual((a.window, a.days), ("days", 3))
        self.assertEqual(hu.parse_args([]).days, 30)
        self.assertTrue(hu.parse_args(["--user"]).user)
        with self.assertRaises(SystemExit):
            hu.parse_args(["--week", "--all"])


class Tasks(unittest.TestCase):
    NOW = 1_700_000_001.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_tasks_split(self):
        out = hu.render_tasks(self.conn, None, ())
        self.assertIn("(main)", out)
        self.assertIn("approval", out)
        # fixture s2 approval row: 2 calls, in 50, out 1, est 0.2 -> $0.20
        line = next(l for l in out.splitlines() if "approval" in l)
        self.assertEqual(line.split()[1:], ["2", "50", "1", "$0.20"])
        main_opus = next(l for l in out.splitlines()
                         if "claude-opus-5-5" in l and "(main)" in l)
        # expected 10: 7 (s2 main) + 3 (s3 main); in 900+400=1300
        self.assertEqual(main_opus.split()[1], "10")

    def test_tasks_source_filter_drops_approval_row(self):
        out = hu.render_tasks(self.conn, None, ("subagent",))
        self.assertIn("approval", out)  # s2 approval survives
        s3_main = [l for l in out.splitlines() if "(main)" in l and "claude-opus" in l]
        # after dropping s3: opus main = 7 calls (only s2), single opus main line
        self.assertEqual(len(s3_main), 1)
        self.assertEqual(s3_main[0].split()[1], "7")


class Live(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_live_loop_survives_locked_tick(self):
        conns = [self.conn, "LOCK", self.conn]

        def factory():
            item = conns.pop(0)
            if item == "LOCK":
                raise sqlite3.OperationalError("database is locked")
            return item

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hu.run_live(factory, hu.parse_args([]), lambda c: "REPORT",
                             sleep=lambda s: None, clear=lambda: "CLEAR", ticks=3)
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("REPORT"), 2)
        self.assertIn("usage: tick skipped (database busy)", out)
        self.assertEqual(out.count("CLEAR"), 3)  # screen cleared every tick


class Trend(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path, self.conn = make_fixture(self._tmp.name)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import rollup  # noqa: local import so path bootstrap applies
        self.rollup = rollup

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_rollup_day_pinned_sums(self):
        from datetime import datetime
        # oracle: fixture rows all first_seen at/beside t=1.7e9, same local calendar day
        date_str = datetime.fromtimestamp(1_700_000_000).date().isoformat()
        rows = {r["model"]: r for r in self.rollup.rollup_day(self.conn, date_str)}
        swift = rows["swift-1.5-iq3_s"]
        # expected 12: 10 (s1 main) + 1 (s2 window row) + 1 (s1 vision) after fixture add
        self.assertEqual((swift["sessions"], swift["api_calls"], swift["in_tok"], swift["out_tok"]),
                         (2, 12, 1000, 200))
        opus = rows["claude-opus-5-5"]
        self.assertEqual((opus["sessions"], opus["api_calls"]), (2, 12))
        self.assertAlmostEqual(opus["est_cost"], 2.2, places=6)  # 1.5 + 0.2 + 0.5

    def test_write_rollup_idempotent(self):
        from datetime import datetime
        date_str = datetime.fromtimestamp(1_700_000_000).date().isoformat()
        out = Path(self._tmp.name) / "usage_daily.jsonl"
        n1 = self.rollup.write_rollup(self.conn, out, date_str)
        n2 = self.rollup.write_rollup(self.conn, out, date_str)
        self.assertEqual((n1, n2), (2, 2))
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(len(lines), 2)  # rerun replaces the day, never duplicates

    def test_sparkline(self):
        self.assertEqual(hu.sparkline([0, 5, 10], 3), "▁▅█")  # idx 0, round(3.5)=4, 7
        self.assertEqual(hu.sparkline([0, 0], 2), "▁▁")

    def test_render_trend_missing_file(self):
        missing = Path(self._tmp.name) / "nope.jsonl"
        self.assertIn(str(missing), hu.render_trend(missing, 30))

    def test_trend_last_shows_recent_nonzero_day(self):
        # steady state: cron fills YESTERDAY at 03:17, today is empty until night.
        # "last" must name the most recent day with cost, with its date label.
        today = datetime.now().date()
        y = (today - timedelta(days=1)).isoformat()
        t = today.isoformat()
        p = Path(self._tmp.name) / "u.jsonl"
        p.write_text(
            json.dumps({"date": y, "model": "m1", "est_cost": 5.0, "act_cost": 0.0, "out_tok": 10}) + "\n"
            + json.dumps({"date": t, "model": "m1", "est_cost": 0.0, "act_cost": 0.0, "out_tok": 0}) + "\n")
        out = hu.render_trend(p, 7)
        self.assertIn(f"on {y}", out)
        self.assertNotIn("last $0.00", out)

    def test_rollup_main_multi_dates_and_connect_path(self):
        # backfill: main accepts multiple dates; must connect via hu.connect_ro
        # (busy_timeout) — connect path pinned with a spy (unavoidable: no unit to inspect)
        calls = []
        real = self.rollup.hu.connect_ro
        def spy(home=None):
            calls.append(home)
            return real(home)
        self.rollup.hu.connect_ro = spy
        try:
            os.environ["HERMES_HOME"] = self._tmp.name  # fixture state.db sits at tmp root
            d0 = datetime.fromtimestamp(1_700_000_000).date().isoformat()
            d1 = (datetime.fromtimestamp(1_700_000_000) + timedelta(days=1)).date().isoformat()
            rc = self.rollup.main([self.rollup.__file__, d0, d1])
        finally:
            self.rollup.hu.connect_ro = real
            os.environ.pop("HERMES_HOME", None)
        self.assertEqual(rc, 0)
        self.assertTrue(calls)
        dates = {json.loads(l)["date"] for l in
                 Path(self._tmp.name, "cache", "usage_daily.jsonl").read_text().splitlines()}
        self.assertIn(d0, dates)

    def test_main_trend_exit_zero(self):
        # --trend must render the trend view (note line pre-rollup), not the table
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = hu.main(["hermes-usage", "--trend", "30"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertTrue("no rollup yet:" in out or "Daily est-cost trend" in out,
                        f"got table/other output: {out[:80]!r}")
        self.assertNotIn("══ TOTAL", out)


if __name__ == "__main__":
    unittest.main()
