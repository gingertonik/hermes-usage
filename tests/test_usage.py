"""Tests for hermes_usage. Fixtures are real temp sqlite files (WAL), not mocks.

Oracle notes: every expected value below is a hand calculation from the fixture
INSERTs, cited in comments; none was produced by running the implementation.
"""
import sqlite3
import tempfile
import time
import unittest
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


def make_fixture(tmpdir: str, wal: bool = True):
    p = Path(tmpdir) / "state.db"
    conn = sqlite3.connect(p)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"CREATE TABLE sessions ({SESSIONS_COLS})")
    conn.execute(f"CREATE TABLE session_model_usage ({USAGE_COLS})")
    t = 1_700_000_000.0
    conn.execute("INSERT INTO sessions VALUES ('s1','local','swift-1.5-iq3_s',?,0,0,0,0,'unknown',10)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s2','local','claude-opus-5-5[1m]',?,0,0,0,0,'estimated',20)", (t,))
    conn.execute("INSERT INTO sessions VALUES ('s3','subagent','claude-opus-5-5',?,0,0,0,0,'estimated',5)", (t,))
    conn.execute("INSERT INTO session_model_usage VALUES ('s1','swift-1.5-iq3_s','custom','','','',"
                 "10,1000,200,50,20,5,0,0,'none',?,?)", (t, t + 100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s2','claude-opus-5-5[1m]','claude-subscription-directsdk-experimental','','',"
                 "'',7,900,300,0,0,100,1.5,0,'',?,?)", (t, t + 100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s2','claude-opus-5-5[1m]','claude-subscription-directsdk-experimental','','',"
                 "'approval',2,50,1,0,0,0,0.2,0,'',?,?)", (t, t + 100))
    conn.execute("INSERT INTO session_model_usage VALUES ('s3','claude-opus-5-5','claude-subscription-directsdk-experimental','','',"
                 "'',3,400,40,0,0,0,0.5,0,'',?,?)", (t, t + 100))
    conn.commit()
    return p, conn


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
        # expected 12: 7 (s2 main) + 2 (s2 approval) + 3 (s3 main)
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
        self.conn.execute("INSERT INTO session_model_usage VALUES ('s9','','','','','',"
                          "1,100,0,0,0,0,0,0,'',?,?)", (t, t + 10))
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


if __name__ == "__main__":
    unittest.main()
