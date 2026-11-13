#!/usr/bin/env python3
"""Nightly usage rollup: append per-model daily totals to usage_daily.jsonl.

Reads Hermes state.db read-only; writes only the JSONL. Idempotent per date:
rerunning for a date replaces that date's lines.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hermes_usage as hu  # noqa: E402


def day_bounds(date_str: str) -> tuple[float, float]:
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    return d0.timestamp(), (d0 + timedelta(days=1)).timestamp()


def rollup_day(conn: sqlite3.Connection, date_str: str) -> list[dict]:
    lo, hi = day_bounds(date_str)
    conn.row_factory = sqlite3.Row
    q = ("SELECT model, billing_provider, COUNT(DISTINCT session_id) AS sessions, "
         "SUM(api_call_count) AS api_calls, SUM(input_tokens) AS in_tok, "
         "SUM(output_tokens) AS out_tok, SUM(cache_read_tokens) AS cache_r, "
         "SUM(cache_write_tokens) AS cache_w, SUM(reasoning_tokens) AS think, "
         "SUM(estimated_cost_usd) AS est_cost, SUM(actual_cost_usd) AS act_cost "
         "FROM session_model_usage WHERE first_seen >= ? AND first_seen < ? "
         "GROUP BY model, billing_provider")
    merged: dict[tuple[str, str], dict] = {}
    for r in conn.execute(q, (lo, hi)):
        key = (hu.normalize_model(r["model"]), r["billing_provider"] or "")
        m = merged.get(key)
        if m is None:
            merged[key] = {"date": date_str, "model": key[0], "provider": key[1],
                           "sessions": r["sessions"], "api_calls": r["api_calls"] or 0,
                           "in_tok": r["in_tok"] or 0, "out_tok": r["out_tok"] or 0,
                           "cache_r": r["cache_r"] or 0, "cache_w": r["cache_w"] or 0,
                           "think": r["think"] or 0, "est_cost": r["est_cost"] or 0.0,
                           "act_cost": r["act_cost"] or 0.0}
            continue
        # context-suffix variants (model[1m] swaps) normalize into one row; sessions summed
        for f in ("sessions", "api_calls", "in_tok", "out_tok", "cache_r", "cache_w",
                  "think", "est_cost", "act_cost"):
            m[f] += r[f] or 0
    return list(merged.values())


def write_rollup(db_conn: sqlite3.Connection, out_path: Path, date_str: str) -> int:
    fresh = rollup_day(db_conn, date_str)
    kept = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                if json.loads(line).get("date") != date_str:
                    kept.append(line)
            except json.JSONDecodeError:
                continue  # drop corrupt lines on rewrite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(l + "\n" for l in kept)
                        + "".join(json.dumps(r) + "\n" for r in fresh))
    return len(fresh)


def main(argv: list[str]) -> int:
    today = datetime.now().date()
    if len(argv) > 1:
        date_str = argv[1]
    else:
        date_str = (today - timedelta(days=1)).isoformat()  # nightly: roll up yesterday
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    out = home / "cache" / "usage_daily.jsonl"
    conn = sqlite3.connect(f"{(home / 'state.db').resolve().as_uri()}?mode=ro", uri=True)
    print(write_rollup(conn, out, date_str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
