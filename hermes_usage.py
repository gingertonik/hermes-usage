#!/usr/bin/env python3
"""hermes-usage: terminal usage dashboard for Hermes, over read-only state.db."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_BRACKET_SUFFIX = re.compile(r"\[[^\]]*\]$")


def connect_ro(hermes_home: Path | None = None) -> sqlite3.Connection:
    """Open <hermes_home>/state.db read-only. Rows come back as sqlite3.Row.

    Accepts the Hermes home dir (not the db file), mirroring the tests'
    connect_ro(dir) contract.
    """
    home = Path(hermes_home) if hermes_home else (
        Path(os.environ["HERMES_HOME"]) if os.environ.get("HERMES_HOME")
        else Path.home() / ".hermes")
    target = (home / "state.db").resolve()
    conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def normalize_model(model: str | None) -> str:
    """Strip bracket context suffixes: 'claude-opus-5-5[1m]' -> 'claude-opus-5-5'."""
    if not model:
        return "(unknown)"
    return _BRACKET_SUFFIX.sub("", model).strip() or "(unknown)"


def window_start(kind: str, days: int, now: float) -> float | None:
    """Cutoff epoch for a window kind: today|week|month|days|all. None = no cutoff.

    'week' and 'month' carry fixed spans (7/30 days); `days` is used only for kind='days'.
    """
    if kind == "all":
        return None
    if kind == "today":
        local = datetime.fromtimestamp(now).astimezone()
        return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if kind == "week":
        return now - 7 * 86400
    if kind == "month":
        return now - 30 * 86400
    return now - days * 86400


@dataclass
class UsageRow:
    model: str
    provider: str
    family: str
    sessions: int
    msgs: int | None
    in_tok: int
    out_tok: int
    cache_r: int
    cache_w: int
    think: int
    api_calls: int
    est_cost: float
    act_cost: float
    cost_source: str


def family_for(provider: str, base_url: str, model: str) -> str:
    """Billing family from exact DB values (spec-pinned mapping)."""
    if provider.startswith("claude-subscription"):
        return "Claude subscription"
    if provider == "custom":
        return "Local"
    if not provider:
        return "Unknown"
    return provider


FAMILY_ORDER = ["Claude subscription", "Local", "Unknown"]


def assign_family(rows: list[UsageRow], conn: sqlite3.Connection) -> None:
    """Fill row.family in place, keyed on the dominant billing_base_url per (model, provider)."""
    conn.row_factory = sqlite3.Row
    urls: dict[tuple[str, str], tuple[str, int]] = {}
    q = ("SELECT model, billing_provider, billing_base_url, "
         "SUM(input_tokens)+SUM(output_tokens) AS toks, COUNT(*) AS n "
         "FROM session_model_usage GROUP BY 1, 2, 3")
    for r in conn.execute(q):
        key = (normalize_model(r["model"]), r["billing_provider"] or "")
        weight = (r["toks"] or 0) + r["n"]  # +n: break all-zero-token ties by row count
        best = urls.get(key)
        if best is None or weight > best[1]:
            urls[key] = (r["billing_base_url"] or "", weight)
    for row in rows:
        base_url = urls.get((row.model, row.provider), ("", 0))[0]
        row.family = family_for(row.provider, base_url, row.model)


def cost_cell(est: float, act: float, cost_source: str | None) -> str:
    """Money cell: actual wins over estimated; unpriced (cost_source='none', zero cost) is an em dash."""
    if act > 0:
        return f"${act:.2f}"
    if est > 0:
        return f"${est:.2f}"
    if cost_source == "none":
        return "—"
    return "$0.00"


def fmt_tokens(n: int) -> str:
    """Token count: M at >=1M (2dp), integer K at >=100K, raw below (0dp K would print 2350 as '2K')."""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 100_000:
        return f"{n // 1000}K"
    return str(n)


def fetch_msgs(conn: sqlite3.Connection, cutoff: float | None,
               exclude_sources: tuple[str, ...] = ()) -> dict[tuple[str, str], int]:
    """Msg counts per (normalized model, provider) by window attribution.

    Approximate by construction (spec): each user/assistant message is attributed to the
    latest-starting main-conversation window (task='') of its session covering the message
    timestamp; messages outside every window are dropped.
    """
    conn.row_factory = sqlite3.Row
    wsql = ("SELECT session_id, model, billing_provider, first_seen, last_seen "
            "FROM session_model_usage WHERE task=''")
    args: list[object] = []
    if cutoff is not None:
        wsql += " AND first_seen >= ?"
        args.append(cutoff)
    if exclude_sources:
        marks = ",".join("?" for _ in exclude_sources)
        wsql += f" AND session_id IN (SELECT id FROM sessions WHERE source NOT IN ({marks}))"
        args.extend(exclude_sources)
    windows: dict[str, list[tuple[float, float, tuple[str, str]]]] = {}
    for r in conn.execute(wsql, args):
        key = (normalize_model(r["model"]), r["billing_provider"] or "")
        sess = windows.setdefault(r["session_id"], [])
        win = next((w for w in sess if w[2] == key), None)
        if win is None:
            sess.append((r["first_seen"] or 0.0, r["last_seen"] or 0.0, key))
        else:  # merge duplicate (session, model, provider) rows: widen [min, max]
            i = sess.index(win)
            sess[i] = (min(win[0], r["first_seen"] or 0.0), max(win[1], r["last_seen"] or 0.0), key)

    counts: dict[tuple[str, str], int] = {}
    seen: set[tuple] = set()
    for m in conn.execute("SELECT session_id, role, timestamp, display_order, content, tool_calls "
                          "FROM messages WHERE role IN ('user','assistant') AND (active=1 OR compacted=1)"):
        dedupe = (m["session_id"], m["display_order"], m["role"], m["content"], m["tool_calls"])
        if dedupe in seen:
            continue
        seen.add(dedupe)
        ts = m["timestamp"]
        covering = [w for w in windows.get(m["session_id"], []) if w[0] <= ts <= w[1]]
        if not covering:
            continue
        key = max(covering, key=lambda w: w[0])[2]  # tie/overlap: latest window wins
        counts[key] = counts.get(key, 0) + 1
    return counts


USER_SOURCES = ("subagent", "cron", "tool", "oneshot")

_COLUMNS = (("Model", 34, "<"), ("Sess", 5, ">"), ("Msgs", 6, ">"), ("In", 8, ">"),
            ("Out", 8, ">"), ("Cache R", 9, ">"), ("Cache W", 9, ">"), ("Think", 8, ">"),
            ("Est. $", 8, ">"))


def _cells(model: str, sessions: int, msgs: int, r_in: int, out: int, cr: int, cw: int,
           think: int, cost: str) -> str:
    name = model if len(model) <= 34 else model[:33] + "…"
    vals = (name, str(sessions), str(msgs), fmt_tokens(r_in), fmt_tokens(out),
            fmt_tokens(cr), fmt_tokens(cw), fmt_tokens(think), cost)
    return " ".join(v.rjust(w) if al == ">" else v.ljust(w)
                    for v, (_, w, al) in zip(vals, _COLUMNS)).rstrip()


def _sum_row(rows: list[UsageRow]) -> tuple:
    cost = cost_cell(sum(r.est_cost for r in rows), sum(r.act_cost for r in rows),
                     "none" if rows and all(r.cost_source == "none" for r in rows) else "")
    return (sum(r.sessions for r in rows), sum(r.msgs or 0 for r in rows),
            sum(r.in_tok for r in rows), sum(r.out_tok for r in rows),
            sum(r.cache_r for r in rows), sum(r.cache_w for r in rows),
            sum(r.think for r in rows), cost)


def family_sort_key(fam: str) -> tuple[int, str]:
    return (FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER), fam)


def render_table(rows: list[UsageRow], period_label: str) -> str:
    header = " ".join(h.ljust(w) if i == 0 else h.rjust(w)
                      for i, (h, w, _) in enumerate(_COLUMNS)).rstrip()
    lines = [f"Hermes usage — {period_label}", header]
    groups: dict[str, list[UsageRow]] = {}
    for r in rows:
        groups.setdefault(r.family, []).append(r)
    for fam in sorted(groups, key=family_sort_key):
        lines.append(fam)
        for r in groups[fam]:
            lines.append(_cells(r.model, r.sessions, r.msgs or 0, r.in_tok, r.out_tok,
                                r.cache_r, r.cache_w, r.think, cost_cell(r.est_cost, r.act_cost, r.cost_source)))
        s = _sum_row(groups[fam])
        lines.append(_cells(f"── {fam} subtotal", *s))
    if groups:
        t = _sum_row([r for g in groups.values() for r in g])
        lines.append(_cells("══ TOTAL", *t))
        lines.append("note: sessions counted per model (a model-switching session counts twice); "
                     "header shows distinct sessions")
    return "\n".join(lines)


class _DaysAction(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, values)
        ns.window = "days"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hermes-usage",
                                description="Per-model Hermes usage over read-only state.db")
    w = p.add_mutually_exclusive_group()
    w.add_argument("--today", dest="window", action="store_const", const="today")
    w.add_argument("--week", dest="window", action="store_const", const="week")
    w.add_argument("--month", dest="window", action="store_const", const="month")
    w.add_argument("--all", dest="window", action="store_const", const="all")
    p.add_argument("--days", type=int, default=30, action=_DaysAction)
    p.add_argument("--user", action="store_true",
                   help="exclude subagent/cron/tool/oneshot sessions")
    p.add_argument("--tasks", action="store_true", help="per-model aux-task split")
    p.add_argument("--live", action="store_true", help="re-render every 5s")
    p.add_argument("--trend", type=int, metavar="N", default=None,
                   help="daily rollup sparklines, last N days")
    p.add_argument("--ticks", type=int, default=None, help=argparse.SUPPRESS)
    p.set_defaults(window="month")
    return p.parse_args(argv)


def build_report(conn: sqlite3.Connection, args: argparse.Namespace, now: float) -> str:
    cutoff = window_start(args.window, args.days, now)
    excl = USER_SOURCES if args.user else ()
    rows = fetch_usage(conn, cutoff, excl)
    assign_family(rows, conn)
    counts = fetch_msgs(conn, cutoff, excl)
    for r in rows:
        r.msgs = counts.get((r.model, r.provider), 0)
    if args.window == "all":
        label = "all time"
    elif args.window == "today":
        label = "today"
    elif args.window == "week":
        label = "last 7 days"
    elif args.window == "month":
        label = "last 30 days"
    else:
        label = f"last {args.days} days"
    q = "SELECT COUNT(DISTINCT session_id) FROM session_model_usage"
    where, qargs = [], []
    if cutoff is not None:
        where.append("first_seen >= ?")
        qargs.append(cutoff)
    if excl:
        marks = ",".join("?" for _ in excl)
        where.append(f"session_id IN (SELECT id FROM sessions WHERE source NOT IN ({marks}))")
        qargs.extend(excl)
    if where:
        q += " WHERE " + " AND ".join(where)
    n = conn.execute(q, qargs).fetchone()[0]
    scope = "user sessions only" if args.user else "incl. subagents, cron, one-shots"
    return render_table(rows, f"{label} · {n} sessions · {scope}")


def render_tasks(conn: sqlite3.Connection, cutoff: float | None,
                 exclude_sources: tuple[str, ...]) -> str:
    """Per-model aux-task split: one section per billing family, (model:task) rows by calls desc.

    Raw-row Python aggregation, same cost_source rule as fetch_usage.
    """
    sql = ("SELECT session_id, model, billing_provider, task, cost_source, api_call_count, "
           "input_tokens, output_tokens, estimated_cost_usd, actual_cost_usd "
           "FROM session_model_usage WHERE 1=1")
    args: list[object] = []
    if cutoff is not None:
        sql += " AND first_seen >= ?"
        args.append(cutoff)
    if exclude_sources:
        marks = ",".join("?" for _ in exclude_sources)
        sql += f" AND session_id IN (SELECT id FROM sessions WHERE source NOT IN ({marks}))"
        args.extend(exclude_sources)
    conn.row_factory = sqlite3.Row
    merged: dict[tuple[str, str, str], list] = {}
    for r in conn.execute(sql, args):
        key = (normalize_model(r["model"]), r["billing_provider"] or "",
               r["task"] or "(main)")
        cell = merged.setdefault(key, [0, 0, 0, 0.0, 0.0, False])
        cell[0] += r["api_call_count"] or 0
        cell[1] += r["input_tokens"] or 0
        cell[2] += r["output_tokens"] or 0
        cell[3] += r["estimated_cost_usd"] or 0.0
        cell[4] += r["actual_cost_usd"] or 0.0
        cell[5] = cell[5] or (r["cost_source"] == "none")
    rows = [UsageRow(model=m, provider=p, family="", msgs=None, sessions=0,
                     in_tok=c[1], out_tok=c[2], cache_r=0, cache_w=0, think=0,
                     api_calls=c[0], est_cost=c[3], act_cost=c[4],
                     cost_source="none" if (c[3] + c[4] == 0 and c[5]) else "")
            for (m, p, _t), c in merged.items()]
    assign_family(rows, conn)
    fam_by_mp: dict[tuple[str, str], str] = {}
    for (m, p, _t), r in zip(merged.keys(), rows):  # dicts preserve insertion order
        fam_by_mp[(m, p)] = r.family
    groups: dict[str, list[tuple[tuple, list]]] = {}
    for key, c in merged.items():
        groups.setdefault(fam_by_mp[(key[0], key[1])], []).append((key, c))
    lines = ["Per-task split (calls / in / out / est $)"]
    for fam in sorted(groups, key=family_sort_key):
        lines.append(fam)
        for (m, _p, t), c in sorted(groups[fam], key=lambda kv: (kv[1][0], kv[0][0], kv[0][2]),
                                    reverse=True):
            label = f"{m}:{t}"
            cost = cost_cell(c[3], c[4], c[5])
            lines.append(f"{label:<42} {c[0]:>6} {fmt_tokens(c[1]):>8} {fmt_tokens(c[2]):>8} {cost:>8}")
    return "\n".join(lines)


def run_live(conn_factory, args: argparse.Namespace, render, sleep=time.sleep,
             clear=None, ticks: int | None = None) -> int:
    """Re-render loop: clear, render, sleep 5s. Locked DB -> one-line skip, loop continues."""
    count = 0
    while ticks is None or count < ticks:
        count += 1
        sys.stdout.write(clear() if clear else "\x1b[2J\x1b[H")
        sys.stdout.flush()
        try:
            print(render(conn_factory()))
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc):
                print("usage: tick skipped (database busy)")
            else:
                raise
        sleep(5)
    return 0


_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int) -> str:
    """Eight-block spark, linear to max; all-zero series is all '▁'."""
    vals = (values + [0] * width)[:width]
    mx = max(vals)
    if mx <= 0:
        return _SPARK[0] * len(vals)
    return "".join(_SPARK[min(7, max(0, round(v / mx * 7)))] for v in vals)


def render_trend(path: Path, days: int) -> str:
    """Per-model daily est-cost sparklines over the last N days from the rollup JSONL."""
    if not path.exists():
        return f"no rollup yet: {path}"
    today = datetime.now().date()
    window = [(today - timedelta(days=i)).isoformat() for i in range(days)][::-1]
    by_model: dict[str, dict[str, list]] = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("date") not in window:
            continue
        m = by_model.setdefault(r.get("model", "(unknown)"),
                                {"cost": {d: 0.0 for d in window}, "out": 0})
        m["cost"][r["date"]] += r.get("est_cost", 0.0) + r.get("act_cost", 0.0)
        m["out"] += r.get("out_tok", 0)
    width = min(days, 60)
    lines = [f"Daily est-cost trend, last {days} days (spark width {width})"]
    for model in sorted(by_model, key=lambda k: -sum(by_model[k]["cost"].values())):
        m = by_model[model]
        series = [m["cost"][d] for d in window]
        bucket = -(-len(series) // width)
        cell = [sum(series[i * bucket:(i + 1) * bucket]) for i in range(width)]
        total = sum(series)
        nz = [d for d in window if m["cost"][d] > 0]
        last_txt = (f"last ${m['cost'][nz[-1]]:.2f} on {nz[-1]}" if nz else "last $0.00")
        lines.append(f"{model:<32} {sparkline(cell, width)}  ${total:8.2f}  "
                     f"{last_txt}  out {fmt_tokens(m['out'])}")
    return "\n".join(lines)


def trend_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return home / "cache" / "usage_daily.jsonl"


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    conn = connect_ro()
    if args.trend is not None:
        print(render_trend(trend_path(), args.trend))
        return 0
    if args.tasks:
        cutoff = window_start(args.window, args.days, datetime.now().timestamp())
        print(render_tasks(conn, cutoff, USER_SOURCES if args.user else ()))
        return 0
    if args.live:
        return run_live(lambda: connect_ro(), args,
                        lambda c: build_report(c, args, datetime.now().timestamp()),
                        ticks=args.ticks)
    print(build_report(conn, args, datetime.now().timestamp()))
    return 0


def fetch_usage(conn: sqlite3.Connection, cutoff: float | None,
                exclude_sources: tuple[str, ...] = ()) -> list[UsageRow]:
    """Aggregate session_model_usage per (normalized model, provider), sorted by tokens desc.

    One Python pass over raw usage rows: COUNT DISTINCT semantics for sessions are exact
    across normalized-model and cost_source group splits (SQL group-sum double-counts —
    live DB: swift 198 distinct vs 355 printed). cost_source='none' wins when the merged
    cost is zero and any row is unpriced, so unpriced models never render $0.00.
    """
    conn.row_factory = sqlite3.Row
    sql = ("SELECT session_id, model, billing_provider, cost_source, api_call_count, "
           "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
           "reasoning_tokens, estimated_cost_usd, actual_cost_usd "
           "FROM session_model_usage WHERE 1=1")
    args: list[object] = []
    if cutoff is not None:
        sql += " AND first_seen >= ?"
        args.append(cutoff)
    if exclude_sources:
        marks = ",".join("?" for _ in exclude_sources)
        sql += f" AND session_id IN (SELECT id FROM sessions WHERE source NOT IN ({marks}))"
        args.extend(exclude_sources)

    agg: dict[tuple[str, str], dict] = {}
    for r in conn.execute(sql, args):
        key = (normalize_model(r["model"]), r["billing_provider"] or "")
        a = agg.get(key)
        if a is None:
            a = agg[key] = {"sess": set(), "calls": 0, "in": 0, "out": 0, "cr": 0, "cw": 0,
                            "th": 0, "est": 0.0, "act": 0.0, "has_none": False}
        a["sess"].add(r["session_id"])
        a["calls"] += r["api_call_count"] or 0
        a["in"] += r["input_tokens"] or 0
        a["out"] += r["output_tokens"] or 0
        a["cr"] += r["cache_read_tokens"] or 0
        a["cw"] += r["cache_write_tokens"] or 0
        a["th"] += r["reasoning_tokens"] or 0
        a["est"] += r["estimated_cost_usd"] or 0.0
        a["act"] += r["actual_cost_usd"] or 0.0
        a["has_none"] = a["has_none"] or (r["cost_source"] == "none")
    rows = [UsageRow(model=m, provider=p, family="", msgs=None, sessions=len(a["sess"]),
                     in_tok=a["in"], out_tok=a["out"], cache_r=a["cr"], cache_w=a["cw"],
                     think=a["th"], api_calls=a["calls"], est_cost=a["est"], act_cost=a["act"],
                     cost_source="none" if (a["est"] + a["act"] == 0 and a["has_none"]) else "")
            for (m, p), a in agg.items()]
    return sorted(rows, key=lambda r: r.in_tok + r.out_tok, reverse=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
