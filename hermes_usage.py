"""hermes-usage: terminal usage dashboard for Hermes, over read-only state.db."""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
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


def fetch_usage(conn: sqlite3.Connection, cutoff: float | None,
                exclude_sources: tuple[str, ...] = ()) -> list[UsageRow]:
    """Aggregate session_model_usage per (normalized model, provider), sorted by tokens desc."""
    sql = ("SELECT model, billing_provider, cost_source, "
           "COUNT(DISTINCT session_id) AS sessions, SUM(api_call_count) AS api_calls, "
           "SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok, "
           "SUM(cache_read_tokens) AS cache_r, SUM(cache_write_tokens) AS cache_w, "
           "SUM(reasoning_tokens) AS think, SUM(estimated_cost_usd) AS est_cost, "
           "SUM(actual_cost_usd) AS act_cost "
           "FROM session_model_usage WHERE 1=1")
    args: list[object] = []
    if cutoff is not None:
        sql += " AND first_seen >= ?"
        args.append(cutoff)
    if exclude_sources:
        marks = ",".join("?" for _ in exclude_sources)
        sql += f" AND session_id IN (SELECT id FROM sessions WHERE source NOT IN ({marks}))"
        args.extend(exclude_sources)
    sql += " GROUP BY model, billing_provider, cost_source"

    conn.row_factory = sqlite3.Row  # callers may pass a factory-less conn (fixtures, CDP-style reuse)
    merged: dict[tuple[str, str], UsageRow] = {}
    for r in conn.execute(sql, args):
        key = (normalize_model(r["model"]), r["billing_provider"] or "")
        row = merged.get(key)
        if row is None:
            merged[key] = UsageRow(
                model=key[0], provider=key[1], family="", msgs=None,
                sessions=r["sessions"], in_tok=r["in_tok"] or 0, out_tok=r["out_tok"] or 0,
                cache_r=r["cache_r"] or 0, cache_w=r["cache_w"] or 0, think=r["think"] or 0,
                api_calls=r["api_calls"] or 0, est_cost=r["est_cost"] or 0.0,
                act_cost=r["act_cost"] or 0.0, cost_source=r["cost_source"] or "")
            continue
        # sessions: a session landing in two cost_source groups of one provider/model is
        # counted twice here; cost_source is provider-stable in practice, so exposure is nil.
        row.sessions += r["sessions"]
        row.in_tok += r["in_tok"] or 0
        row.out_tok += r["out_tok"] or 0
        row.cache_r += r["cache_r"] or 0
        row.cache_w += r["cache_w"] or 0
        row.think += r["think"] or 0
        row.api_calls += r["api_calls"] or 0
        row.est_cost += r["est_cost"] or 0.0
        row.act_cost += r["act_cost"] or 0.0
    return sorted(merged.values(), key=lambda r: r.in_tok + r.out_tok, reverse=True)
