"""
Fetches live Maryland Multi-Match draw results from mdlottery.com.

Falls back gracefully if the site is unreachable or the page structure changes:
network fetch -> local cache -> caller's static history.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime

RESULTS_URL = "https://www.mdlottery.com/player-tools/winning-numbers/"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_cache.json")
# Multi-Match draws twice a week, so re-hitting the site more often than this buys nothing.
CACHE_TTL_SECONDS = 3 * 60 * 60


def fetch_live_results() -> tuple:
    """
    Return recent Multi-Match draw records, preferring a fresh network fetch
    and falling back to a local cache (fresh or stale) if the network fails.

    Returns (records, note, source):
      records: tuple[DrawRecord, ...] on success, None if nothing was available.
      note:    None on a clean live fetch, otherwise a human-readable explanation.
      source:  "live", "cache", or None.
    """
    from lottery_game import DrawRecord, SUPPORTED_DRAW_DAYS  # local import avoids circular dep

    cached = _read_cache()
    if cached is not None and (time.time() - cached.get("fetched_at", 0)) < CACHE_TTL_SECONDS:
        records = _records_from_cache(cached, DrawRecord)
        if records:
            return records, None, "cache"

    html, fetch_err = _download(RESULTS_URL)
    if html is not None:
        records = _parse_html(html, DrawRecord, SUPPORTED_DRAW_DAYS)
        if records:
            _write_cache(records)
            return records, None, "live"
        fetch_err = "parse failed: Multi-Match table not found or empty"

    records = _records_from_cache(cached, DrawRecord) if cached is not None else None
    if records:
        return records, f"live fetch failed ({fetch_err}); using cached data", "cache"

    return None, fetch_err or "unknown error", None


def _download(url: str) -> tuple[str | None, str | None]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore"), None
    except Exception as exc:
        return None, f"network error: {exc}"


def _parse_html(html: str, DrawRecord, SUPPORTED_DRAW_DAYS) -> tuple | None:
    """Extract Multi-Match draw records from the winning-numbers table."""
    table_match = re.search(
        r'<table[^>]*id="table_multi-match"[^>]*>(.*?)</table>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    if not table_match:
        return None
    table_html = table_match.group(1)

    row_re = re.compile(
        r'<td class="date">\s*([\d/]+)\s*</td>\s*<td class="numbers">(.*?)</td>',
        re.IGNORECASE | re.DOTALL,
    )
    ball_re = re.compile(r"<li>\s*(\d{1,2})\s*</li>", re.IGNORECASE)

    records: list = []
    for date_str, numbers_html in row_re.findall(table_html):
        try:
            draw_date = datetime.strptime(date_str.strip(), "%m/%d/%y").date()
        except ValueError:
            continue
        if draw_date.strftime("%A") not in SUPPORTED_DRAW_DAYS:
            continue

        numbers = [int(n) for n in ball_re.findall(numbers_html)]
        if len(numbers) != 6:
            continue
        try:
            records.append(DrawRecord(draw_date, tuple(sorted(numbers))))
        except (ValueError, TypeError):
            continue

    if not records:
        return None

    # Deduplicate by draw_date and sort chronologically
    seen: dict = {}
    for r in records:
        seen.setdefault(r.draw_date, r)
    return tuple(sorted(seen.values(), key=lambda r: r.draw_date))


def _read_cache() -> dict | None:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _records_from_cache(cached: dict, DrawRecord) -> tuple | None:
    try:
        records = tuple(
            DrawRecord(datetime.strptime(row["date"], "%Y-%m-%d").date(), tuple(row["numbers"]))
            for row in cached["records"]
        )
        return records or None
    except (KeyError, ValueError, TypeError):
        return None


def _write_cache(records: tuple) -> None:
    payload = {
        "fetched_at": time.time(),
        "records": [
            {"date": r.draw_date.isoformat(), "numbers": list(r.numbers)}
            for r in records
        ],
    }
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError:
        pass


def merge_history(static: tuple, live: tuple) -> tuple:
    """Combine static and live records; live records win on date conflicts."""
    if live is None:
        return static
    live_dates = {r.draw_date for r in live}
    merged = list(live) + [r for r in static if r.draw_date not in live_dates]
    merged.sort(key=lambda r: r.draw_date)
    return tuple(merged)
