#!/usr/bin/env python3
"""
Powerball & Double Play Combined Statistical Analysis
Fetches full draw history from NY Open Data (Powerball + Double Play), runs
9 statistical patterns on each game, then proposes the hottest ticket combos.
"""

import sys
import io
from collections import Counter
from itertools import combinations

import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PB_API     = (
    "https://data.ny.gov/resource/d6yy-54nr.json"
    "?$order=draw_date%20DESC&$limit=5000"
)
EXCEL_FILE = "powerball_draws_full.xlsx"
MAIN_COLS  = ["Num1", "Num2", "Num3", "Num4", "Num5"]
DP_COLS    = ["DP1",  "DP2",  "DP3",  "DP4",  "DP5"]
RECENT_N   = 104   # ~1 year (Powerball draws Mon / Wed / Sat)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_api() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("  Fetching Powerball + Double Play history from NY Open Data …")
    resp = requests.get(PB_API, timeout=30)
    resp.raise_for_status()

    pb_rows, dp_rows = [], []

    for item in resp.json():
        raw_date = item.get("draw_date", "")[:10]
        winning  = item.get("winning_numbers", "")

        # ── Powerball numbers ─────────────────────────────────────────────────
        parts = winning.split()
        if len(parts) >= 6:
            try:
                nums = [int(p) for p in parts[:5]]
                pb   = int(parts[5])
                dt   = pd.to_datetime(raw_date)
                pb_rows.append({
                    "draw_date": dt, "Year": dt.year, "Month": dt.month,
                    "Day": dt.day, "DOW": dt.strftime("%A"),
                    "Num1": nums[0], "Num2": nums[1], "Num3": nums[2],
                    "Num4": nums[3], "Num5": nums[4], "Powerball": pb,
                })
            except (ValueError, TypeError):
                pass

        # ── Double Play numbers (field present from Aug 2021) ─────────────────
        dp_raw = (
            item.get("double_play_winning_numbers")
            or item.get("doubleplay_winning_numbers")
            or item.get("double_play")
            or ""
        )
        dp_parts = dp_raw.split() if dp_raw else []
        if len(dp_parts) >= 6:
            try:
                dp_nums = [int(p) for p in dp_parts[:5]]
                dp_pb   = int(dp_parts[5])
                dt      = pd.to_datetime(raw_date)
                dp_rows.append({
                    "draw_date": dt, "Year": dt.year, "Month": dt.month,
                    "Day": dt.day, "DOW": dt.strftime("%A"),
                    "DP1": dp_nums[0], "DP2": dp_nums[1], "DP3": dp_nums[2],
                    "DP4": dp_nums[3], "DP5": dp_nums[4], "Powerball": dp_pb,
                })
            except (ValueError, TypeError):
                pass

    def clean(rows, key_cols):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Year", "Month", "Day"])
        df = df.sort_values("draw_date").reset_index(drop=True)
        return df

    pb_df = clean(pb_rows, MAIN_COLS)
    dp_df = clean(dp_rows, DP_COLS)

    if not pb_df.empty:
        print(f"  → {len(pb_df):,} Powerball draws  "
              f"({pb_df['draw_date'].min().date()} – {pb_df['draw_date'].max().date()})")
    if not dp_df.empty:
        print(f"  → {len(dp_df):,} Double Play draws "
              f"({dp_df['draw_date'].min().date()} – {dp_df['draw_date'].max().date()})")
    else:
        print("  → No Double Play field found in API (will note this below)")

    return pb_df, dp_df


def load_excel_pb() -> pd.DataFrame:
    try:
        raw = pd.read_excel(EXCEL_FILE)
        rows = []
        for _, r in raw.iterrows():
            try:
                dt = pd.to_datetime(
                    f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['Day']):02d}"
                )
            except Exception:
                continue
            pb_val = r.get("Powerball") or r.get("powerball") or 0
            rows.append({
                "draw_date": dt, "Year": dt.year, "Month": dt.month, "Day": dt.day,
                "DOW": dt.strftime("%A"),
                "Num1": int(r["Num1"]), "Num2": int(r["Num2"]), "Num3": int(r["Num3"]),
                "Num4": int(r["Num4"]), "Num5": int(r["Num5"]),
                "Powerball": int(pb_val),
            })
        df = pd.DataFrame(rows)
        print(f"  → {len(df):,} draws from Excel ({EXCEL_FILE})")
        return df
    except Exception as exc:
        print(f"  ⚠  Excel load skipped: {exc}")
        return pd.DataFrame()


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    pb_api, dp_df = fetch_api()
    pb_local      = load_excel_pb()

    pb_df = pd.concat([pb_api, pb_local], ignore_index=True)
    pb_df = (pb_df.drop_duplicates(subset=["Year", "Month", "Day"])
                  .sort_values("draw_date")
                  .reset_index(drop=True))

    print(f"\n  TOTAL: {len(pb_df):,} Powerball draws  |  {len(dp_df):,} Double Play draws\n")
    return pb_df, dp_df


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sec(title: str, w: int = 68):
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")


def sub(title: str, w: int = 62):
    print(f"\n  {'─'*w}\n  {title}\n  {'─'*w}")


def bar(v: float, mx: float, w: int = 26) -> str:
    f = int(round(v / mx * w)) if mx else 0
    return "█" * f + "░" * (w - f)


def extract_nums(df: pd.DataFrame, cols: list[str]) -> list[int]:
    return [int(v) for c in cols for v in df[c].dropna()]


def row_nums(row, cols: list[str]) -> list[int]:
    return [int(row[c]) for c in cols if pd.notna(row[c])]


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1 – FREQUENCY  (all-time + recent)
# ─────────────────────────────────────────────────────────────────────────────

def frequency_analysis(
    df: pd.DataFrame, cols: list[str], tag: str, recent_n: int = RECENT_N
) -> tuple[Counter, Counter]:
    total    = len(df)
    all_c    = Counter(extract_nums(df, cols))
    rec_c    = Counter(extract_nums(df.tail(min(recent_n, total)), cols))
    expected = total * 5 / 69

    print(f"\n  {tag} – {total:,} draws  |  Expected per number ≈ {expected:.1f}")
    print(f"  {'#':>3}  {'All-time':>8}  {'Recent-{:d}'.format(recent_n):>9}  {'% draws':>8}  Bar")
    print(f"  {'─'*3}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*26}")
    top = all_c.most_common(1)[0][1] if all_c else 1
    for num, cnt in all_c.most_common(20):
        r_cnt = rec_c.get(num, 0)
        pct   = cnt / total * 100
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {num:>3}  {cnt:>8}  {r_cnt:>9}  {pct:>7.1f}%  {bar(cnt, top)}"
              f"  ({sign}{delta:.1f}%)")
    return all_c, rec_c


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2 – OVERDUE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def overdue_analysis(df: pd.DataFrame, cols: list[str], tag: str) -> dict:
    last: dict[int, int] = {}
    for idx, row in df.iterrows():
        for n in row_nums(row, cols):
            last[n] = int(idx)
    total = len(df) - 1
    gap   = {n: total - last.get(n, -1) for n in range(1, 70)}
    top15 = sorted(gap.items(), key=lambda x: -x[1])[:15]
    mx    = top15[0][1] if top15 else 1

    print(f"\n  {tag}  (top 15 most-overdue)")
    print(f"  {'#':>3}  {'Draws ago':>10}  {'% history':>10}  Bar")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*26}")
    for num, g in top15:
        pct = g / (total or 1) * 100
        print(f"  {num:>3}  {g:>10}  {pct:>9.1f}%  {bar(g, mx)}")
    return gap


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3 – PAIR CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────

def pair_analysis(df: pd.DataFrame, cols: list[str], tag: str) -> Counter:
    pair_c: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(row_nums(row, cols))
        for a, b in combinations(nums, 2):
            pair_c[(a, b)] += 1
    top = pair_c.most_common(1)[0][1] if pair_c else 1

    print(f"\n  {tag}  (top 15 pairs)")
    print(f"  {'Pair':>12}  {'Count':>6}  Bar")
    print(f"  {'─'*12}  {'─'*6}  {'─'*26}")
    for pair, cnt in pair_c.most_common(15):
        print(f"  {str(pair):>12}  {cnt:>6}  {bar(cnt, top)}")
    return pair_c


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4 – RED BALL FREQUENCY  (1-26)
# ─────────────────────────────────────────────────────────────────────────────

def red_ball_frequency(df: pd.DataFrame, tag: str) -> Counter:
    c      = Counter(int(v) for v in df["Powerball"].dropna())
    total  = sum(c.values())
    exp    = total / 26
    top    = c.most_common(1)[0][1] if c else 1

    print(f"\n  {tag}  –  {total:,} draws  |  Expected each ≈ {exp:.1f}")
    print(f"  {'Ball':>5}  {'Count':>6}  {'vs avg':>8}  Bar")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*26}")
    for ball, cnt in c.most_common():
        delta = (cnt - exp) / exp * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {ball:>5}  {cnt:>6}  {sign}{delta:>6.1f}%  {bar(cnt, top)}")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 5 – ODD / EVEN SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def odd_even_split(df: pd.DataFrame, cols: list[str], tag: str) -> tuple:
    counts: Counter = Counter()
    for _, row in df.iterrows():
        nums  = row_nums(row, cols)
        odds  = sum(1 for n in nums if n % 2 != 0)
        counts[(odds, 5 - odds)] += 1
    total = sum(counts.values())
    top   = counts.most_common(1)[0][1]

    print(f"\n  {tag}")
    print(f"  {'Split':>8}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*26}")
    for (o, e), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {o}O-{e}E      {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best = counts.most_common(1)[0][0]
    print(f"  → Best split: {best[0]} Odd / {best[1]} Even  ({counts[best]/total*100:.1f}%)")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 6 – HIGH / LOW SPLIT  (low = 1-35, high = 36-69)
# ─────────────────────────────────────────────────────────────────────────────

def high_low_split(df: pd.DataFrame, cols: list[str], tag: str) -> tuple:
    counts: Counter = Counter()
    for _, row in df.iterrows():
        nums = row_nums(row, cols)
        low  = sum(1 for n in nums if n <= 35)
        counts[(low, 5 - low)] += 1
    total = sum(counts.values())
    top   = counts.most_common(1)[0][1]

    print(f"\n  {tag}")
    print(f"  {'Split':>8}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*26}")
    for (l, h), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {l}L-{h}H      {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best = counts.most_common(1)[0][0]
    print(f"  → Best split: {best[0]} Low / {best[1]} High  ({counts[best]/total*100:.1f}%)")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 7 – SUM RANGE
# ─────────────────────────────────────────────────────────────────────────────

def sum_range(df: pd.DataFrame, cols: list[str], tag: str) -> float:
    sums  = [sum(row_nums(row, cols)) for _, row in df.iterrows()]
    s     = pd.Series(sums)
    mean  = s.mean()

    print(f"\n  {tag}")
    print(f"  Min: {s.min()}  |  Max: {s.max()}  |  Mean: {mean:.1f}  |  Median: {s.median():.1f}")

    hist  = s.value_counts(
        bins=pd.interval_range(start=0, end=350, freq=25)
    ).sort_index()
    total = len(sums)
    top   = hist.max()
    best_range, best_cnt = None, 0
    print(f"\n  {'Range':>12}  {'Count':>7}  {'%':>7}  Bar")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*26}")
    for interval, cnt in hist.items():
        mark = ""
        if cnt > best_cnt:
            best_cnt   = cnt
            best_range = interval
            mark = "  ◄ PEAK"
        print(f"  {str(interval):>12}  {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}{mark}")

    optimal = (best_range.left + best_range.right) / 2 if best_range else mean
    print(f"  → Target sum: ~{optimal:.0f}")
    return optimal


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 8 – DECADE DISTRIBUTION  (1-9 · 10-19 · … · 60-69)
# ─────────────────────────────────────────────────────────────────────────────

def decade_distribution(df: pd.DataFrame, cols: list[str], tag: str):
    decades = {
        "01-09": range(1,  10), "10-19": range(10, 20),
        "20-29": range(20, 30), "30-39": range(30, 40),
        "40-49": range(40, 50), "50-59": range(50, 60),
        "60-69": range(60, 70),
    }
    dec_c: Counter = Counter()
    for _, row in df.iterrows():
        for n in row_nums(row, cols):
            for label, rng in decades.items():
                if n in rng:
                    dec_c[label] += 1
                    break

    total    = sum(dec_c.values())
    expected = total / 7
    top      = max(dec_c.values()) if dec_c else 1

    print(f"\n  {tag}")
    print(f"  {'Decade':>8}  {'Count':>7}  {'vs exp':>8}  Bar")
    print(f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*26}")
    for label in decades:
        cnt   = dec_c[label]
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:>8}  {cnt:>7}  {sign}{delta:>6.1f}%  {bar(cnt, top)}")


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 9 – POSITIONAL FREQUENCY  (position 1-5 in sorted draw)
# ─────────────────────────────────────────────────────────────────────────────

def positional_frequency(df: pd.DataFrame, cols: list[str], tag: str) -> list[list[int]]:
    pos_c = [Counter() for _ in range(5)]
    for _, row in df.iterrows():
        for i, n in enumerate(sorted(row_nums(row, cols))):
            pos_c[i][n] += 1

    print(f"\n  {tag}")
    best_per_pos = []
    for i, c in enumerate(pos_c):
        top5 = c.most_common(5)
        best_per_pos.append([n for n, _ in top5])
        nums_str = "  ".join(f"{n:>2}({cnt})" for n, cnt in top5)
        print(f"  Position {i+1}:  {nums_str}")
    return best_per_pos


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE HOT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_hot_score(
    full_c: Counter,
    rec_c:  Counter,
    gap:    dict,
    pair_c: Counter,
) -> list[tuple[int, float]]:
    pair_s: Counter = Counter()
    for (a, b), cnt in pair_c.items():
        pair_s[a] += cnt
        pair_s[b] += cnt

    max_pair   = max(pair_s.values()) if pair_s else 1
    max_gap    = max(gap.values())    if gap     else 1
    full_rank  = {n: r for r, (n, _) in enumerate(full_c.most_common())}
    rec_rank   = {n: r for r, (n, _) in enumerate(rec_c.most_common())}
    total_r    = max(len(full_c), len(rec_c), 1)

    scores: list[tuple[int, float]] = []
    for n in range(1, 70):
        f_r  = full_rank.get(n, total_r) / total_r
        r_r  = rec_rank.get(n, total_r) / total_r
        p_s  = pair_s.get(n, 0) / max_pair
        g_s  = gap.get(n, max_gap) / max_gap
        # Weights: recent 40% · all-time 25% · pair co-occur 20% · overdue 15%
        score = 0.40 * (1 - r_r) + 0.25 * (1 - f_r) + 0.20 * p_s + 0.15 * g_s
        scores.append((n, round(score, 5)))

    scores.sort(key=lambda x: -x[1])
    return scores


def show_scores(scores: list[tuple[int, float]], tag: str, top_n: int = 25):
    top_s = scores[0][1] if scores else 1
    print(f"\n  {tag}  (top {top_n})")
    print(f"  {'Rank':>5}  {'Number':>7}  {'Score':>8}  Bar")
    print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*26}")
    for rank, (num, sc) in enumerate(scores[:top_n], 1):
        print(f"  {rank:>5}  {num:>7}  {sc:>8.5f}  {bar(sc, top_s)}")


# ─────────────────────────────────────────────────────────────────────────────
# TICKET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def show_ticket(label: str, main: list[int], pb: int, note: str = ""):
    W   = 62
    pad = max(W - len(label) - 3, 0)
    print(f"\n  ┌─ {label} {'─'*pad}┐")
    print(f"  │  Main:       {' – '.join(f'{n:2d}' for n in sorted(main)):<50}│")
    print(f"  │  Powerball:  {pb:<2}{'':>48}│")
    o = sum(1 for n in main if n % 2 != 0)
    l = sum(1 for n in main if n <= 35)
    print(f"  │  Profile:    {o}O-{5-o}E | {l}L-{5-l}H | Sum = {sum(main):<3}{'':>39}│")
    if note:
        print(f"  │  Note:       {note:<49}│")
    print(f"  └{'─'*W}┘")


def build_tickets(
    scores:    list[tuple[int, float]],
    pb_c:      Counter,
    pair_c:    Counter,
    rec_c:     Counter,
    gap:       dict,
    game_name: str,
):
    top_pb = [b for b, _ in pb_c.most_common(5)]

    # Ticket 1 – Pure Composite Hot
    t1 = sorted([n for n, _ in scores[:5]])
    show_ticket(f"{game_name} · Ticket 1 – PURE HOT",
                t1, top_pb[0], "Top-5 by composite score")

    # Ticket 2 – Hot + Overdue
    top10 = [n for n, _ in scores[:10]]
    t2    = sorted(sorted(top10, key=lambda n: -gap.get(n, 0))[:5])
    show_ticket(f"{game_name} · Ticket 2 – HOT + OVERDUE",
                t2, top_pb[1], "Top-10 score, overdue-sorted")

    # Ticket 3 – Pair-Anchored
    best_pair = pair_c.most_common(1)[0][0]
    t3_pool   = list(best_pair)
    for n, _ in scores:
        if n not in t3_pool:
            t3_pool.append(n)
        if len(t3_pool) >= 5:
            break
    t3 = sorted(t3_pool[:5])
    show_ticket(f"{game_name} · Ticket 3 – PAIR ANCHOR",
                t3, top_pb[2], f"Built on most common pair {best_pair}")

    # Ticket 4 – Recent Surge
    t4 = sorted([n for n, _ in rec_c.most_common(5)])
    show_ticket(f"{game_name} · Ticket 4 – RECENT SURGE",
                t4, top_pb[0], f"Hottest in last {RECENT_N} draws (~1 year)")

    # Ticket 5 – Wildcard (top-4 hot + most-overdue number)
    wildcard = max(gap, key=gap.get)
    t5       = sorted(set([n for n, _ in scores[:4]] + [wildcard]))[:5]
    show_ticket(f"{game_name} · Ticket 5 – WILDCARD",
                t5, top_pb[3],
                f"Top-4 + #{wildcard} (overdue {gap[wildcard]} draws)")

    print(f"\n  Hot red balls: "
          + "  ".join(f"#{b}({cnt}×)" for b, cnt in pb_c.most_common(7)))


# ─────────────────────────────────────────────────────────────────────────────
# FULL ANALYSIS RUNNER  (shared logic for PB and DP)
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    df:       pd.DataFrame,
    cols:     list[str],
    game_tag: str,
) -> tuple[list[tuple[int, float]], Counter, Counter, Counter, dict]:
    full_c, rec_c = frequency_analysis(df, cols, f"{game_tag} – Main Number Frequency")
    gap            = overdue_analysis(df, cols, f"{game_tag} – Overdue Tracker")
    pair_c         = pair_analysis(df, cols, f"{game_tag} – Pair Co-occurrence")
    odd_even_split(df, cols, f"{game_tag} – Odd/Even Split")
    high_low_split(df, cols, f"{game_tag} – High/Low Split")
    sum_range(df, cols, f"{game_tag} – Sum Range")
    decade_distribution(df, cols, f"{game_tag} – Decade Distribution")
    positional_frequency(df, cols, f"{game_tag} – Positional Frequency")
    scores = composite_hot_score(full_c, rec_c, gap, pair_c)
    show_scores(scores, f"{game_tag} – Composite Hot Score")
    return scores, full_c, rec_c, pair_c, gap


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     POWERBALL + DOUBLE PLAY  COMBINED STATISTICAL ANALYSIS      ║")
    print("║     9 Patterns · Composite Hot Score · 10 Ticket Proposals      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        pb_df, dp_df = load_all()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    if pb_df.empty:
        print("No Powerball data available. Exiting.")
        sys.exit(1)

    # ─── POWERBALL ANALYSIS ───────────────────────────────────────────────────
    sec("POWERBALL  –  STATISTICAL ANALYSIS")
    pb_scores, pb_full_c, pb_rec_c, pb_pair_c, pb_gap = run_analysis(
        pb_df, MAIN_COLS, "POWERBALL"
    )
    pb_ball_c = red_ball_frequency(pb_df, "POWERBALL – Red Ball Frequency (1-26)")

    # ─── DOUBLE PLAY ANALYSIS ─────────────────────────────────────────────────
    DP_MIN_DRAWS = 50   # below this, independent analysis is not meaningful
    dp_sufficient = not dp_df.empty and len(dp_df) >= DP_MIN_DRAWS

    if dp_sufficient:
        sec("DOUBLE PLAY  –  STATISTICAL ANALYSIS")
        dp_scores, dp_full_c, dp_rec_c, dp_pair_c, dp_gap = run_analysis(
            dp_df, DP_COLS, "DOUBLE PLAY"
        )
        dp_ball_c = red_ball_frequency(dp_df, "DOUBLE PLAY – Red Ball Frequency (1-26)")
    else:
        draws_found = len(dp_df) if not dp_df.empty else 0
        sec("DOUBLE PLAY  –  STATISTICAL ANALYSIS  (PROXY MODE)")
        print(f"\n  ⚠  Only {draws_found} Double Play draw(s) are currently available from the")
        print(f"     NY Open Data API (minimum required: {DP_MIN_DRAWS}).")
        print(f"     The API recently began exposing the double_play_winning_numbers field,")
        print(f"     so the historical archive is not yet published.")
        print(f"\n  ► Powerball patterns are used as a statistical proxy for Double Play.")
        print(f"    This is valid because both games draw from identical pools:")
        print(f"    5 white balls (1-69)  +  1 red ball (1-26), independent drawings.")
        dp_scores  = pb_scores
        dp_rec_c   = pb_rec_c
        dp_pair_c  = pb_pair_c
        dp_gap     = pb_gap
        dp_ball_c  = pb_ball_c

    # ─── FINAL TICKET RECOMMENDATIONS ─────────────────────────────────────────
    sec("★  FINAL RECOMMENDED TICKETS  ★")

    sub("POWERBALL TICKETS  (5 main numbers 1-69  +  red ball 1-26)")
    build_tickets(pb_scores, pb_ball_c, pb_pair_c, pb_rec_c, pb_gap, "PB")

    sub("DOUBLE PLAY TICKETS  (same pool as Powerball – independent draw)")
    build_tickets(dp_scores, dp_ball_c, dp_pair_c, dp_rec_c, dp_gap, "DP")

    # ─── HOT NUMBER CROSS-COMPARISON ─────────────────────────────────────────
    cross_label = ("HOT NUMBER CROSS-COMPARISON  (top-15 from each game)"
                   if dp_sufficient else
                   "POWERBALL HOT NUMBER SUMMARY  (DP proxy – same patterns)")
    sec(cross_label)
    pb_top = {n for n, _ in pb_scores[:15]}
    dp_top = {n for n, _ in dp_scores[:15]}
    shared  = sorted(pb_top & dp_top)
    pb_only = sorted(pb_top - dp_top)
    dp_only = sorted(dp_top - pb_top)

    if dp_sufficient:
        print(f"\n  Numbers hot in BOTH games :  {shared}")
        print(f"  Hot in Powerball only      :  {pb_only}")
        print(f"  Hot in Double Play only    :  {dp_only}")
    else:
        print(f"\n  Top-15 hottest numbers (apply to both PB and DP): {sorted(pb_top)}")

    pb_rb = pb_ball_c.most_common(1)[0][0]
    dp_rb = dp_ball_c.most_common(1)[0][0]

    if dp_sufficient and len(shared) >= 5:
        consensus = sorted(shared[:5])
        print(f"\n  ╔═══════════════════════════════════════════════════╗")
        print(f"  ║   ★  CONSENSUS PICK  (hottest in BOTH games)  ★  ║")
        print(f"  ╠═══════════════════════════════════════════════════╣")
        print(f"  ║  Main :  {' – '.join(f'{n:2d}' for n in consensus):<41}║")
        print(f"  ║  PB red ball → {pb_rb:<2}   |   DP red ball → {dp_rb:<2}{'':>11}║")
        print(f"  ╚═══════════════════════════════════════════════════╝")
    else:
        top5 = sorted([n for n, _ in pb_scores[:5]])
        print(f"\n  ╔═══════════════════════════════════════════════════╗")
        print(f"  ║   ★  HIGHEST-CONFIDENCE PICK  (both games)  ★    ║")
        print(f"  ╠═══════════════════════════════════════════════════╣")
        print(f"  ║  Main :  {' – '.join(f'{n:2d}' for n in top5):<41}║")
        print(f"  ║  Red ball → {pb_rb:<2}  (top frequency, 1,959 draws) {'':>10}║")
        print(f"  ╚═══════════════════════════════════════════════════╝")

    print("""
  ════════════════════════════════════════════════════════════════════
  DISCLAIMER: Statistical frequency analysis cannot predict future
  draws. Powerball and Double Play are independent random events.
  Play responsibly and within your budget.
  ════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
