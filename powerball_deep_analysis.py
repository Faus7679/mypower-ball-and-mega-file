#!/usr/bin/env python3
"""
Advanced Powerball Pattern Analysis
Combines: frequency, recency, overdue, pair-co-occurrence, sum-range,
odd/even split, high/low split, consecutive patterns, delta sequences,
positional frequency, decade distribution, and a multi-factor composite
score to propose the highest-confidence ticket combinations.
"""

import sys
import io
from collections import Counter, defaultdict
from itertools import combinations
import math

import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PB_API = (
    "https://data.ny.gov/resource/d6yy-54nr.json"
    "?$order=draw_date%20DESC&$limit=5000"
)
EXCEL_FILE  = "powerball_game.xlsx"
MAIN_COLS   = ["Num1", "Num2", "Num3", "Num4", "Num5"]
RECENT_N    = 104   # ~2 years of draws


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────

def fetch_api() -> pd.DataFrame:
    print("  Fetching Powerball history from NY Open Data …")
    resp = requests.get(PB_API, timeout=25)
    resp.raise_for_status()
    rows = []
    for item in resp.json():
        raw_date = item.get("draw_date", "")[:10]
        parts    = item.get("winning_numbers", "").split()
        if len(parts) < 6:
            continue
        try:
            nums = [int(p) for p in parts[:5]]
            pb   = int(parts[5])
            dt   = pd.to_datetime(raw_date)
        except (ValueError, TypeError):
            continue
        rows.append({
            "draw_date": dt, "Year": dt.year, "Month": dt.month,
            "Day": dt.day, "DOW": dt.strftime("%A"),
            "Num1": nums[0], "Num2": nums[1], "Num3": nums[2],
            "Num4": nums[3], "Num5": nums[4], "Powerball": pb,
        })
    df = pd.DataFrame(rows)
    print(f"  → {len(df)} draws from API")
    return df


def load_excel() -> pd.DataFrame:
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
            rows.append({
                "draw_date": dt, "Year": int(r["Year"]),
                "Month": int(r["Month"]), "Day": int(r["Day"]),
                "DOW": dt.strftime("%A"),
                "Num1": int(r["Num1"]), "Num2": int(r["Num2"]),
                "Num3": int(r["Num3"]), "Num4": int(r["Num4"]),
                "Num5": int(r["Num5"]), "Powerball": int(r["powerball"]),
            })
        df = pd.DataFrame(rows)
        print(f"  → {len(df)} draws from Excel")
        return df
    except Exception as exc:
        print(f"  ⚠  Excel load failed: {exc}")
        return pd.DataFrame()


def load_data() -> pd.DataFrame:
    api_df   = fetch_api()
    excel_df = load_excel()
    combined = pd.concat([api_df, excel_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Year", "Month", "Day", "Powerball"])
    combined = combined.sort_values("draw_date").reset_index(drop=True)
    print(f"  → {len(combined)} unique draws  "
          f"({combined['draw_date'].min().date()} – {combined['draw_date'].max().date()})")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def row_nums(row) -> list[int]:
    return [int(row[c]) for c in MAIN_COLS]


def all_nums(df: pd.DataFrame) -> list[int]:
    return [int(v) for c in MAIN_COLS for v in df[c].dropna()]


def sec(title: str, w: int = 66):
    print(f"\n{'═'*w}\n  {title}\n{'═'*w}")


def bar(v: float, mx: float, w: int = 28) -> str:
    f = int(round(v / mx * w)) if mx else 0
    return "█" * f + "░" * (w - f)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1 – FULL & RECENT FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def frequency_analysis(df: pd.DataFrame, recent_n: int = RECENT_N):
    sec("PATTERN 1 – NUMBER FREQUENCY  (all-time top-25 + recent hot-20)")
    total  = len(df)
    full_c = Counter(all_nums(df))
    rec_c  = Counter(all_nums(df.tail(recent_n)))
    exp    = total * 5 / 69

    print(f"  Total draws: {total:,}  |  Expected per number ≈ {exp:.1f}")
    print(f"\n  {'#':>3}  {'All-time':>8}  {'Recent':>7}  Bar (all-time)")
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*28}")
    top = full_c.most_common(1)[0][1]
    for num, cnt in full_c.most_common(25):
        r_cnt = rec_c.get(num, 0)
        pct   = cnt / total * 100
        print(f"  {num:>3}  {cnt:>6} ({pct:4.1f}%)  {r_cnt:>6}   {bar(cnt, top)}")
    return full_c, rec_c


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2 – OVERDUE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def overdue_analysis(df: pd.DataFrame) -> dict:
    sec("PATTERN 2 – OVERDUE TRACKER  (draws since last appearance)")
    last: dict[int, int] = {}
    for idx, row in df.iterrows():
        for n in row_nums(row):
            last[n] = int(idx)
    total = len(df) - 1
    gap   = {n: total - last.get(n, -1) for n in range(1, 70)}
    top15 = sorted(gap.items(), key=lambda x: -x[1])[:15]
    mx    = top15[0][1]
    print(f"  {'#':>3}  {'Draws ago':>10}  {'Draws back %':>13}  Bar")
    print(f"  {'─'*3}  {'─'*10}  {'─'*13}  {'─'*28}")
    for num, g in top15:
        pct = g / total * 100
        print(f"  {num:>3}  {g:>10}  {pct:>12.1f}%  {bar(g, mx)}")
    return gap


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3 – ODD / EVEN  DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def odd_even_analysis(df: pd.DataFrame):
    sec("PATTERN 3 – ODD / EVEN SPLIT  (best historical ratio)")
    counts = Counter()
    for _, row in df.iterrows():
        nums = row_nums(row)
        odds  = sum(1 for n in nums if n % 2 != 0)
        evens = 5 - odds
        counts[(odds, evens)] += 1
    total = sum(counts.values())
    print(f"  {'Odd-Even':>10}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*28}")
    top = counts.most_common(1)[0][1]
    for (o, e), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {o}O-{e}E        {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best_oe = counts.most_common(1)[0][0]
    print(f"\n  → Best ratio: {best_oe[0]} Odd / {best_oe[1]} Even  "
          f"({counts[best_oe]/total*100:.1f}% of draws)")
    return best_oe


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4 – HIGH / LOW SPLIT  (1-35 = low, 36-69 = high)
# ─────────────────────────────────────────────────────────────────────────────

def high_low_analysis(df: pd.DataFrame):
    sec("PATTERN 4 – HIGH / LOW SPLIT  (1-35 low, 36-69 high)")
    counts = Counter()
    for _, row in df.iterrows():
        nums = row_nums(row)
        low  = sum(1 for n in nums if n <= 35)
        high = 5 - low
        counts[(low, high)] += 1
    total = sum(counts.values())
    print(f"  {'Low-High':>10}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*28}")
    top = counts.most_common(1)[0][1]
    for (l, h), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {l}L-{h}H        {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best_lh = counts.most_common(1)[0][0]
    print(f"\n  → Best ratio: {best_lh[0]} Low / {best_lh[1]} High")
    return best_lh


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 5 – SUM RANGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def sum_range_analysis(df: pd.DataFrame):
    sec("PATTERN 5 – SUM RANGE  (sum of 5 main numbers)")
    sums = [sum(row_nums(row)) for _, row in df.iterrows()]
    s_arr = pd.Series(sums)
    print(f"  Min: {s_arr.min()}  |  Max: {s_arr.max()}  |  Mean: {s_arr.mean():.1f}  |  Median: {s_arr.median():.1f}")

    bins  = range(0, 350, 25)
    hist  = s_arr.value_counts(bins=pd.interval_range(start=0, end=350, freq=25))
    total = len(sums)
    best_sum_range = None
    best_count     = 0
    print(f"\n  {'Range':>12}  {'Count':>7}  {'%':>7}  Bar")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*28}")
    top = hist.max()
    for interval, cnt in hist.sort_index().items():
        pct = cnt / total * 100
        mark = " ◄ PEAK" if cnt == top else ""
        print(f"  {str(interval):>12}  {cnt:>7}  {pct:>6.1f}%  {bar(cnt, top)}{mark}")
        if cnt > best_count:
            best_count     = cnt
            best_sum_range = interval
    optimal_sum = (best_sum_range.left + best_sum_range.right) / 2
    print(f"\n  → Optimal sum target: ~{optimal_sum:.0f}  (range {best_sum_range})")
    return optimal_sum


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 6 – CONSECUTIVE NUMBER PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def consecutive_analysis(df: pd.DataFrame):
    sec("PATTERN 6 – CONSECUTIVE NUMBER PAIRS  (how often drawn together)")
    consec_counts  = Counter()  # how many consec pairs in a draw
    consec_present = Counter()  # which consecutive pair appeared

    for _, row in df.iterrows():
        nums = sorted(row_nums(row))
        pairs = 0
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                pairs += 1
                consec_present[(nums[i], nums[i + 1])] += 1
        consec_counts[pairs] += 1

    total = sum(consec_counts.values())
    print(f"\n  Consecutive pairs in a draw:")
    print(f"  {'# Pairs':>8}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*28}")
    top = consec_counts.most_common(1)[0][1]
    for k in sorted(consec_counts):
        cnt = consec_counts[k]
        print(f"  {k:>8}  {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")

    best_consec = consec_counts.most_common(1)[0][0]
    print(f"\n  → Most draws have {best_consec} consecutive pair(s)")
    print(f"\n  Top 10 most common consecutive pairs:")
    for pair, cnt in consec_present.most_common(10):
        print(f"    {pair}  ×{cnt}")
    return best_consec, consec_present


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 7 – DELTA (GAP) SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def delta_analysis(df: pd.DataFrame):
    sec("PATTERN 7 – DELTA / SPACING PATTERN  (gaps between sorted numbers)")
    delta_freq: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(row_nums(row))
        for i in range(1, len(nums)):
            delta_freq[nums[i] - nums[i - 1]] += 1

    total = sum(delta_freq.values())
    print(f"  {'Delta':>6}  {'Count':>7}  {'%':>7}  Bar")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*28}")
    top = delta_freq.most_common(1)[0][1]
    for d, cnt in sorted(delta_freq.items()):
        if cnt / total > 0.01:
            print(f"  {d:>6}  {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")

    avg_delta = sum(d * cnt for d, cnt in delta_freq.items()) / total
    top5_deltas = [d for d, _ in delta_freq.most_common(5)]
    print(f"\n  → Average delta: {avg_delta:.2f}  |  Top-5 deltas: {sorted(top5_deltas)}")
    return avg_delta, top5_deltas


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 8 – DECADE / RANGE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def decade_analysis(df: pd.DataFrame):
    sec("PATTERN 8 – DECADE DISTRIBUTION  (1-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69)")
    decades = {f"{lo}-{lo+9}": range(lo, lo + 10) for lo in range(1, 69, 10)}
    # last bucket is 61-69
    decades = {
        "01-09": range(1, 10),
        "10-19": range(10, 20),
        "20-29": range(20, 30),
        "30-39": range(30, 40),
        "40-49": range(40, 50),
        "50-59": range(50, 60),
        "60-69": range(60, 70),
    }
    dec_count: Counter = Counter()
    for _, row in df.iterrows():
        for n in row_nums(row):
            for label, rng in decades.items():
                if n in rng:
                    dec_count[label] += 1
                    break

    total = sum(dec_count.values())
    expected = total / 7
    print(f"  {'Decade':>8}  {'Count':>7}  {'vs exp':>8}  Bar")
    print(f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*28}")
    top = max(dec_count.values())
    for label in decades:
        cnt   = dec_count[label]
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:>8}  {cnt:>7}  {sign}{delta:>6.1f}%  {bar(cnt, top)}")
    return dec_count, decades


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 9 – POSITIONAL FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def positional_frequency(df: pd.DataFrame):
    sec("PATTERN 9 – POSITIONAL FREQUENCY  (hottest number at each position)")
    pos_counters = [Counter() for _ in range(5)]
    for _, row in df.iterrows():
        nums = sorted(row_nums(row))
        for i, n in enumerate(nums):
            pos_counters[i][n] += 1

    best_per_pos = []
    for i, c in enumerate(pos_counters):
        top5 = c.most_common(5)
        best_per_pos.append([n for n, _ in top5])
        nums_str = "  ".join(f"{n:>2}({cnt})" for n, cnt in top5)
        print(f"  Position {i+1}:  {nums_str}")
    return best_per_pos


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 10 – PAIR CO-OCCURRENCE
# ─────────────────────────────────────────────────────────────────────────────

def pair_analysis(df: pd.DataFrame) -> Counter:
    sec("PATTERN 10 – PAIR CO-OCCURRENCE  (top 20 pairs)")
    pair_c: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(row_nums(row))
        for a, b in combinations(nums, 2):
            pair_c[(a, b)] += 1
    top = pair_c.most_common(1)[0][1]
    print(f"  {'Pair':>12}  {'Count':>6}  Bar")
    print(f"  {'─'*12}  {'─'*6}  {'─'*28}")
    for pair, cnt in pair_c.most_common(20):
        print(f"  {str(pair):>12}  {cnt:>6}  {bar(cnt, top)}")
    return pair_c


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 11 – POWERBALL BALL FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def pb_frequency(df: pd.DataFrame) -> Counter:
    sec("PATTERN 11 – POWERBALL BALL FREQUENCY  (1-26 ranked)")
    pb_draws = df[df["Powerball"].notna()]
    c        = Counter(int(v) for v in pb_draws["Powerball"])
    total    = len(pb_draws)
    expected = total / 26
    top      = c.most_common(1)[0][1]
    print(f"  Total: {total:,}  |  Expected each: {expected:.1f}")
    print(f"\n  {'Ball':>5}  {'Count':>6}  {'vs avg':>8}  Bar")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*28}")
    for ball, cnt in c.most_common():
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {ball:>5}  {cnt:>6}  {sign}{delta:>6.1f}%  {bar(cnt, top)}")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_score(
    full_c:  Counter,
    rec_c:   Counter,
    gap:     dict,
    pair_c:  Counter,
    df:      pd.DataFrame,
) -> list[tuple[int, float]]:
    sec("COMPOSITE MULTI-FACTOR SCORE  (top 30)")

    pair_s: Counter = Counter()
    for (a, b), cnt in pair_c.items():
        pair_s[a] += cnt
        pair_s[b] += cnt
    max_pair = max(pair_s.values()) or 1
    max_gap  = max(gap.values()) or 1

    full_rank   = {n: r for r, (n, _) in enumerate(full_c.most_common())}
    rec_rank    = {n: r for r, (n, _) in enumerate(rec_c.most_common())}
    total_ranks = max(len(full_c), len(rec_c), 1)

    scores: list[tuple[int, float]] = []
    for n in range(1, 70):
        f_r = full_rank.get(n, total_ranks) / total_ranks
        r_r = rec_rank.get(n, total_ranks) / total_ranks
        p_s = pair_s.get(n, 0) / max_pair
        g_s = gap.get(n, max_gap) / max_gap
        # Overdue-weighted: gap since last hit now dominates (was 15%, hot-recency
        # was 35%). Numbers that are "due" outrank numbers that are merely hot.
        score = 0.40 * g_s + 0.25 * (1 - f_r) + 0.20 * p_s + 0.15 * (1 - r_r)
        scores.append((n, round(score, 5)))

    scores.sort(key=lambda x: -x[1])
    top_s = scores[0][1]
    print(f"  {'#':>3}  {'Score':>8}  Bar")
    print(f"  {'─'*3}  {'─'*8}  {'─'*28}")
    for num, sc in scores[:30]:
        print(f"  {num:>3}  {sc:>8.5f}  {bar(sc, top_s)}")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TICKET BUILDER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def show_ticket(label: str, main: list[int], pb: int, note: str = ""):
    w = 56
    print(f"\n  ┌─ {label} {'─'*(w-len(label)-3)}┐")
    print(f"  │  Main  :  {' – '.join(f'{n:2d}' for n in sorted(main)):<44}│")
    print(f"  │  Powerball:  {pb:<2}{'':>41}│")
    o = sum(1 for n in main if n % 2 != 0)
    l = sum(1 for n in main if n <= 35)
    h = 5 - l
    s = sum(main)
    print(f"  │  Profile: {o}O-{5-o}E | {l}L-{h}H | Sum={s:<3}{'':>35}│")
    if note:
        print(f"  │  Note  : {note:<46}│")
    print(f"  └{'─'*w}┘")


# ─────────────────────────────────────────────────────────────────────────────
# DRAW AUDIT – analyse a specific set of numbers against history
# ─────────────────────────────────────────────────────────────────────────────

def analyze_specific_ticket(
    pick_nums: list,
    pick_pb:   int,
    full_c:    Counter,
    recent_c:  Counter,
    gap:       dict,
    pair_c:    Counter,
    pb_c:      Counter,
    scores:    list,
    df:        "pd.DataFrame",
):
    sorted_nums = sorted(pick_nums)
    sec(f"DRAW AUDIT:  {' – '.join(f'{n:02d}' for n in sorted_nums)}  PB {pick_pb:02d}")

    total      = len(df)
    score_dict = dict(scores)
    score_rank = {n: r + 1 for r, (n, _) in enumerate(scores)}
    full_rank  = {n: r + 1 for r, (n, _) in enumerate(full_c.most_common())}
    rec_rank   = {n: r + 1 for r, (n, _) in enumerate(recent_c.most_common())}

    # ── Per-number breakdown ──────────────────────────────────────────────────
    print(f"\n  ── PER-NUMBER BREAKDOWN ──────────────────────────────────────────────")
    print(f"  {'Num':>4}  {'All-time cnt':>13}  {'Recent-52':>9}  {'Draws ago':>10}  {'Hot score':>10}  {'Rank/69':>7}")
    print(f"  {'─'*4}  {'─'*13}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*7}")
    for n in sorted_nums:
        f_cnt = full_c.get(n, 0)
        r_cnt = recent_c.get(n, 0)
        g     = gap.get(n, total)
        sc    = score_dict.get(n, 0.0)
        srk   = score_rank.get(n, 69)
        pct   = f_cnt / total * 100 if total else 0
        tag   = (" [HOT]"     if srk <= 15 else
                 " [WARM]"    if srk <= 30 else
                 " [OVERDUE]" if g >= 25   else
                 " [COLD]"    if srk >= 55 else "")
        print(f"  {n:>4}  {f_cnt:>5} ({pct:4.1f}%)  {r_cnt:>9}  {g:>10}  {sc:>10.5f}  #{srk:>2}{tag}")

    # ── Profile ───────────────────────────────────────────────────────────────
    odds   = sum(1 for n in pick_nums if n % 2 != 0)
    evens  = 5 - odds
    lows   = sum(1 for n in pick_nums if n <= 35)
    highs  = 5 - lows
    s      = sum(pick_nums)
    decades_map = [
        ("01-09", range(1,  10)), ("10-19", range(10, 20)),
        ("20-29", range(20, 30)), ("30-39", range(30, 40)),
        ("40-49", range(40, 50)), ("50-59", range(50, 60)),
        ("60-69", range(60, 70)),
    ]
    used_dec = []
    for n in sorted_nums:
        for label, rng in decades_map:
            if n in rng:
                used_dec.append(f"{n}→{label}")
                break
    consec = [(sorted_nums[i], sorted_nums[i+1])
              for i in range(len(sorted_nums)-1)
              if sorted_nums[i+1] - sorted_nums[i] == 1]

    print(f"\n  ── TICKET PROFILE ────────────────────────────────────────────────────")
    print(f"  Odd / Even   : {odds}O – {evens}E")
    print(f"  Low / High   : {lows}L (≤35) – {highs}H (36-69)")
    print(f"  Sum          : {s}  (historical peak ≈ 125–175)")
    print(f"  Consecutive  : {len(consec)} pair(s)  {'none' if not consec else consec}")
    print(f"  Decades      : {' | '.join(used_dec)}")

    # ── Pair strength ─────────────────────────────────────────────────────────
    all_pairs      = list(pair_c.most_common())
    pair_rank_map  = {p: r + 1 for r, (p, _) in enumerate(all_pairs)}
    print(f"\n  ── PAIR STRENGTH (within this ticket) ─────────────────────────────")
    print(f"  {'Pair':>12}  {'Historical count':>17}  {'Rank':>6}  Strength")
    print(f"  {'─'*12}  {'─'*17}  {'─'*6}  {'─'*10}")
    for a, b in combinations(sorted_nums, 2):
        pair  = (a, b)
        cnt   = pair_c.get(pair, 0)
        rk    = pair_rank_map.get(pair, len(all_pairs))
        total_pairs = len(all_pairs)
        pct_rk = rk / total_pairs * 100 if total_pairs else 100
        strength = ("★ TOP TIER" if pct_rk <= 5  else
                    "STRONG"    if pct_rk <= 15 else
                    "medium"    if pct_rk <= 40 else "weak")
        print(f"  {str(pair):>12}  {cnt:>17}  #{rk:>5}  {strength}")

    # ── Powerball ball ────────────────────────────────────────────────────────
    pb_total  = sum(pb_c.values())
    pb_cnt    = pb_c.get(pick_pb, 0)
    pb_ranks  = {b: r + 1 for r, (b, _) in enumerate(pb_c.most_common())}
    pb_rank   = pb_ranks.get(pick_pb, 26)
    pb_pct    = pb_cnt / pb_total * 100 if pb_total else 0
    exp_pb    = pb_total / 26 if pb_total else 0
    delta_pb  = (pb_cnt - exp_pb) / exp_pb * 100 if exp_pb else 0
    sign_pb   = "+" if delta_pb >= 0 else ""
    pb_label  = ("ABOVE avg" if delta_pb > 5 else
                 "BELOW avg" if delta_pb < -5 else "NEAR avg")

    print(f"\n  ── POWERBALL BALL {pick_pb:02d} ─────────────────────────────────────────────")
    print(f"  Appeared     : {pb_cnt}×  ({pb_pct:.1f}% of draws)")
    print(f"  Expected     : {exp_pb:.1f}×")
    print(f"  vs Average   : {sign_pb}{delta_pb:.1f}%  [{pb_label}]")
    print(f"  Frequency Rank: #{pb_rank} out of 26 Powerballs")

    # ── Overall assessment ────────────────────────────────────────────────────
    hot_count    = sum(1 for n in pick_nums if score_rank.get(n, 69) <= 20)
    overdue_cnt  = sum(1 for n in pick_nums if gap.get(n, 0) >= 20)
    avg_sc       = sum(score_dict.get(n, 0) for n in pick_nums) / len(pick_nums)
    # normalise roughly: top composite is ~0.60, random ~0.40
    rel = min(max((avg_sc - 0.30) / 0.30, 0), 1)
    filled = int(round(rel * 30))
    meter = "█" * filled + "░" * (30 - filled)
    level = "HIGH" if rel >= 0.65 else ("MEDIUM" if rel >= 0.35 else "LOW")

    print(f"\n  ── OVERALL STATISTICAL ASSESSMENT ─────────────────────────────────")
    print(f"  Hot numbers  (top-20 score) : {hot_count}/5")
    print(f"  Overdue numbers (≥20 draws) : {overdue_cnt}/5")
    print(f"  Avg composite score         : {avg_sc:.5f}")
    print(f"  Statistical alignment       : {level}  [{meter}]")
    print(f"\n  ⚑  Each Powerball draw is fully independent – statistics do NOT")
    print(f"     predict the next result.  Play responsibly.")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_diverse_profile_tickets(
    scores:      list[tuple[int, float]],
    best_oe:     tuple,
    best_lh:     tuple,
    optimal_sum: float,
    pool_size:   int = 45,
    n_tickets:   int = 5,
    max_reuse:   int = 2,
) -> list[list[int]]:
    """Build N tickets that each match the single strongest historical
    profile (dominant odd/even split, dominant low/high split, sum near
    the peak bucket, zero consecutive pairs — the most common shape a
    real draw takes), picked from the hottest composite-score numbers,
    while capping how many times any one number can reuse across the N
    tickets so the set is genuinely diversified rather than near-clones
    of the single best combination.
    """
    score_dict = dict(scores)
    pool       = [n for n, _ in scores[:pool_size]]
    used       = Counter()
    tickets: list[list[int]] = []

    for _ in range(n_tickets):
        best_combo, best_sc = None, -1.0
        for combo in combinations(pool, 5):
            if any(used[n] >= max_reuse for n in combo):
                continue
            odds  = sum(1 for n in combo if n % 2 != 0)
            low   = sum(1 for n in combo if n <= 35)
            s     = sum(combo)
            nums  = sorted(combo)
            consec = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
            if odds != best_oe[0] or low != best_lh[0]:
                continue
            if abs(s - optimal_sum) > 25:
                continue
            if consec > 0:
                continue
            sc = sum(score_dict[n] for n in combo) - sum(used[n] for n in combo) * 0.20
            if sc > best_sc:
                best_sc, best_combo = sc, combo
        if best_combo is None:
            break
        tickets.append(sorted(best_combo))
        for n in best_combo:
            used[n] += 1
    return tickets


def final_picks(
    scores:      list[tuple[int, float]],
    pb_c:        Counter,
    best_oe:     tuple,
    best_lh:     tuple,
    optimal_sum: float,
):
    sec("★  FIVE STRONG TICKETS  (overdue-weighted profile-match pattern)  ★")

    print(f"  Pattern used: composite score now led by overdue gap (40%) rather than\n"
          f"  recent hotness (15%), pool constrained to the single most common draw\n"
          f"  shape — {best_oe[0]}O-{best_oe[1]}E, {best_lh[0]}L-{best_lh[1]}H, "
          f"sum≈{optimal_sum:.0f}, 0 consecutive pairs —\n"
          f"  with number reuse capped so the 5 tickets stay genuinely distinct.\n")

    top_pb  = [b for b, _ in pb_c.most_common(5)]
    tickets = build_diverse_profile_tickets(scores, best_oe, best_lh, optimal_sum)
    score_dict = dict(scores)

    for i, t in enumerate(tickets):
        avg_sc = sum(score_dict[n] for n in t) / len(t)
        show_ticket(f"Ticket {i+1} – STRONG PATTERN MATCH", t, top_pb[i % len(top_pb)],
                    f"Profile-matched, diversity-capped pick | avg composite score={avg_sc:.4f}")

    # ── Powerball Shortlist ───────────────────────────────────────────────
    print(f"\n  ─── TOP POWERBALL BALLS (by frequency) ───")
    for ball, cnt in pb_c.most_common(7):
        pct = cnt / sum(pb_c.values()) * 100
        print(f"    Ball {ball:>2}  appeared {cnt}×  ({pct:.1f}%)")

    print("""
  ════════════════════════════════════════════════════════
  DISCLAIMER: Statistical analysis does NOT predict draws.
  Each Powerball draw is an independent random event.
  Play responsibly and within your budget.
  ════════════════════════════════════════════════════════
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         POWERBALL  ADVANCED PATTERN ANALYSIS                ║")
    print("║         11 Statistical Patterns → 5 Strong Ticket Picks     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        df = load_data()
    except Exception as exc:
        print(f"\nERROR loading data: {exc}")
        sys.exit(1)

    if df.empty:
        print("No data. Exiting.")
        sys.exit(1)

    full_c, rec_c = frequency_analysis(df)
    gap           = overdue_analysis(df)
    best_oe       = odd_even_analysis(df)
    best_lh       = high_low_analysis(df)
    optimal_sum   = sum_range_analysis(df)
    consecutive_analysis(df)
    delta_avg, _  = delta_analysis(df)
    decade_analysis(df)
    positional_frequency(df)
    pair_c        = pair_analysis(df)
    pb_c          = pb_frequency(df)
    scores        = composite_score(full_c, rec_c, gap, pair_c, df)

    final_picks(scores, pb_c, best_oe, best_lh, optimal_sum)

    # ── Specific draw audit ──────────────────────────────────────────────────
    analyze_specific_ticket(
        pick_nums=[30, 36, 40, 42, 57],
        pick_pb=2,
        full_c=full_c,
        recent_c=rec_c,
        gap=gap,
        pair_c=pair_c,
        pb_c=pb_c,
        scores=scores,
        df=df,
    )


if __name__ == "__main__":
    main()
