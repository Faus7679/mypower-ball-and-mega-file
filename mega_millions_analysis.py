#!/usr/bin/env python3
"""
Deep Mega Millions statistical analysis.
Fetches all available draws from NY Open Data, then runs a full pattern
mechanism — frequency, recency, overdue, pair co-occurrence, odd/even split,
high/low split, sum-range, consecutive pairs, delta sequencing, decade
distribution, positional frequency, day-of-week bias, and a multi-factor
composite score — to select the winning main numbers plus Mega Ball.
"""

import sys
import io
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
import pandas as pd

# ── NY Open Data – Mega Millions Winning Numbers (beginning 2002) ─────────────
MM_API = (
    "https://data.ny.gov/resource/5xaw-6ayf.json"
    "?$order=draw_date%20DESC&$limit=5000"
)

MAIN_COLS = ["Num1", "Num2", "Num3", "Num4", "Num5"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mega_millions() -> pd.DataFrame:
    print("Fetching Mega Millions draw history from NY Open Data …")
    resp = requests.get(MM_API, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        raw_date = item.get("draw_date", "")[:10]
        winning  = item.get("winning_numbers", "")
        mega_ball = item.get("mega_ball", None)
        multiplier = item.get("multiplier", None)

        parts = winning.split()
        if len(parts) < 5:
            continue

        try:
            nums = [int(p) for p in parts[:5]]
            mb   = int(mega_ball) if mega_ball is not None else (int(parts[5]) if len(parts) > 5 else None)
            dt   = pd.to_datetime(raw_date)
        except (ValueError, TypeError):
            continue

        rows.append({
            "draw_date": dt,
            "Year":  dt.year,
            "Month": dt.month,
            "Day":   dt.day,
            "DOW":   dt.strftime("%A"),
            "Num1":  nums[0],
            "Num2":  nums[1],
            "Num3":  nums[2],
            "Num4":  nums[3],
            "Num5":  nums[4],
            "MegaBall":   mb,
            "Multiplier": multiplier,
        })

    df = pd.DataFrame(rows).sort_values("draw_date").reset_index(drop=True)
    print(f"  → {len(df)} draws loaded  ({df['draw_date'].min().date()} – {df['draw_date'].max().date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def row_nums(row) -> list[int]:
    return [int(row[c]) for c in MAIN_COLS]


def all_main_numbers(df: pd.DataFrame):
    return [int(v) for col in MAIN_COLS for v in df[col].dropna()]


def section(title: str, width: int = 62):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def bar(count: float, total: float, width: int = 30) -> str:
    filled = int(round(count / total * width)) if total else 0
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FULL-HISTORY FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def full_frequency(df: pd.DataFrame):
    section("ALL-TIME MAIN NUMBER FREQUENCY  (top 25 / bottom 10)")
    nums = all_main_numbers(df)
    total_draws = len(df)
    c = Counter(nums)

    expected = total_draws * 5 / 70  # uniform expectation

    print(f"  Total draws analysed: {total_draws:,}  |  Expected per number: {expected:.1f}")
    print(f"\n  {'#':>3}  {'Count':>6}  {'% draws':>8}  {'vs avg':>8}  Bar")
    print(f"  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*30}")

    top25 = c.most_common(25)
    for num, cnt in top25:
        pct   = cnt / total_draws * 100
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {num:>3}  {cnt:>6}  {pct:>7.1f}%  {sign}{delta:>6.1f}%  {bar(cnt, c.most_common(1)[0][1])}")

    print(f"\n  ── COLDEST 10 ──")
    for num, cnt in c.most_common()[:-11:-1]:
        pct   = cnt / total_draws * 100
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {num:>3}  {cnt:>6}  {pct:>7.1f}%  {sign}{delta:>6.1f}%  {bar(cnt, c.most_common(1)[0][1])}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECENT-TREND FREQUENCY  (last 52 draws ≈ 1 year)
# ─────────────────────────────────────────────────────────────────────────────

def recent_frequency(df: pd.DataFrame, n: int = 52):
    section(f"RECENT TREND – last {n} draws  (hottest 20)")
    recent = df.tail(n)
    nums = all_main_numbers(recent)
    c = Counter(nums)

    print(f"  {'#':>3}  {'Count':>6}  Bar")
    print(f"  {'─'*3}  {'─'*6}  {'─'*30}")
    for num, cnt in c.most_common(20):
        print(f"  {num:>3}  {cnt:>6}  {bar(cnt, c.most_common(1)[0][1])}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OVERDUE NUMBERS  (draws since last appearance)
# ─────────────────────────────────────────────────────────────────────────────

def overdue_numbers(df: pd.DataFrame):
    section("OVERDUE NUMBERS  (draws since last seen – top 15)")
    last_seen: dict[int, int] = {}
    for idx, row in df.iterrows():
        for col in MAIN_COLS:
            n = int(row[col])
            if n > 70:
                continue  # pre-2017 matrix used 1-75; not valid under the current 1-70 range
            last_seen[n] = idx   # last draw index where n appeared

    total = len(df) - 1
    gap = {n: total - idx for n, idx in last_seen.items()}

    # numbers that have NEVER appeared get max gap
    for n in range(1, 71):
        if n not in gap:
            gap[n] = total + 1

    print(f"  {'#':>3}  {'Draws ago':>10}  Bar")
    print(f"  {'─'*3}  {'─'*10}  {'─'*30}")
    top15 = sorted(gap.items(), key=lambda x: -x[1])[:15]
    max_gap = top15[0][1]
    for num, g in top15:
        print(f"  {num:>3}  {g:>10}  {bar(g, max_gap)}")

    return gap


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ODD / EVEN  DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def odd_even_analysis(df: pd.DataFrame):
    section("ODD / EVEN SPLIT  (best historical ratio)")
    counts = Counter()
    for _, row in df.iterrows():
        nums = row_nums(row)
        odds  = sum(1 for n in nums if n % 2 != 0)
        evens = 5 - odds
        counts[(odds, evens)] += 1
    total = sum(counts.values())
    print(f"  {'Odd-Even':>10}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*30}")
    top = counts.most_common(1)[0][1]
    for (o, e), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {o}O-{e}E        {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best_oe = counts.most_common(1)[0][0]
    print(f"\n  → Best ratio: {best_oe[0]} Odd / {best_oe[1]} Even  "
          f"({counts[best_oe]/total*100:.1f}% of draws)")
    return best_oe


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HIGH / LOW SPLIT  (1-35 = low, 36-70 = high)
# ─────────────────────────────────────────────────────────────────────────────

def high_low_analysis(df: pd.DataFrame):
    section("HIGH / LOW SPLIT  (1-35 low, 36-70 high)")
    counts = Counter()
    for _, row in df.iterrows():
        nums = row_nums(row)
        low  = sum(1 for n in nums if n <= 35)
        high = 5 - low
        counts[(low, high)] += 1
    total = sum(counts.values())
    print(f"  {'Low-High':>10}  {'Draws':>7}  {'%':>7}  Bar")
    print(f"  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*30}")
    top = counts.most_common(1)[0][1]
    for (l, h), cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {l}L-{h}H        {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")
    best_lh = counts.most_common(1)[0][0]
    print(f"\n  → Best ratio: {best_lh[0]} Low / {best_lh[1]} High")
    return best_lh


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SUM RANGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def sum_range_analysis(df: pd.DataFrame):
    section("SUM RANGE  (sum of 5 main numbers)")
    sums = [sum(row_nums(row)) for _, row in df.iterrows()]
    s_arr = pd.Series(sums)
    print(f"  Min: {s_arr.min()}  |  Max: {s_arr.max()}  |  Mean: {s_arr.mean():.1f}  |  Median: {s_arr.median():.1f}")

    hist  = s_arr.value_counts(bins=pd.interval_range(start=0, end=350, freq=25))
    total = len(sums)
    best_sum_range = None
    best_count     = 0
    print(f"\n  {'Range':>12}  {'Count':>7}  {'%':>7}  Bar")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*30}")
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
# 9.  CONSECUTIVE NUMBER PATTERN
# ─────────────────────────────────────────────────────────────────────────────

def consecutive_analysis(df: pd.DataFrame):
    section("CONSECUTIVE NUMBER PAIRS  (how often drawn together)")
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
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*30}")
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
# 10.  DELTA (GAP) SEQUENCE
# ─────────────────────────────────────────────────────────────────────────────

def delta_analysis(df: pd.DataFrame):
    section("DELTA / SPACING PATTERN  (gaps between sorted numbers)")
    delta_freq: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(row_nums(row))
        for i in range(1, len(nums)):
            delta_freq[nums[i] - nums[i - 1]] += 1

    total = sum(delta_freq.values())
    print(f"  {'Delta':>6}  {'Count':>7}  {'%':>7}  Bar")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*30}")
    top = delta_freq.most_common(1)[0][1]
    for d, cnt in sorted(delta_freq.items()):
        if cnt / total > 0.01:
            print(f"  {d:>6}  {cnt:>7}  {cnt/total*100:>6.1f}%  {bar(cnt, top)}")

    avg_delta = sum(d * cnt for d, cnt in delta_freq.items()) / total
    top5_deltas = [d for d, _ in delta_freq.most_common(5)]
    print(f"\n  → Average delta: {avg_delta:.2f}  |  Top-5 deltas: {sorted(top5_deltas)}")
    return avg_delta, top5_deltas


# ─────────────────────────────────────────────────────────────────────────────
# 11.  DECADE / RANGE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def decade_analysis(df: pd.DataFrame):
    section("DECADE DISTRIBUTION  (1-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-70)")
    decades = {
        "01-09": range(1, 10),
        "10-19": range(10, 20),
        "20-29": range(20, 30),
        "30-39": range(30, 40),
        "40-49": range(40, 50),
        "50-59": range(50, 60),
        "60-70": range(60, 71),
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
    print(f"  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*30}")
    top = max(dec_count.values())
    for label in decades:
        cnt   = dec_count[label]
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:>8}  {cnt:>7}  {sign}{delta:>6.1f}%  {bar(cnt, top)}")
    return dec_count, decades


# ─────────────────────────────────────────────────────────────────────────────
# 12.  POSITIONAL FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def positional_frequency(df: pd.DataFrame):
    section("POSITIONAL FREQUENCY  (hottest number at each position)")
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
# 13.  PAIR CO-OCCURRENCE  (top 20 most common pairs)
# ─────────────────────────────────────────────────────────────────────────────

def pair_cooccurrence(df: pd.DataFrame):
    section("TOP 20 MOST COMMON PAIRS  (appear together most often)")
    pair_counts: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(int(row[col]) for col in MAIN_COLS)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counts[(nums[i], nums[j])] += 1

    print(f"  {'Pair':>12}  {'Count':>6}  Bar")
    print(f"  {'─'*12}  {'─'*6}  {'─'*30}")
    top_count = pair_counts.most_common(1)[0][1]
    for pair, cnt in pair_counts.most_common(20):
        print(f"  {str(pair):>12}  {cnt:>6}  {bar(cnt, top_count)}")

    return pair_counts


# ─────────────────────────────────────────────────────────────────────────────
# 14.  MEGA BALL FREQUENCY
# ─────────────────────────────────────────────────────────────────────────────

def mega_ball_frequency(df: pd.DataFrame):
    section("MEGA BALL FREQUENCY  (all 25 balls ranked)")
    mb_draws = df[df["MegaBall"].notna()]
    c = Counter(int(v) for v in mb_draws["MegaBall"])
    total = len(mb_draws)
    expected = total / 25

    print(f"  Total draws with Mega Ball: {total:,}  |  Expected per ball: {expected:.1f}")
    print(f"\n  {'Ball':>5}  {'Count':>6}  {'vs avg':>8}  Bar")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*30}")
    top_count = c.most_common(1)[0][1]
    for ball, cnt in c.most_common():
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {ball:>5}  {cnt:>6}  {sign}{delta:>6.1f}%  {bar(cnt, top_count)}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 15.  DAY-OF-WEEK BIAS
# ─────────────────────────────────────────────────────────────────────────────

def dow_bias(df: pd.DataFrame):
    section("DAY-OF-WEEK BIAS  (which days produce higher numbers?)")
    dow_totals = defaultdict(list)
    for _, row in df.iterrows():
        nums = [int(row[col]) for col in MAIN_COLS if pd.notna(row[col])]
        dow_totals[row["DOW"]].extend(nums)

    order = ["Tuesday", "Friday", "Saturday", "Sunday", "Monday", "Wednesday", "Thursday"]
    print(f"  {'Day':>12}  {'Draws':>6}  {'Avg number':>11}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*11}")
    for day in order:
        if day in dow_totals:
            vals = dow_totals[day]
            draws = len(vals) // 5
            avg = sum(vals) / len(vals)
            print(f"  {day:>12}  {draws:>6}  {avg:>11.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 16.  COMPOSITE HOT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_hot_score(
    full_c: Counter,
    recent_c: Counter,
    gap: dict,
    pair_c: Counter,
    df: pd.DataFrame,
) -> list[tuple[int, float]]:
    """
    Score = 0.40*recent_rank + 0.25*full_rank + 0.20*pair_score + 0.15*overdue
    Higher = hotter.
    """
    section("COMPOSITE HOT-SCORE  (weighted ranking)")

    all_nums = list(range(1, 71))

    recent_order = {n: rank for rank, (n, _) in enumerate(recent_c.most_common())}
    full_order   = {n: rank for rank, (n, _) in enumerate(full_c.most_common())}

    pair_score: dict[int, int] = Counter()
    for (a, b), cnt in pair_c.items():
        pair_score[a] += cnt
        pair_score[b] += cnt
    max_pair = max(pair_score.values()) if pair_score else 1

    max_gap = max(gap.values()) if gap else 1

    scores: list[tuple[int, float]] = []
    for n in all_nums:
        r_rank = recent_order.get(n, len(recent_c))       # lower=hotter
        f_rank = full_order.get(n, len(full_c))
        p_sc   = pair_score.get(n, 0) / max_pair          # higher=better
        g      = gap.get(n, max_gap) / max_gap             # higher=more overdue (small bonus)

        score = (
            0.40 * (1 - r_rank / 70) +
            0.25 * (1 - f_rank / 70) +
            0.20 * p_sc +
            0.15 * g
        )
        scores.append((n, round(score, 4)))

    scores.sort(key=lambda x: -x[1])

    print(f"\n  {'#':>3}  {'Score':>7}  Bar")
    print(f"  {'─'*3}  {'─'*7}  {'─'*30}")
    top_score = scores[0][1]
    for num, sc in scores[:25]:
        print(f"  {num:>3}  {sc:>7.4f}  {bar(int(sc * 1000), int(top_score * 1000))}")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TICKET BUILDER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def fits_profile(nums: list[int], target_oe: tuple, target_lh: tuple,
                  optimal_sum: float) -> bool:
    odds  = sum(1 for n in nums if n % 2 != 0)
    low   = sum(1 for n in nums if n <= 35)
    s     = sum(nums)
    # allow ±1 on odd/even, ±1 on low/high, ±30 around optimal sum
    oe_ok  = abs(odds - target_oe[0]) <= 1
    lh_ok  = abs(low  - target_lh[0]) <= 1
    sum_ok = abs(s - optimal_sum) <= 30
    return oe_ok and lh_ok and sum_ok


def show_ticket(label: str, main: list[int], mb: int, note: str = ""):
    w = 56
    print(f"\n  ┌─ {label} {'─'*(w-len(label)-3)}┐")
    print(f"  │  Main  :  {' – '.join(f'{n:2d}' for n in sorted(main)):<44}│")
    print(f"  │  MegaBall:  {mb:<2}{'':>41}│")
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
    pick_mb:   int,
    full_c:    Counter,
    recent_c:  Counter,
    gap:       dict,
    pair_c:    Counter,
    mb_c:      Counter,
    scores:    list,
    df:        "pd.DataFrame",
):
    sorted_nums = sorted(pick_nums)
    section(f"DRAW AUDIT:  {' – '.join(f'{n:02d}' for n in sorted_nums)}  MB {pick_mb:02d}")

    total      = len(df)
    score_dict = dict(scores)
    score_rank = {n: r + 1 for r, (n, _) in enumerate(scores)}
    full_rank  = {n: r + 1 for r, (n, _) in enumerate(full_c.most_common())}

    # ── Per-number breakdown ──────────────────────────────────────────────────
    print(f"\n  ── PER-NUMBER BREAKDOWN ──────────────────────────────────────────────")
    print(f"  {'Num':>4}  {'All-time cnt':>13}  {'Recent-52':>9}  {'Draws ago':>10}  {'Hot score':>10}  {'Rank/70':>7}")
    print(f"  {'─'*4}  {'─'*13}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*7}")
    for n in sorted_nums:
        f_cnt = full_c.get(n, 0)
        r_cnt = recent_c.get(n, 0)
        g     = gap.get(n, total)
        sc    = score_dict.get(n, 0.0)
        srk   = score_rank.get(n, 70)
        pct   = f_cnt / total * 100 if total else 0
        tag   = (" [HOT]"     if srk <= 15 else
                 " [WARM]"    if srk <= 30 else
                 " [OVERDUE]" if g >= 25   else
                 " [COLD]"    if srk >= 55 else "")
        print(f"  {n:>4}  {f_cnt:>5} ({pct:4.1f}%)  {r_cnt:>9}  {g:>10}  {sc:>10.4f}  #{srk:>2}{tag}")

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
        ("60-70", range(60, 71)),
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
    print(f"  Low / High   : {lows}L (≤35) – {highs}H (36-70)")
    print(f"  Sum          : {s}  (historical peak ≈ 120–170)")
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

    # ── Mega Ball ─────────────────────────────────────────────────────────────
    mb_total  = sum(mb_c.values())
    mb_cnt    = mb_c.get(pick_mb, 0)
    mb_ranks  = {b: r + 1 for r, (b, _) in enumerate(mb_c.most_common())}
    mb_rank   = mb_ranks.get(pick_mb, 25)
    mb_pct    = mb_cnt / mb_total * 100 if mb_total else 0
    exp_mb    = mb_total / 25 if mb_total else 0
    delta_mb  = (mb_cnt - exp_mb) / exp_mb * 100 if exp_mb else 0
    sign_mb   = "+" if delta_mb >= 0 else ""
    mb_label  = ("ABOVE avg" if delta_mb > 5 else
                 "BELOW avg" if delta_mb < -5 else "NEAR avg")

    print(f"\n  ── MEGA BALL {pick_mb:02d} ────────────────────────────────────────────────")
    print(f"  Appeared     : {mb_cnt}×  ({mb_pct:.1f}% of draws)")
    print(f"  Expected     : {exp_mb:.1f}×")
    print(f"  vs Average   : {sign_mb}{delta_mb:.1f}%  [{mb_label}]")
    print(f"  Frequency Rank: #{mb_rank} out of 25 Mega Balls")

    # ── Overall assessment ────────────────────────────────────────────────────
    hot_count    = sum(1 for n in pick_nums if score_rank.get(n, 70) <= 20)
    overdue_cnt  = sum(1 for n in pick_nums if gap.get(n, 0) >= 20)
    avg_sc       = sum(score_dict.get(n, 0) for n in pick_nums) / len(pick_nums)
    rel = min(max((avg_sc - 0.30) / 0.30, 0), 1)
    filled = int(round(rel * 30))
    meter = "█" * filled + "░" * (30 - filled)
    level = "HIGH" if rel >= 0.65 else ("MEDIUM" if rel >= 0.35 else "LOW")

    print(f"\n  ── OVERALL STATISTICAL ASSESSMENT ─────────────────────────────────")
    print(f"  Hot numbers  (top-20 score) : {hot_count}/5")
    print(f"  Overdue numbers (≥20 draws) : {overdue_cnt}/5")
    print(f"  Avg composite score         : {avg_sc:.4f}")
    print(f"  Statistical alignment       : {level}  [{meter}]")
    print(f"\n  ⚑  Each Mega Millions draw is fully independent – statistics do NOT")
    print(f"     predict the next result.  Play responsibly.")


# ─────────────────────────────────────────────────────────────────────────────
# 17.  FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def final_recommendations(
    scores:      list[tuple[int, float]],
    mb_c:        Counter,
    pair_c:      Counter,
    recent_c:    Counter,
    gap:         dict,
    best_oe:     tuple,
    best_lh:     tuple,
    optimal_sum: float,
    pos_best:    list[list[int]],
):
    section("★  FINAL RECOMMENDED TICKETS  ★")

    top_mb = [b for b, _ in mb_c.most_common(7)]

    # ── Ticket 1 – pure HOT (top 5 composite score) ────────────────────────
    ticket1_main = sorted([n for n, _ in scores[:5]])
    show_ticket("Ticket 1 – PURE COMPOSITE HOT", ticket1_main, top_mb[0],
                "Top-5 by multi-factor composite score")

    # ── Ticket 2 – HOT + OVERDUE blend ──────────────────────────────────────
    top10 = [n for n, _ in scores[:10]]
    ticket2_main = sorted(sorted(top10, key=lambda n: -gap.get(n, 0))[:5])
    show_ticket("Ticket 2 – HOT + OVERDUE", ticket2_main, top_mb[1],
                "Top-10 score, sorted by longest absence")

    # ── Ticket 3 – BEST PAIR driven ─────────────────────────────────────────
    best_pair = pair_c.most_common(1)[0][0]
    ticket3_pool = list(best_pair)
    for n, _ in scores:
        if n not in ticket3_pool:
            ticket3_pool.append(n)
        if len(ticket3_pool) >= 5:
            break
    ticket3_main = sorted(ticket3_pool[:5])
    show_ticket("Ticket 3 – PAIR ANCHOR", ticket3_main, top_mb[2],
                f"Anchored on hottest pair {best_pair}")

    # ── Ticket 4 – PROFILE MATCHED (sum + odd/even + high/low) ─────────────
    candidates = [n for n, _ in scores[:25]]
    t4_best, t4_best_score = None, -1
    for combo in combinations(candidates, 5):
        if fits_profile(list(combo), best_oe, best_lh, optimal_sum):
            sc = sum(dict(scores).get(n, 0) for n in combo)
            if sc > t4_best_score:
                t4_best_score = sc
                t4_best = combo
    if t4_best is None:
        t4_best = tuple([n for n, _ in scores[:5]])
    ticket4_main = sorted(t4_best)
    show_ticket("Ticket 4 – PROFILE MATCHED", ticket4_main, top_mb[3],
                f"Best combo matching {best_oe[0]}O-{best_oe[1]}E, "
                f"{best_lh[0]}L-{best_lh[1]}H, sum≈{optimal_sum:.0f}")

    # ── Ticket 5 – POSITIONAL LEADERS ───────────────────────────────────────
    ticket5_main = []
    for pos_list in pos_best:
        for n in pos_list:
            if n not in ticket5_main:
                ticket5_main.append(n)
                break
    ticket5_main = sorted(ticket5_main[:5])
    show_ticket("Ticket 5 – POSITIONAL LEADERS", ticket5_main, top_mb[4],
                "Hottest number at each of the 5 sorted positions")

    # ── Ticket 6 – RECENT SURGE (last ~1 yr) ────────────────────────────────
    ticket6_main = sorted([n for n, _ in recent_c.most_common(5)])
    show_ticket("Ticket 6 – RECENT SURGE (last 52 draws)", ticket6_main, top_mb[5],
                "5 most frequent numbers in last ~1 year")

    # ── Ticket 7 – WILDCARD / OVERDUE ───────────────────────────────────────
    wildcard = max(gap, key=gap.get)
    ticket7_pool = [n for n, _ in scores[:4]] + [wildcard]
    ticket7_main = sorted(set(ticket7_pool))[:5]
    show_ticket("Ticket 7 – WILDCARD OVERDUE", ticket7_main, top_mb[6],
                f"Top-4 hot + #{wildcard} (most overdue ever: {gap[wildcard]} draws)")

    print(f"\n  ─── MEGA BALL SHORTLIST  (by frequency) ───")
    for ball, cnt in mb_c.most_common(7):
        pct = cnt / sum(mb_c.values()) * 100
        print(f"    Ball {ball:>2}  appeared {cnt}×  ({pct:.1f}%)")

    print("""
  ─────────────────────────────────────────────────
  DISCLAIMER
  Statistical frequency analysis does NOT predict
  future lottery draws.  Each draw is independent.
  Play responsibly. Good luck!
  ─────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          MEGA MILLIONS  DEEP STATISTICAL ANALYSIS        ║")
    print("║       12 Pattern Mechanisms → 7 Ticket Proposals         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        df = fetch_mega_millions()
    except Exception as exc:
        print(f"\nERROR fetching data: {exc}")
        sys.exit(1)

    if df.empty:
        print("No data retrieved. Exiting.")
        sys.exit(1)

    # Save raw data locally for reference
    df.to_excel("mega_millions_draws.xlsx", index=False)
    print(f"  → Raw data saved to mega_millions_draws.xlsx")

    full_c        = full_frequency(df)
    recent_c      = recent_frequency(df, n=52)
    gap           = overdue_numbers(df)
    best_oe       = odd_even_analysis(df)
    best_lh       = high_low_analysis(df)
    optimal_sum   = sum_range_analysis(df)
    consec_tgt, _ = consecutive_analysis(df)
    delta_avg, _  = delta_analysis(df)
    decade_analysis(df)
    pos_best      = positional_frequency(df)
    pair_c        = pair_cooccurrence(df)
    mb_c          = mega_ball_frequency(df)
    dow_bias(df)
    scores        = composite_hot_score(full_c, recent_c, gap, pair_c, df)

    final_recommendations(
        scores, mb_c, pair_c, recent_c, gap,
        best_oe, best_lh, optimal_sum, pos_best
    )

    # ── Specific draw audit ──────────────────────────────────────────────────
    analyze_specific_ticket(
        pick_nums=[n for n, _ in scores[:5]],
        pick_mb=mb_c.most_common(1)[0][0],
        full_c=full_c,
        recent_c=recent_c,
        gap=gap,
        pair_c=pair_c,
        mb_c=mb_c,
        scores=scores,
        df=df,
    )


if __name__ == "__main__":
    main()
