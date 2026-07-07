#!/usr/bin/env python3
"""
Deep Mega Millions statistical analysis.
Fetches all available draws from NY Open Data, then runs:
  - Full-history frequency ranking
  - Recent-trend (last 52 draws / ~1 year) frequency
  - Overdue / cold number detection
  - Pair & trio co-occurrence heatmap
  - Day-of-week & month-of-year bias
  - Composite HOT score and final recommendations
"""

import sys
import io
from collections import Counter, defaultdict
from datetime import datetime, timedelta

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

def all_main_numbers(df: pd.DataFrame):
    return [int(v) for col in MAIN_COLS for v in df[col].dropna()]


def freq_table(numbers: list[int], label: str) -> Counter:
    c = Counter(numbers)
    return c


def section(title: str, width: int = 62):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def bar(count: int, total: int, width: int = 30) -> str:
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
# 6.  PAIR CO-OCCURRENCE  (top 20 most common pairs)
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
# 7.  MEGA BALL FREQUENCY
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
# 8.  DAY-OF-WEEK BIAS
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
# 9.  COMPOSITE HOT SCORE & FINAL PICKS
# ─────────────────────────────────────────────────────────────────────────────

def composite_hot_score(
    full_c: Counter,
    recent_c: Counter,
    gap: dict,
    pair_c: Counter,
    df: pd.DataFrame,
) -> list[tuple[int, float]]:
    """
    Score = 0.35*recent_rank + 0.25*full_rank + 0.20*pair_score + 0.20*overdue_penalty
    Higher = hotter.
    """
    section("COMPOSITE HOT-SCORE  (weighted ranking)")

    # Normalise all signals to 0-1
    all_nums = list(range(1, 71))
    total    = len(df)

    # recent rank normalised
    recent_order = {n: rank for rank, (n, _) in enumerate(recent_c.most_common())}
    full_order   = {n: rank for rank, (n, _) in enumerate(full_c.most_common())}

    # pair score: sum of counts for all pairs that include this number
    pair_score: dict[int, int] = Counter()
    for (a, b), cnt in pair_c.items():
        pair_score[a] += cnt
        pair_score[b] += cnt
    max_pair = max(pair_score.values()) if pair_score else 1

    # overdue penalty (very overdue = penalty)
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
# 10.  FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def final_recommendations(
    scores: list[tuple[int, float]],
    mb_c: Counter,
    pair_c: Counter,
    recent_c: Counter,
    gap: dict,
):
    section("★  FINAL RECOMMENDED TICKETS  ★")

    top_pool = [n for n, _ in scores[:15]]   # top-15 composite pool

    # Ticket 1 – pure HOT (top 5 composite score)
    ticket1_main = sorted([n for n, _ in scores[:5]])
    ticket1_mb   = mb_c.most_common(1)[0][0]

    # Ticket 2 – HOT + OVERDUE blend
    # Take top-10 by score, then sort by overdue (draw gap) within that set
    top10 = [n for n, _ in scores[:10]]
    overdue_in_top10 = sorted(top10, key=lambda n: -gap.get(n, 0))
    ticket2_main = sorted(overdue_in_top10[:5])
    ticket2_mb   = mb_c.most_common(3)[1][0]   # 2nd hottest mega ball

    # Ticket 3 – BEST PAIR driven (anchor the most common pair, fill from scores)
    best_pair = pair_c.most_common(1)[0][0]
    ticket3_pool = list(best_pair)
    for n, _ in scores:
        if n not in ticket3_pool:
            ticket3_pool.append(n)
        if len(ticket3_pool) >= 5:
            break
    ticket3_main = sorted(ticket3_pool[:5])
    ticket3_mb   = mb_c.most_common(3)[2][0]   # 3rd hottest mega ball

    # Ticket 4 – RECENT SURGE  (hottest in last 52 draws)
    recent_top = [n for n, _ in recent_c.most_common(5)]
    ticket4_main = sorted(recent_top)
    ticket4_mb   = mb_c.most_common(2)[0][0]

    # Ticket 5 – BALANCED  (mix hot + one cold wildcard)
    wildcard = max(gap, key=gap.get)  # most overdue number
    ticket5_pool = [n for n, _ in scores[:4]] + [wildcard]
    ticket5_main = sorted(set(ticket5_pool))[:5]
    ticket5_mb   = mb_c.most_common(5)[4][0]

    def show(label, main, mb, note=""):
        print(f"\n  ┌─ {label} {'─'*(54-len(label))}┐")
        print(f"  │  Main  :  {' – '.join(f'{n:2d}' for n in main):<40}  │")
        print(f"  │  MegaBall: {mb:<2}                                      │")
        if note:
            print(f"  │  Note  : {note:<43}  │")
        print(f"  └{'─'*56}┘")

    show("Ticket 1 – PURE HOT",         ticket1_main, ticket1_mb,
         "Top-5 composite score")
    show("Ticket 2 – HOT + OVERDUE",    ticket2_main, ticket2_mb,
         "Top-10 score sorted by gap")
    show("Ticket 3 – PAIR ANCHOR",      ticket3_main, ticket3_mb,
         f"Anchored on most common pair {best_pair}")
    show("Ticket 4 – RECENT SURGE",     ticket4_main, ticket4_mb,
         "Hottest numbers in last 52 draws")
    show("Ticket 5 – BALANCED WILDCARD",ticket5_main, ticket5_mb,
         f"Top-4 hot + wildcard #{wildcard} (most overdue)")

    print(f"\n  ─── MEGA BALL SHORTLIST  (by frequency) ───")
    for ball, cnt in mb_c.most_common(5):
        print(f"    Ball {ball:>2}  appeared {cnt}×")

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

    full_c   = full_frequency(df)
    recent_c = recent_frequency(df, n=52)
    gap      = overdue_numbers(df)
    pair_c   = pair_cooccurrence(df)
    mb_c     = mega_ball_frequency(df)
    dow_bias(df)
    scores   = composite_hot_score(full_c, recent_c, gap, pair_c, df)

    final_recommendations(scores, mb_c, pair_c, recent_c, gap)


if __name__ == "__main__":
    main()
