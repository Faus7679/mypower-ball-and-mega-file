#!/usr/bin/env python3
"""
Powerball + Double Play Smart Ticket Generator – Next Draw Edition
Fetches live draw history, computes fresh composite scores for both games,
outputs 7 strategy tickets + 3 smart weighted random picks per game.
"""

import sys
import io
import random
from collections import Counter
from itertools import combinations

import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PB_API     = (
    "https://data.ny.gov/resource/d6yy-54nr.json"
    "?$order=draw_date%20DESC&$limit=5000"
)
EXCEL_FILE = "powerball_game.xlsx"
MAIN_COLS  = ["Num1", "Num2", "Num3", "Num4", "Num5"]
DP_COLS    = ["DP1",  "DP2",  "DP3",  "DP4",  "DP5"]
RECENT_N   = 104   # ~2 years of Powerball draws
DP_MIN     = 50    # minimum DP draws needed for independent analysis


# ── Data loading ──────────────────────────────────────────────────────────────

def fetch_api() -> tuple:
    r = requests.get(PB_API, timeout=25)
    r.raise_for_status()
    pb_rows, dp_rows = [], []

    for item in r.json():
        raw_date = item.get("draw_date", "")[:10]

        # Powerball numbers
        parts = item.get("winning_numbers", "").split()
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

        # Double Play numbers (field exposed from Aug 2021 onward)
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

    def to_df(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Year", "Month", "Day"])
        return df.sort_values("draw_date").reset_index(drop=True)

    return to_df(pb_rows), to_df(dp_rows)


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
            rows.append({
                "draw_date": dt, "Year": int(r["Year"]),
                "Month": int(r["Month"]), "Day": int(r["Day"]),
                "DOW": dt.strftime("%A"),
                "Num1": int(r["Num1"]), "Num2": int(r["Num2"]),
                "Num3": int(r["Num3"]), "Num4": int(r["Num4"]),
                "Num5": int(r["Num5"]), "Powerball": int(r["powerball"]),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def load_data() -> tuple:
    pb_api, dp_df = fetch_api()
    pb_local      = load_excel_pb()

    pb_df = pd.concat([pb_api, pb_local], ignore_index=True)
    pb_df = (pb_df.drop_duplicates(subset=["Year", "Month", "Day"])
                  .sort_values("draw_date")
                  .reset_index(drop=True))
    return pb_df, dp_df


# ── Score computation ─────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, cols: list) -> tuple:
    total  = len(df)
    full_c = Counter(int(v) for c in cols for v in df[c].dropna())
    rec_c  = Counter(int(v) for c in cols
                     for v in df.tail(min(RECENT_N, total))[c].dropna())

    last: dict = {}
    for idx, row in df.iterrows():
        for n in [int(row[c]) for c in cols if pd.notna(row[c])]:
            last[n] = int(idx)
    gap = {n: total - 1 - last.get(n, -1) for n in range(1, 70)}

    pair_c: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(int(row[c]) for c in cols if pd.notna(row[c]))
        for a, b in combinations(nums, 2):
            pair_c[(a, b)] += 1

    pair_s: Counter = Counter()
    for (a, b), cnt in pair_c.items():
        pair_s[a] += cnt
        pair_s[b] += cnt
    max_pair = max(pair_s.values()) or 1
    max_gap  = max(gap.values()) or 1

    full_rank = {n: r for r, (n, _) in enumerate(full_c.most_common())}
    rec_rank  = {n: r for r, (n, _) in enumerate(rec_c.most_common())}
    n_ranks   = max(len(full_c), len(rec_c), 1)

    scores: dict = {}
    for n in range(1, 70):
        f_r = full_rank.get(n, n_ranks) / n_ranks
        r_r = rec_rank.get(n, n_ranks)  / n_ranks
        p_s = pair_s.get(n, 0) / max_pair
        g_s = gap.get(n, max_gap) / max_gap
        scores[n] = round(0.35*(1-r_r) + 0.25*(1-f_r) + 0.25*p_s + 0.15*g_s, 5)

    pb_c = Counter(int(v) for v in df["Powerball"].dropna())

    oe_cnt: Counter = Counter()
    lh_cnt: Counter = Counter()
    sums = []
    for _, row in df.iterrows():
        nums = [int(row[c]) for c in cols if pd.notna(row[c])]
        if not nums:
            continue
        odds = sum(1 for n in nums if n % 2 != 0)
        low  = sum(1 for n in nums if n <= 35)
        oe_cnt[(odds, 5 - odds)] += 1
        lh_cnt[(low,  5 - low)]  += 1
        sums.append(sum(nums))

    best_oe = oe_cnt.most_common(1)[0][0]
    best_lh = lh_cnt.most_common(1)[0][0]

    s_ser = pd.Series(sums)
    hist  = s_ser.value_counts(bins=pd.interval_range(start=0, end=350, freq=25))
    peak  = hist.idxmax()
    opt_sum = (peak.left + peak.right) / 2

    return scores, gap, pair_c, rec_c, pb_c, best_oe, best_lh, opt_sum


# ── Utilities ─────────────────────────────────────────────────────────────────

def weighted_sample(population: list, weights: list, k: int) -> list:
    pool   = list(zip(population, weights))
    result = []
    while len(result) < k:
        total = sum(w for _, w in pool)
        r     = random.uniform(0, total)
        cum   = 0.0
        for i, (item, w) in enumerate(pool):
            cum += w
            if r <= cum:
                result.append(item)
                pool.pop(i)
                break
    return result


def profile(nums: list) -> str:
    o = sum(1 for n in nums if n % 2 != 0)
    l = sum(1 for n in nums if n <= 35)
    return f"{o}O-{5-o}E  |  {l}L-{5-l}H  |  Sum={sum(nums)}"


def fits(nums: list, oe: tuple, lh: tuple, opt_sum: float) -> bool:
    o = sum(1 for n in nums if n % 2 != 0)
    l = sum(1 for n in nums if n <= 35)
    return (abs(o - oe[0]) <= 1 and abs(l - lh[0]) <= 1
            and abs(sum(nums) - opt_sum) <= 30)


def show(label: str, main: list, pb: int, strategy: str = ""):
    w = 58
    print(f"\n  ┌─ {label} {'─'*(w - len(label) - 3)}┐")
    print(f"  │  Numbers  :  {' – '.join(f'{n:2d}' for n in sorted(main)):<45}│")
    print(f"  │  Powerball:  {pb:<2}{'':>43}│")
    print(f"  │  Profile  :  {profile(sorted(main)):<45}│")
    if strategy:
        print(f"  │  Strategy :  {strategy:<45}│")
    print(f"  └{'─'*w}┘")


def divider(title: str):
    line = "═" * 64
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


# ── Ticket builder ────────────────────────────────────────────────────────────

def build_tickets(scores: dict, gap: dict, pair_c: Counter, rec_c: Counter,
                  pb_c: Counter, best_oe: tuple, best_lh: tuple,
                  opt_sum: float, game: str, proxy: bool = False):

    ranked    = sorted(scores.items(), key=lambda x: -x[1])
    top5      = [n for n, _ in ranked[:5]]
    pb_ranked = [b for b, _ in pb_c.most_common()]
    hot15     = [n for n, _ in ranked[:15]]

    proxy_note = "  [Using Powerball patterns as proxy — same number pool]" if proxy else ""
    if proxy_note:
        print(proxy_note)

    print(f"\n  Composite top-15 : {', '.join(map(str, hot15))}")
    print(f"  Best draw profile: {best_oe[0]}O-{best_oe[1]}E  |  "
          f"{best_lh[0]}L-{best_lh[1]}H  |  sum ≈ {opt_sum:.0f}")
    print(f"  Hottest PB balls : {', '.join(str(b) for b, _ in pb_c.most_common(7))}")

    # Ticket 1: Pure Composite Hot
    t1 = sorted(top5)
    show(f"{game} Ticket 1 – COMPOSITE HOT", t1, pb_ranked[0],
         "Top-5 multi-factor composite score")

    # Ticket 2: Hot + Overdue
    top10 = [n for n, _ in ranked[:10]]
    t2    = sorted(sorted(top10, key=lambda n: -gap.get(n, 0))[:5])
    show(f"{game} Ticket 2 – HOT + OVERDUE", t2, pb_ranked[1],
         "Top-10 composite re-sorted by longest absence")

    # Ticket 3: Profile-Matched
    cands   = [n for n, _ in ranked[:25]]
    t3_best = None
    t3_sc   = -1.0
    for combo in combinations(cands, 5):
        if fits(list(combo), best_oe, best_lh, opt_sum):
            sc = sum(scores.get(n, 0) for n in combo)
            if sc > t3_sc:
                t3_sc, t3_best = sc, combo
    if t3_best is None:
        t3_best = tuple(top5)
    show(f"{game} Ticket 3 – PROFILE MATCHED", sorted(t3_best), pb_ranked[2],
         f"{best_oe[0]}O-{best_oe[1]}E | {best_lh[0]}L-{best_lh[1]}H | sum≈{opt_sum:.0f}")

    # Ticket 4: Recent Surge
    t4 = sorted([n for n, _ in rec_c.most_common(5)])
    show(f"{game} Ticket 4 – RECENT SURGE", t4, pb_ranked[0],
         "5 most frequent numbers in the last ~2 years")

    # Ticket 5: Pair Anchor
    best_pair = pair_c.most_common(1)[0][0]
    t5_pool   = list(best_pair)
    for n, _ in ranked:
        if n not in t5_pool:
            t5_pool.append(n)
        if len(t5_pool) >= 5:
            break
    show(f"{game} Ticket 5 – PAIR ANCHOR", sorted(t5_pool[:5]), pb_ranked[2],
         f"Anchored on hottest pair {best_pair}")

    # Ticket 6: Decade Balanced
    decade_map = [("1-9", 1, 9), ("10-29", 10, 29), ("30-49", 30, 49), ("50-69", 50, 69)]
    t6_pool: list = []
    for _, lo, hi in decade_map:
        for n, _ in ranked:
            if lo <= n <= hi and n not in t6_pool:
                t6_pool.append(n)
                break
    for n, _ in ranked:
        if n not in t6_pool:
            t6_pool.append(n)
        if len(t6_pool) >= 5:
            break
    show(f"{game} Ticket 6 – DECADE BALANCED", sorted(t6_pool[:5]), pb_ranked[3],
         "Best hot pick from each active decade range")

    # Ticket 7: Wildcard / Most Overdue
    wildcard = max(gap, key=gap.get)
    t7_pool  = [n for n, _ in ranked[:4]] + [wildcard]
    t7       = sorted(set(t7_pool))[:5]
    show(f"{game} Ticket 7 – WILDCARD OVERDUE", t7, pb_ranked[1],
         f"Top-4 hot + #{wildcard} (most overdue: {gap[wildcard]} draws ago)")

    # Smart Weighted Random Picks
    print(f"\n  ─── {game} SMART WEIGHTED RANDOM PICKS ───")
    nums_pool   = list(range(1, 70))
    num_weights = [scores[n] for n in nums_pool]
    red_pool    = list(range(1, 27))
    red_weights = [pb_c.get(n, 1) for n in red_pool]

    for i in range(3):
        sp_main = sorted(weighted_sample(nums_pool, num_weights, 5))
        sp_pb   = weighted_sample(red_pool, red_weights, 1)[0]
        show(f"{game} Smart Pick {i + 1}", sp_main, sp_pb,
             "Hot-weighted random — unique combination each run")

    # Red ball shortlist
    print(f"\n  ─── {game} POWERBALL RED BALL RANKING ───")
    total_pb = sum(pb_c.values())
    expected = total_pb / 26
    for ball, cnt in pb_c.most_common(10):
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"    PB {ball:>2}  ·  {cnt:>3}× drawn  ·  {sign}{delta:.1f}% vs avg")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   POWERBALL + DOUBLE PLAY SMART TICKETS – NEXT DRAW EDITION  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n  Loading live draw history … ", end="", flush=True)

    try:
        pb_df, dp_df = load_data()
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    pb_total   = len(pb_df)
    dp_total   = len(dp_df)
    pb_min     = pb_df["draw_date"].min().date()
    pb_max     = pb_df["draw_date"].max().date()

    print(f"done.")
    print(f"  Powerball   : {pb_total:,} draws  ({pb_min} – {pb_max})")

    if dp_total >= DP_MIN:
        dp_min = dp_df["draw_date"].min().date()
        dp_max = dp_df["draw_date"].max().date()
        print(f"  Double Play : {dp_total:,} draws  ({dp_min} – {dp_max})")
    else:
        print(f"  Double Play : {dp_total} draws in API — using Powerball as proxy")

    # ── Powerball scores ──────────────────────────────────────────────────────
    pb_scores, pb_gap, pb_pair_c, pb_rec_c, pb_pb_c, pb_oe, pb_lh, pb_opt = \
        compute_scores(pb_df, MAIN_COLS)

    # ── Double Play scores ────────────────────────────────────────────────────
    dp_proxy = dp_total < DP_MIN
    if not dp_proxy:
        dp_scores, dp_gap, dp_pair_c, dp_rec_c, dp_pb_c, dp_oe, dp_lh, dp_opt = \
            compute_scores(dp_df, DP_COLS)
    else:
        dp_scores, dp_gap, dp_pair_c, dp_rec_c = pb_scores, pb_gap, pb_pair_c, pb_rec_c
        dp_pb_c, dp_oe, dp_lh, dp_opt = pb_pb_c, pb_oe, pb_lh, pb_opt

    # ── Powerball tickets ─────────────────────────────────────────────────────
    divider(f"POWERBALL TICKETS  ({pb_total:,} draws through {pb_max})")
    build_tickets(pb_scores, pb_gap, pb_pair_c, pb_rec_c,
                  pb_pb_c, pb_oe, pb_lh, pb_opt, "PB")

    # ── Double Play tickets ───────────────────────────────────────────────────
    dp_label = (
        f"DOUBLE PLAY TICKETS  ({dp_total:,} draws)"
        if not dp_proxy
        else f"DOUBLE PLAY TICKETS  (Powerball proxy — {pb_total:,} draws)"
    )
    divider(dp_label)
    build_tickets(dp_scores, dp_gap, dp_pair_c, dp_rec_c,
                  dp_pb_c, dp_oe, dp_lh, dp_opt, "DP", proxy=dp_proxy)

    # ── Cross-game consensus ──────────────────────────────────────────────────
    divider("CONSENSUS PICK  (top numbers confirmed across both games)")
    pb_top15 = {n for n, _ in sorted(pb_scores.items(), key=lambda x: -x[1])[:15]}
    dp_top15 = {n for n, _ in sorted(dp_scores.items(), key=lambda x: -x[1])[:15]}
    shared   = sorted(pb_top15 & dp_top15)
    consensus = shared[:5] if len(shared) >= 5 else \
                sorted(pb_scores, key=pb_scores.get, reverse=True)[:5]
    best_pb_ball = pb_pb_c.most_common(1)[0][0]
    best_dp_ball = dp_pb_c.most_common(1)[0][0]

    print(f"\n  Numbers hot in both games : {shared}")
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║         ★  CONSENSUS POWER TICKET  ★                ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  Main Numbers :  {' – '.join(f'{n:2d}' for n in consensus):<35}║")
    print(f"  ║  PB  red ball :  {best_pb_ball:<35}║")
    print(f"  ║  DP  red ball :  {best_dp_ball:<35}║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    print(f"""
  ══════════════════════════════════════════════════════════════
  Analysis: {pb_total:,} Powerball draws through {pb_max}
  Composite score weights:
    Recent frequency (last 2 yrs)  ·  35%
    All-time frequency             ·  25%
    Pair co-occurrence strength    ·  25%
    Overdue bonus                  ·  15%

  DISCLAIMER: Lottery draws are independent random events.
  Statistical analysis cannot predict future outcomes.
  Play responsibly and within your budget.
  ══════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
