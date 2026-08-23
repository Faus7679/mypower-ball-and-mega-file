#!/usr/bin/env python3
"""
Propose N Powerball tickets for the next draw (N is configurable, default 3).

Rules applied:
  - LOW TO HIGH : each ticket draws one number from each of 5 equal bands
                   spanning 1-69, so every ticket spreads across the full range.
  - COLD TO HOT : across the N tickets, picks progress from cold-leaning (most
                   overdue numbers) -> balanced -> hot-leaning (most frequent
                   recently), so collectively the set covers the cold->hot
                   spectrum.
  - AVOID STATIC: numbers are never reused across tickets (each main number and
                   each Powerball is unique) as long as N does not exceed the
                   smallest band's size (13); beyond that, repeats are allowed
                   with a warning.
"""
import argparse
import sys, io
from collections import Counter
import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PB_API = ("https://data.ny.gov/resource/d6yy-54nr.json"
          "?$order=draw_date%20DESC&$limit=5000")
EXCEL_FILE = "powerball_game.xlsx"
MAIN_COLS  = ["Num1", "Num2", "Num3", "Num4", "Num5"]
RECENT_N   = 104  # ~2 years


def fetch_api():
    r = requests.get(PB_API, timeout=25)
    r.raise_for_status()
    rows = []
    for item in r.json():
        raw_date = item.get("draw_date", "")[:10]
        parts = item.get("winning_numbers", "").split()
        if len(parts) < 6:
            continue
        try:
            nums = [int(p) for p in parts[:5]]
            pb = int(parts[5])
            dt = pd.to_datetime(raw_date)
        except (ValueError, TypeError):
            continue
        rows.append({"draw_date": dt, "Num1": nums[0], "Num2": nums[1],
                      "Num3": nums[2], "Num4": nums[3], "Num5": nums[4],
                      "Powerball": pb})
    return pd.DataFrame(rows)


def load_excel():
    try:
        raw = pd.read_excel(EXCEL_FILE)
        rows = []
        for _, r in raw.iterrows():
            try:
                dt = pd.to_datetime(f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['Day']):02d}")
            except Exception:
                continue
            rows.append({"draw_date": dt, "Num1": int(r["Num1"]), "Num2": int(r["Num2"]),
                         "Num3": int(r["Num3"]), "Num4": int(r["Num4"]), "Num5": int(r["Num5"]),
                         "Powerball": int(r["powerball"])})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def load_data():
    api_df, excel_df = fetch_api(), load_excel()
    df = pd.concat([api_df, excel_df], ignore_index=True)
    df = df.drop_duplicates(subset=["draw_date", "Num1", "Num2", "Num3", "Num4", "Num5"])
    return df.sort_values("draw_date").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Propose N Powerball tickets for the next draw.")
    parser.add_argument("-n", "--tickets", type=int, default=3,
                         help="number of tickets to generate (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    n_tickets = max(1, args.tickets)

    df = load_data()
    total = len(df)
    dmax = df["draw_date"].max().date()
    print(f"Loaded {total:,} unique Powerball draws through {dmax}\n")

    full_c = Counter(int(v) for c in MAIN_COLS for v in df[c].dropna())
    rec_c  = Counter(int(v) for c in MAIN_COLS for v in df.tail(min(RECENT_N, total))[c].dropna())
    pb_c   = Counter(int(v) for v in df["Powerball"].dropna())

    # gap: draws since each number's last appearance (cold = big gap)
    last = {}
    for idx, row in df.iterrows():
        for n in [int(row[c]) for c in MAIN_COLS if pd.notna(row[c])]:
            last[n] = int(idx)
    gap = {n: total - 1 - last.get(n, -1) for n in range(1, 70)}

    last_pb = {}
    for idx, row in df.iterrows():
        if pd.notna(row["Powerball"]):
            last_pb[int(row["Powerball"])] = int(idx)
    pb_gap = {n: total - 1 - last_pb.get(n, -1) for n in range(1, 27)}

    # hotness score for main numbers: recent-frequency weighted
    hot_rank = sorted(range(1, 70), key=lambda n: (-rec_c.get(n, 0), -full_c.get(n, 0)))
    cold_rank = sorted(range(1, 70), key=lambda n: -gap.get(n, 0))

    hot_score = {n: r for r, n in enumerate(hot_rank)}   # 0 = hottest
    cold_score = {n: r for r, n in enumerate(cold_rank)}  # 0 = coldest/most overdue

    # 5 equal-ish bands across 1-69 for low->high spread
    bands = [(1, 14), (15, 27), (28, 41), (42, 55), (56, 69)]

    used = set()
    band_exhausted = False

    def pick_from_band(lo, hi, w):
        """w in [0,1]: 0 = coldest pick, 1 = hottest pick, in between = blended rank."""
        nonlocal band_exhausted
        candidates = [n for n in range(lo, hi + 1) if n not in used]
        if not candidates:
            band_exhausted = True
            candidates = [n for n in range(lo, hi + 1)]  # fallback if band exhausted
        candidates.sort(key=lambda n: (1 - w) * cold_score[n] + w * hot_score[n])
        pick = candidates[0]
        used.add(pick)
        return pick

    used_pb = set()
    pb_exhausted = False
    pb_cold_rank = {b: r for r, (b, _) in enumerate(sorted(pb_gap.items(), key=lambda x: -x[1]))}
    pb_hot_rank = {b: r for r, (b, _) in enumerate(pb_c.most_common())}

    def pick_pb(w):
        nonlocal pb_exhausted
        pool = [n for n in range(1, 27) if n not in used_pb]
        if not pool:
            pb_exhausted = True
            pool = list(range(1, 27))
        pool.sort(key=lambda n: (1 - w) * pb_cold_rank.get(n, 25) + w * pb_hot_rank.get(n, 25))
        pick = pool[0]
        used_pb.add(pick)
        return pick

    def ticket_position(t):
        """0 = cold-leaning ticket, 1 = hot-leaning ticket, spread evenly across n_tickets."""
        return (t - 1) / (n_tickets - 1) if n_tickets > 1 else 0.5

    def ticket_label(pos):
        if pos < 0.34:
            return "COLD-LEANING"
        if pos > 0.66:
            return "HOT-LEANING"
        return "BALANCED (COLD -> HOT)"

    print("=" * 64)
    print(f"  {n_tickets} PROPOSED POWERBALL TICKET{'S' if n_tickets != 1 else ''} - NEXT DRAW")
    print("  (Low->High spread | Cold->Hot progression | No repeated numbers)")
    print("=" * 64)

    for t in range(1, n_tickets + 1):
        pos = ticket_position(t)
        # blend the ticket's overall cold->hot lean with each band's own position
        # (low bands lean colder first, high bands lean hotter first)
        weights = [min(1.0, max(0.0, (pos + b_pos) / 2 + (pos - 0.5) * 0.5))
                   for b_pos in (0.0, 0.25, 0.5, 0.75, 1.0)]
        nums = [pick_from_band(lo, hi, w) for (lo, hi), w in zip(bands, weights)]
        nums.sort()
        pb = pick_pb(pos)
        print(f"\n  Ticket {t} - {ticket_label(pos)}")
        print(f"    Numbers   : {' - '.join(f'{n:02d}' for n in nums)}")
        print(f"    Powerball : {pb:02d}")
        detail = []
        for n in nums:
            tag = "HOT" if hot_score[n] < 15 else ("COLD" if cold_score[n] < 15 else "mid")
            detail.append(f"{n:02d}({tag},gap={gap[n]})")
        print(f"    Detail    : {', '.join(detail)}")

    if band_exhausted or pb_exhausted:
        print(f"\n  NOTE: {n_tickets} tickets exceeded the pool of unique numbers in at")
        print("  least one band or the Powerball range, so some repeats were unavoidable.")
    else:
        print(f"\n  All {n_tickets * 5} main numbers and all {n_tickets} Powerballs above are unique -")
        print("  no number is repeated across the tickets.")
    print("\nDISCLAIMER: Past frequency does not predict future draws (each draw is")
    print("independent and random). For entertainment purposes only - play responsibly.")


if __name__ == "__main__":
    main()
