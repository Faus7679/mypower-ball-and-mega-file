#!/usr/bin/env python3
"""
Propose exactly 3 Powerball tickets for the next draw.

Rules applied:
  - LOW TO HIGH : each ticket draws one number from each of 5 equal bands
                   spanning 1-69, so every ticket spreads across the full range.
  - COLD TO HOT : the 3 tickets progress from cold-leaning (most overdue
                   numbers) -> balanced -> hot-leaning (most frequent recently),
                   so collectively the set covers the cold->hot spectrum.
  - AVOID STATIC: numbers are never reused across the 3 tickets (each of the
                   15 main numbers and each of the 3 Powerballs is unique),
                   so no "safe" repeated static pick appears twice.
"""
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


def main():
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

    def pick_from_band(lo, hi, mode):
        candidates = [n for n in range(lo, hi + 1) if n not in used]
        if not candidates:
            candidates = [n for n in range(lo, hi + 1)]  # fallback if band exhausted
        if mode == "cold":
            candidates.sort(key=lambda n: cold_score[n])
        elif mode == "hot":
            candidates.sort(key=lambda n: hot_score[n])
        else:  # balanced: minimize combined rank
            candidates.sort(key=lambda n: cold_score[n] + hot_score[n])
        pick = candidates[0]
        used.add(pick)
        return pick

    used_pb = set()

    def pick_pb(mode):
        pool = [n for n in range(1, 27) if n not in used_pb]
        if mode == "cold":
            pool.sort(key=lambda n: -pb_gap.get(n, 0))
        elif mode == "hot":
            pool.sort(key=lambda n: -pb_c.get(n, 0))
        else:
            full_rank = {b: r for r, (b, _) in enumerate(sorted(pb_gap.items(), key=lambda x: -x[1]))}
            hot_rank_pb = {b: r for r, (b, _) in enumerate(pb_c.most_common())}
            pool.sort(key=lambda n: full_rank.get(n, 99) + hot_rank_pb.get(n, 99))
        pick = pool[0]
        used_pb.add(pick)
        return pick

    # Ticket modes per band, ticket-by-ticket: cold-leaning -> balanced -> hot-leaning
    band_modes = {
        1: ["cold", "cold", "balanced", "balanced", "hot"],   # Ticket 1: cold-leaning
        2: ["cold", "balanced", "balanced", "hot", "hot"],    # Ticket 2: balanced, tilts cold->hot
        3: ["balanced", "hot", "hot", "hot", "hot"],          # Ticket 3: hot-leaning
    }
    pb_modes = {1: "cold", 2: "balanced", 3: "hot"}
    labels = {1: "COLD-LEANING", 2: "BALANCED (COLD -> HOT)", 3: "HOT-LEANING"}

    print("=" * 64)
    print("  3 PROPOSED POWERBALL TICKETS - NEXT DRAW")
    print("  (Low->High spread | Cold->Hot progression | No repeated numbers)")
    print("=" * 64)

    for t in (1, 2, 3):
        modes = band_modes[t]
        nums = [pick_from_band(lo, hi, m) for (lo, hi), m in zip(bands, modes)]
        nums.sort()
        pb = pick_pb(pb_modes[t])
        print(f"\n  Ticket {t} - {labels[t]}")
        print(f"    Numbers   : {' - '.join(f'{n:02d}' for n in nums)}")
        print(f"    Powerball : {pb:02d}")
        detail = []
        for n in nums:
            tag = "HOT" if hot_score[n] < 15 else ("COLD" if cold_score[n] < 15 else "mid")
            detail.append(f"{n:02d}({tag},gap={gap[n]})")
        print(f"    Detail    : {', '.join(detail)}")

    print(f"\n  All 15 main numbers and all 3 Powerballs above are unique -")
    print(f"  no number is repeated across the three tickets.")
    print("\nDISCLAIMER: Past frequency does not predict future draws (each draw is")
    print("independent and random). For entertainment purposes only - play responsibly.")


if __name__ == "__main__":
    main()
