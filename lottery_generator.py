#!/usr/bin/env python3
"""
Lottery Number Generator
Random and Smart Pick modes for Powerball, Double Play, and Mega Millions.
Smart Pick hot scores are derived from powerball_doubleplay_analysis.py
using 1,959 historical draws (Feb 2010 – Jun 2026).
"""

import sys
import io
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Hot score data from 1,959-draw analysis (June 2026) ──────────────────────
# Composite score per main number (1-69): recent freq 40% · all-time 25%
# · pair co-occurrence 20% · overdue bonus 15%.  Higher = hotter.
_PB_SCORES = {
     1: 0.31608,  2: 0.46072,  3: 0.75207,  4: 0.54549,  5: 0.56394,
     6: 0.67769,  7: 0.65852,  8: 0.54485,  9: 0.34761, 10: 0.52442,
    11: 0.70204, 12: 0.48637, 13: 0.36259, 14: 0.52663, 15: 0.26295,
    16: 0.60830, 17: 0.55168, 18: 0.76206, 19: 0.64245, 20: 0.50106,
    21: 0.74020, 22: 0.40775, 23: 0.57273, 24: 0.60504, 25: 0.33469,
    26: 0.39965, 27: 0.71056, 28: 0.87459, 29: 0.44158, 30: 0.61167,
    31: 0.67517, 32: 0.69571, 33: 0.59208, 34: 0.32331, 35: 0.35893,
    36: 0.81619, 37: 0.58248, 38: 0.42987, 39: 0.63618, 40: 0.56364,
    41: 0.47196, 42: 0.52049, 43: 0.60002, 44: 0.49918, 45: 0.44109,
    46: 0.26622, 47: 0.77404, 48: 0.43409, 49: 0.41661, 50: 0.33961,
    51: 0.63828, 52: 0.82687, 53: 0.59150, 54: 0.51944, 55: 0.36655,
    56: 0.66850, 57: 0.59798, 58: 0.63898, 59: 0.67274, 60: 0.47602,
    61: 0.24989, 62: 0.30285, 63: 0.52066, 64: 0.56167, 65: 0.39849,
    66: 0.35581, 67: 0.18459, 68: 0.25911, 69: 0.28681,
}

# Red ball (1-26) frequency counts from 1,959 draws
_PB_RED = {
     1: 75,  2: 68,  3: 67,  4: 77,  5: 76,  6: 67,  7: 64,  8: 62,
     9: 66, 10: 63, 11: 65, 12: 64, 13: 66, 14: 79, 15: 64, 16: 57,
    17: 61, 18: 78, 19: 67, 20: 77, 21: 71, 22: 59, 23: 67,
    24: 82, 25: 76, 26: 68,
}

# Pre-ranked reference lists
_HOT_MAIN = sorted(_PB_SCORES, key=_PB_SCORES.get, reverse=True)[:15]
_HOT_RED  = sorted(_PB_RED,    key=_PB_RED.get,    reverse=True)[:5]
_TOP5     = sorted(_PB_SCORES, key=_PB_SCORES.get, reverse=True)[:5]


# ── Weighted sampling (no replacement) ───────────────────────────────────────

def _weighted_sample(population: list, weights: list, k: int) -> list:
    pool   = list(zip(population, weights))
    result = []
    while len(result) < k:
        total   = sum(w for _, w in pool)
        r       = random.uniform(0, total)
        cumsum  = 0.0
        for i, (item, w) in enumerate(pool):
            cumsum += w
            if r <= cumsum:
                result.append(item)
                pool.pop(i)
                break
    return result


# ── Number generators ─────────────────────────────────────────────────────────

def generate_powerball():
    """5 unique random numbers 1-69  +  1 red ball 1-26."""
    return sorted(random.sample(range(1, 70), 5)), random.randint(1, 26)


def generate_double_play():
    """Independent draw: same pool as Powerball."""
    return sorted(random.sample(range(1, 70), 5)), random.randint(1, 26)


def generate_mega_millions():
    """5 unique random numbers 1-70  +  1 Mega Ball 1-25."""
    return sorted(random.sample(range(1, 71), 5)), random.randint(1, 25)


def generate_smart_pick():
    """
    Weighted pick: hot numbers are more likely but all 69 are in the pool.
    Red ball weighted by historical frequency.
    Returns (main_numbers, powerball).
    """
    nums    = list(range(1, 70))
    weights = [_PB_SCORES[n] for n in nums]
    main    = sorted(_weighted_sample(nums, weights, 5))

    red_nums    = list(range(1, 27))
    red_weights = [_PB_RED[n] for n in red_nums]
    pb          = _weighted_sample(red_nums, red_weights, 1)[0]

    return main, pb


def generate_top_confidence():
    """Deterministic top-5 composite score numbers + hottest red ball."""
    return sorted(_TOP5), _HOT_RED[0]


# ── Display helpers ───────────────────────────────────────────────────────────

def display_powerball(main, pb):
    print("\n" + "=" * 52)
    print("           POWERBALL NUMBERS")
    print("=" * 52)
    print(f"  Main Numbers :  {' - '.join(map(str, main))}")
    print(f"  Powerball    :  {pb}")
    print("=" * 52)


def display_double_play(main, pb):
    print("\n" + "-" * 52)
    print("        DOUBLE PLAY WINNING NUMBERS")
    print("-" * 52)
    print(f"  Main Numbers :  {' - '.join(map(str, main))}")
    print(f"  Powerball    :  {pb}")
    print("-" * 52)


def display_mega_millions(main, mega_ball):
    print("\n" + "=" * 52)
    print("         MEGA MILLIONS NUMBERS")
    print("=" * 52)
    print(f"  Main Numbers :  {' - '.join(map(str, main))}")
    print(f"  Mega Ball    :  {mega_ball}")
    print("=" * 52)


def display_smart_pick(game: str, main: list, pb: int, top_main: list, top_pb: int):
    """Show the weighted pick, the top-confidence ticket, and reference lists."""
    border = "★" * 52
    print(f"\n{border}")
    print(f"   ★  {game} SMART PICK  ★")
    print(f"   Based on 1,959-draw analysis  ·  June 2026")
    print(border)
    print(f"\n  Weighted Pick   :  {' – '.join(f'{n:2d}' for n in main)}"
          f"   |  Red: {pb}")
    print(f"  Top Confidence  :  {' – '.join(f'{n:2d}' for n in top_main)}"
          f"   |  Red: {top_pb}")
    print(f"\n  Hot main numbers  :  {', '.join(map(str, _HOT_MAIN))}")
    print(f"  Hottest red balls :  {', '.join(map(str, _HOT_RED))}")
    print(border)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║          LOTTERY NUMBER GENERATOR                ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Random picks  ·  Smart Picks (hot-weighted)    ║")
    print("╚══════════════════════════════════════════════════╝")

    while True:
        print("\n  ── Random ───────────────────────────────────")
        print("  1. Powerball           (random)")
        print("  2. Mega Millions       (random)")
        print("  3. Both Games          (random)")
        print("\n  ── Smart Pick (hot-number weighted) ─────────")
        print("  4. ★ Powerball Smart Pick")
        print("  5. ★ Double Play Smart Pick")
        print("  6. ★ Both Smart Picks")
        print("\n  7. Exit")

        choice = input("\n  Enter your choice (1-7): ").strip()

        # ── Random picks ──────────────────────────────────────────────────────
        if choice == "1":
            main_n, pb = generate_powerball()
            display_powerball(main_n, pb)
            dp_main, dp_pb = generate_double_play()
            display_double_play(dp_main, dp_pb)

        elif choice == "2":
            main_n, mb = generate_mega_millions()
            display_mega_millions(main_n, mb)

        elif choice == "3":
            main_n, pb = generate_powerball()
            display_powerball(main_n, pb)
            dp_main, dp_pb = generate_double_play()
            display_double_play(dp_main, dp_pb)
            main_n, mb = generate_mega_millions()
            display_mega_millions(main_n, mb)

        # ── Smart picks ───────────────────────────────────────────────────────
        elif choice == "4":
            sp_main, sp_pb       = generate_smart_pick()
            top_main, top_pb     = generate_top_confidence()
            display_smart_pick("POWERBALL", sp_main, sp_pb, top_main, top_pb)

        elif choice == "5":
            sp_main, sp_pb       = generate_smart_pick()
            top_main, top_pb     = generate_top_confidence()
            display_smart_pick("DOUBLE PLAY", sp_main, sp_pb, top_main, top_pb)

        elif choice == "6":
            print()
            sp_main, sp_pb   = generate_smart_pick()
            top_main, top_pb = generate_top_confidence()
            display_smart_pick("POWERBALL", sp_main, sp_pb, top_main, top_pb)

            sp_main2, sp_pb2 = generate_smart_pick()
            display_smart_pick("DOUBLE PLAY", sp_main2, sp_pb2, top_main, top_pb)

        elif choice == "7":
            print("\n  Thank you for using Lottery Number Generator!")
            print("  Good luck!\n")
            break

        else:
            print("\n  Invalid choice. Please enter 1 through 7.")


if __name__ == "__main__":
    main()
