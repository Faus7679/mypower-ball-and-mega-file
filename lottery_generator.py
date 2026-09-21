#!/usr/bin/env python3
"""
Lottery Number Generator
Random and Smart Pick modes for Powerball, Double Play, and Mega Millions.
Smart Pick hot scores are derived from powerball_doubleplay_analysis.py
using 1,959 historical draws (Feb 2010 – Jun 2026).
ML Position Pick trains a model per ball position on drawn-order Powerball
history and generates tickets in drawn order (not sorted ascending).
"""

import sys
import io
import random

try:
    import numpy as np          # only the ML position pick needs it
except ImportError:
    np = None

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


# ── ML position model (drawn order, not sorted) ──────────────────────────────
# One logistic regression per ball position (1st..5th drawn) plus one for the
# red ball.  Rows are (draw t, candidate number n); the label is "n came out in
# this position at draw t".  Every feature uses draws < t only, so the model
# never sees the draw it is scored on.  Needs numpy/pandas/scikit-learn/openpyxl,
# imported lazily so the rest of the menu works without them.

_ORDER_FILE  = "powerball_game.xlsx"      # only file that keeps drawn order
_MAIN_COLS   = ["Num1", "Num2", "Num3", "Num4", "Num5"]
_RULES_START = "2015-10-07"               # current 69/26 matrix
_WARMUP      = 104                        # draws of history before a row is trainable
_HOLDOUT_N   = 100                        # newest draws held out for evaluation
_GAP_CAP     = 120
_ML_CACHE    = {}


def _load_drawn_order():
    """Draws in the order the balls came out, plus the last date that holds."""
    import pandas as pd
    from pathlib import Path

    raw = pd.read_excel(Path(__file__).with_name(_ORDER_FILE))
    raw["draw_date"] = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"], day=raw["Day"]))
    raw = raw.sort_values("draw_date", kind="stable").reset_index(drop=True)
    # Rows appended after the last unsorted draw were stored ascending, so their
    # positions carry no information; stop at the last row that is not ascending.
    ascending = (raw[_MAIN_COLS].diff(axis=1).iloc[:, 1:] > 0).all(axis=1)
    last_order = raw.loc[~ascending, "draw_date"].max()
    raw = raw[(raw["draw_date"] >= _RULES_START) & (raw["draw_date"] <= last_order)]
    return raw.reset_index(drop=True), last_order


def _one_hot(values, pool):
    v = values.reshape(len(values), -1)
    out = np.zeros((len(v), pool + 1))
    out[np.arange(len(v))[:, None], v] = 1.0
    return out


def _cum(presence):
    return np.vstack([np.zeros((1, presence.shape[1])), np.cumsum(presence, axis=0)])


def _last_seen(presence):
    """out[t, n] = index of n's last appearance before draw t (or -1); t runs 0..n_draws."""
    n_draws, width = presence.shape
    out  = np.full((n_draws + 1, width), -1, dtype=np.int64)
    seen = np.full(width, -1, dtype=np.int64)
    for t in range(n_draws):
        out[t] = seen
        seen[presence[t] > 0] = t
    out[n_draws] = seen
    return out


def _window_freq(cum, ts, w):
    return (cum[ts] - cum[np.maximum(ts - w, 0)]) / np.minimum(w, ts)[:, None]


def _gap_norm(last_seen, ts):
    ls  = last_seen[ts]
    gap = np.where(ls >= 0, ts[:, None] - ls, ts[:, None])
    return np.minimum(gap, _GAP_CAP) / _GAP_CAP


def _position_features(values, p):
    """Features for position p: (train X, y, row draw index, next-draw X)."""
    n_draws = len(values)
    pos, anyb = _one_hot(values[:, p], 69), _one_hot(values, 69)
    cpos, cany = _cum(pos), _cum(anyb)
    lpos, lany = _last_seen(pos), _last_seen(anyb)

    ts   = np.arange(_WARMUP, n_draws + 1)          # last entry = the unplayed next draw
    prev = values[ts - 1, p]
    feats = np.stack([
        cpos[ts] / ts[:, None],                     # all-time freq in this position
        _window_freq(cpos, ts, 52),                 # recent freq in this position
        _window_freq(cpos, ts, 104),
        _window_freq(cany, ts, 26),                 # recent freq in any position
        _gap_norm(lpos, ts),                        # draws since seen in this position
        _gap_norm(lany, ts),                        # draws since drawn at all
        np.abs(np.arange(70)[None, :] - prev[:, None]) / 69,   # distance from last draw's ball here
    ], axis=2)[:, 1:, :]                            # drop dummy number 0 -> (T, 69, k)

    X, y = feats[:-1].reshape(-1, feats.shape[2]), pos[ts[:-1]][:, 1:].reshape(-1)
    return X, y, np.repeat(ts[:-1], 69), feats[-1]


def _red_features(red):
    n_draws = len(red)
    pres = _one_hot(red, 26)
    cum, last = _cum(pres), _last_seen(pres)
    ts = np.arange(_WARMUP, n_draws + 1)
    feats = np.stack([
        cum[ts] / ts[:, None], _window_freq(cum, ts, 26), _window_freq(cum, ts, 52),
        _gap_norm(last, ts),
    ], axis=2)[:, 1:, :]
    X, y = feats[:-1].reshape(-1, feats.shape[2]), pres[ts[:-1]][:, 1:].reshape(-1)
    return X, y, np.repeat(ts[:-1], 26), feats[-1]


def _fit_and_score(X, y, row_t, n_draws, next_X, pool):
    """Fit on older draws, score on the held-out newest; return next-draw probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = row_t < n_draws - _HOLDOUT_N
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X[train], y[train])

    p_test = model.predict_proba(X[~train])[:, 1]
    auc = roc_auc_score(y[~train], p_test)
    # Log-loss of "which number came out here" after normalising over the pool.
    dist   = p_test.reshape(-1, pool)
    dist   = dist / dist.sum(axis=1, keepdims=True)
    actual = y[~train].reshape(-1, pool).argmax(axis=1)
    loss   = -np.log(dist[np.arange(len(dist)), actual]).mean()

    probs = model.predict_proba(next_X)[:, 1]
    return probs / probs.sum(), auc, loss, np.log(pool)


def train_position_model():
    """Train (once per session) the five position models and the red-ball model."""
    if _ML_CACHE:
        return _ML_CACHE
    if np is None:
        raise ImportError("numpy", name="numpy")
    df, last_order = _load_drawn_order()
    n_draws = len(df)
    if n_draws < _WARMUP + _HOLDOUT_N + 10:
        raise RuntimeError(f"only {n_draws} drawn-order draws; need more history")

    values = df[_MAIN_COLS].values.astype(int)
    _ML_CACHE["n_draws"], _ML_CACHE["last"] = n_draws, last_order.date()
    _ML_CACHE["positions"] = []
    for p in range(5):
        X, y, row_t, nxt = _position_features(values, p)
        _ML_CACHE["positions"].append(_fit_and_score(X, y, row_t, n_draws, nxt, 69))
    X, y, row_t, nxt = _red_features(df["powerball"].values.astype(int))
    _ML_CACHE["red"] = _fit_and_score(X, y, row_t, n_draws, nxt, 26)
    return _ML_CACHE


def _ml_pick(ml, greedy=False):
    """One ticket in drawn order: position 1..5 each picked from its own distribution."""
    ticket = []
    for probs, *_ in ml["positions"]:
        cand    = [n for n in range(1, 70) if n not in ticket]
        weights = [probs[n - 1] for n in cand]
        ticket.append(cand[weights.index(max(weights))] if greedy
                      else _weighted_sample(cand, weights, 1)[0])
    red_probs = ml["red"][0]
    pb = (int(red_probs.argmax()) + 1 if greedy
          else _weighted_sample(list(range(1, 27)), list(red_probs), 1)[0])
    return ticket, pb


def generate_ml_position_picks(count: int):
    """Returns (model info, top-confidence ticket, [count weighted tickets])."""
    ml = train_position_model()
    return ml, _ml_pick(ml, greedy=True), [_ml_pick(ml) for _ in range(count)]


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


def _show_position_ticket(label: str, ticket: list, pb: int):
    cells = "   ".join(f"P{i}: {n:2d}" for i, n in enumerate(ticket, 1))
    print(f"  {label:<16}:  {cells}   |  Red: {pb}")


def display_ml_position(game: str, ml: dict, top, picks: list):
    """Show model quality first, then the tickets in drawn order (not sorted)."""
    border = "=" * 70
    print(f"\n{border}")
    print(f"   ML POSITION PICK  -  {game}  (drawn order, not sorted)")
    print(f"   Trained on {ml['n_draws']:,} drawn-order draws through {ml['last']}")
    print(border)
    print("\n  Held-out check, last 100 draws (AUC 0.5 and log-loss = uniform are chance):")
    for i, (_, auc, loss, base) in enumerate(ml["positions"], 1):
        print(f"    Position {i}: AUC {auc:.3f}   log-loss {loss:.3f} vs uniform {base:.3f}")
    _, auc, loss, base = ml["red"]
    print(f"    Red ball  : AUC {auc:.3f}   log-loss {loss:.3f} vs uniform {base:.3f}")
    print("\n  Position P1..P5 = order the balls came out; each is picked from")
    print("  that position's own model distribution, without repeating a number.\n")
    _show_position_ticket("Top Confidence", *top)
    for i, (ticket, pb) in enumerate(picks, 1):
        _show_position_ticket(f"Weighted Pick {i}", ticket, pb)
    print("\n  Note: ball order is random too, so a score near chance is expected.")
    print(border)


def run_ml_position(game: str):
    raw = input("  How many weighted tickets? [3]: ").strip()
    count = int(raw) if raw.isdigit() and int(raw) > 0 else 3
    print("\n  Training position models ... ", end="", flush=True)
    try:
        ml, top, picks = generate_ml_position_picks(count)
    except ImportError as e:
        print(f"\n  Missing dependency ({e.name}). Run: pip install numpy pandas scikit-learn openpyxl")
        return
    except (OSError, RuntimeError, ValueError) as e:
        print(f"\n  Could not train: {e}")
        return
    print("done.")
    display_ml_position(game, ml, top, picks)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║          LOTTERY NUMBER GENERATOR                ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Random · Smart Picks · ML Position Picks       ║")
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
        print("\n  ── ML Position Pick (drawn order, not sorted) ─")
        print("  7. ★ Powerball / Double Play ML Position Pick")
        print("\n  8. Exit")

        choice = input("\n  Enter your choice (1-8): ").strip()

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
            run_ml_position("POWERBALL / DOUBLE PLAY")

        elif choice == "8":
            print("\n  Thank you for using Lottery Number Generator!")
            print("  Good luck!\n")
            break

        else:
            print("\n  Invalid choice. Please enter 1 through 8.")


if __name__ == "__main__":
    main()
