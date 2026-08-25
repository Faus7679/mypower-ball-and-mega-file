#!/usr/bin/env python3
"""
Machine-Learning Mega Millions Ticket Generator.

Trains a logistic regression per number (main pool 1-70, mega ball 1-25) on
engineered features built causally from draw history -- rolling frequency
(recent 26 / 52 draws), all-time frequency, overdue gap, and a pairing-
momentum score vs. the immediately preceding draw. Every feature for a
training row at draw t is computed only from draws before t, so the model
never sees the future it is trying to predict.

Reports held-out ROC-AUC so the model's real (near-chance) skill is visible
rather than implied. Lottery draws are independent random events; nothing
here predicts the next draw.
"""
import argparse
import random
import sys, io
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MM_API = ("https://data.ny.gov/resource/5xaw-6ayf.json"
          "?$order=draw_date%20DESC&$limit=5000")
EXCEL_FILE   = "mega_millions_draws.xlsx"
MAIN_COLS    = ["Num1", "Num2", "Num3", "Num4", "Num5"]
# Mega Millions switched to the current 70-white/25-mega-ball matrix on
# 2017-10-31; earlier draws used different pools (e.g. white up to 75) and
# would corrupt the fixed-width one-hot feature matrices below.
CURRENT_RULES_START = pd.Timestamp("2017-10-31")
WARMUP       = 104   # draws of history required before a row is trainable
HOLDOUT_N    = 100   # most recent draws held out for AUC evaluation
REC_WINDOWS  = (26, 52)
GAP_CAP      = 120
WHITE_POOL   = 70
MEGA_POOL    = 25


# ── Data loading (same pattern as ml_ticket_generator.py) ────────────────

def fetch_api() -> pd.DataFrame:
    r = requests.get(MM_API, timeout=25)
    r.raise_for_status()
    rows = []
    for item in r.json():
        raw_date = item.get("draw_date", "")[:10]
        parts = item.get("winning_numbers", "").split()
        mega_ball = item.get("mega_ball", None)
        if len(parts) < 5:
            continue
        try:
            nums = [int(p) for p in parts[:5]]
            mb = int(mega_ball) if mega_ball is not None else int(parts[5])
            dt = pd.to_datetime(raw_date)
        except (ValueError, TypeError, IndexError):
            continue
        rows.append({"draw_date": dt, "Num1": nums[0], "Num2": nums[1],
                      "Num3": nums[2], "Num4": nums[3], "Num5": nums[4],
                      "MegaBall": mb})
    return pd.DataFrame(rows)


def load_excel() -> pd.DataFrame:
    try:
        raw = pd.read_excel(EXCEL_FILE)
        if "draw_date" in raw.columns:
            dt = pd.to_datetime(raw["draw_date"])
        else:
            dt = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"], day=raw["Day"]))
        return pd.DataFrame({
            "draw_date": dt,
            "Num1": raw["Num1"].astype(int), "Num2": raw["Num2"].astype(int),
            "Num3": raw["Num3"].astype(int), "Num4": raw["Num4"].astype(int),
            "Num5": raw["Num5"].astype(int), "MegaBall": raw["MegaBall"].astype(int),
        })
    except Exception:
        return pd.DataFrame()


def load_data() -> pd.DataFrame:
    try:
        api_df = fetch_api()
    except Exception:
        api_df = pd.DataFrame()
    df = pd.concat([api_df, load_excel()], ignore_index=True)
    df = df.drop_duplicates(subset=["draw_date", "Num1", "Num2", "Num3", "Num4", "Num5"])
    df = df[df["draw_date"] >= CURRENT_RULES_START]
    return df.sort_values("draw_date").reset_index(drop=True)


# ── Causal feature engineering ────────────────────────────────────────────────

def _presence_matrix(values: np.ndarray, pool_size: int) -> np.ndarray:
    """values: (n_draws, k) int array of drawn numbers -> (n_draws, pool_size+1) one-hot."""
    n_draws = values.shape[0]
    presence = np.zeros((n_draws, pool_size + 1), dtype=np.float64)
    for t in range(n_draws):
        presence[t, values[t]] = 1.0
    return presence


def _last_seen_before(presence: np.ndarray) -> np.ndarray:
    """last_seen_before[t, n] = draw index of n's last appearance in [0, t), or -1."""
    n_draws, width = presence.shape
    out = np.full((n_draws, width), -1, dtype=np.int64)
    seen = np.full(width, -1, dtype=np.int64)
    for t in range(n_draws):
        out[t] = seen
        hit = presence[t] > 0
        seen[hit] = t
    return out


def _pair_momentum(main_matrix: np.ndarray, pool_size: int) -> np.ndarray:
    """pair_score_before[t, n] = historical co-occurrence strength of n with the
    numbers drawn in draw t-1, using only pairs formed by draws < t."""
    n_draws = main_matrix.shape[0]
    pair_counts = np.zeros((pool_size + 1, pool_size + 1), dtype=np.float64)
    out = np.zeros((n_draws, pool_size + 1), dtype=np.float64)
    for t in range(1, n_draws):
        prev = main_matrix[t - 1]
        for a, b in combinations(sorted(prev), 2):
            pair_counts[a, b] += 1
            pair_counts[b, a] += 1
        out[t] = pair_counts[:, prev].sum(axis=1)
    return out, pair_counts


def build_main_features(df: pd.DataFrame):
    main_matrix = df[MAIN_COLS].values.astype(int)
    n_draws = main_matrix.shape[0]

    presence = _presence_matrix(main_matrix, WHITE_POOL)
    cum = np.vstack([np.zeros((1, WHITE_POOL + 1)), np.cumsum(presence, axis=0)])
    last_seen_before = _last_seen_before(presence)
    pair_before, pair_counts_final = _pair_momentum(main_matrix, WHITE_POOL)

    ts = np.arange(WARMUP, n_draws)
    T = len(ts)
    freq_all = cum[ts] / ts[:, None]
    rec = {}
    for w in REC_WINDOWS:
        rec[w] = (cum[ts] - cum[np.maximum(ts - w, 0)]) / np.minimum(w, ts)[:, None]
    gap = np.where(last_seen_before[ts] >= 0, ts[:, None] - last_seen_before[ts], ts[:, None])
    gap_norm = np.minimum(gap, GAP_CAP) / GAP_CAP
    pair = pair_before[ts]
    target = presence[ts]

    X_blocks, y_blocks, ts_blocks = [], [], []
    for n in range(1, WHITE_POOL + 1):
        Xn = np.stack([freq_all[:, n], rec[26][:, n], rec[52][:, n],
                        gap_norm[:, n], pair[:, n]], axis=1)
        X_blocks.append(Xn)
        y_blocks.append(target[:, n])
        ts_blocks.append(ts)
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ts_full = np.concatenate(ts_blocks)

    # Feature vector for the not-yet-played next draw, using all n_draws of history.
    last_draw = main_matrix[-1]
    for a, b in combinations(sorted(last_draw), 2):
        pair_counts_final[a, b] += 1
        pair_counts_final[b, a] += 1
    pair_next = pair_counts_final[:, last_draw].sum(axis=1)

    next_feats = {}
    for n in range(1, WHITE_POOL + 1):
        f_all = cum[n_draws, n] / n_draws
        f26 = (cum[n_draws, n] - cum[max(n_draws - 26, 0), n]) / min(26, n_draws)
        f52 = (cum[n_draws, n] - cum[max(n_draws - 52, 0), n]) / min(52, n_draws)
        seen = last_seen_before[-1, n] if last_seen_before[-1, n] >= 0 else -1
        # roll the "last seen" one draw forward to include the final draw itself
        seen = n_draws - 1 if n in last_draw else seen
        g = (n_draws - seen) if seen >= 0 else n_draws
        g_norm = min(g, GAP_CAP) / GAP_CAP
        next_feats[n] = [f_all, f26, f52, g_norm, pair_next[n]]

    return X, y, ts_full, next_feats, n_draws


def build_mega_features(df: pd.DataFrame):
    mega = df["MegaBall"].values.astype(int)
    n_draws = len(mega)
    presence = np.zeros((n_draws, MEGA_POOL + 1), dtype=np.float64)
    presence[np.arange(n_draws), mega] = 1.0
    cum = np.vstack([np.zeros((1, MEGA_POOL + 1)), np.cumsum(presence, axis=0)])
    last_seen_before = _last_seen_before(presence)

    ts = np.arange(WARMUP, n_draws)
    freq_all = cum[ts] / ts[:, None]
    rec26 = (cum[ts] - cum[np.maximum(ts - 26, 0)]) / np.minimum(26, ts)[:, None]
    rec52 = (cum[ts] - cum[np.maximum(ts - 52, 0)]) / np.minimum(52, ts)[:, None]
    gap = np.where(last_seen_before[ts] >= 0, ts[:, None] - last_seen_before[ts], ts[:, None])
    gap_norm = np.minimum(gap, GAP_CAP) / GAP_CAP
    target = presence[ts]

    X_blocks, y_blocks, ts_blocks = [], [], []
    for b in range(1, MEGA_POOL + 1):
        Xb = np.stack([freq_all[:, b], rec26[:, b], rec52[:, b], gap_norm[:, b]], axis=1)
        X_blocks.append(Xb)
        y_blocks.append(target[:, b])
        ts_blocks.append(ts)
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ts_full = np.concatenate(ts_blocks)

    next_feats = {}
    for b in range(1, MEGA_POOL + 1):
        f_all = cum[n_draws, b] / n_draws
        f26 = (cum[n_draws, b] - cum[max(n_draws - 26, 0), b]) / min(26, n_draws)
        f52 = (cum[n_draws, b] - cum[max(n_draws - 52, 0), b]) / min(52, n_draws)
        seen = last_seen_before[-1, b] if last_seen_before[-1, b] >= 0 else -1
        seen = n_draws - 1 if mega[-1] == b else seen
        g = (n_draws - seen) if seen >= 0 else n_draws
        g_norm = min(g, GAP_CAP) / GAP_CAP
        next_feats[b] = [f_all, f26, f52, g_norm]

    return X, y, ts_full, next_feats


# ── Model training / evaluation ───────────────────────────────────────────────

def train_and_score(X, y, ts_full, n_draws, next_feats):
    split_t = max(n_draws - HOLDOUT_N, WARMUP + 1)
    train_mask = ts_full < split_t
    test_mask = ~train_mask

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(X[train_mask], y[train_mask])

    auc = None
    if test_mask.sum() > 0 and len(set(y[test_mask])) > 1:
        preds = model.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y[test_mask], preds)

    prob = {n: model.predict_proba([feats])[0, 1] for n, feats in next_feats.items()}
    return prob, auc


# ── Sampling / display ────────────────────────────────────────────────────────

def weighted_sample(population: list, weights: list, k: int) -> list:
    pool = list(zip(population, weights))
    result = []
    while len(result) < k:
        total = sum(w for _, w in pool)
        r = random.uniform(0, total)
        cum = 0.0
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


def show(label: str, main: list, mb: int, strategy: str = ""):
    w = 58
    print(f"\n  +- {label} {'-'*(w - len(label) - 3)}+")
    print(f"  |  Numbers  :  {' - '.join(f'{n:2d}' for n in sorted(main)):<45}|")
    print(f"  |  Mega Ball:  {mb:<2}{'':>43}|")
    print(f"  |  Profile  :  {profile(sorted(main)):<45}|")
    if strategy:
        print(f"  |  Strategy :  {strategy:<45}|")
    print(f"  +{'-'*w}+")


def parse_args():
    p = argparse.ArgumentParser(description="Generate ML-scored Mega Millions tickets for the next draw.")
    p.add_argument("-n", "--tickets", type=int, default=3,
                   help="number of weighted-random ML tickets to generate (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    n_tickets = max(1, args.tickets)
    random.seed()

    print("=" * 66)
    print("  MACHINE-LEARNING MEGA MILLIONS TICKET GENERATOR")
    print("  (logistic regression, causal rolling features, held-out AUC)")
    print("=" * 66)

    print("\n  Loading draw history ... ", end="", flush=True)
    df = load_data()
    n_draws = len(df)
    dmax = df["draw_date"].max().date()
    print(f"done. {n_draws:,} draws through {dmax}")

    if n_draws < WARMUP + HOLDOUT_N + 10:
        print("\n  Not enough draw history to train reliably. Exiting.")
        sys.exit(1)

    print("\n  Training main-number model (70-way, 5 causal features/number) ...")
    X_main, y_main, ts_main, next_main, n_draws_main = build_main_features(df)
    main_prob, main_auc = train_and_score(X_main, y_main, ts_main, n_draws_main, next_main)

    print("  Training mega-ball model (25-way, 4 causal features/ball) ...")
    X_mega, y_mega, ts_mega, next_mega = build_mega_features(df)
    mega_prob, mega_auc = train_and_score(X_mega, y_mega, ts_mega, n_draws_main, next_mega)

    ranked_main = sorted(main_prob.items(), key=lambda x: -x[1])
    ranked_mega = sorted(mega_prob.items(), key=lambda x: -x[1])

    print(f"\n  Held-out ROC-AUC (last {min(HOLDOUT_N, n_draws - WARMUP)} draws, 0.5 = coin flip):")
    print(f"    Main numbers : {main_auc:.3f}" if main_auc is not None else "    Main numbers : n/a")
    print(f"    Mega ball    : {mega_auc:.3f}" if mega_auc is not None else "    Mega ball    : n/a")
    print("  An AUC near 0.5 is expected -- Mega Millions draws are independent")
    print("  random events, so no model, this one included, has real predictive power.")

    print(f"\n  Top-15 main numbers by model probability:")
    print(f"    {', '.join(f'{n}({p:.3f})' for n, p in ranked_main[:15])}")
    print(f"  Top-7 mega balls by model probability:")
    print(f"    {', '.join(f'{b}({p:.3f})' for b, p in ranked_mega[:7])}")

    # Deterministic top-confidence ticket.
    top5 = sorted(n for n, _ in ranked_main[:5])
    top_mega = ranked_mega[0][0]
    show("ML TOP CONFIDENCE", top5, top_mega,
         "Highest model-predicted probability, no randomness")

    # N stochastic tickets, weighted by predicted probability.
    print(f"\n  --- {n_tickets} ML WEIGHTED-RANDOM TICKET{'S' if n_tickets != 1 else ''} ---")
    pool_main = [n for n, _ in ranked_main]
    w_main = [max(p, 1e-6) for _, p in ranked_main]
    pool_mega = [b for b, _ in ranked_mega]
    w_mega = [max(p, 1e-6) for _, p in ranked_mega]

    for i in range(n_tickets):
        picked = sorted(weighted_sample(pool_main, w_main, 5))
        mb = weighted_sample(pool_mega, w_mega, 1)[0]
        show(f"ML Pick {i + 1}", picked, mb,
             "Weighted random sample from model-predicted probabilities")

    print("""
  ==================================================================
  DISCLAIMER: This model is trained on past draws only and reports
  its own near-chance held-out AUC above. Mega Millions draws are
  independent random events -- no model can predict them.
  Play responsibly and within your budget.
  ==================================================================
""")


if __name__ == "__main__":
    main()
