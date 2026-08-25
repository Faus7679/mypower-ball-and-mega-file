#!/usr/bin/env python3
"""
Machine-Learning Powerball Ticket Generator.

Trains a logistic regression per number (main pool 1-69, red ball 1-26) on
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

PB_API = ("https://data.ny.gov/resource/d6yy-54nr.json"
          "?$order=draw_date%20DESC&$limit=5000")
EXCEL_FILE   = "powerball_game.xlsx"
MAIN_COLS    = ["Num1", "Num2", "Num3", "Num4", "Num5"]
# Powerball switched to the current 69-white/26-red matrix on 2015-10-07;
# earlier draws used different pools (e.g. red ball up to 35) and would
# corrupt the fixed-width one-hot feature matrices below.
CURRENT_RULES_START = pd.Timestamp("2015-10-07")
WARMUP       = 104   # draws of history required before a row is trainable
HOLDOUT_N    = 100   # most recent draws held out for AUC evaluation
REC_WINDOWS  = (26, 52)
GAP_CAP      = 120


# ── Data loading (same pattern as powerball_three_tickets.py) ────────────────

def fetch_api() -> pd.DataFrame:
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


def load_excel() -> pd.DataFrame:
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

    presence = _presence_matrix(main_matrix, 69)
    cum = np.vstack([np.zeros((1, 70)), np.cumsum(presence, axis=0)])  # (n_draws+1, 70)
    last_seen_before = _last_seen_before(presence)
    pair_before, pair_counts_final = _pair_momentum(main_matrix, 69)

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
    for n in range(1, 70):
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
    for n in range(1, 70):
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


def build_red_features(df: pd.DataFrame):
    red = df["Powerball"].values.astype(int)
    n_draws = len(red)
    presence = np.zeros((n_draws, 27), dtype=np.float64)
    presence[np.arange(n_draws), red] = 1.0
    cum = np.vstack([np.zeros((1, 27)), np.cumsum(presence, axis=0)])
    last_seen_before = _last_seen_before(presence)

    ts = np.arange(WARMUP, n_draws)
    freq_all = cum[ts] / ts[:, None]
    rec26 = (cum[ts] - cum[np.maximum(ts - 26, 0)]) / np.minimum(26, ts)[:, None]
    rec52 = (cum[ts] - cum[np.maximum(ts - 52, 0)]) / np.minimum(52, ts)[:, None]
    gap = np.where(last_seen_before[ts] >= 0, ts[:, None] - last_seen_before[ts], ts[:, None])
    gap_norm = np.minimum(gap, GAP_CAP) / GAP_CAP
    target = presence[ts]

    X_blocks, y_blocks, ts_blocks = [], [], []
    for b in range(1, 27):
        Xb = np.stack([freq_all[:, b], rec26[:, b], rec52[:, b], gap_norm[:, b]], axis=1)
        X_blocks.append(Xb)
        y_blocks.append(target[:, b])
        ts_blocks.append(ts)
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    ts_full = np.concatenate(ts_blocks)

    next_feats = {}
    for b in range(1, 27):
        f_all = cum[n_draws, b] / n_draws
        f26 = (cum[n_draws, b] - cum[max(n_draws - 26, 0), b]) / min(26, n_draws)
        f52 = (cum[n_draws, b] - cum[max(n_draws - 52, 0), b]) / min(52, n_draws)
        seen = last_seen_before[-1, b] if last_seen_before[-1, b] >= 0 else -1
        seen = n_draws - 1 if red[-1] == b else seen
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


def show(label: str, main: list, pb: int, strategy: str = ""):
    w = 58
    print(f"\n  +- {label} {'-'*(w - len(label) - 3)}+")
    print(f"  |  Numbers  :  {' - '.join(f'{n:2d}' for n in sorted(main)):<45}|")
    print(f"  |  Powerball:  {pb:<2}{'':>43}|")
    print(f"  |  Profile  :  {profile(sorted(main)):<45}|")
    if strategy:
        print(f"  |  Strategy :  {strategy:<45}|")
    print(f"  +{'-'*w}+")


def parse_args():
    p = argparse.ArgumentParser(description="Generate ML-scored Powerball tickets for the next draw.")
    p.add_argument("-n", "--tickets", type=int, default=3,
                   help="number of weighted-random ML tickets to generate (default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    n_tickets = max(1, args.tickets)
    random.seed()

    print("=" * 66)
    print("  MACHINE-LEARNING POWERBALL TICKET GENERATOR")
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

    print("\n  Training main-number model (69-way, 5 causal features/number) ...")
    X_main, y_main, ts_main, next_main, n_draws_main = build_main_features(df)
    main_prob, main_auc = train_and_score(X_main, y_main, ts_main, n_draws_main, next_main)

    print("  Training red-ball model (26-way, 4 causal features/ball) ...")
    X_red, y_red, ts_red, next_red = build_red_features(df)
    red_prob, red_auc = train_and_score(X_red, y_red, ts_red, n_draws_main, next_red)

    ranked_main = sorted(main_prob.items(), key=lambda x: -x[1])
    ranked_red = sorted(red_prob.items(), key=lambda x: -x[1])

    print(f"\n  Held-out ROC-AUC (last {min(HOLDOUT_N, n_draws - WARMUP)} draws, 0.5 = coin flip):")
    print(f"    Main numbers : {main_auc:.3f}" if main_auc is not None else "    Main numbers : n/a")
    print(f"    Red ball     : {red_auc:.3f}" if red_auc is not None else "    Red ball     : n/a")
    print("  An AUC near 0.5 is expected -- Powerball draws are independent random")
    print("  events, so no model, this one included, has real predictive power.")

    print(f"\n  Top-15 main numbers by model probability:")
    print(f"    {', '.join(f'{n}({p:.3f})' for n, p in ranked_main[:15])}")
    print(f"  Top-7 red balls by model probability:")
    print(f"    {', '.join(f'{b}({p:.3f})' for b, p in ranked_red[:7])}")

    # Deterministic top-confidence ticket.
    top5 = sorted(n for n, _ in ranked_main[:5])
    top_red = ranked_red[0][0]
    show("ML TOP CONFIDENCE", top5, top_red,
         "Highest model-predicted probability, no randomness")

    # N stochastic tickets, weighted by predicted probability.
    print(f"\n  --- {n_tickets} ML WEIGHTED-RANDOM TICKET{'S' if n_tickets != 1 else ''} ---")
    pool_main = [n for n, _ in ranked_main]
    w_main = [max(p, 1e-6) for _, p in ranked_main]
    pool_red = [b for b, _ in ranked_red]
    w_red = [max(p, 1e-6) for _, p in ranked_red]

    for i in range(n_tickets):
        picked = sorted(weighted_sample(pool_main, w_main, 5))
        pb = weighted_sample(pool_red, w_red, 1)[0]
        show(f"ML Pick {i + 1}", picked, pb,
             "Weighted random sample from model-predicted probabilities")

    print("""
  ==================================================================
  DISCLAIMER: This model is trained on past draws only and reports
  its own near-chance held-out AUC above. Lottery draws are
  independent random events -- no model can predict them.
  Play responsibly and within your budget.
  ==================================================================
""")


if __name__ == "__main__":
    main()
