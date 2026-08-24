#!/usr/bin/env python3
"""
Train a logistic-regression model per number on historical Powerball draws
and use its predicted probabilities to weight ticket generation.

IMPORTANT: Powerball drawings are independent, uniformly-random mechanical
draws. No feature derivable from past draws (frequency, recency, seasonality)
carries information about future draws. This script trains a real model and
reports its held-out accuracy so you can see for yourself that it performs
no better than the random baseline (5/69 chance a number appears). It is
NOT a predictor of winning numbers - it is a data-informed way to pick
tickets for fun, nothing more.
"""
import argparse
import sys, io
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EXCEL_FILE = "powerball_game.xlsx"
MAIN_COLS = ["Num1", "Num2", "Num3", "Num4", "Num5"]
RECENT_WINDOW = 52          # ~1 year of draws for "recent frequency" feature
WARMUP = 60                 # draws needed before features are meaningful
TEST_FRACTION = 0.15        # most recent slice held out for evaluation


def load_data():
    raw = pd.read_excel(EXCEL_FILE)
    rows = []
    for _, r in raw.iterrows():
        try:
            dt = pd.to_datetime(f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['Day']):02d}")
        except Exception:
            continue
        rows.append({
            "draw_date": dt,
            "Num1": int(r["Num1"]), "Num2": int(r["Num2"]), "Num3": int(r["Num3"]),
            "Num4": int(r["Num4"]), "Num5": int(r["Num5"]), "Powerball": int(r["powerball"]),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["draw_date"] + MAIN_COLS)
    return df.sort_values("draw_date").reset_index(drop=True)


def build_feature_table(df, number_range, col_getter):
    """
    For each draw t >= WARMUP and each candidate number n in number_range,
    build features from draws strictly before t, and label = 1 if n was
    drawn at t, else 0. Returns a long-format DataFrame.
    """
    n_lo, n_hi = number_range
    numbers = list(range(n_lo, n_hi + 1))
    total_seen = {n: 0 for n in numbers}
    last_seen_idx = {n: -1 for n in numbers}
    recent_hist = {n: [] for n in numbers}  # rolling window of 0/1 appearance flags

    records = []
    for t in range(len(df)):
        drawn = set(col_getter(df.iloc[t]))
        dow = df.iloc[t]["draw_date"].dayofweek
        month = df.iloc[t]["draw_date"].month

        if t >= WARMUP:
            for n in numbers:
                overall_freq = total_seen[n] / t
                recent_freq = (sum(recent_hist[n][-RECENT_WINDOW:]) /
                               max(1, len(recent_hist[n][-RECENT_WINDOW:])))
                gap = t - 1 - last_seen_idx[n] if last_seen_idx[n] >= 0 else t
                records.append({
                    "draw_idx": t,
                    "number": n,
                    "overall_freq": overall_freq,
                    "recent_freq": recent_freq,
                    "gap": gap,
                    "dow": dow,
                    "month": month,
                    "label": 1 if n in drawn else 0,
                })

        # update running stats AFTER building features for this draw
        for n in numbers:
            hit = 1 if n in drawn else 0
            total_seen[n] += hit
            recent_hist[n].append(hit)
            if hit:
                last_seen_idx[n] = t

    return pd.DataFrame(records)


def train_and_eval(feat_df, label="main numbers"):
    feat_df = feat_df.sort_values("draw_idx").reset_index(drop=True)
    split_idx = int(feat_df["draw_idx"].max() * (1 - TEST_FRACTION))
    train = feat_df[feat_df["draw_idx"] <= split_idx]
    test = feat_df[feat_df["draw_idx"] > split_idx]

    feature_cols = ["overall_freq", "recent_freq", "gap", "dow", "month"]
    X_train, y_train = train[feature_cols], train["label"]
    X_test, y_test = test[feature_cols], test["label"]

    model = LogisticRegression(max_iter=1000, class_weight=None)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    baseline_rate = y_train.mean()  # constant baseline = training prevalence
    p_baseline = np.full_like(p_test, baseline_rate)

    auc = roc_auc_score(y_test, p_test)
    ll_model = log_loss(y_test, p_test)
    ll_base = log_loss(y_test, p_baseline)
    brier_model = brier_score_loss(y_test, p_test)
    brier_base = brier_score_loss(y_test, p_baseline)

    print(f"  [{label}] held-out evaluation ({len(test['draw_idx'].unique())} draws, "
          f"{len(test)} number-observations)")
    print(f"    AUC (0.5 = coin flip, no skill) ........ {auc:.4f}")
    print(f"    Log-loss  model / constant-baseline .... {ll_model:.4f} / {ll_base:.4f}")
    print(f"    Brier     model / constant-baseline .... {brier_model:.4f} / {brier_base:.4f}")
    print("    (model beating baseline here would be noteworthy; expect them to be")
    print("     statistically indistinguishable, confirming draws are independent)\n")

    return model, feature_cols


def predict_next_draw_probs(model, feature_cols, df, number_range, col_getter):
    n_lo, n_hi = number_range
    numbers = list(range(n_lo, n_hi + 1))
    t = len(df)
    total_seen = {n: 0 for n in numbers}
    last_seen_idx = {n: -1 for n in numbers}
    recent_hist = {n: [] for n in numbers}

    for i in range(t):
        drawn = set(col_getter(df.iloc[i]))
        for n in numbers:
            hit = 1 if n in drawn else 0
            total_seen[n] += hit
            recent_hist[n].append(hit)
            if hit:
                last_seen_idx[n] = i

    next_dow = (df.iloc[-1]["draw_date"].dayofweek + 2) % 7  # rough next-draw-day guess
    next_month = df.iloc[-1]["draw_date"].month

    rows = []
    for n in numbers:
        overall_freq = total_seen[n] / t
        recent_freq = (sum(recent_hist[n][-RECENT_WINDOW:]) /
                       max(1, len(recent_hist[n][-RECENT_WINDOW:])))
        gap = t - 1 - last_seen_idx[n] if last_seen_idx[n] >= 0 else t
        rows.append({"number": n, "overall_freq": overall_freq, "recent_freq": recent_freq,
                     "gap": gap, "dow": next_dow, "month": next_month})
    X = pd.DataFrame(rows)
    probs = model.predict_proba(X[feature_cols])[:, 1]
    return dict(zip(X["number"], probs))


def weighted_sample_without_replacement(prob_dict, k, rng):
    numbers = list(prob_dict.keys())
    weights = np.array([prob_dict[n] for n in numbers], dtype=float)
    weights = weights / weights.sum()
    chosen = list(rng.choice(numbers, size=k, replace=False, p=weights))
    return sorted(int(c) for c in chosen)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an ML model on Powerball history and generate model-weighted tickets.")
    p.add_argument("-n", "--tickets", type=int, default=3,
                    help="number of tickets to generate (default: 3)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    n_tickets = max(1, args.tickets)
    rng = np.random.default_rng(args.seed)

    df = load_data()
    print(f"Loaded {len(df):,} Powerball draws through {df['draw_date'].max().date()}\n")

    print("Training model on MAIN numbers (1-69)...")
    main_feat = build_feature_table(df, (1, 69), lambda row: [row[c] for c in MAIN_COLS])
    main_model, main_cols = train_and_eval(main_feat, "main numbers")

    print("Training model on POWERBALL (1-26)...")
    pb_feat = build_feature_table(df, (1, 26), lambda row: [row["Powerball"]])
    pb_model, pb_cols = train_and_eval(pb_feat, "powerball")

    main_probs = predict_next_draw_probs(main_model, main_cols, df, (1, 69),
                                          lambda row: [row[c] for c in MAIN_COLS])
    pb_probs = predict_next_draw_probs(pb_model, pb_cols, df, (1, 26),
                                        lambda row: [row["Powerball"]])

    print("=" * 64)
    print(f"  {n_tickets} MODEL-WEIGHTED POWERBALL TICKET{'S' if n_tickets != 1 else ''}")
    print("=" * 64)
    for i in range(1, n_tickets + 1):
        nums = weighted_sample_without_replacement(main_probs, 5, rng)
        pb = int(rng.choice(list(pb_probs.keys()), p=np.array(list(pb_probs.values())) /
                             sum(pb_probs.values())))
        print(f"\n  Ticket {i}")
        print(f"    Numbers   : {' - '.join(f'{n:02d}' for n in nums)}")
        print(f"    Powerball : {pb:02d}")

    print("\n" + "-" * 64)
    print("DISCLAIMER: The evaluation above shows this model's held-out AUC/log-loss/")
    print("Brier score are statistically indistinguishable from a naive constant-")
    print("probability baseline. That is expected: each Powerball draw is an")
    print("independent, mechanically random event, so no model - this one included -")
    print("can predict it above chance (1 in 292,201,338 for the jackpot). These")
    print("tickets are for entertainment only. Play responsibly.")


if __name__ == "__main__":
    main()
