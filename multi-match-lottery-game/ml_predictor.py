"""Machine-learning predictor for the Maryland Multi-Match demo.

This trains a per-number logistic regression classifier ("does number N
appear in the next draw?") on engineered features from the historical
draw sequence in ``lottery_game.sample_maryland_history``, then ranks all
43 numbers by predicted probability to build a line.

Lottery draws are independent random events, so no model can beat chance
in reality. This module exists to demonstrate a real train/predict/
backtest ML workflow on top of the existing project, not to make any
claim of predictive power.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from lottery_game import (
    DRAW_SIZE,
    DrawRecord,
    NUMBER_RANGE,
    sample_maryland_history,
    validate_draw_day,
)

MIN_TRAINING_DRAWS = 8
FEATURE_NAMES = (
    "number_normalized",
    "career_frequency",
    "hits_last_3",
    "hits_last_5",
    "gap_ratio",
    "history_length",
)


def build_feature_row(history_before: tuple[DrawRecord, ...], number: int) -> list[float]:
    """Engineer features for `number` using only draws that happened before it."""
    hits = [1.0 if number in record.numbers else 0.0 for record in history_before]
    history_length = len(hits)
    career_frequency = sum(hits) / history_length if history_length else 0.0
    hits_last_3 = sum(hits[-3:])
    hits_last_5 = sum(hits[-5:])
    gap = next(
        (offset for offset, hit in enumerate(reversed(hits), start=1) if hit),
        history_length + 1,
    )
    return [
        number / max(NUMBER_RANGE),
        career_frequency,
        hits_last_3,
        hits_last_5,
        gap / (history_length + 1),
        float(history_length),
    ]


def build_training_set(history_by_day: tuple[DrawRecord, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a chronological single-weekday draw history into (X, y) training rows.

    Each draw after the first `MIN_TRAINING_DRAWS` draws contributes one row per
    number (43 rows), labeled 1 if that number appeared in that draw.
    """
    features: list[list[float]] = []
    labels: list[int] = []
    for index in range(MIN_TRAINING_DRAWS, len(history_by_day)):
        history_before = history_by_day[:index]
        draw = history_by_day[index]
        for number in NUMBER_RANGE:
            features.append(build_feature_row(history_before, number))
            labels.append(1 if number in draw.numbers else 0)
    return np.array(features), np.array(labels)


def train_model(history: Iterable[DrawRecord], draw_day: str) -> tuple[LogisticRegression, tuple[DrawRecord, ...]]:
    """Train a logistic regression model on all available history for `draw_day`."""
    validate_draw_day(draw_day)
    history_by_day = tuple(record for record in history if record.weekday == draw_day)
    if len(history_by_day) <= MIN_TRAINING_DRAWS + 1:
        raise ValueError(
            f"Need more than {MIN_TRAINING_DRAWS + 1} {draw_day} draws to train a model; "
            f"found {len(history_by_day)}."
        )
    features, labels = build_training_set(history_by_day)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(features, labels)
    return model, history_by_day


def predict_probabilities(
    model: LogisticRegression, history_by_day: tuple[DrawRecord, ...]
) -> dict[int, float]:
    """Predict each number's probability of appearing in the next draw."""
    features = np.array([build_feature_row(history_by_day, number) for number in NUMBER_RANGE])
    probabilities = model.predict_proba(features)[:, 1]
    return dict(zip(NUMBER_RANGE, probabilities))


def predict_ml_line(
    model: LogisticRegression, history_by_day: tuple[DrawRecord, ...]
) -> tuple[int, ...]:
    """Rank all numbers by predicted probability and return the top DRAW_SIZE."""
    probabilities = predict_probabilities(model, history_by_day)
    ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    return tuple(sorted(number for number, _ in ranked[:DRAW_SIZE]))


def walk_forward_backtest(history_by_day: tuple[DrawRecord, ...]) -> dict[str, object]:
    """Chronologically retrain the ML model and score each draw it wasn't trained on.

    For every draw after the initial training window, only earlier draws are used
    to produce a prediction for it, so the backtest never sees the future.
    """
    matches: list[int] = []
    eval_start = MIN_TRAINING_DRAWS + 5
    for index in range(eval_start, len(history_by_day)):
        history_before = history_by_day[:index]
        actual = history_by_day[index].numbers
        features, labels = build_training_set(history_before)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(features, labels)
        predicted = predict_ml_line(model, history_before)
        matches.append(len(set(predicted) & set(actual)))

    random_baseline = DRAW_SIZE * DRAW_SIZE / max(NUMBER_RANGE)
    return {
        "predictor": "ml",
        "draws_evaluated": len(matches),
        "average_matches": sum(matches) / len(matches) if matches else 0.0,
        "best_match": max(matches, default=0),
        "match_distribution": matches,
        "random_baseline_average_matches": random_baseline,
    }


def format_backtest(result: dict[str, object]) -> str:
    return (
        f"  {result['predictor']:>9} model: "
        f"avg {result['average_matches']:.2f} / {DRAW_SIZE} matches "
        f"over {result['draws_evaluated']} draws "
        f"(best single draw: {result['best_match']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and predict Multi-Match numbers with ML.")
    parser.add_argument("--day", choices=("Monday", "Thursday"), default="Thursday")
    args = parser.parse_args()

    history = sample_maryland_history()
    model, history_by_day = train_model(history, args.day)

    probabilities = predict_probabilities(model, history_by_day)
    ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    predicted_line = tuple(sorted(number for number, _ in ranked[:DRAW_SIZE]))

    print(f"Multi-Match ML predictor ({args.day})")
    print("=" * 40)
    print(f"Trained on {len(history_by_day)} {args.day} draws.")
    print()
    print(f"Predicted line: {', '.join(str(n) for n in predicted_line)}")
    print()
    print("Top 10 numbers by predicted probability:")
    for number, probability in ranked[:10]:
        print(f"    {number:>2}: {probability:.3f}")
    print()

    print("Walk-forward backtest (chronological, no lookahead):")
    ml_result = walk_forward_backtest(history_by_day)
    print(format_backtest(ml_result))
    print(f"  random baseline:  avg {ml_result['random_baseline_average_matches']:.2f} / {DRAW_SIZE} matches (expected)")
    print()
    print(
        "Note: lottery draws are independent random events. This backtest shows "
        "whether the model beats chance historically, not a guarantee of future results."
    )


if __name__ == "__main__":
    main()
