import unittest

from lottery_game import DRAW_SIZE, NUMBER_RANGE, sample_maryland_history
from ml_predictor import (
    build_feature_row,
    build_training_set,
    predict_ml_line,
    predict_probabilities,
    train_model,
    walk_forward_backtest,
)


class MlPredictorTests(unittest.TestCase):
    def test_build_feature_row_on_empty_history(self) -> None:
        row = build_feature_row((), 7)

        self.assertEqual(len(row), 6)
        self.assertEqual(row[1], 0.0)  # career_frequency
        self.assertEqual(row[3], 0.0)  # hits_last_5

    def test_build_training_set_shapes_match(self) -> None:
        history_by_day = tuple(
            record for record in sample_maryland_history() if record.weekday == "Thursday"
        )

        features, labels = build_training_set(history_by_day)

        expected_rows = (len(history_by_day) - 8) * len(NUMBER_RANGE)
        self.assertEqual(features.shape, (expected_rows, 6))
        self.assertEqual(labels.shape, (expected_rows,))
        self.assertEqual(set(labels.tolist()), {0, 1})

    def test_train_model_predicts_a_valid_line(self) -> None:
        history = sample_maryland_history()

        model, history_by_day = train_model(history, "Thursday")
        line = predict_ml_line(model, history_by_day)

        self.assertEqual(len(line), DRAW_SIZE)
        self.assertEqual(line, tuple(sorted(set(line))))
        for number in line:
            self.assertIn(number, NUMBER_RANGE)

    def test_predict_probabilities_covers_all_numbers(self) -> None:
        history = sample_maryland_history()

        model, history_by_day = train_model(history, "Monday")
        probabilities = predict_probabilities(model, history_by_day)

        self.assertEqual(set(probabilities.keys()), set(NUMBER_RANGE))
        for probability in probabilities.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_train_model_rejects_insufficient_history(self) -> None:
        history = sample_maryland_history()
        short_history = tuple(
            record for record in history if record.weekday == "Thursday"
        )[:5]

        with self.assertRaisesRegex(ValueError, "Need more than"):
            train_model(short_history, "Thursday")

    def test_walk_forward_backtest_returns_a_result_per_draw(self) -> None:
        history_by_day = tuple(
            record for record in sample_maryland_history() if record.weekday == "Thursday"
        )

        result = walk_forward_backtest(history_by_day)

        self.assertEqual(result["draws_evaluated"], len(result["match_distribution"]))
        self.assertGreater(result["draws_evaluated"], 0)
        for match_count in result["match_distribution"]:
            self.assertGreaterEqual(match_count, 0)
            self.assertLessEqual(match_count, DRAW_SIZE)


if __name__ == "__main__":
    unittest.main()
