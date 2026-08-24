import unittest
from datetime import date
from unittest.mock import patch

from lottery_game import (
    DRAW_SIZE,
    DrawRecord,
    analyze_draw_day,
    build_ticket,
    evaluate_ticket,
    find_actual_result,
    generate_smart_tickets,
    next_draw_day,
    normalize_line,
    predict_winning_line,
    sample_maryland_history,
)


class _FixedDate(date):
    _fixed: date

    @classmethod
    def today(cls) -> date:
        return cls._fixed


class LotteryGameTests(unittest.TestCase):
    def test_draw_record_rejects_invalid_draw_day(self) -> None:
        with self.assertRaisesRegex(ValueError, "Monday and Thursday"):
            DrawRecord.fromisoformat("2026-05-19", (1, 2, 3, 4, 5, 6))

    def test_predict_winning_line_returns_a_valid_line(self) -> None:
        line = predict_winning_line("Monday", seed=1)

        self.assertEqual(normalize_line(line), line)
        self.assertEqual(len(line), DRAW_SIZE)

    def test_predict_winning_line_is_reproducible_with_a_seed(self) -> None:
        self.assertEqual(predict_winning_line("Thursday", seed=7), predict_winning_line("Thursday", seed=7))

    def test_predict_winning_line_rejects_invalid_draw_day(self) -> None:
        with self.assertRaisesRegex(ValueError, "Monday or Thursday"):
            predict_winning_line("Tuesday")

    def test_thursday_analysis_returns_valid_line_and_historical_stats(self) -> None:
        history = sample_maryland_history()

        analysis = analyze_draw_day(history, "Thursday", seed=1)

        self.assertEqual(analysis.recommended_line, predict_winning_line("Thursday", seed=1))
        self.assertEqual(analysis.hottest_numbers, (11, 12, 16, 23, 24, 41))
        self.assertEqual(len(analysis.overdue_numbers), DRAW_SIZE)

    def test_build_ticket_is_reproducible_with_a_seed(self) -> None:
        ticket = build_ticket("Thursday", seed=1)

        self.assertEqual(ticket, build_ticket("Thursday", seed=1))
        self.assertEqual(len(ticket), 3)
        seen: set[tuple[int, ...]] = set()
        for line in ticket:
            self.assertEqual(len(line), DRAW_SIZE)
            self.assertEqual(line, tuple(sorted(line)))
            self.assertNotIn(line, seen)
            seen.add(line)

    def test_next_draw_day_reports_today_when_today_is_a_draw_day(self) -> None:
        fixed = type("_Monday", (_FixedDate,), {"_fixed": date(2026, 7, 27)})  # Monday
        with patch("lottery_game.date", fixed):
            name, draw_date = next_draw_day()

        self.assertEqual((name, draw_date), ("Monday", date(2026, 7, 27)))

    def test_next_draw_day_skips_ahead_to_the_nearest_draw_day(self) -> None:
        fixed = type("_Wednesday", (_FixedDate,), {"_fixed": date(2026, 7, 29)})  # Wednesday
        with patch("lottery_game.date", fixed):
            name, draw_date = next_draw_day()

        self.assertEqual((name, draw_date), ("Thursday", date(2026, 7, 30)))

    def test_generate_smart_tickets_returns_three_valid_tickets(self) -> None:
        tickets = generate_smart_tickets("Thursday", seed=1)

        self.assertEqual(len(tickets), 3)
        for ticket in tickets:
            self.assertEqual(len(ticket), 3)
            lines_in_ticket: set[tuple[int, ...]] = set()
            for line in ticket:
                self.assertEqual(normalize_line(line), line)
                self.assertEqual(len(line), DRAW_SIZE)
                self.assertNotIn(line, lines_in_ticket)
                lines_in_ticket.add(line)

    def test_generate_smart_tickets_varies_between_runs(self) -> None:
        first = generate_smart_tickets("Thursday")
        second = generate_smart_tickets("Thursday")

        self.assertNotEqual(first, second)

    def test_find_actual_result_returns_published_record_for_the_draw_day_and_date(self) -> None:
        history = (
            DrawRecord(date(2026, 7, 27), (2, 24, 32, 33, 37, 39)),
            DrawRecord(date(2026, 7, 30), (1, 2, 3, 4, 5, 6)),
        )

        result = find_actual_result(history, "Thursday", date(2026, 7, 30))

        self.assertEqual(result, history[1])

    def test_find_actual_result_returns_none_when_draw_not_yet_published(self) -> None:
        history = (DrawRecord(date(2026, 7, 27), (2, 24, 32, 33, 37, 39)),)

        result = find_actual_result(history, "Thursday", date(2026, 7, 30))

        self.assertIsNone(result)

    def test_find_actual_result_ignores_same_date_on_a_different_weekday(self) -> None:
        history = (DrawRecord(date(2026, 7, 30), (1, 2, 3, 4, 5, 6)),)

        result = find_actual_result(history, "Monday", date(2026, 7, 30))

        self.assertIsNone(result)

    def test_evaluate_ticket_counts_matches_per_line(self) -> None:
        ticket = (
            (1, 2, 3, 4, 5, 6),
            (7, 8, 9, 10, 11, 12),
            (13, 14, 15, 16, 17, 18),
        )

        result = evaluate_ticket(ticket, (1, 3, 5, 7, 9, 11))

        self.assertEqual(result["per_line_matches"], (3, 3, 0))
        self.assertEqual(result["best_line_match_count"], 3)
        self.assertEqual(result["total_matched_numbers"], 6)


if __name__ == "__main__":
    unittest.main()
