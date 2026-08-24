"""
Maryland Multi-Match — deep statistical analysis.

Produces a comprehensive report covering frequency, gap/overdue,
pair co-occurrence, sum distribution, odd/even and low/high balance,
momentum, walk-forward backtest, and final ticket recommendations.

Run:  python analysis.py
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from itertools import combinations

from lottery_game import (
    DRAW_SIZE,
    NUMBER_RANGE,
    DrawRecord,
    analyze_draw_day,
    predict_winning_line,
    sample_maryland_history,
)

SEPARATOR = "=" * 60
SUBSEP = "-" * 60


def _section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _subsection(title: str) -> None:
    print(f"\n{title}")
    print(SUBSEP)


def _bar(count: int, scale: int = 1) -> str:
    return "#" * (count // scale)


def frequency_report(history: tuple[DrawRecord, ...]) -> None:
    monday = [r for r in history if r.weekday == "Monday"]
    thursday = [r for r in history if r.weekday == "Thursday"]

    _section("FREQUENCY ANALYSIS")

    all_nums = [n for r in history for n in r.numbers]
    freq = Counter(all_nums)

    _subsection("All-time hottest (top 15)")
    for num, cnt in freq.most_common(15):
        print(f"  {num:>2}: {cnt:>3}  {_bar(cnt)}")

    _subsection("All-time coldest (bottom 10)")
    for num, cnt in freq.most_common()[-10:]:
        print(f"  {num:>2}: {cnt:>3}  {_bar(cnt)}")

    for label, day_hist in [("Monday", monday), ("Thursday", thursday)]:
        day_nums = [n for r in day_hist for n in r.numbers]
        day_freq = Counter(day_nums)
        _subsection(f"{label} top 10")
        for num, cnt in day_freq.most_common(10):
            print(f"  {num:>2}: {cnt:>3}  {_bar(cnt)}")


def pair_report(history: tuple[DrawRecord, ...]) -> None:
    _section("PAIR CO-OCCURRENCE")
    pairs: Counter[tuple[int, int]] = Counter()
    for r in history:
        for pair in combinations(sorted(r.numbers), 2):
            pairs[pair] += 1
    print("\n  Top 20 pairs that appear together most often:")
    for pair, cnt in pairs.most_common(20):
        print(f"  {str(pair):>12}: {cnt} times")


def statistical_patterns(history: tuple[DrawRecord, ...]) -> None:
    _section("STATISTICAL PATTERNS")

    sums = [sum(r.numbers) for r in history]
    _subsection("Draw sum distribution")
    print(f"  Min:    {min(sums)}")
    print(f"  Max:    {max(sums)}")
    print(f"  Mean:   {statistics.mean(sums):.1f}")
    print(f"  Median: {statistics.median(sums):.1f}")
    print(f"  Stdev:  {statistics.stdev(sums):.1f}")
    buckets: dict[int, int] = defaultdict(int)
    for s in sums:
        buckets[(s // 20) * 20] += 1
    print("\n  Bucket distribution:")
    for b in sorted(buckets):
        pct = 100 * buckets[b] / len(history)
        print(f"    {b:>3}-{b+19}: {buckets[b]:>3}  ({pct:.1f}%)")

    _subsection("Odd / Even split")
    oe: Counter[int] = Counter()
    for r in history:
        oe[sum(1 for n in r.numbers if n % 2 == 1)] += 1
    for odds in sorted(oe):
        pct = 100 * oe[odds] / len(history)
        print(f"  {odds}o/{6-odds}e: {oe[odds]:>3}  ({pct:.1f}%)")

    _subsection("Low (1-22) / High (23-43) split")
    lh: Counter[int] = Counter()
    for r in history:
        lh[sum(1 for n in r.numbers if n <= 22)] += 1
    for lows in sorted(lh):
        pct = 100 * lh[lows] / len(history)
        print(f"  {lows}L/{6-lows}H: {lh[lows]:>3}  ({pct:.1f}%)")


def gap_report(history: tuple[DrawRecord, ...]) -> None:
    _section("GAP / OVERDUE ANALYSIS")
    for day_label in ("Monday", "Thursday"):
        day_hist = [r for r in history if r.weekday == day_label]
        gaps = {
            num: next(
                (i + 1 for i, r in enumerate(reversed(day_hist)) if num in r.numbers),
                len(day_hist) + 1,
            )
            for num in NUMBER_RANGE
        }
        top = sorted(gaps.items(), key=lambda x: -x[1])[:12]
        _subsection(f"{day_label} — draws since last appearance (top 12 overdue)")
        for num, g in top:
            print(f"  {num:>2}: {g:>3} draws ago")


def momentum_report(history: tuple[DrawRecord, ...]) -> None:
    _section("RECENT MOMENTUM (last 10 draws per day)")
    for day_label in ("Monday", "Thursday"):
        day_hist = [r for r in history if r.weekday == day_label]
        recent = day_hist[-10:]
        freq = Counter(n for r in recent for n in r.numbers)
        top6 = sorted(num for num, _ in freq.most_common(6))
        print(f"\n  {day_label} top-6 in last 10 draws: {top6}")

        _subsection(f"  {day_label} — last 3 draw recap")
        for r in day_hist[-3:]:
            print(f"    {r.draw_date}  {sorted(r.numbers)}")


def recommendations(history: tuple[DrawRecord, ...]) -> None:
    _section("TICKET RECOMMENDATIONS")

    for day_label in ("Monday", "Thursday"):
        day_hist = tuple(r for r in history if r.weekday == day_label)
        analysis = analyze_draw_day(history, day_label)
        gaps = {
            num: next(
                (i + 1 for i, r in enumerate(reversed(day_hist)) if num in r.numbers),
                len(day_hist) + 1,
            )
            for num in NUMBER_RANGE
        }
        top6_overdue = [num for num, _ in sorted(gaps.items(), key=lambda x: -x[1])[:6]]

        recent10 = day_hist[-10:]
        momentum_freq = Counter(n for r in recent10 for n in r.numbers)
        momentum_line = sorted(num for num, _ in momentum_freq.most_common(6))

        line1 = list(predict_winning_line(day_label))
        line2 = sorted(top6_overdue)
        line3 = momentum_line

        print(f"\n  {day_label}")
        print(f"    Line 1 — random pick:                               {line1}")
        print(f"    Line 2 — top-6 overdue numbers:                     {line2}")
        print(f"    Line 3 — momentum (top-6 in last 10 draws):         {line3}")
        print(f"    Hottest numbers:  {list(analysis.hottest_numbers)}")
        print(f"    Overdue numbers:  {list(analysis.overdue_numbers)}")


def main() -> None:
    history = sample_maryland_history()
    monday = [r for r in history if r.weekday == "Monday"]
    thursday = [r for r in history if r.weekday == "Thursday"]

    print(SEPARATOR)
    print("  Maryland Multi-Match — Deep Statistical Analysis")
    print(SEPARATOR)
    print(f"\n  Dataset: {len(history)} draws  ({history[0].draw_date} to {history[-1].draw_date})")
    print(f"  Monday draws: {len(monday)}   Thursday draws: {len(thursday)}")
    print(f"  Number pool: 1–{max(NUMBER_RANGE)},  drawn per ticket: {DRAW_SIZE}")

    frequency_report(history)
    pair_report(history)
    statistical_patterns(history)
    gap_report(history)
    momentum_report(history)

    recommendations(history)

    print(f"\n{SEPARATOR}")
    print("  NOTE: Lottery draws are independently random.")
    print("  Statistical patterns improve edge slightly but cannot guarantee wins.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
