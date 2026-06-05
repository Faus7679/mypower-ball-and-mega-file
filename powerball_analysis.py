#!/usr/bin/env python3
"""
Deep Powerball statistical analysis.
Fetches all available draws from NY Open Data, merges with local Excel,
then runs:
  - Full-history frequency ranking
  - Recent-trend (last 52 draws / ~1 year) frequency
  - Overdue / cold number detection
  - Pair co-occurrence (top 20)
  - Powerball ball frequency (1-26)
  - Day-of-week bias
  - Composite HOT score and final recommendations
"""

import sys
import io
from collections import Counter, defaultdict

import requests
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── NY Open Data – Powerball Winning Numbers (beginning 2010) ────────────────
PB_API = (
    "https://data.ny.gov/resource/d6yy-54nr.json"
    "?$order=draw_date%20DESC&$limit=5000"
)

EXCEL_FILE = "powerball_game.xlsx"
MAIN_COLS  = ["Num1", "Num2", "Num3", "Num4", "Num5"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA FETCH & MERGE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_powerball_api() -> pd.DataFrame:
    print("Fetching Powerball draw history from NY Open Data …")
    resp = requests.get(PB_API, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for item in data:
        raw_date   = item.get("draw_date", "")[:10]
        winning    = item.get("winning_numbers", "")
        multiplier = item.get("multiplier", None)

        parts = winning.split()
        if len(parts) < 6:
            continue
        try:
            nums = [int(p) for p in parts[:5]]
            pb   = int(parts[5])
            dt   = pd.to_datetime(raw_date)
        except (ValueError, TypeError):
            continue

        rows.append({
            "draw_date":  dt,
            "Year":  dt.year,
            "Month": dt.month,
            "Day":   dt.day,
            "DOW":   dt.strftime("%A"),
            "Num1":  nums[0],
            "Num2":  nums[1],
            "Num3":  nums[2],
            "Num4":  nums[3],
            "Num5":  nums[4],
            "Powerball":  pb,
            "Multiplier": multiplier,
        })

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} draws from API")
    return df


def load_excel() -> pd.DataFrame:
    try:
        raw = pd.read_excel(EXCEL_FILE)
        rows = []
        for _, r in raw.iterrows():
            try:
                dt = pd.to_datetime(
                    f"{int(r['Year'])}-{int(r['Month']):02d}-{int(r['Day']):02d}"
                )
            except Exception:
                continue
            rows.append({
                "draw_date":  dt,
                "Year":  int(r["Year"]),
                "Month": int(r["Month"]),
                "Day":   int(r["Day"]),
                "DOW":   dt.strftime("%A"),
                "Num1":  int(r["Num1"]),
                "Num2":  int(r["Num2"]),
                "Num3":  int(r["Num3"]),
                "Num4":  int(r["Num4"]),
                "Num5":  int(r["Num5"]),
                "Powerball":  int(r["powerball"]),
                "Multiplier": r.get("Power Play", None),
            })
        df = pd.DataFrame(rows)
        print(f"  → {len(df)} draws from Excel ({EXCEL_FILE})")
        return df
    except Exception as exc:
        print(f"  ⚠  Could not load Excel: {exc}")
        return pd.DataFrame()


def load_data() -> pd.DataFrame:
    api_df   = fetch_powerball_api()
    excel_df = load_excel()

    combined = pd.concat([api_df, excel_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Year", "Month", "Day", "Powerball"])
    combined = combined.sort_values("draw_date").reset_index(drop=True)

    print(f"  → {len(combined)} unique draws after merge  "
          f"({combined['draw_date'].min().date()} – {combined['draw_date'].max().date()})")
    combined.to_excel("powerball_draws_full.xlsx", index=False)
    print(f"  → Full dataset saved to powerball_draws_full.xlsx")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def all_main_numbers(df: pd.DataFrame) -> list[int]:
    return [int(v) for col in MAIN_COLS for v in df[col].dropna()]


def section(title: str, width: int = 62):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def bar(count: int, total: int, width: int = 30) -> str:
    filled = int(round(count / total * width)) if total else 0
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FULL-HISTORY FREQUENCY  (main numbers 1-69)
# ─────────────────────────────────────────────────────────────────────────────

def full_frequency(df: pd.DataFrame) -> Counter:
    section("ALL-TIME MAIN NUMBER FREQUENCY  (top 25 / bottom 10)")
    nums  = all_main_numbers(df)
    total = len(df)
    c     = Counter(nums)
    expected = total * 5 / 69

    print(f"  Total draws analysed: {total:,}  |  Expected per number: {expected:.1f}")
    print(f"\n  {'#':>3}  {'Count':>6}  {'% draws':>8}  {'vs avg':>8}  Bar")
    print(f"  {'─'*3}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*30}")

    top_count = c.most_common(1)[0][1]
    for num, cnt in c.most_common(25):
        pct   = cnt / total * 100
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {num:>3}  {cnt:>6}  {pct:>7.1f}%  {sign}{delta:>6.1f}%  {bar(cnt, top_count)}")

    print(f"\n  ── COLDEST 10 ──")
    for num, cnt in c.most_common()[:-11:-1]:
        pct   = cnt / total * 100
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {num:>3}  {cnt:>6}  {pct:>7.1f}%  {sign}{delta:>6.1f}%  {bar(cnt, top_count)}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECENT-TREND FREQUENCY  (last 52 draws ≈ 1 year)
# ─────────────────────────────────────────────────────────────────────────────

def recent_frequency(df: pd.DataFrame, n: int = 52) -> Counter:
    section(f"RECENT TREND – last {n} draws  (hottest 20)")
    recent = df.tail(n)
    nums   = all_main_numbers(recent)
    c      = Counter(nums)

    top_count = c.most_common(1)[0][1]
    print(f"  {'#':>3}  {'Count':>6}  Bar")
    print(f"  {'─'*3}  {'─'*6}  {'─'*30}")
    for num, cnt in c.most_common(20):
        print(f"  {num:>3}  {cnt:>6}  {bar(cnt, top_count)}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OVERDUE NUMBERS  (draws since last appearance)
# ─────────────────────────────────────────────────────────────────────────────

def overdue_numbers(df: pd.DataFrame) -> dict:
    section("OVERDUE NUMBERS  (draws since last seen – top 15)")
    last_seen: dict[int, int] = {}
    for idx, row in df.iterrows():
        for col in MAIN_COLS:
            n = int(row[col])
            last_seen[n] = idx

    total = len(df) - 1
    gap   = {n: total - idx for n, idx in last_seen.items()}

    for n in range(1, 70):
        if n not in gap:
            gap[n] = total + 1

    top15   = sorted(gap.items(), key=lambda x: -x[1])[:15]
    max_gap = top15[0][1]

    print(f"  {'#':>3}  {'Draws ago':>10}  Bar")
    print(f"  {'─'*3}  {'─'*10}  {'─'*30}")
    for num, g in top15:
        print(f"  {num:>3}  {g:>10}  {bar(g, max_gap)}")

    return gap


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PAIR CO-OCCURRENCE  (top 20 most common pairs)
# ─────────────────────────────────────────────────────────────────────────────

def pair_cooccurrence(df: pd.DataFrame) -> Counter:
    section("TOP 20 MOST COMMON PAIRS  (appear together most often)")
    pair_counts: Counter = Counter()
    for _, row in df.iterrows():
        nums = sorted(int(row[col]) for col in MAIN_COLS)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counts[(nums[i], nums[j])] += 1

    top_count = pair_counts.most_common(1)[0][1]
    print(f"  {'Pair':>12}  {'Count':>6}  Bar")
    print(f"  {'─'*12}  {'─'*6}  {'─'*30}")
    for pair, cnt in pair_counts.most_common(20):
        print(f"  {str(pair):>12}  {cnt:>6}  {bar(cnt, top_count)}")

    return pair_counts


# ─────────────────────────────────────────────────────────────────────────────
# 7.  POWERBALL BALL FREQUENCY  (1-26)
# ─────────────────────────────────────────────────────────────────────────────

def powerball_frequency(df: pd.DataFrame) -> Counter:
    section("POWERBALL BALL FREQUENCY  (all 26 balls ranked)")
    pb_draws = df[df["Powerball"].notna()]
    c        = Counter(int(v) for v in pb_draws["Powerball"])
    total    = len(pb_draws)
    expected = total / 26

    print(f"  Total draws with Powerball: {total:,}  |  Expected per ball: {expected:.1f}")
    print(f"\n  {'Ball':>5}  {'Count':>6}  {'vs avg':>8}  Bar")
    print(f"  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*30}")
    top_count = c.most_common(1)[0][1]
    for ball, cnt in c.most_common():
        delta = (cnt - expected) / expected * 100
        sign  = "+" if delta >= 0 else ""
        print(f"  {ball:>5}  {cnt:>6}  {sign}{delta:>6.1f}%  {bar(cnt, top_count)}")

    return c


# ─────────────────────────────────────────────────────────────────────────────
# 8.  DAY-OF-WEEK BIAS
# ─────────────────────────────────────────────────────────────────────────────

def dow_bias(df: pd.DataFrame):
    section("DAY-OF-WEEK BIAS  (draw days and average number drawn)")
    dow_totals: dict[str, list[int]] = defaultdict(list)
    for _, row in df.iterrows():
        nums = [int(row[col]) for col in MAIN_COLS if pd.notna(row[col])]
        dow_totals[row["DOW"]].extend(nums)

    order = ["Monday", "Wednesday", "Saturday", "Sunday",
             "Tuesday", "Thursday", "Friday"]
    print(f"  {'Day':>12}  {'Draws':>6}  {'Avg number':>11}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*11}")
    for day in order:
        if day in dow_totals:
            vals  = dow_totals[day]
            draws = len(vals) // 5
            avg   = sum(vals) / len(vals)
            print(f"  {day:>12}  {draws:>6}  {avg:>11.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  COMPOSITE HOT SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite_hot_score(
    full_c:   Counter,
    recent_c: Counter,
    gap:      dict,
    pair_c:   Counter,
    df:       pd.DataFrame,
) -> list[tuple[int, float]]:
    section("COMPOSITE HOT-SCORE  (weighted ranking – top 25)")

    pair_score: dict[int, int] = Counter()
    for (a, b), cnt in pair_c.items():
        pair_score[a] += cnt
        pair_score[b] += cnt
    max_pair = max(pair_score.values()) if pair_score else 1
    max_gap  = max(gap.values()) if gap else 1

    recent_order = {n: rank for rank, (n, _) in enumerate(recent_c.most_common())}
    full_order   = {n: rank for rank, (n, _) in enumerate(full_c.most_common())}

    scores: list[tuple[int, float]] = []
    for n in range(1, 70):
        r_rank = recent_order.get(n, len(recent_c))
        f_rank = full_order.get(n, len(full_c))
        p_sc   = pair_score.get(n, 0) / max_pair
        g      = gap.get(n, max_gap) / max_gap

        score = (
            0.40 * (1 - r_rank / 69) +
            0.25 * (1 - f_rank / 69) +
            0.20 * p_sc +
            0.15 * g
        )
        scores.append((n, round(score, 4)))

    scores.sort(key=lambda x: -x[1])

    top_score = scores[0][1]
    print(f"\n  {'#':>3}  {'Score':>7}  Bar")
    print(f"  {'─'*3}  {'─'*7}  {'─'*30}")
    for num, sc in scores[:25]:
        print(f"  {num:>3}  {sc:>7.4f}  {bar(int(sc * 1000), int(top_score * 1000))}")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 10.  FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def final_recommendations(
    scores:   list[tuple[int, float]],
    pb_c:     Counter,
    pair_c:   Counter,
    recent_c: Counter,
    gap:      dict,
):
    section("★  FINAL RECOMMENDED TICKETS  ★")

    # Ticket 1 – pure HOT (top-5 composite)
    t1_main = sorted([n for n, _ in scores[:5]])
    t1_pb   = pb_c.most_common(1)[0][0]

    # Ticket 2 – HOT + OVERDUE (top-10 score, sorted by gap)
    top10   = [n for n, _ in scores[:10]]
    t2_main = sorted(sorted(top10, key=lambda n: -gap.get(n, 0))[:5])
    t2_pb   = pb_c.most_common(3)[1][0]

    # Ticket 3 – PAIR ANCHOR
    best_pair  = pair_c.most_common(1)[0][0]
    t3_pool    = list(best_pair)
    for n, _ in scores:
        if n not in t3_pool:
            t3_pool.append(n)
        if len(t3_pool) >= 5:
            break
    t3_main = sorted(t3_pool[:5])
    t3_pb   = pb_c.most_common(3)[2][0]

    # Ticket 4 – RECENT SURGE
    t4_main = sorted([n for n, _ in recent_c.most_common(5)])
    t4_pb   = pb_c.most_common(2)[0][0]

    # Ticket 5 – BALANCED + WILDCARD (most overdue valid number)
    wildcard = max(gap, key=gap.get)
    t5_pool  = [n for n, _ in scores[:4]] + [wildcard]
    t5_main  = sorted(set(t5_pool))[:5]
    t5_pb    = pb_c.most_common(5)[4][0]

    def show(label, main, pb, note=""):
        pad = 54 - len(label)
        print(f"\n  ┌─ {label} {'─'*pad}┐")
        print(f"  │  Main  :  {' – '.join(f'{n:2d}' for n in main):<40}  │")
        print(f"  │  Powerball:  {pb:<2}                                    │")
        if note:
            print(f"  │  Note  : {note:<43}  │")
        print(f"  └{'─'*56}┘")

    show("Ticket 1 – PURE HOT",          t1_main, t1_pb, "Top-5 composite score")
    show("Ticket 2 – HOT + OVERDUE",     t2_main, t2_pb, "Top-10 score sorted by gap")
    show("Ticket 3 – PAIR ANCHOR",       t3_main, t3_pb, f"Anchored on most common pair {best_pair}")
    show("Ticket 4 – RECENT SURGE",      t4_main, t4_pb, "Hottest numbers in last 52 draws")
    show("Ticket 5 – BALANCED WILDCARD", t5_main, t5_pb, f"Top-4 hot + wildcard #{wildcard} (most overdue)")

    print(f"\n  ─── POWERBALL SHORTLIST  (by frequency) ───")
    for ball, cnt in pb_c.most_common(5):
        print(f"    Ball {ball:>2}  appeared {cnt}×")

    print("""
  ─────────────────────────────────────────────────
  DISCLAIMER
  Statistical frequency analysis does NOT predict
  future lottery draws.  Each draw is independent.
  Play responsibly. Good luck!
  ─────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            POWERBALL  DEEP STATISTICAL ANALYSIS          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        df = load_data()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import sys; sys.exit(1)

    if df.empty:
        print("No data retrieved. Exiting.")
        import sys; sys.exit(1)

    full_c   = full_frequency(df)
    recent_c = recent_frequency(df, n=52)
    gap      = overdue_numbers(df)
    pair_c   = pair_cooccurrence(df)
    pb_c     = powerball_frequency(df)
    dow_bias(df)
    scores   = composite_hot_score(full_c, recent_c, gap, pair_c, df)

    final_recommendations(scores, pb_c, pair_c, recent_c, gap)


if __name__ == "__main__":
    main()
