#!/usr/bin/env python3
"""
Powerball data updater and dynamic number suggester.
Uses the public NY Open Data API (no key required) for recent results.
"""

from collections import Counter

import pandas as pd # type: ignore
import requests # type: ignore

EXCEL_FILE = "powerball_game.xlsx"

# NY State Gaming Commission – Powerball Winning Numbers: Beginning 2010
NY_API_URL = (
    "https://data.ny.gov/resource/d6yy-54nr.json"
    "?$order=draw_date%20DESC&$limit=100"
)


def fetch_latest_powerball() -> list[dict]:
    """Fetch the 50 most recent Powerball draws from NY Open Data."""
    resp = requests.get(NY_API_URL, timeout=10)
    resp.raise_for_status()
    rows = []
    for item in resp.json():
        date = item.get("draw_date", "")[:10]           # "YYYY-MM-DD"
        winning = item.get("winning_numbers", "")        # "n1 n2 n3 n4 n5 pb"
        parts = winning.split()
        if len(parts) < 6:
            continue
        nums = [int(p) for p in parts[:5]]
        pb = int(parts[5])
        multiplier = item.get("multiplier")
        dt = pd.to_datetime(date)
        rows.append({
            "GameName":   "Powerball",
            "Month":      dt.month,
            "Day":        dt.day,
            "Year":       dt.year,
            "Num1":       nums[0],
            "Num2":       nums[1],
            "Num3":       nums[2],
            "Num4":       nums[3],
            "Num5":       nums[4],
            "powerball":  pb,
            "Power Play": int(multiplier) if multiplier else None,
        })
    return rows


def update_excel(filename: str = EXCEL_FILE) -> None:
    """Append any draws not yet in the Excel file."""
    df = pd.read_excel(filename)
    new_rows = fetch_latest_powerball()
    added = 0
    for row in new_rows:
        mask = (
            (df["Year"]      == row["Year"])
            & (df["Month"]   == row["Month"])
            & (df["Day"]     == row["Day"])
            & (df["powerball"] == row["powerball"])
        )
        if not mask.any():
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            added += 1
    if added:
        df.to_excel(filename, index=False)
        print(f"Excel updated — {added} new draw(s) added.")
    else:
        print("No new results to add.")
    return df


def suggest_numbers(df: pd.DataFrame, strategy: str = "hot") -> tuple[list[int], int]:
    """
    Suggest numbers based on historical frequency.

    strategy:
      'hot'   – top 5 most-frequent main numbers + hottest powerball
      'fresh' – hot numbers that were NOT drawn in the last 3 games
    """
    recent = df[pd.to_datetime(df[["Year", "Month", "Day"]].rename(
        columns={"Year": "year", "Month": "month", "Day": "day"})) >= "2024-01-01"]

    main_cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]
    main_freq = Counter(
        int(v) for col in main_cols for v in recent[col].dropna()
    )
    pb_freq = Counter(int(v) for v in recent["powerball"].dropna())

    hot_main = [n for n, _ in main_freq.most_common(10)]
    hot_pb   = pb_freq.most_common(1)[0][0]

    if strategy == "hot":
        main = sorted(hot_main[:5])
        pb   = hot_pb
    else:  # fresh – exclude last 3 draws
        last3: set[int] = set()
        for _, row in df.tail(3).iterrows():
            for c in main_cols:
                last3.add(int(row[c]))
        fresh = [n for n in hot_main if n not in last3]
        while len(fresh) < 5:
            for n in hot_main:
                if n not in fresh:
                    fresh.append(n)
                if len(fresh) >= 5:
                    break
        main = sorted(fresh[:5])
        pb   = pb_freq.most_common(2)[1][0]

    return main, pb


def display_suggestion(label: str, main: list[int], pb: int) -> None:
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  Main Numbers : {' - '.join(map(str, main))}")
    print(f"  Powerball    : {pb}")
    print(f"{'='*52}")


if __name__ == "__main__":
    print("Fetching latest Powerball results…")
    try:
        df = update_excel(EXCEL_FILE)
    except Exception as exc:
        print(f"Could not fetch live data ({exc}). Using existing file.")
        df = pd.read_excel(EXCEL_FILE)

    last = df.sort_values(["Year", "Month", "Day"]).iloc[-1]
    print(
        f"\nMost recent draw on file: "
        f"{int(last['Month'])}/{int(last['Day'])}/{int(last['Year'])}  "
        f"Nums: {int(last['Num1'])} {int(last['Num2'])} "
        f"{int(last['Num3'])} {int(last['Num4'])} {int(last['Num5'])}  "
        f"PB: {int(last['powerball'])}"
    )

    print("\n=== DYNAMIC SUGGESTIONS FOR TODAY ===")
    hot_main, hot_pb     = suggest_numbers(df, strategy="hot")
    fresh_main, fresh_pb = suggest_numbers(df, strategy="fresh")

    display_suggestion("Strategy 1 – HOT (most frequent since 2024)", hot_main, hot_pb)
    display_suggestion("Strategy 2 – HOT + FRESH (skip last 3 draws)", fresh_main, fresh_pb)

    print("\nDisclaimer: These are statistically-informed suggestions, not guarantees.")
