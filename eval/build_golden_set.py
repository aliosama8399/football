"""
Golden Evaluation Set Builder (Phase 0)
========================================
Generates programmatic and curated ground-truth evaluation datasets
from held-out matches (season 2024-25, Date >= 2024-08-01) with strict
as-of date timestamps to prevent look-ahead data leakage.

Output:
  eval/golden/rag_golden_set.json
  eval/golden/slm_golden_set.json

Usage:
  python eval/build_golden_set.py --count 200
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "processed_matches.csv"
OUT_DIR = BASE_DIR / "eval" / "golden"

# Curated tactical query templates
TACTICAL_TEMPLATES = [
    ("How does {team} set up their offensive transition against high-pressing teams?", "attack_tactic"),
    ("What are the defensive vulnerabilities of {team} when defending set pieces and wide crosses?", "defense_tactic"),
    ("Analyze the tactical matchup between {home} and {away} in terms of pressing intensity and shape.", "matchup"),
    ("How does {team}'s defensive structure change when playing away from home?", "defense_structure"),
    ("What key tactical instructions should a coach issue when attacking against {team}'s backline?", "coach_advice")
]


def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Matches CSV missing at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_h2h_item(idx: int, match: pd.Series, history_df: pd.DataFrame) -> dict:
    home, away = match["HomeTeam"], match["AwayTeam"]
    as_of = match["Date"].strftime("%Y-%m-%d")

    h2h_matches = history_df[
        ((history_df["HomeTeam"] == home) & (history_df["AwayTeam"] == away)) |
        ((history_df["HomeTeam"] == away) & (history_df["AwayTeam"] == home))
    ]

    home_wins = len(h2h_matches[
        ((h2h_matches["HomeTeam"] == home) & (h2h_matches["FTR"] == "H")) |
        ((h2h_matches["AwayTeam"] == home) & (h2h_matches["FTR"] == "A"))
    ])
    away_wins = len(h2h_matches[
        ((h2h_matches["HomeTeam"] == away) & (h2h_matches["FTR"] == "H")) |
        ((h2h_matches["AwayTeam"] == away) & (h2h_matches["FTR"] == "A"))
    ])
    draws = len(h2h_matches[h2h_matches["FTR"] == "D"])

    query = f"What is the head-to-head record between {home} and {away} prior to {as_of}?"
    ground_truth = (
        f"Prior to {as_of}, {home} and {away} played {len(h2h_matches)} matches. "
        f"{home} won {home_wins}, {away} won {away_wins}, and {draws} ended in a draw."
    )

    return {
        "id": f"h2h_{idx:03d}",
        "type": "h2h",
        "as_of_date": as_of,
        "query": query,
        "teams": [home, away],
        "ground_truth": ground_truth,
        "expected_numbers": {
            "total_matches": len(h2h_matches),
            "home_wins": home_wins,
            "away_wins": away_wins,
            "draws": draws,
        }
    }


def build_form_item(idx: int, match: pd.Series, history_df: pd.DataFrame) -> dict:
    home = match["HomeTeam"]
    as_of = match["Date"].strftime("%Y-%m-%d")

    team_matches = history_df[
        (history_df["HomeTeam"] == home) | (history_df["AwayTeam"] == home)
    ].tail(5)

    results = []
    gf, ga = 0, 0
    for _, m in team_matches.iterrows():
        is_h = (m["HomeTeam"] == home)
        fthg, ftag = int(m["FTHG"]), int(m["FTAG"])
        if is_h:
            gf += fthg
            ga += ftag
            results.append("W" if m["FTR"] == "H" else ("D" if m["FTR"] == "D" else "L"))
        else:
            gf += ftag
            ga += fthg
            results.append("W" if m["FTR"] == "A" else ("D" if m["FTR"] == "D" else "L"))

    form_str = "-".join(results) if results else "N/A"
    query = f"What was {home}'s 5-match form leading into their match on {as_of}?"
    ground_truth = f"In the 5 matches prior to {as_of}, {home} had a record of {form_str}, scoring {gf} goals and conceding {ga}."

    return {
        "id": f"form_{idx:03d}",
        "type": "recent_form",
        "as_of_date": as_of,
        "query": query,
        "teams": [home],
        "ground_truth": ground_truth,
        "expected_numbers": {
            "goals_for": gf,
            "goals_against": ga,
        }
    }


def build_prediction_item(idx: int, match: pd.Series) -> dict:
    home, away = match["HomeTeam"], match["AwayTeam"]
    as_of = match["Date"].strftime("%Y-%m-%d")
    result = match["FTR"]
    fthg, ftag = int(match["FTHG"]), int(match["FTAG"])

    query = f"Predict the tactical outcome and key performance factors for {home} vs {away} on {as_of}."
    ground_truth = (
        f"Match played on {as_of} ended with {home} {fthg}-{ftag} {away} (Outcome: {result})."
    )

    return {
        "id": f"pred_{idx:03d}",
        "type": "prediction",
        "as_of_date": as_of,
        "query": query,
        "teams": [home, away],
        "ground_truth": ground_truth,
        "actual_result": result,
        "actual_score": f"{fthg}-{ftag}"
    }


def main(target_count: int = 200):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # Split: held-out matches from season 2024-25
    held_out = df[df["Date"] >= "2024-08-01"].copy()
    if held_out.empty:
        # Fallback to the last 15% of records
        split_idx = int(len(df) * 0.85)
        held_out = df.iloc[split_idx:].copy()

    logger.info("Found %d held-out test matches.", len(held_out))

    sample_size = min(len(held_out), max(1, -(-target_count // 3)))
    sampled = held_out.sample(n=sample_size, random_state=42).sort_values("Date")

    rag_items = []
    item_idx = 1

    for _, match in sampled.iterrows():
        history = df[df["Date"] < match["Date"]]

        # 1. Head-to-Head
        rag_items.append(build_h2h_item(item_idx, match, history))
        item_idx += 1

        # 2. Recent form
        rag_items.append(build_form_item(item_idx, match, history))
        item_idx += 1

        # 3. Match prediction
        rag_items.append(build_prediction_item(item_idx, match))
        item_idx += 1

    # 4. Add curated tactical queries
    teams = list(held_out["HomeTeam"].drop_duplicates().sample(
        n=min(15, held_out["HomeTeam"].nunique()), random_state=42))
    for i, t in enumerate(teams):
        q = f"Describe {t}'s primary attacking pattern, pressing intensity, and defensive vulnerabilities."
        rag_items.append({
            "id": f"tactic_{i+1:03d}",
            "type": "tactical_profile",
            "as_of_date": "2024-08-01",
            "query": q,
            "teams": [t],
            "ground_truth": f"Grounded in {t}'s tactical profile, set piece stats, and season aggregates."
        })

    rag_path = OUT_DIR / "rag_golden_set.json"
    with open(rag_path, "w", encoding="utf-8") as f:
        json.dump(rag_items, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d golden RAG evaluation items → %s", len(rag_items), rag_path)

    # Build SLM validation set sample
    slm_items = []
    val_json_path = BASE_DIR / "data" / "finetune" / "football_val.json"
    if val_json_path.exists():
        with open(val_json_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        slm_items = val_data[:100]  # Take 100 representative held-out val samples
    else:
        # Synthesize from sampled matches
        for _, m in sampled.head(50).iterrows():
            slm_items.append({
                "instruction": f"Generate a coach tactical summary for {m['HomeTeam']} vs {m['AwayTeam']}.",
                "input": f"Date: {m['Date'].strftime('%Y-%m-%d')}, League: {m['League']}, Score: {m['FTHG']}-{m['FTAG']}",
                "expected_schema": ["match_state", "analysis", "coach_recommendations"]
            })

    slm_path = OUT_DIR / "slm_golden_set.json"
    with open(slm_path, "w", encoding="utf-8") as f:
        json.dump(slm_items, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d golden SLM evaluation items → %s", len(slm_items), slm_path)
    logger.info("✅ Phase 0 Golden Set build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Golden Evaluation Set")
    parser.add_argument("--count", type=int, default=200, help="Target question count")
    args = parser.parse_args()
    main(target_count=args.count)
