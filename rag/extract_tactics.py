"""
Tactical Extractor
====================
Parses football_tactical.jsonl and extracts per-team attack/defense tactics.

Each JSONL row has:
  - messages[0].content : the user prompt (match stats)
  - messages[1].content : the LLM assistant analysis

This script scans the analysis for paragraphs that clearly describe either
the home team's or away team's play style (attack / defense), then
aggregates them by team name.

Output: rag/knowledge_base/team_tactics.json
  {
    "Arsenal": {
      "attack_tactic":  "...",
      "defense_tactic": "..."
    },
    ...
  }

Usage:
    python rag/extract_tactics.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
JSONL_PATH   = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"
OUT_DIR      = BASE_DIR / "rag" / "knowledge_base"
OUT_PATH     = OUT_DIR / "team_tactics.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Keywords for attack / defense detection ───────────────────────────────────
ATTACK_KEYWORDS = [
    "attack", "offensive", "scoring", "goals", "shots", "press",
    "high press", "forward", "striker", "chance creation", "xg",
    "expected goals", "penetrat", "build-up", "possession", "dribbl",
]
DEFENSE_KEYWORDS = [
    "defense", "defensive", "defending", "backline", "clean sheet",
    "concede", "tackle", "interception", "block", "goalkeeper", "compact",
    "low block", "counter-press", "transition", "vulnerable", "defensive shape",
]


def _score_paragraph(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in a paragraph (case-insensitive)."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _extract_team_sections(analysis: str, home_team: str, away_team: str) -> dict:
    """
    Split an analysis into paragraphs and assign each to home/away.
    Returns {home_team: {attack, defense}, away_team: {attack, defense}}.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", analysis) if len(p.strip()) > 80]

    home_attack, home_defense = [], []
    away_attack, away_defense = [], []

    for para in paragraphs:
        para_lower = para.lower()
        is_home = home_team.lower() in para_lower
        is_away = away_team.lower() in para_lower

        atk_score = _score_paragraph(para, ATTACK_KEYWORDS)
        def_score = _score_paragraph(para, DEFENSE_KEYWORDS)

        # Assign paragraph to most likely category + team
        if is_home and not is_away:
            if atk_score >= def_score and atk_score > 0:
                home_attack.append(para)
            elif def_score > 0:
                home_defense.append(para)
        elif is_away and not is_home:
            if atk_score >= def_score and atk_score > 0:
                away_attack.append(para)
            elif def_score > 0:
                away_defense.append(para)
        else:
            # Shared paragraph — skip to avoid mixing
            pass

    return {
        home_team: {
            "attack_paragraphs":  home_attack,
            "defense_paragraphs": home_defense,
        },
        away_team: {
            "attack_paragraphs":  away_attack,
            "defense_paragraphs": away_defense,
        },
    }


def _pick_best_paragraph(paragraphs: list[str], keywords: list[str]) -> str:
    """Pick the single most keyword-rich paragraph from a list."""
    if not paragraphs:
        return ""
    scored = [(p, _score_paragraph(p, keywords)) for p in paragraphs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def extract_tactics(jsonl_path: Path) -> dict:
    """
    Main extraction loop.
    Returns team_tactics dict keyed by team name.
    """
    # Accumulate multiple paragraphs per team across many matches
    accumulator: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"attack_paragraphs": [], "defense_paragraphs": []}
    )

    n_processed = 0
    n_skipped   = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            match_id = row.get("match_id", "")
            messages = row.get("messages", [])
            if len(messages) < 2:
                n_skipped += 1
                continue

            # Parse team names from match_id: "2024-08-15 00:00:00_Arsenal_vs_Chelsea"
            parts = match_id.split("_vs_")
            if len(parts) < 2:
                n_skipped += 1
                continue
            home_team = parts[0].rsplit("_", maxsplit=1)[-1].strip()
            away_team = parts[1].strip()

            analysis = messages[1].get("content", "")
            if not analysis:
                n_skipped += 1
                continue

            sections = _extract_team_sections(analysis, home_team, away_team)
            for team, data in sections.items():
                accumulator[team]["attack_paragraphs"].extend(data["attack_paragraphs"])
                accumulator[team]["defense_paragraphs"].extend(data["defense_paragraphs"])

            n_processed += 1

    logger.info("Processed %d entries | skipped %d", n_processed, n_skipped)

    # Collapse: pick the best single paragraph per team per category
    team_tactics = {}
    for team, data in accumulator.items():
        best_attack  = _pick_best_paragraph(data["attack_paragraphs"],  ATTACK_KEYWORDS)
        best_defense = _pick_best_paragraph(data["defense_paragraphs"], DEFENSE_KEYWORDS)
        if best_attack or best_defense:
            team_tactics[team] = {
                "attack_tactic":  best_attack,
                "defense_tactic": best_defense,
            }

    logger.info("Extracted tactics for %d teams", len(team_tactics))
    return team_tactics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading %s …", JSONL_PATH)
    tactics = extract_tactics(JSONL_PATH)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tactics, f, ensure_ascii=False, indent=2)

    logger.info("Saved team_tactics.json → %s", OUT_PATH)
    logger.info("Sample teams: %s", list(tactics.keys())[:10])
