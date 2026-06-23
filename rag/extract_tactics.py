"""
Tactical Extractor
====================
Parses football_tactical.jsonl and extracts per-team tactical profiles.

Each JSONL row has:
  - messages[0].content : the user prompt (match stats)
  - messages[1].content : the LLM assistant analysis

Extracted fields per team:
  - attack_tactic    : best paragraph describing attacking play
  - defense_tactic   : best paragraph describing defensive play
  - attack_headline  : one-sentence attack summary
  - defense_headline : one-sentence defense summary
  - strengths        : bullet list of strength phrases found in the text
  - weaknesses       : bullet list of weakness phrases found in the text

Output: rag/knowledge_base/team_tactics.json

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

_MD_STRIP = re.compile(r'\*+|#{1,6}\s*|_{2,}|\[|\]|`')


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting characters and collapse whitespace."""
    text = _MD_STRIP.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


BASE_DIR   = Path(__file__).parent.parent
JSONL_PATH = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"
OUT_DIR    = BASE_DIR / "rag" / "knowledge_base"
OUT_PATH   = OUT_DIR / "team_tactics.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Keyword lists ──────────────────────────────────────────────────────────────
ATTACK_KEYWORDS = [
    "attack", "offensive", "scoring", "goals", "shots", "press",
    "high press", "forward", "striker", "chance creation", "xg",
    "expected goals", "penetrat", "build-up", "possession", "dribbl",
    "counter", "through ball", "link-up", "creativity", "pace",
]
DEFENSE_KEYWORDS = [
    "defense", "defensive", "defending", "backline", "clean sheet",
    "concede", "tackle", "interception", "block", "goalkeeper", "compact",
    "low block", "counter-press", "transition", "vulnerable", "defensive shape",
    "pressing", "recovery", "aerial", "set piece", "organization",
]
STRENGTH_KEYWORDS = [
    "strength", "strong", "excellent", "dominant", "clinical", "prolific",
    "solid", "effective", "dangerous", "potent", "impressive", "consistent",
    "reliable", "threat", "quality", "sharp", "lethal",
]
WEAKNESS_KEYWORDS = [
    "weakness", "weak", "struggle", "vulnerable", "concede", "susceptible",
    "poor", "lack", "inconsistent", "exposed", "fragile", "trouble",
    "inability", "fails", "prone", "leaky",
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _score_paragraph(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _extract_headline(
    paragraphs: list[str],
    keywords: list[str],
    exclude_keywords: list[str] | None = None,
) -> str:
    """
    Extract the single most representative sentence as a headline.
    Sentences that score higher on exclude_keywords than on keywords are skipped.
    """
    exclude_keywords = exclude_keywords or []
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(SENTENCE_SPLIT.split(para))

    scored = []
    for s in all_sentences:
        s = s.strip()
        if not (20 < len(s) < 220):
            continue
        score = _score_paragraph(s, keywords)
        if score == 0:
            continue
        # Skip sentences dominated by the opposite category
        if exclude_keywords and _score_paragraph(s, exclude_keywords) >= score:
            continue
        cleaned = _clean_markdown(s)
        if cleaned:
            scored.append((cleaned, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else ""


def _extract_phrases(paragraphs: list[str], keywords: list[str]) -> list[str]:
    """
    Extract short sentences (40-180 chars) that score highly for the
    given keyword list. Returns up to 4 unique clean phrases.
    """
    all_sentences = []
    for para in paragraphs:
        all_sentences.extend(SENTENCE_SPLIT.split(para))

    seen, results = set(), []
    scored = sorted(
        [(s.strip(), _score_paragraph(s, keywords)) for s in all_sentences
         if 40 < len(s.strip()) < 180 and _score_paragraph(s, keywords) > 0],
        key=lambda x: x[1], reverse=True,
    )
    for s, _ in scored:
        cleaned = _clean_markdown(s)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)
        if len(results) >= 4:
            break
    return results


def _extract_team_sections(analysis: str, home_team: str, away_team: str) -> dict:
    """
    Split an analysis into paragraphs and assign each to home/away × attack/defense.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", analysis) if len(p.strip()) > 60]

    home_attack, home_defense = [], []
    away_attack, away_defense = [], []

    for para in paragraphs:
        para_lower = para.lower()
        is_home = home_team.lower() in para_lower
        is_away = away_team.lower() in para_lower

        atk_score = _score_paragraph(para, ATTACK_KEYWORDS)
        def_score = _score_paragraph(para, DEFENSE_KEYWORDS)

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

    return {
        home_team: {"attack_paragraphs": home_attack, "defense_paragraphs": home_defense},
        away_team: {"attack_paragraphs": away_attack, "defense_paragraphs": away_defense},
    }


def _pick_best_paragraph(paragraphs: list[str], keywords: list[str]) -> str:
    if not paragraphs:
        return ""
    scored = [(p, _score_paragraph(p, keywords)) for p in paragraphs]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def extract_tactics(jsonl_path: Path) -> dict:
    """
    Main extraction loop. Returns team_tactics dict keyed by team name.
    Each value contains:
        attack_tactic, defense_tactic,
        attack_headline, defense_headline,
        strengths, weaknesses
    """
    accumulator: dict[str, dict[str, list]] = defaultdict(lambda: {
        "attack_paragraphs":  [],
        "defense_paragraphs": [],
        "all_paragraphs":     [],  # all text mentioning this team
    })

    n_processed, n_skipped = 0, 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
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

            # Parse team names: "2024-08-15 00:00:00_Arsenal_vs_Chelsea"
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
            all_paras = [p.strip() for p in re.split(r"\n{2,}", analysis) if len(p.strip()) > 60]

            for team, data in sections.items():
                accumulator[team]["attack_paragraphs"].extend(data["attack_paragraphs"])
                accumulator[team]["defense_paragraphs"].extend(data["defense_paragraphs"])
                # Collect paragraphs mentioning this team for strength/weakness extraction
                accumulator[team]["all_paragraphs"].extend(
                    [p for p in all_paras if team.lower() in p.lower()]
                )

            n_processed += 1

    logger.info("Processed %d entries | skipped %d", n_processed, n_skipped)

    # Build final output
    team_tactics = {}
    for team, data in accumulator.items():
        atk_paras  = data["attack_paragraphs"]
        def_paras  = data["defense_paragraphs"]
        all_paras  = data["all_paragraphs"]

        attack_tactic    = _pick_best_paragraph(atk_paras, ATTACK_KEYWORDS)
        defense_tactic   = _pick_best_paragraph(def_paras, DEFENSE_KEYWORDS)
        # Headlines: exclude sentences dominated by the opposite polarity
        attack_headline  = _extract_headline(atk_paras,  ATTACK_KEYWORDS,  exclude_keywords=WEAKNESS_KEYWORDS)
        defense_headline = _extract_headline(def_paras, DEFENSE_KEYWORDS,  exclude_keywords=STRENGTH_KEYWORDS)
        strengths        = _extract_phrases(all_paras, STRENGTH_KEYWORDS)
        weaknesses       = _extract_phrases(all_paras, WEAKNESS_KEYWORDS)

        if any([attack_tactic, defense_tactic, strengths, weaknesses]):
            team_tactics[team] = {
                "attack_tactic":    attack_tactic,
                "defense_tactic":   defense_tactic,
                "attack_headline":  attack_headline,
                "defense_headline": defense_headline,
                "strengths":        strengths,
                "weaknesses":       weaknesses,
            }

    logger.info("Extracted tactics for %d teams", len(team_tactics))
    return team_tactics


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Loading %s …", JSONL_PATH)
    tactics = extract_tactics(JSONL_PATH)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tactics, f, ensure_ascii=False, indent=2)

    logger.info("Saved team_tactics.json → %s", OUT_PATH)
    logger.info("Sample teams: %s", list(tactics.keys())[:10])
