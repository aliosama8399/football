"""
Tactical Extractor & Grounded Profile Generator
================================================
Combines fine-tuning tactical analyses (data/finetune/football_tactical.jsonl)
with hybrid recency-weighted empirical match statistics (data/team_stats.py)
to generate accurate, factually grounded, leak-free team tactical profiles.

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
import pandas as pd

from data.team_stats import build_team_aggregates
from data.team_registry import normalize_team_name
from rag.retrieval.team_linker import _ALIAS_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
CSV_PATH   = BASE_DIR / "data" / "processed" / "processed_matches.csv"
JSONL_PATH = BASE_DIR / "data" / "finetune" / "football_tactical.jsonl"
OUT_DIR    = BASE_DIR / "rag" / "knowledge_base"
OUT_PATH   = OUT_DIR / "team_tactics.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cleaners & Patterns ────────────────────────────────────────────────────────
_MD_STRIP = re.compile(r'\*+|#{1,6}\s*|_{2,}|\[|\]|`|[\x00-\x08\x0b\x0c\x0e-\x1f]')
_PREFIX_STRIP = re.compile(r'^(strengths|weaknesses|conclusion|attack|defense|tactical analysis)[\s:&*—–-]*', re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _clean_markdown(text: str) -> str:
    """Strip markdown formatting characters, header junk, and collapse whitespace."""
    text = _MD_STRIP.sub('', text)
    text = _PREFIX_STRIP.sub('', text.strip())
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading bullets or dashes
    text = re.sub(r'^[-•*\d.]+\s*', '', text).strip()
    return text


def _canonicalize_team(name: str) -> str:
    name_clean = name.strip()
    return _ALIAS_MAP.get(name_clean.lower(), normalize_team_name(name_clean))


# ── Keyword lists ──────────────────────────────────────────────────────────────
ATTACK_KEYWORDS = [
    "attack", "offensive", "scoring", "goals", "shots", "press",
    "high press", "forward", "chance creation", "xg", "penetrat",
    "build-up", "possession", "counter", "through ball", "creativity", "pace"
]
DEFENSE_KEYWORDS = [
    "defense", "defensive", "defending", "backline", "clean sheet",
    "concede", "tackle", "interception", "block", "compact",
    "low block", "defensive shape", "pressing", "organization"
]
STRENGTH_KEYWORDS = [
    "strength", "strong", "excellent", "dominant", "clinical", "prolific",
    "solid", "effective", "dangerous", "potent", "impressive", "consistent"
]
WEAKNESS_KEYWORDS = [
    "weakness", "weak", "struggle", "vulnerable", "concede", "susceptible",
    "poor", "lack", "inconsistent", "exposed", "fragile", "trouble", "prone"
]


def _score_text(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def _extract_clean_paragraphs(analysis: str, home_team: str, away_team: str) -> dict:
    """
    Split analysis into paragraphs cleanly attributed to home or away team.
    Crucially drops paragraphs that reference the opponent to prevent leakage.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", analysis) if len(p.strip()) > 50]
    home_attack, home_defense = [], []
    away_attack, away_defense = [], []

    h_lower = home_team.lower()
    a_lower = away_team.lower()

    for para in paragraphs:
        p_clean = _clean_markdown(para)
        p_lower = p_clean.lower()

        # Strict attribution: paragraph must mention one team and NOT mention the other
        has_home = h_lower in p_lower
        has_away = a_lower in p_lower

        if has_home and not has_away:
            atk_score = _score_text(p_lower, ATTACK_KEYWORDS)
            def_score = _score_text(p_lower, DEFENSE_KEYWORDS)
            if atk_score >= def_score and atk_score > 0:
                home_attack.append(p_clean)
            elif def_score > 0:
                home_defense.append(p_clean)
        elif has_away and not has_home:
            atk_score = _score_text(p_lower, ATTACK_KEYWORDS)
            def_score = _score_text(p_lower, DEFENSE_KEYWORDS)
            if atk_score >= def_score and atk_score > 0:
                away_attack.append(p_clean)
            elif def_score > 0:
                away_defense.append(p_clean)

    return {
        home_team: {"attack": home_attack, "defense": home_defense},
        away_team: {"attack": away_attack, "defense": away_defense},
    }


def _is_consistent_and_unpolluted(sentence: str, team_name: str, agg: dict, is_attack: bool, all_teams: set = None) -> bool:
    s_clean = sentence.strip()
    if not s_clean.endswith((".", "!", "?")):
        return False
    last_word = s_clean.rstrip(".!?").split()[-1].lower() if s_clean.rstrip(".!?").split() else ""
    if last_word in ("vs", "vs.", "and", "or", "but", "with", "than", "that", "the", "to", "for", "of"):
        return False

    s_lower = s_clean.lower()
    t_lower = team_name.lower()

    # Contamination markers where sentence discusses the opponent or comparative conflict
    bad_markers = [
        "their opponents", "opposing", "tested by", "frustrate", "equipped to handle",
        "trouble a team of", "on the other hand", "conceding nearly two goals",
        "struggles:", "concerns:", "flaw:", "worries:", "not the ", "vs.", " vs ",
        "defensive gap", "defensive leaks", "neutralizing", "could slice them open",
        "neutralize", "slice them", "capable of"
    ]
    for bm in bad_markers:
        if bm in s_lower:
            return False

    # Team name token match (e.g. "Arsenal", "Barcelona", "Madrid", "City")
    tokens = [tok for tok in re.findall(r'[a-z]{4,}', t_lower)]
    if tokens and not any(tok in s_lower for tok in tokens):
        return False

    # Check for ANY other team mention or alias
    generic_words = {"city", "united", "real", "club", "town", "athletic", "borussia", "deportivo", "sporting", "west", "east", "north", "south", "inter", "saint"}
    for alias, canon in _ALIAS_MAP.items():
        if canon.lower() == t_lower or alias.lower() == t_lower:
            continue
        alias_clean = alias.lower()
        if len(alias_clean) >= 4 and alias_clean not in t_lower and alias_clean not in generic_words:
            if re.search(r'\b' + re.escape(alias_clean) + r'\b', s_lower):
                return False

    if all_teams:
        for other in all_teams:
            if other.lower() == t_lower:
                continue
            other_clean = other.lower()
            other_words = [w for w in re.findall(r'[a-z]{4,}', other_clean) if w not in tokens and w not in generic_words]
            for ow in other_words:
                if re.search(r'\b' + re.escape(ow) + r'\b', s_lower):
                    return False

    # Factual consistency checks
    avg_goals = agg.get("avg_goals", 1.5)
    avg_xg = agg.get("avg_xg", 1.4)
    avg_conceded = agg.get("avg_conceded", 1.2)
    avg_xga = agg.get("avg_xga", 1.2)
    cs_rate = agg.get("clean_sheet_rate", 0.3)

    if is_attack:
        if avg_goals >= 1.65 or avg_xg >= 1.6:
            negatives = ["struggle", "poor return", "limited attack", "anemic", "lack of penetration", "inability to score", "scoring only", "not a prolific"]
            if any(n in s_lower for n in negatives):
                return False
    else:
        if avg_conceded <= 1.15 or avg_xga <= 1.20 or cs_rate >= 0.33:
            negatives = ["porous", "vulnerability", "defensive struggle", "leaky", "fragile defense", "conceding regularly"]
            if any(n in s_lower for n in negatives):
                return False

    return True


def _pick_headline(paragraphs: list[str], keywords: list[str], team_name: str, agg: dict, is_attack: bool, all_teams: set = None) -> str:
    """Extract a single concise, punchy sentence as a headline."""
    scored = []
    for para in paragraphs:
        sentences = SENTENCE_SPLIT.split(para)
        for s in sentences:
            s_clean = _clean_markdown(s)
            if not (35 <= len(s_clean) <= 150):
                continue
            if s_clean.lower().startswith(("however,", "therefore,", "moreover,", "in conclusion")):
                continue
            if not _is_consistent_and_unpolluted(s_clean, team_name, agg, is_attack, all_teams):
                continue
            score = _score_text(s_clean, keywords)
            if score > 0:
                scored.append((s_clean, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else ""


def _pick_concise_tactic(paragraphs: list[str], keywords: list[str], team_name: str, agg: dict, is_attack: bool, all_teams: set = None) -> str:
    """Select the best concise tactical description (1-2 sentences, max 260 chars)."""
    scored = []
    for para in paragraphs:
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(para) if len(s.strip()) > 25]
        if not sentences:
            continue
        candidate = " ".join(sentences[:2])
        candidate = _clean_markdown(candidate)
        if len(candidate) > 280:
            candidate = sentences[0]
            candidate = _clean_markdown(candidate)
        if 40 <= len(candidate) <= 280:
            if not _is_consistent_and_unpolluted(candidate, team_name, agg, is_attack, all_teams):
                continue
            score = _score_text(candidate, keywords)
            scored.append((candidate, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else ""


def build_grounded_strengths(agg: dict) -> list[str]:
    """Derive accurate, empirically grounded strengths from recency-weighted match statistics."""
    strengths = []
    avg_goals = agg.get("avg_goals", 0.0)
    avg_xg    = agg.get("avg_xg", 0.0)
    avg_sot   = agg.get("avg_sot", 0.0)
    avg_shots = agg.get("avg_shots", 0.0)
    avg_xga   = agg.get("avg_xga", 0.0)
    avg_sota  = agg.get("avg_sot_against", 0.0)
    avg_gh    = agg.get("avg_goals_home", 0.0)
    win_rate  = agg.get("win_rate", 0.0)
    cs_rate   = agg.get("clean_sheet_rate", 0.0)

    # 1. Attacking output
    if avg_goals >= 1.95 or avg_xg >= 1.90:
        strengths.append(f"Prolific attacking output: averages {avg_goals:.2f} goals and {avg_xg:.2f} xG per match.")
    elif avg_goals >= 1.50 or avg_xg >= 1.45:
        strengths.append(f"Consistent offensive threat: averages {avg_goals:.2f} goals and {avg_xg:.2f} xG per match.")
    elif avg_gh >= 1.80:
        strengths.append(f"Potent home attacking presence: scores {avg_gh:.2f} goals per match at home.")

    # 2. Shot generation / precision
    if avg_sot >= 4.8:
        strengths.append(f"High-volume chance creation: tests opposition goalkeepers with {avg_sot:.1f} shots on target per match.")
    elif avg_shots >= 13.5:
        strengths.append(f"Active offensive pressure: generates {avg_shots:.1f} total shots per match.")

    # 3. Defensive resilience
    if cs_rate >= 0.35 or avg_xga <= 1.12:
        strengths.append(f"Resilient defensive structure: keeps clean sheets in {cs_rate*100:.1f}% of matches and limits opponents to {avg_xga:.2f} xGA.")
    elif avg_sota <= 3.6:
        strengths.append(f"Disciplined defensive containment: restricts opponents to just {avg_sota:.1f} shots on target per match.")
    elif cs_rate >= 0.28:
        strengths.append(f"Dependable defensive baseline: records clean sheets in {cs_rate*100:.1f}% of fixtures.")

    # 4. Overall control & win efficiency
    if win_rate >= 0.55:
        strengths.append(f"Commanding game control: maintains a {win_rate*100:.1f}% overall victory rate.")
    elif avg_gh >= 2.05:
        strengths.append(f"Formidable home fortress: averages {avg_gh:.2f} goals scored per home match.")
    elif win_rate >= 0.42:
        strengths.append(f"Competitive match management: secures points consistently with a {win_rate*100:.1f}% win rate.")

    return strengths[:4]


def build_grounded_weaknesses(agg: dict) -> list[str]:
    """Derive accurate, empirically grounded weaknesses from recency-weighted match statistics."""
    weaknesses = []
    avg_goals = agg.get("avg_goals", 0.0)
    avg_conceded = agg.get("avg_conceded", 0.0)
    avg_xg    = agg.get("avg_xg", 0.0)
    avg_xga   = agg.get("avg_xga", 0.0)
    avg_sot   = agg.get("avg_sot", 0.0)
    avg_sota  = agg.get("avg_sot_against", 0.0)
    avg_gh    = agg.get("avg_goals_home", 0.0)
    avg_ga    = agg.get("avg_goals_away", 0.0)
    avg_cards = agg.get("avg_yellows", 0.0)
    cs_rate   = agg.get("clean_sheet_rate", 0.0)

    # 1. Defensive vulnerabilities
    if avg_conceded >= 1.45 or avg_xga >= 1.45:
        weaknesses.append(f"Defensive vulnerability: concedes {avg_conceded:.2f} goals and {avg_xga:.2f} xGA per match.")
    elif cs_rate <= 0.22:
        weaknesses.append(f"Struggles to maintain clean sheets: records shutouts in only {cs_rate*100:.1f}% of matches.")
    elif avg_sota >= 4.6:
        weaknesses.append(f"Susceptible to shot pressure: allows opponents {avg_sota:.1f} shots on target per match.")
    elif avg_conceded >= 1.10:
        weaknesses.append(f"Occasional defensive lapses: concedes an average of {avg_conceded:.2f} goals per match.")
    elif avg_xga >= 1.05:
        weaknesses.append(f"Transitional susceptibility: concedes {avg_xga:.2f} expected goals against per match.")
    else:
        weaknesses.append(f"Rare defensive concession on rapid counter-attacks ({avg_conceded:.2f} goals conceded per match).")

    # 2. Attacking limitations / Away drop-off / Discipline
    if avg_goals <= 1.25 or avg_xg <= 1.20:
        weaknesses.append(f"Limited offensive punch: produces only {avg_goals:.2f} goals and {avg_xg:.2f} xG per match.")
    elif avg_sot <= 3.8:
        weaknesses.append(f"Subdued penetration in final third: registers only {avg_sot:.1f} shots on target per match.")
    elif avg_gh > 0 and avg_ga < (avg_gh * 0.72):
        weaknesses.append(f"Away form drop-off: scoring output decreases away from home ({avg_ga:.2f} goals away vs {avg_gh:.2f} home).")
    elif avg_cards >= 2.1:
        weaknesses.append(f"Disciplinary liability: averages {avg_cards:.1f} yellow cards per match during high-tempo pressing.")
    else:
        weaknesses.append(f"Slight dip in scoring fluency during tough away fixtures ({avg_ga:.2f} goals away per match).")

    return weaknesses[:3]


def generate_tactics_profiles() -> dict:
    """Main generation routine: merges stats + leak-free qualitative tactical summaries."""
    logger.info("Loading match statistics from %s ...", CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    aggregates = build_team_aggregates(df, recency_weighted=True)
    logger.info("Computed hybrid recency-weighted statistics for %d teams.", len(aggregates))

    qualitative_acc = defaultdict(lambda: {"attack": [], "defense": []})

    if JSONL_PATH.exists():
        logger.info("Parsing qualitative tactical analyses from %s ...", JSONL_PATH)
        n_processed = 0
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                match_id = row.get("match_id", "")
                messages = row.get("messages", [])
                if len(messages) < 2:
                    continue

                parts = match_id.split("_vs_")
                if len(parts) < 2:
                    continue
                raw_home = parts[0].rsplit("_", maxsplit=1)[-1].strip()
                raw_away = parts[1].strip()

                home_canon = _canonicalize_team(raw_home)
                away_canon = _canonicalize_team(raw_away)

                analysis = messages[1].get("content", "")
                if not analysis:
                    continue

                sec = _extract_clean_paragraphs(analysis, home_canon, away_canon)
                for t, data in sec.items():
                    qualitative_acc[t]["attack"].extend(data["attack"])
                    qualitative_acc[t]["defense"].extend(data["defense"])
                n_processed += 1

        logger.info("Parsed %d matches into qualitative tactical accumulator.", n_processed)
    else:
        logger.warning("JSONL file not found at %s. Proceeding with pure statistical profiles.", JSONL_PATH)

    tactics_db = {}
    all_team_names = set(aggregates.keys())

    # Iterate over all teams present in match dataset
    for team, agg in aggregates.items():
        q_data = qualitative_acc.get(team, {"attack": [], "defense": []})
        atk_paras = q_data["attack"]
        def_paras = q_data["defense"]

        # Qualitative headlines
        atk_headline = _pick_headline(atk_paras, ATTACK_KEYWORDS, team, agg, is_attack=True, all_teams=all_team_names)
        def_headline = _pick_headline(def_paras, DEFENSE_KEYWORDS, team, agg, is_attack=False, all_teams=all_team_names)

        if not atk_headline:
            avg_g = agg.get("avg_goals", 1.5)
            avg_xg = agg.get("avg_xg", 1.4)
            atk_headline = f"{team} executes structured offensive build-up, averaging {avg_g:.2f} goals and {avg_xg:.2f} xG per match."

        if not def_headline:
            avg_con = agg.get("avg_conceded", 1.2)
            cs_pct = agg.get("clean_sheet_rate", 0.3) * 100
            def_headline = f"{team} maintains a disciplined defensive block with a {cs_pct:.1f}% clean sheet rate ({avg_con:.2f} goals conceded/match)."

        # Concise tactical paragraphs
        atk_tactic = _pick_concise_tactic(atk_paras, ATTACK_KEYWORDS, team, agg, is_attack=True, all_teams=all_team_names)
        def_tactic = _pick_concise_tactic(def_paras, DEFENSE_KEYWORDS, team, agg, is_attack=False, all_teams=all_team_names)

        if not atk_tactic:
            atk_tactic = (
                f"{team}'s attacking approach relies on patient progression and sustained pressure in the final third, "
                f"creating {agg.get('avg_sot', 4.5):.1f} shots on target per match."
            )
        if not def_tactic:
            def_tactic = (
                f"{team} organizes in a compact zonal block during defensive transitions, "
                f"limiting opposition opportunities to {agg.get('avg_xga', 1.2):.2f} xGA per match."
            )

        # Grounded strengths & weaknesses
        strengths = build_grounded_strengths(agg)
        weaknesses = build_grounded_weaknesses(agg)

        profile = {
            "attack_tactic":    atk_tactic,
            "defense_tactic":   def_tactic,
            "attack_headline":  atk_headline,
            "defense_headline": def_headline,
            "strengths":        strengths,
            "weaknesses":       weaknesses,
        }

        tactics_db[team] = profile

    logger.info("Generated clean, grounded tactical profiles for %d teams.", len(tactics_db))
    return tactics_db


if __name__ == "__main__":
    tactics = generate_tactics_profiles()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tactics, f, ensure_ascii=False, indent=2)

    logger.info("Saved team_tactics.json -> %s", OUT_PATH)
    # Print sample teams
    for sample in ["Arsenal", "Barcelona", "Manchester City", "Real Madrid"]:
        if sample in tactics:
            print(f"\n--- {sample} ---")
            print("Strengths:", tactics[sample]["strengths"])
            print("Weaknesses:", tactics[sample]["weaknesses"])
            print("Attack Headline:", tactics[sample]["attack_headline"])
            print("Defense Headline:", tactics[sample]["defense_headline"])
