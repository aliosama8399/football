"""
Football RAG Orchestrator
==========================
The single entry point for all RAG queries. Wires together:
  - KG Provider  (Neo4j or PostgreSQL)
  - Vector Provider (FAISS)
  - LLM Provider (HuggingFace / Gemini / Ollama / etc.)

All provider choices are read from llm_config.yaml (rag section) by default,
but can be overridden at runtime:

    # Default — reads from config
    rag = FootballRAGSystem()

    # Switch providers on the fly
    rag = FootballRAGSystem(kg="postgres", vector="faiss", llm="gemini")

    # Run a query
    answer = rag.query("How does Arsenal attack compared to Chelsea?")
    answer = rag.predict_match("Liverpool", "Manchester City")
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import re
import yaml
from pathlib import Path
from typing import Optional


from rag.providers.kg_provider         import get_kg_provider
from rag.providers.vector_provider     import get_vector_provider
from rag.providers.gnn_provider        import BasePredictionProvider, GNNPredictionProvider
from models.llm_providers              import get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CFG_PATH = Path(__file__).parent.parent / "models" / "llm_config.yaml"


def _load_rag_cfg() -> dict:
    if not _CFG_PATH.exists():
        return {}
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f).get("rag", {})


# ─────────────────────────────────────────────────────────────────────────────
# Team Name Extractor
# ─────────────────────────────────────────────────────────────────────────────

# A lightweight list of known teams (loaded once from Neo4j / PostgreSQL at startup)
_TEAM_CACHE: list[str] = []

def _extract_team_names(query: str, known_teams: list[str]) -> list[str]:
    """
    Scan the query for team names from the known_teams list.
    Returns up to 2 matched team names.
    """
    found = []
    q_lower = query.lower()
    for team in known_teams:
        if team.lower() in q_lower:
            found.append(team)
        if len(found) >= 2:
            break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# RAG Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class FootballRAGSystem:
    """
    Full hybrid RAG pipeline:
      1. Extract team entities from the query
      2. Pull structured facts from Neo4j / PostgreSQL (team profiles, H2H, form, tactics)
      3. Semantic search over FAISS (similar past analyses + match stats)
      4. Assemble context → LLM generation
    """

    def __init__(
        self,
        kg: str = None,
        vector: str = None,
        llm: str = None,
        predictor: Optional[BasePredictionProvider] = None,
    ):
        cfg = _load_rag_cfg()
        kg_name     = kg     or cfg.get("kg_provider",     "neo4j")
        vector_name = vector or cfg.get("vector_provider", "faiss")
        # None-safe LLM: "" / None / "none" disables the LLM (graceful degradation).
        llm_name    = llm if llm else cfg.get("llm_provider", "huggingface")

        logger.info("Initializing RAG — KG=%s | Vector=%s | LLM=%s",
                    kg_name, vector_name, llm_name)

        self.kg      = get_kg_provider(kg_name)
        self.vector  = get_vector_provider(vector_name)
        self.llm     = get_llm_provider(llm_name)  # returns None when llm_name in ("", None, "none")
        self.cfg     = cfg
        # Expert 1: pluggable prediction model (defaults to TEA-GNN).
        self.predictor = predictor if predictor is not None else GNNPredictionProvider()

        # Cache team names for entity extraction
        global _TEAM_CACHE
        if not _TEAM_CACHE:
            try:
                for league in ["Premier_League", "La_Liga", "Serie_A", "Bundesliga", "Ligue_1"]:
                    _TEAM_CACHE.extend(self.kg.get_league_teams(league))
                _TEAM_CACHE = list(set(_TEAM_CACHE))
                logger.info("Team cache loaded: %d teams", len(_TEAM_CACHE))
            except Exception as e:
                logger.warning("Could not load team cache: %s", e)

    # ── Context Assembly ──────────────────────────────────────────────────────

    def _get_kg_context(self, teams: list[str], query: str) -> str:
        """Build a KG context block for the given teams."""
        parts = []

        for team in teams[:2]:  # Max 2 teams per query
            try:
                parts.append(self.kg.format_team_context(team))
            except Exception as e:
                logger.warning("KG retrieval failed for '%s': %s", team, e)

        if len(teams) == 2:
            try:
                h2h = self.kg.get_head_to_head(teams[0], teams[1], limit=5)
                if h2h:
                    lines = [f"\n=== Head-to-Head: {teams[0]} vs {teams[1]} (last 5) ==="]
                    for m in h2h:
                        lines.append(
                            f"  {m.get('date','?')} | "
                            f"{m.get('home_team','?')} {m.get('home_goals','?')}-"
                            f"{m.get('away_goals','?')} {m.get('away_team','?')} "
                            f"[{m.get('result','?')}] | {m.get('league','?')}"
                        )
                    parts.append("\n".join(lines))
            except Exception as e:
                logger.warning("H2H retrieval failed: %s", e)

        return "\n\n".join(parts)

    def _get_vector_context(self, query: str, teams: list[str]) -> str:
        """Semantic search for similar analyses and match stats."""
        # Build enriched query (add team names for better recall)
        enriched = query
        if teams:
            enriched = f"{' '.join(teams)} {query}"

        k = self.cfg.get("vector_top_k", 5)
        return self.vector.format_vector_context(enriched, k=k)

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """
        General-purpose RAG query.
        Automatically extracts teams, retrieves context, and generates an answer.

        Args:
            question: Free-form natural language question.

        Returns:
            LLM-generated answer with grounding in retrieved facts.
        """
        teams = _extract_team_names(question, _TEAM_CACHE)
        logger.info("Teams detected: %s", teams)

        kg_context  = self._get_kg_context(teams, question)
        vec_context = self._get_vector_context(question, teams)

        # None-safe LLM: when no LLM is configured, return the assembled KG + vector
        # context as a structured JSON response instead of calling a generator.
        if self.llm is None:
            return json.dumps({
                "question": question,
                "teams_detected": teams,
                "kg_context": kg_context,
                "vector_context": vec_context,
                "note": "LLM not configured (set rag.llm_provider in llm_config.yaml).",
            }, ensure_ascii=False)

        # Generate with context (works for both BaseLLMProvider and HuggingFaceProvider)
        try:
            if hasattr(self.llm, "generate_with_context"):
                return self.llm.generate_with_context(
                    prompt=question,
                    kg_context=kg_context,
                    vector_context=vec_context,
                )
            # Generic fallback: HF exposes generate(); other providers expose _call_api().
            full_prompt = self._build_prompt(question, kg_context, vec_context)
            if hasattr(self.llm, "generate"):
                return self.llm.generate(full_prompt)
            if hasattr(self.llm, "_call_api"):
                return self.llm._call_api(full_prompt)
            raise AttributeError(
                f"LLM {type(self.llm).__name__} exposes neither generate() nor _call_api()."
            )
        except Exception as e:
            logger.error("LLM inference failed in query(): %s. Returning structured context fallback.", e)
            return json.dumps({
                "question": question,
                "teams_detected": teams,
                "kg_context": kg_context,
                "vector_context": vec_context,
                "note": "LLM runtime failure; degraded to quantitative/context fallback.",
            }, ensure_ascii=False)

    def _get_gnn_prediction(self, home_team: str, away_team: str) -> str:
        """Return a human-readable GNN prediction block (for LLM prompts). Empty on failure."""
        result = self.predict_structured(home_team, away_team)
        if result is None:
            return ""
        probs = result["probabilities"]
        return (
            f"\n=== Expert 1 (Graph Neural Network) Prediction ===\n"
            f"Predicted Outcome: {result['predicted_result']}\n"
            f"Probabilities: Home Win={probs['H']:.1%} | Draw={probs['D']:.1%} | Away Win={probs['A']:.1%}\n"
            f"=================================================="
        )

    def predict_structured(self, home_team: str, away_team: str) -> Optional[dict]:
        """
        Return Expert 1's structured prediction (dict) or None on failure.

        Delegates to the pluggable BasePredictionProvider; safe to call even
        when the LLM is disabled (`llm=None`) and even when the GNN checkpoint
        is missing (the provider warns once and returns None).
        """
        return self.predictor.predict(home_team, away_team)

    def predict_match(self, home_team: str, away_team: str) -> str:
        """
        Structured match prediction with full RAG context and GNN ensemble.

        None-safe: when no LLM is configured, returns the GNN structured
        prediction as JSON (no narrative), so the API can still answer.
        """
        gnn_result = self.predict_structured(home_team, away_team)

        # LLM absent: return structured GNN result only.
        if self.llm is None:
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "note": "LLM not configured; returning Expert 1 (GNN) result only.",
            }, ensure_ascii=False)

        gnn_pred_text = self._get_gnn_prediction(home_team, away_team)
        question = (
            f"You are Expert 2 (Fine-tuned LLM). Analyze the upcoming match between {home_team} (Home) and {away_team} (Away).\n"
            f"Consider their recent form, head-to-head history, and tactical styles (attack and defense).\n"
            f"{gnn_pred_text}\n\n"
            f"Provide your final match prediction with reasoning, integrating your analysis with Expert 1's prediction."
        )
        try:
            return self.query(question)
        except Exception as e:
            logger.error("Expert 2 (LLM) failed in predict_match: %s. Returning Expert 1 GNN only.", e)
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "note": "LLM runtime failure; returning Expert 1 (GNN) result only.",
            }, ensure_ascii=False)

    def predict_live_match(self, home_team: str, away_team: str, live_context: dict) -> str:
        """
        Expert 2 live-match narrative: pre-match GNN block + current match
        state (minute, score, live probs, drivers) -> coach-actionable analysis.

        None-safe: when no LLM is configured, returns the structured context
        as JSON so the API still answers with the live model numbers.
        """
        gnn_result = self.predict_structured(home_team, away_team)
        gnn_pred_text = self._get_gnn_prediction(home_team, away_team)

        live_block = (
            f"LIVE STATE (minute {live_context.get('minute', '?')}):\n"
            f"Current score: {live_context.get('home_goals', 0)} - {live_context.get('away_goals', 0)}\n"
            f"Live probabilities: Home Win={live_context.get('live_probs', {}).get('H', 0):.1%} | "
            f"Draw={live_context.get('live_probs', {}).get('D', 0):.1%} | "
            f"Away Win={live_context.get('live_probs', {}).get('A', 0):.1%}\n"
            f"Pre-match probabilities: Home Win={live_context.get('pre_probs', {}).get('H', 0):.1%} | "
            f"Draw={live_context.get('pre_probs', {}).get('D', 0):.1%} | "
            f"Away Win={live_context.get('pre_probs', {}).get('A', 0):.1%}\n"
            f"Expected final score: {live_context.get('expected_score', {}).get('home', '?')} - "
            f"{live_context.get('expected_score', {}).get('away', '?')}\n"
        )

        drivers = live_context.get("drivers", [])
        if drivers:
            lines = ["LIVE STAT DRIVERS (deviation from season averages):"]
            for d in drivers:
                lines.append(
                    f"  - {d.get('side', '?')} {d.get('label', d.get('stat', '?'))}: "
                    f"{d.get('direction', 'neutral')} (live {d.get('live', '?')}/min vs season {d.get('season_avg', '?')}/min, "
                    f"pace x{d.get('pace', '?')})"
                )
            live_block += "\n" + "\n".join(lines)

        if self.llm is None:
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "live_context": live_context,
                "note": "LLM not configured; returning structured live context only.",
            }, ensure_ascii=False)

        question = (
            f"You are Expert 2 (Fine-tuned LLM), an in-match tactical advisor to a coach.\n"
            f"You are watching {home_team} (Home) vs {away_team} (Away).\n"
            f"{gnn_pred_text}\n\n"
            f"{live_block}\n\n"
            f"Give a concise, coach-actionable analysis grounded ONLY in the live driver data.\n\n"
            f"Respond with a SINGLE JSON object and nothing else: no markdown fences, no prose "
            f"outside the object. Use exactly this structure:\n"
            f"{{\n"
            f'  "match_state": {{"minute": {live_context.get("minute", "?")}, '
            f'"score": "{live_context.get("home_goals", 0)}-{live_context.get("away_goals", 0)}"}},\n'
            f'  "analysis": {{\n'
            f'    "who_controls_now": "2-3 sentences: who is actually controlling the match right now and why, from the driver numbers.",\n'
            f'    "why": ["bullet 1", "bullet 2", "bullet 3"],\n'
            f'    "how_outlook_changed": "2-3 sentences: how the live picture shifted vs the pre-match prediction.",\n'
            f'    "coach_recommendations": [\n'
            f'      {{"priority": 1, "action": "specific action", "reason": "grounded reason"}},\n'
            f'      {{"priority": 2, "action": "specific action", "reason": "grounded reason"}},\n'
            f'      {{"priority": 3, "action": "specific action", "reason": "grounded reason"}}\n'
            f"    ]\n"
            f"  }}\n"
            f"}}\n\n"
            f"Rules: who_controls_now, why and how_outlook_changed must be plain strings/bullets "
            f"without markdown. coach_recommendations must have 2-4 items with integer priorities "
            f"starting at 1, each with a concrete action (substitution, formation/tactical tweak, "
            f"pressing/intensity) and a reason tied to a specific driver (e.g. shots/SOT/xG pace, "
            f"corners, yellows, reds). If coaching advice would differ per side, keep both "
            f"perspectives inside the same actions."
        )
        try:
            return self.query(question)
        except Exception as e:
            logger.error("Expert 2 (LLM) failed in predict_live_match: %s. Returning structured fallback.", e)
            return json.dumps({
                "home_team": home_team,
                "away_team": away_team,
                "gnn_prediction": gnn_result,
                "live_context": live_context,
                "note": "LLM runtime failure; returning structured live context only.",
            }, ensure_ascii=False)

    def get_team_report(self, team_name: str) -> str:
        """
        Generate a full team tactical report.

        Args:
            team_name: Team name as stored in the database.

        Returns:
            Comprehensive tactical and statistical report.
        """
        question = (
            f"Provide a comprehensive tactical analysis of {team_name}. "
            f"Describe their attacking style, defensive organization, "
            f"recent form, strengths, and weaknesses."
        )
        return self.query(question)

    def compare_teams(self, team_a: str, team_b: str) -> str:
        """Compare two teams tactically and statistically."""
        question = (
            f"Compare {team_a} and {team_b} in detail: "
            f"attacking tactics, defensive organization, recent form, "
            f"head-to-head record, and key statistical differences."
        )
        return self.query(question)

    def _build_prompt(self, question: str, kg_context: str, vector_context: str) -> str:
        """Build the final RAG prompt for providers without generate_with_context."""
        parts = [
            "You are an expert football tactical analyst with access to a comprehensive database.",
            "",
        ]
        if kg_context:
            parts += ["## Retrieved Knowledge Graph Context", kg_context, ""]
        if vector_context:
            parts += ["## Retrieved Historical Analyses", vector_context, ""]
        parts += ["## Question", question]
        return "\n".join(parts)

    def get_available_teams(self) -> list[str]:
        """
        Return a sorted list of all available team names from the cached known teams.
        """
        global _TEAM_CACHE
        return sorted(list(_TEAM_CACHE))

    # ── Context Management ────────────────────────────────────────────────────

    def close(self):
        """Close database connections."""
        try:
            self.kg.close()
        except Exception:
            pass
