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

import logging
import re
import yaml
from pathlib import Path

from rag.providers.kg_provider     import get_kg_provider
from rag.providers.vector_provider import get_vector_provider
from models.llm_providers          import get_llm_provider

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
    ):
        cfg = _load_rag_cfg()
        kg_name     = kg     or cfg.get("kg_provider",     "neo4j")
        vector_name = vector or cfg.get("vector_provider", "faiss")
        llm_name    = llm    or cfg.get("llm_provider",    "huggingface")

        logger.info("Initializing RAG — KG=%s | Vector=%s | LLM=%s",
                    kg_name, vector_name, llm_name)

        self.kg     = get_kg_provider(kg_name)
        self.vector = get_vector_provider(vector_name)
        self.llm    = get_llm_provider(llm_name)
        self.cfg    = cfg

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

        # Generate with context (works for both BaseLLMProvider and HuggingFaceProvider)
        if hasattr(self.llm, "generate_with_context"):
            return self.llm.generate_with_context(
                prompt=question,
                kg_context=kg_context,
                vector_context=vec_context,
            )
        else:
            # Fallback for providers that only expose generate()
            full_prompt = self._build_prompt(question, kg_context, vec_context)
            return self.llm.generate(full_prompt)

    def predict_match(self, home_team: str, away_team: str) -> str:
        """
        Structured match prediction with full RAG context.

        Args:
            home_team: Name of the home team (must match names in the database).
            away_team: Name of the away team.

        Returns:
            Tactical analysis and prediction.
        """
        question = (
            f"Analyze the upcoming match between {home_team} (Home) and {away_team} (Away). "
            f"Consider their recent form, head-to-head history, and tactical styles "
            f"(attack and defense). Provide a match prediction with reasoning."
        )
        return self.query(question)

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

    # ── Context Management ────────────────────────────────────────────────────

    def close(self):
        """Close database connections."""
        try:
            self.kg.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Football RAG System — interactive CLI")
    parser.add_argument("--kg",     default=None, help="KG provider: neo4j | postgres")
    parser.add_argument("--vector", default=None, help="Vector provider: faiss")
    parser.add_argument("--llm",    default=None, help="LLM provider: huggingface | gemini | ollama")
    parser.add_argument("--predict", nargs=2, metavar=("HOME", "AWAY"),
                        help="Predict a match: --predict Arsenal Chelsea")
    parser.add_argument("--team",   metavar="TEAM",
                        help="Generate team report: --team Arsenal")
    parser.add_argument("--query",  metavar="QUESTION",
                        help="Free-form query: --query 'How does Liverpool attack?'")
    args = parser.parse_args()

    rag = FootballRAGSystem(kg=args.kg, vector=args.vector, llm=args.llm)

    try:
        if args.predict:
            print(rag.predict_match(args.predict[0], args.predict[1]))
        elif args.team:
            print(rag.get_team_report(args.team))
        elif args.query:
            print(rag.query(args.query))
        else:
            # Interactive mode
            print("⚽  Football RAG System — type 'quit' to exit")
            print(f"    KG: {rag.kg.__class__.__name__} | "
                  f"Vector: {rag.vector.__class__.__name__} | "
                  f"LLM: {rag.llm.__class__.__name__}")
            print("-" * 60)
            while True:
                q = input("\nYour question: ").strip()
                if q.lower() in ("quit", "exit", "q"):
                    break
                if not q:
                    continue
                print("\n" + rag.query(q))
    finally:
        rag.close()


if __name__ == "__main__":
    main()
