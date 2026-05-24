"""
Base Provider Abstract Classes
================================
All KG and Vector providers must subclass these.
Adding a new provider: subclass the relevant base and register it in the factory.
"""

from abc import ABC, abstractmethod
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseKGProvider(ABC):
    """Abstract interface for Knowledge Graph providers (Neo4j, PostgreSQL, etc.)."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    def get_team_profile(self, team_name: str) -> dict:
        """
        Return a team's full profile:
        stats, attack_tactic, defense_tactic, league, avg metrics.
        """
        pass

    @abstractmethod
    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> list[dict]:
        """Return the last N H2H matches between two teams."""
        pass

    @abstractmethod
    def get_recent_form(self, team_name: str, n: int = 5) -> list[dict]:
        """Return the last N matches for a team."""
        pass

    @abstractmethod
    def get_league_teams(self, league: str, season: str = None) -> list[str]:
        """Return all team names in a league, optionally filtered by season."""
        pass

    @abstractmethod
    def get_match(self, home_team: str, away_team: str, date: str = None) -> dict:
        """Look up a specific match."""
        pass

    def format_team_context(self, team_name: str) -> str:
        """
        High-level helper: returns a ready-to-inject text block with
        team profile + recent form. Used by the RAG Orchestrator.
        """
        profile = self.get_team_profile(team_name)
        form    = self.get_recent_form(team_name, n=5)

        lines = [f"=== {team_name} Profile ==="]
        if profile:
            lines.append(f"League          : {profile.get('league', 'N/A')}")
            lines.append(f"Avg Goals (H)   : {profile.get('avg_goals_home', 'N/A'):.2f}")
            lines.append(f"Avg Goals (A)   : {profile.get('avg_goals_away', 'N/A'):.2f}")
            lines.append(f"Avg xG          : {profile.get('avg_xg', 'N/A'):.2f}")
            lines.append(f"Avg xGA         : {profile.get('avg_xga', 'N/A'):.2f}")
            lines.append(f"Total Matches   : {profile.get('total_matches', 'N/A')}")
            if profile.get("attack_tactic"):
                lines.append(f"\n[Attack Style]\n{profile['attack_tactic']}")
            if profile.get("defense_tactic"):
                lines.append(f"\n[Defense Style]\n{profile['defense_tactic']}")

        if form:
            lines.append("\n--- Last 5 Matches ---")
            for m in form:
                lines.append(
                    f"  {m.get('date','?')} | {m.get('home_team','?')} {m.get('home_goals','?')}-"
                    f"{m.get('away_goals','?')} {m.get('away_team','?')} [{m.get('result','?')}]"
                )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseVectorProvider(ABC):
    """Abstract interface for vector store providers (FAISS, ChromaDB, etc.)."""

    @abstractmethod
    def load(self) -> None:
        """Load the index from disk."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5, filter_meta: dict = None) -> list[dict]:
        """
        Semantic search. Returns a list of dicts with keys:
            text      : str   – the source text chunk
            score     : float – similarity score (lower = better for L2)
            metadata  : dict  – match_id, teams, date, league, etc.
        """
        pass

    def format_vector_context(self, query: str, k: int = 5) -> str:
        """
        High-level helper: search and format results as a text block.
        Used by the RAG Orchestrator.
        """
        results = self.search(query, k=k)
        if not results:
            return "[No relevant historical analyses found.]"

        lines = ["=== Similar Historical Analyses ==="]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            lines.append(
                f"\n[{i}] {meta.get('home_team','?')} vs {meta.get('away_team','?')} "
                f"({meta.get('date','?')} | {meta.get('league','?')} | Result: {meta.get('actual_result','?')})"
            )
            lines.append(r.get("text", "")[:600])  # Cap at 600 chars per chunk

        return "\n".join(lines)
