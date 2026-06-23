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
        Rich multi-section context block for the RAG Orchestrator.
        Includes all profile stats, tactical headlines, strengths/weaknesses,
        and the last 5 matches with rolling-form detail where available.
        """
        profile = self.get_team_profile(team_name)

        # Use detailed form if provider supports it, else fall back to simple form
        if hasattr(self, "get_team_detailed_form"):
            form = self.get_team_detailed_form(team_name, n=5)
        else:
            form = self.get_recent_form(team_name, n=5)

        def _pct(v) -> str:
            try:
                return f"{float(v)*100:.1f}%"
            except Exception:
                return "N/A"

        def _f(v, decimals=2) -> str:
            try:
                return f"{float(v):.{decimals}f}"
            except Exception:
                return "N/A"

        lines = [f"=== {team_name} ==="]

        if profile:
            # ── Overview ────────────────────────────────────────────────────
            lines.append("\n[Overview]")
            lines.append(f"  League         : {profile.get('league', 'N/A')}")
            lines.append(f"  Total Matches  : {profile.get('total_matches', 'N/A')}")
            lines.append(
                f"  Record         : "
                f"W {_pct(profile.get('win_rate'))}  "
                f"D {_pct(profile.get('draw_rate'))}  "
                f"L {_pct(profile.get('loss_rate'))}"
            )
            lines.append(f"  Clean Sheets   : {_pct(profile.get('clean_sheet_rate'))}")

            # ── Attack ──────────────────────────────────────────────────────
            lines.append("\n[Attack]")
            lines.append(f"  Avg Goals  (Home/Away) : {_f(profile.get('avg_goals_home'))} / {_f(profile.get('avg_goals_away'))}")
            lines.append(f"  Avg xG                 : {_f(profile.get('avg_xg'))}")
            lines.append(f"  Avg Shots              : {_f(profile.get('avg_shots'))}")
            lines.append(f"  Avg Shots on Target    : {_f(profile.get('avg_sot'))}")
            lines.append(f"  Avg Corners            : {_f(profile.get('avg_corners'))}")
            if profile.get("attack_headline"):
                lines.append(f"  Summary  : {profile['attack_headline']}")
            if profile.get("attack_tactic"):
                lines.append(f"\n  [Attacking Style]\n  {profile['attack_tactic']}")

            # ── Defense ─────────────────────────────────────────────────────
            lines.append("\n[Defense]")
            lines.append(f"  Avg xGA                : {_f(profile.get('avg_xga'))}")
            lines.append(f"  Avg Shots Against      : {_f(profile.get('avg_shots_against'))}")
            lines.append(f"  Avg SOT Against        : {_f(profile.get('avg_sot_against'))}")
            lines.append(f"  Avg Fouls Committed    : {_f(profile.get('avg_fouls'))}")
            lines.append(f"  Avg Yellows            : {_f(profile.get('avg_yellows'))}")
            if profile.get("defense_headline"):
                lines.append(f"  Summary  : {profile['defense_headline']}")
            if profile.get("defense_tactic"):
                lines.append(f"\n  [Defensive Style]\n  {profile['defense_tactic']}")

            # ── Strengths ───────────────────────────────────────────────────
            strengths = profile.get("strengths") or []
            if strengths:
                lines.append("\n[Strengths]")
                for s in strengths:
                    lines.append(f"  • {s}")

            # ── Weaknesses ──────────────────────────────────────────────────
            weaknesses = profile.get("weaknesses") or []
            if weaknesses:
                lines.append("\n[Weaknesses]")
                for w in weaknesses:
                    lines.append(f"  • {w}")

        # ── Recent Form ─────────────────────────────────────────────────────
        if form:
            lines.append("\n[Last 5 Matches]")
            for m in form:
                is_home = m.get("home_team", "") == team_name
                role    = "HOME" if is_home else "AWAY"
                score   = f"{m.get('home_goals','?')}-{m.get('away_goals','?')}"
                result  = m.get("result", "?")
                opp     = m.get("away_team") if is_home else m.get("home_team")

                # Rolling form values (if available from detailed form)
                form_5  = m.get("home_form_5") if is_home else m.get("away_form_5")
                xg_5    = m.get("home_xg_5")   if is_home else m.get("away_xg_5")
                xga_5   = m.get("home_xga_5")  if is_home else m.get("away_xga_5")
                shots_5 = m.get("home_shots_5") if is_home else m.get("away_shots_5")
                sot_5   = m.get("home_sot_5")  if is_home else m.get("away_sot_5")

                line = (
                    f"  {m.get('date','?')} [{role}] vs {opp}  {score}  [{result}]"
                    f"  | {m.get('league','?')}"
                )
                if form_5 is not None:
                    line += (
                        f"\n    Rolling-5 -> Form:{_f(form_5)}  xG:{_f(xg_5)}  "
                        f"xGA:{_f(xga_5)}  Shots:{_f(shots_5)}  SOT:{_f(sot_5)}"
                    )
                lines.append(line)

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
