"""
MatchDataStore — lazy, read-only access to data/processed/processed_matches.csv.

The CSV is loaded on the FIRST query, once per process (boot stays fast), and
all methods return JSON-safe plain Python structures — no pandas leaks out.

This is the structured-facts layer of the knowledge base. It shares the
aggregation math with the PostgreSQL builder via data.team_stats, so DB
tables, GraphQL numbers and KB answers cannot drift.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.team_stats import build_team_aggregates, build_league_table, pyval
from data.team_registry import normalize_team_name
from rag.knowledge_base.config import kb_settings

logger = logging.getLogger(__name__)


class MatchDataStore:
    """Read-only CSV-backed structured facts store."""

    def __init__(self, csv_path: Optional[str] = None):
        cfg = kb_settings()
        self._csv_path = Path(csv_path or cfg["kb_csv_path"])
        self._df: Optional[pd.DataFrame] = None
        self._aggregates: Optional[dict] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle (lazy)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._load()
        return self._df

    def _load(self) -> None:
        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"KB match CSV not found: {self._csv_path}. "
                "Run the data pipeline (data/preprocess.py) or fix kb_csv_path."
            )
        t0 = time.time()
        df = pd.read_csv(self._csv_path, low_memory=False).copy()
        df["_dt"] = pd.to_datetime(df["Date"], errors="coerce")
        self._df = df
        logger.info(
            "MatchDataStore loaded %d rows / %d cols from %s (%.2fs)",
            len(df), len(df.columns), self._csv_path, time.time() - t0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Team resolution
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def _team_set(self) -> set:
        return set(self.df["HomeTeam"].unique()) | set(self.df["AwayTeam"].unique())

    def team_names(self) -> List[str]:
        return sorted(self._team_set)

    def resolve_team(self, query) -> Optional[str]:
        """Resolve a user-provided name to a canonical CSV team name.

        Order: registry aliases (data/team_registry.py) → case-insensitive
        exact → prefix/substring/token fuzzy. Returns None when nothing matches.
        """
        if not query:
            return None
        q = str(query).strip()
        if not q:
            return None

        names = self._team_set

        cand = normalize_team_name(q)
        if cand in names:
            return cand

        ql = q.lower()
        for name in names:
            if name.lower() == ql:
                return name

        q_tokens = set(re.findall(r"[a-z0-9]+", ql))
        best, best_score = None, 0
        for name in names:
            nl = name.lower()
            if nl.startswith(ql) or ql.startswith(nl):
                score = 3
            elif ql in nl or nl in ql:
                score = 2
            else:
                n_tokens = set(re.findall(r"[a-z0-9]+", nl))
                inter = len(q_tokens & n_tokens)
                score = 2 if inter and inter == len(q_tokens) else (1 if inter else 0)
            if score > best_score:
                best, best_score = name, score
        return best if best_score > 0 else None

    # ─────────────────────────────────────────────────────────────────────────
    # Structured facts
    # ─────────────────────────────────────────────────────────────────────────

    def aggregates_for(self, team) -> Optional[dict]:
        """Per-team aggregate stats (same keys as the PostgreSQL `teams` table).

        The full aggregates dict is computed lazily once per process.
        """
        name = self.resolve_team(team)
        if name is None:
            return None
        if self._aggregates is None:
            t0 = time.time()
            self._aggregates = build_team_aggregates(self.df)
            logger.info("Team aggregates computed for %d teams (%.2fs)",
                        len(self._aggregates), time.time() - t0)
        agg = self._aggregates.get(name)
        return dict(agg) if agg else None

    def league_table(self, league=None, season=None) -> List[dict]:
        """Standings: {position, team, played, wins, draws, losses, ...}."""
        return build_league_table(self.df, league=league, season=season)

    def head_to_head(self, team_a, team_b, limit: int = 10,
                     league: Optional[str] = None,
                     season: Optional[str] = None) -> List[dict]:
        """Head-to-head fixtures between two teams.

        league/season (optional) scope the search to one competition run —
        e.g. the Premier_League 2425 season — instead of all seasons.
        """
        a = self.resolve_team(team_a)
        b = self.resolve_team(team_b)
        if a is None or b is None or a == b:
            return []
        df = self.df
        mask = (
            ((df["HomeTeam"] == a) & (df["AwayTeam"] == b))
            | ((df["HomeTeam"] == b) & (df["AwayTeam"] == a))
        )
        if league:
            mask &= df["League"].astype(str) == str(league)
        if season:
            mask &= df["Season"].astype(str) == str(season)
        sub = df[mask].sort_values("_dt", ascending=False).head(limit)
        return [self._match_row(r) for _, r in sub.iterrows()]

    def recent_form(self, team, n: int = 5) -> List[dict]:
        name = self.resolve_team(team)
        if name is None:
            return []
        df = self.df
        mask = (df["HomeTeam"] == name) | (df["AwayTeam"] == name)
        sub = df[mask].sort_values("_dt", ascending=False).head(n)
        return [self._match_row(r) for _, r in sub.iterrows()]

    def last_match(self, team) -> Optional[dict]:
        rows = self.recent_form(team, n=1)
        return rows[0] if rows else None

    def league_names(self) -> List[str]:
        """Distinct league names in the data (for disambiguation in answers)."""
        if "League" in self.df.columns:
            return sorted(self.df["League"].dropna().astype(str).unique().tolist())
        return []

    def stats(self) -> dict:
        return dict(
            rows=len(self.df),
            columns=len(self.df.columns),
            teams=len(self._team_set),
            csv_path=str(self._csv_path),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _match_row(self, row) -> dict:
        return dict(
            date=str(row["Date"]),
            home_team=row["HomeTeam"],
            away_team=row["AwayTeam"],
            home_goals=pyval(row.get("FTHG")),
            away_goals=pyval(row.get("FTAG")),
            result=str(row.get("FTR")) if pd.notna(row.get("FTR")) else None,
            home_xg=pyval(row.get("Home_xG")),
            away_xg=pyval(row.get("Away_xG")),
            league=str(row.get("League")) if pd.notna(row.get("League")) else None,
            season=str(row.get("Season")) if pd.notna(row.get("Season")) else None,
        )
