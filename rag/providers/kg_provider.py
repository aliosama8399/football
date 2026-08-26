"""
Knowledge Graph Providers
==========================
Implements BaseKGProvider for:
  - Neo4jProvider    : uses Cypher queries via the neo4j driver
  - PostgreSQLProvider: uses SQL via psycopg2 / SQLAlchemy

Factory:
    from rag.providers.kg_provider import get_kg_provider
    kg = get_kg_provider("neo4j")   # or "postgres"
"""

import yaml
import logging
from pathlib import Path
from typing import Optional
from neo4j import GraphDatabase

from rag.providers.base import BaseKGProvider

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).parent.parent.parent / "models" / "llm_config.yaml"


def _load_rag_cfg() -> dict:
    if not _CFG_PATH.exists():
        return {}
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j Provider
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jProvider(BaseKGProvider):
    """Reads from a Neo4j database using Cypher queries."""

    def __init__(self):
        import os
        cfg  = _load_rag_cfg().get("rag", {})
        # env overrides first (docker-compose), then llm_config.yaml, then defaults
        self.uri      = os.getenv("NEO4J_URI")      or cfg.get("neo4j_uri",      "bolt://localhost:7687")
        self.user     = os.getenv("NEO4J_USER")     or cfg.get("neo4j_user",     "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD") or cfg.get("neo4j_password",  "password")
        self._driver  = None

    # ── Connection ─────────────────────────────────────────────────────────
    def connect(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info("Neo4j connected → %s", self.uri)

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def _run(self, query: str, **params) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    # ── Queries ────────────────────────────────────────────────────────────
    def get_team_profile(self, team_name: str) -> dict:
        rows = self._run(
            """
            MATCH (t:Team {name: $name})
            RETURN t.name          AS name,
                   t.league        AS league,
                   t.total_matches AS total_matches,
                   t.avg_xg        AS avg_xg,
                   t.avg_xga       AS avg_xga,
                   t.avg_goals_home AS avg_goals_home,
                   t.avg_goals_away AS avg_goals_away,
                   t.attack_tactic  AS attack_tactic,
                   t.defense_tactic AS defense_tactic
            """,
            name=team_name,
        )
        return rows[0] if rows else {}

    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> list[dict]:
        return self._run(
            """
            MATCH (h:Team {name: $a})-[m:MATCH]->(aw:Team {name: $b})
            RETURN m.date        AS date,
                   h.name        AS home_team,
                   aw.name       AS away_team,
                   m.home_goals  AS home_goals,
                   m.away_goals  AS away_goals,
                   m.result      AS result,
                   m.home_xg     AS home_xg,
                   m.away_xg     AS away_xg,
                   m.league      AS league,
                   m.season      AS season
            ORDER BY m.date DESC
            LIMIT $limit
            """,
            a=team_a, b=team_b, limit=limit,
        )

    def get_recent_form(self, team_name: str, n: int = 5) -> list[dict]:
        return self._run(
            """
            MATCH (t:Team {name: $name})-[m:MATCH]->(:Team)
            RETURN m.date       AS date,
                   t.name       AS home_team,
                   m.away_team  AS away_team,
                   m.home_goals AS home_goals,
                   m.away_goals AS away_goals,
                   m.result     AS result,
                   m.league     AS league
            ORDER BY m.date DESC
            LIMIT $n
            UNION
            MATCH (:Team)-[m:MATCH]->(t:Team {name: $name})
            RETURN m.home_team   AS home_team,
                   t.name        AS away_team,
                   m.home_goals  AS home_goals,
                   m.away_goals  AS away_goals,
                   m.result      AS result,
                   m.date        AS date,
                   m.league      AS league
            ORDER BY m.date DESC
            LIMIT $n
            """,
            name=team_name, n=n,
        )[:n]

    def get_league_teams(self, league: str, season: str = None) -> list[str]:
        if season:
            rows = self._run(
                "MATCH (t:Team)-[:PLAYED_IN]->(l:League {name: $league}) "
                "WHERE $season IN t.seasons RETURN t.name AS name",
                league=league, season=season,
            )
        else:
            rows = self._run(
                "MATCH (t:Team)-[:PLAYED_IN]->(l:League {name: $league}) RETURN t.name AS name",
                league=league,
            )
        return [r["name"] for r in rows]

    def get_match(self, home_team: str, away_team: str, date: str = None) -> dict:
        if date:
            rows = self._run(
                "MATCH (h:Team {name:$h})-[m:MATCH {date:$date}]->(a:Team {name:$a}) RETURN m",
                h=home_team, a=away_team, date=date,
            )
        else:
            rows = self._run(
                "MATCH (h:Team {name:$h})-[m:MATCH]->(a:Team {name:$a}) "
                "RETURN m ORDER BY m.date DESC LIMIT 1",
                h=home_team, a=away_team,
            )
        return rows[0] if rows else {}


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL Provider
# ─────────────────────────────────────────────────────────────────────────────

class PostgreSQLProvider(BaseKGProvider):
    """Reads from a PostgreSQL database using psycopg2 / SQLAlchemy."""

    def __init__(self):
        import os
        cfg = _load_rag_cfg().get("rag", {})
        self.dsn = os.getenv("POSTGRES_DSN") or cfg.get(
            "postgres_dsn",
            "postgresql://postgres:123@localhost:5432/football_rag",
        )
        self._conn   = None
        self._engine = None

    # ── Connection ─────────────────────────────────────────────────────────
    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(self.dsn)
        logger.info("PostgreSQL connected → %s", self.dsn.split("@")[-1])

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Queries ────────────────────────────────────────────────────────────
    def get_team_profile(self, team_name: str) -> dict:
        rows = self._query(
            """
            SELECT name, league, total_matches,
                   avg_goals_home, avg_goals_away,
                   avg_xg, avg_xga,
                   avg_shots, avg_shots_against,
                   avg_sot, avg_sot_against,
                   avg_corners, avg_fouls, avg_yellows,
                   win_rate, draw_rate, loss_rate, clean_sheet_rate,
                   attack_tactic, defense_tactic,
                   attack_headline, defense_headline,
                   strengths, weaknesses
            FROM teams
            WHERE name = %s
            """,
            (team_name,),
        )
        return rows[0] if rows else {}

    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> list[dict]:
        return self._query(
            """
            SELECT date, home_team, away_team,
                   home_goals, away_goals, result,
                   home_xg, away_xg, league, season
            FROM matches
            WHERE home_team = %s AND away_team = %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (team_a, team_b, limit),
        )

    def get_recent_form(self, team_name: str, n: int = 5) -> list[dict]:
        return self._query(
            """
            SELECT date, home_team, away_team, home_goals, away_goals, result, league
            FROM matches
            WHERE home_team = %s OR away_team = %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (team_name, team_name, n),
        )

    def get_league_teams(self, league: str, season: str = None) -> list[str]:
        if season:
            rows = self._query(
                "SELECT DISTINCT home_team AS name FROM matches WHERE league=%s AND season=%s",
                (league, season),
            )
        else:
            rows = self._query(
                "SELECT DISTINCT home_team AS name FROM matches WHERE league=%s",
                (league,),
            )
        return [r["name"] for r in rows]

    def get_match(self, home_team: str, away_team: str, date: str = None) -> dict:
        if date:
            rows = self._query(
                "SELECT * FROM matches WHERE home_team=%s AND away_team=%s AND date=%s",
                (home_team, away_team, date),
            )
        else:
            rows = self._query(
                "SELECT * FROM matches WHERE home_team=%s AND away_team=%s ORDER BY date DESC LIMIT 1",
                (home_team, away_team),
            )
        return rows[0] if rows else {}

    def get_team_detailed_form(self, team_name: str, n: int = 5) -> list[dict]:
        """
        Return the last N matches for a team with all rolling-form columns.
        Includes xG, xGA, shots, SOT, corners, fouls, yellows rolling averages.
        """
        return self._query(
            """
            SELECT date, home_team, away_team,
                   home_goals, away_goals, result,
                   home_xg, away_xg,
                   home_shots, away_shots,
                   home_sot, away_sot,
                   home_corners, away_corners,
                   home_form_5, away_form_5,
                   home_gf_5, away_gf_5,
                   home_ga_5, away_ga_5,
                   home_xg_5, away_xg_5,
                   home_xga_5, away_xga_5,
                   home_shots_5, away_shots_5,
                   home_sot_5, away_sot_5,
                   home_corners_5, away_corners_5,
                   home_yellows_5, away_yellows_5,
                   league, season
            FROM matches
            WHERE home_team = %s OR away_team = %s
            ORDER BY date DESC
            LIMIT %s
            """,
            (team_name, team_name, n),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_KG_REGISTRY = {
    "neo4j":    Neo4jProvider,
    "postgres": PostgreSQLProvider,
}


def get_kg_provider(name: str = None) -> BaseKGProvider:
    """
    Return and connect a KG provider.

    Args:
        name: "neo4j" | "postgres". Defaults to llm_config.yaml rag.kg_provider.
    """
    if name is None:
        cfg  = _load_rag_cfg()
        name = cfg.get("rag", {}).get("kg_provider", "neo4j")

    name = name.lower()
    if name not in _KG_REGISTRY:
        raise ValueError(f"Unknown KG provider '{name}'. Choose from: {list(_KG_REGISTRY)}")

    provider = _KG_REGISTRY[name]()
    provider.connect()
    return provider
