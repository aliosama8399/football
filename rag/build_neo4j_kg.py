"""
Neo4j Knowledge Graph Builder
================================
Reads processed_matches.csv + team_tactics.json → populates Neo4j with:

  Nodes  : Team, League, Season, Referee
  Edges  : (Team)-[:MATCH {stats}]->(Team)
           (Team)-[:PLAYED_IN]->(League)
           (Referee)-[:OFFICIATED]->(Team)  [implicit via match props]

Each Team node gets attack_tactic + defense_tactic from extract_tactics.py.

Usage:
    # Make sure Neo4j is running, then:
    python rag/build_neo4j_kg.py
    python rag/build_neo4j_kg.py --clear  # wipe and rebuild
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
CSV_PATH     = BASE_DIR / "data" / "processed" / "processed_matches.csv"
TACTICS_PATH = BASE_DIR / "rag" / "knowledge_base" / "team_tactics.json"
CFG_PATH     = BASE_DIR / "models" / "llm_config.yaml"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f) or {}


def _safe(val):
    """Convert numpy types / NaN to Python-native types safe for Neo4j."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Build Functions
# ─────────────────────────────────────────────────────────────────────────────

def clear_database(session):
    logger.warning("Clearing entire Neo4j database …")
    session.run("MATCH (n) DETACH DELETE n")
    logger.info("Database cleared.")


def create_constraints(session):
    """Create uniqueness constraints for faster upserts."""
    constraints = [
        "CREATE CONSTRAINT team_name   IF NOT EXISTS FOR (t:Team)    REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT league_name IF NOT EXISTS FOR (l:League)  REQUIRE l.name IS UNIQUE",
        "CREATE CONSTRAINT season_id   IF NOT EXISTS FOR (s:Season)  REQUIRE s.id   IS UNIQUE",
        "CREATE CONSTRAINT ref_name    IF NOT EXISTS FOR (r:Referee) REQUIRE r.name IS UNIQUE",
    ]
    for c in constraints:
        try:
            session.run(c)
        except Exception:
            pass  # Constraint may already exist
    logger.info("Constraints ensured.")


def build_team_aggregates(df: pd.DataFrame) -> dict:
    """Compute per-team aggregate stats from the matches dataframe."""
    aggregates = {}

    for team in set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()):
        home_rows = df[df["HomeTeam"] == team]
        away_rows = df[df["AwayTeam"] == team]

        total  = len(home_rows) + len(away_rows)
        league = (home_rows["League"].mode()[0] if len(home_rows) > 0 else
                  away_rows["League"].mode()[0] if len(away_rows) > 0 else "Unknown")

        avg_gh  = home_rows["FTHG"].mean() if len(home_rows) > 0 else 0.0
        avg_ga  = away_rows["FTAG"].mean() if len(away_rows) > 0 else 0.0
        avg_xg  = pd.concat([home_rows["Home_xG"], away_rows["Away_xG"]]).mean()
        avg_xga = pd.concat([home_rows["Away_xG"], away_rows["Home_xG"]]).mean()

        seasons = sorted(set(df[
            (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
        ]["Season"].astype(str).unique()))

        aggregates[team] = {
            "league":         league,
            "total_matches":  total,
            "avg_goals_home": float(avg_gh)  if not np.isnan(avg_gh)  else 0.0,
            "avg_goals_away": float(avg_ga)  if not np.isnan(avg_ga)  else 0.0,
            "avg_xg":         float(avg_xg)  if not np.isnan(avg_xg)  else 0.0,
            "avg_xga":        float(avg_xga) if not np.isnan(avg_xga) else 0.0,
            "seasons":        seasons,
        }

    return aggregates


def create_nodes(session, df: pd.DataFrame, tactics: dict):
    """Create League, Season, Referee, and Team nodes."""

    # ── League nodes ───────────────────────────────────────────────────────
    leagues = df["League"].unique().tolist()
    for lg in leagues:
        session.run("MERGE (l:League {name: $name})", name=lg)
    logger.info("Upserted %d League nodes.", len(leagues))

    # ── Season nodes ───────────────────────────────────────────────────────
    seasons = df["Season"].unique().tolist()
    for s in seasons:
        session.run("MERGE (s:Season {id: $id})", id=str(s))
    logger.info("Upserted %d Season nodes.", len(seasons))

    # ── Referee nodes ──────────────────────────────────────────────────────
    if "Referee" in df.columns:
        refs = df["Referee"].dropna().unique().tolist()
        for ref in refs:
            avg_y  = df[df["Referee"] == ref]["HY"].mean() + df[df["Referee"] == ref]["AY"].mean()
            avg_r  = df[df["Referee"] == ref]["HR"].mean() + df[df["Referee"] == ref]["AR"].mean()
            session.run(
                "MERGE (r:Referee {name: $name}) SET r.avg_yellows=$y, r.avg_reds=$r",
                name=ref, y=_safe(avg_y), r=_safe(avg_r),
            )
        logger.info("Upserted %d Referee nodes.", len(refs))

    # ── Team nodes ────────────────────────────────────────────────────────
    team_aggs = build_team_aggregates(df)
    for team, agg in team_aggs.items():
        tactic = tactics.get(team, {})
        session.run(
            """
            MERGE (t:Team {name: $name})
            SET   t.league         = $league,
                  t.total_matches  = $total_matches,
                  t.avg_goals_home = $avg_goals_home,
                  t.avg_goals_away = $avg_goals_away,
                  t.avg_xg         = $avg_xg,
                  t.avg_xga        = $avg_xga,
                  t.seasons        = $seasons,
                  t.attack_tactic  = $attack_tactic,
                  t.defense_tactic = $defense_tactic
            """,
            name=team,
            league=agg["league"],
            total_matches=agg["total_matches"],
            avg_goals_home=agg["avg_goals_home"],
            avg_goals_away=agg["avg_goals_away"],
            avg_xg=agg["avg_xg"],
            avg_xga=agg["avg_xga"],
            seasons=agg["seasons"],
            attack_tactic=tactic.get("attack_tactic", ""),
            defense_tactic=tactic.get("defense_tactic", ""),
        )
        # PLAYED_IN relationship
        session.run(
            """
            MATCH (t:Team {name:$team}), (l:League {name:$league})
            MERGE (t)-[:PLAYED_IN]->(l)
            """,
            team=team, league=agg["league"],
        )

    logger.info("Upserted %d Team nodes with tactics.", len(team_aggs))


def create_match_edges(session, df: pd.DataFrame, batch_size: int = 500):
    """Create (Team)-[:MATCH {stats}]->(Team) edges in batches."""

    MATCH_COLS = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR",
        "Home_xG", "Away_xG",
        "HS", "AS", "HST", "AST",
        "HC", "AC", "HF", "AF",
        "HY", "AY", "HR", "AR",
        "League", "Season",
    ]

    # Only keep columns that actually exist
    cols = [c for c in MATCH_COLS if c in df.columns]
    chunk_df = df[cols].copy()
    chunk_df["Date"] = chunk_df["Date"].astype(str)
    chunk_df["Season"] = chunk_df["Season"].astype(str)

    records = chunk_df.to_dict("records")
    total   = len(records)

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        # Sanitise NaN
        clean = [{k: _safe(v) for k, v in r.items()} for r in batch]

        session.run(
            """
            UNWIND $rows AS row
            MATCH (h:Team {name: row.HomeTeam})
            MATCH (a:Team {name: row.AwayTeam})
            MERGE (h)-[m:MATCH {date: row.Date, home_team: row.HomeTeam, away_team: row.AwayTeam}]->(a)
            SET   m.home_goals  = row.FTHG,
                  m.away_goals  = row.FTAG,
                  m.result      = row.FTR,
                  m.home_xg     = row.Home_xG,
                  m.away_xg     = row.Away_xG,
                  m.home_shots  = row.HS,
                  m.away_shots  = row.AS,
                  m.home_sot    = row.HST,
                  m.away_sot    = row.AST,
                  m.home_corners= row.HC,
                  m.away_corners= row.AC,
                  m.home_fouls  = row.HF,
                  m.away_fouls  = row.AF,
                  m.home_yellows= row.HY,
                  m.away_yellows= row.AY,
                  m.home_reds   = row.HR,
                  m.away_reds   = row.AR,
                  m.league      = row.League,
                  m.season      = row.Season
            """,
            rows=clean,
        )
        logger.info("  Match edges: %d / %d", min(start + batch_size, total), total)

    logger.info("All %d MATCH edges created.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(clear: bool = False):
    cfg = _load_cfg().get("rag", {})
    uri      = os.getenv("NEO4J_URI")      or cfg.get("neo4j_uri",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER")     or cfg.get("neo4j_user",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD") or cfg.get("neo4j_password",  "password")

    logger.info("Connecting to Neo4j at %s …", uri)
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Load data
    logger.info("Loading %s …", CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    logger.info("  %d rows, %d columns", *df.shape)

    tactics = {}
    if TACTICS_PATH.exists():
        with open(TACTICS_PATH) as f:
            tactics = json.load(f)
        logger.info("Loaded tactics for %d teams.", len(tactics))
    else:
        logger.warning("team_tactics.json not found. Run extract_tactics.py first. Continuing without tactics.")

    with driver.session() as session:
        if clear:
            clear_database(session)

        create_constraints(session)
        create_nodes(session, df, tactics)
        create_match_edges(session, df)

    driver.close()
    logger.info("✅  Neo4j Knowledge Graph build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Neo4j football knowledge graph")
    parser.add_argument("--clear", action="store_true", help="Clear the database before building")
    args = parser.parse_args()
    main(clear=args.clear)
