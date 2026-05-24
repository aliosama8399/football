"""
PostgreSQL Database Builder
==============================
Creates and populates two tables in PostgreSQL:

  teams   : team name, league, aggregate stats, attack_tactic, defense_tactic
  matches : every match row from processed_matches.csv

Usage:
    python rag/build_postgres_db.py
    python rag/build_postgres_db.py --drop  # drop+recreate tables
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import numpy as np
import pandas as pd
import psycopg2
import yaml
import json
from pathlib import Path
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR     = Path(__file__).parent.parent
CSV_PATH     = BASE_DIR / "data" / "processed" / "processed_matches.csv"
TACTICS_PATH = BASE_DIR / "rag" / "knowledge_base" / "team_tactics.json"
CFG_PATH     = BASE_DIR / "models" / "llm_config.yaml"


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f) or {}


def _safe(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TEAMS = """
CREATE TABLE IF NOT EXISTS teams (
    name             TEXT PRIMARY KEY,
    league           TEXT,
    total_matches    INTEGER,
    avg_goals_home   FLOAT,
    avg_goals_away   FLOAT,
    avg_xg           FLOAT,
    avg_xga          FLOAT,
    seasons          TEXT[],
    attack_tactic    TEXT,
    defense_tactic   TEXT,
    created_at       TIMESTAMP DEFAULT NOW()
);
"""

CREATE_MATCHES = """
CREATE TABLE IF NOT EXISTS matches (
    id             SERIAL PRIMARY KEY,
    date           DATE,
    home_team      TEXT,
    away_team      TEXT,
    home_goals     INTEGER,
    away_goals     INTEGER,
    result         TEXT,
    home_xg        FLOAT,
    away_xg        FLOAT,
    home_shots     INTEGER,
    away_shots     INTEGER,
    home_sot       INTEGER,
    away_sot       INTEGER,
    home_corners   INTEGER,
    away_corners   INTEGER,
    home_fouls     INTEGER,
    away_fouls     INTEGER,
    home_yellows   INTEGER,
    away_yellows   INTEGER,
    home_reds      INTEGER,
    away_reds      INTEGER,
    league         TEXT,
    season         TEXT,
    -- Rolling form features
    home_form_5    FLOAT,
    away_form_5    FLOAT,
    home_gf_5      FLOAT,
    away_gf_5      FLOAT,
    home_ga_5      FLOAT,
    away_ga_5      FLOAT,
    home_xg_5      FLOAT,
    away_xg_5      FLOAT,
    -- H2H
    h2h_matches    INTEGER,
    h2h_home_wins  INTEGER,
    h2h_away_wins  INTEGER,
    h2h_draws      INTEGER,
    UNIQUE(date, home_team, away_team)
);

CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_team);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_team);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Populate
# ─────────────────────────────────────────────────────────────────────────────

def build_team_aggregates(df: pd.DataFrame) -> dict:
    aggregates = {}
    for team in set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()):
        home_rows = df[df["HomeTeam"] == team]
        away_rows = df[df["AwayTeam"] == team]
        total  = len(home_rows) + len(away_rows)
        league = (home_rows["League"].mode()[0] if len(home_rows) > 0 else
                  away_rows["League"].mode()[0] if len(away_rows) > 0 else "Unknown")
        avg_gh  = home_rows["FTHG"].mean() if len(home_rows) > 0 else 0.0
        avg_ga  = away_rows["FTAG"].mean() if len(away_rows) > 0 else 0.0
        avg_xg  = pd.concat([home_rows.get("Home_xG", pd.Series(dtype=float)),
                              away_rows.get("Away_xG", pd.Series(dtype=float))]).mean()
        avg_xga = pd.concat([home_rows.get("Away_xG", pd.Series(dtype=float)),
                              away_rows.get("Home_xG", pd.Series(dtype=float))]).mean()
        seasons = sorted(set(df[
            (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
        ]["Season"].astype(str).unique()))
        aggregates[team] = dict(
            league=league, total_matches=total,
            avg_goals_home=float(avg_gh) if not np.isnan(avg_gh) else 0.0,
            avg_goals_away=float(avg_ga) if not np.isnan(avg_ga) else 0.0,
            avg_xg =float(avg_xg)  if not np.isnan(avg_xg)  else 0.0,
            avg_xga=float(avg_xga) if not np.isnan(avg_xga) else 0.0,
            seasons=seasons,
        )
    return aggregates


def insert_teams(cur, df: pd.DataFrame, tactics: dict):
    aggregates = build_team_aggregates(df)
    rows = []
    for team, agg in aggregates.items():
        tactic = tactics.get(team, {})
        rows.append((
            team,
            agg["league"],
            agg["total_matches"],
            agg["avg_goals_home"],
            agg["avg_goals_away"],
            agg["avg_xg"],
            agg["avg_xga"],
            agg["seasons"],
            tactic.get("attack_tactic",  ""),
            tactic.get("defense_tactic", ""),
        ))

    execute_values(cur,
        """
        INSERT INTO teams
            (name, league, total_matches, avg_goals_home, avg_goals_away,
             avg_xg, avg_xga, seasons, attack_tactic, defense_tactic)
        VALUES %s
        ON CONFLICT (name) DO UPDATE SET
            league          = EXCLUDED.league,
            total_matches   = EXCLUDED.total_matches,
            avg_goals_home  = EXCLUDED.avg_goals_home,
            avg_goals_away  = EXCLUDED.avg_goals_away,
            avg_xg          = EXCLUDED.avg_xg,
            avg_xga         = EXCLUDED.avg_xga,
            seasons         = EXCLUDED.seasons,
            attack_tactic   = EXCLUDED.attack_tactic,
            defense_tactic  = EXCLUDED.defense_tactic
        """,
        rows,
    )
    logger.info("Inserted/updated %d team rows.", len(rows))


def insert_matches(cur, df: pd.DataFrame, batch_size: int = 500):
    col_map = {
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "Home_xG": "home_xg", "Away_xG": "away_xg",
        "HS": "home_shots", "AS": "away_shots",
        "HST": "home_sot", "AST": "away_sot",
        "HC": "home_corners", "AC": "away_corners",
        "HF": "home_fouls", "AF": "away_fouls",
        "HY": "home_yellows", "AY": "away_yellows",
        "HR": "home_reds", "AR": "away_reds",
        "League": "league", "Season": "season",
        "HomeForm_5": "home_form_5", "AwayForm_5": "away_form_5",
        "HomeGF_5": "home_gf_5", "AwayGF_5": "away_gf_5",
        "HomeGA_5": "home_ga_5", "AwayGA_5": "away_ga_5",
        "HomexG_5": "home_xg_5", "AwayxG_5": "away_xg_5",
        "H2H_Matches": "h2h_matches",
        "H2H_HomeWins": "h2h_home_wins",
        "H2H_AwayWins": "h2h_away_wins",
        "H2H_Draws": "h2h_draws",
    }

    cols_present = {k: v for k, v in col_map.items() if k in df.columns}
    sub = df[list(cols_present.keys())].rename(columns=cols_present).copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.date

    db_cols = list(sub.columns)
    records = [
        tuple(_safe(row[c]) for c in db_cols)
        for _, row in sub.iterrows()
    ]

    insert_sql = f"""
        INSERT INTO matches ({', '.join(db_cols)})
        VALUES %s
        ON CONFLICT (date, home_team, away_team) DO NOTHING
    """

    total = len(records)
    for start in range(0, total, batch_size):
        execute_values(cur, insert_sql, records[start:start + batch_size])
        logger.info("  Matches: %d / %d", min(start + batch_size, total), total)

    logger.info("Inserted %d match rows.", total)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(drop: bool = False):
    cfg = _load_cfg().get("rag", {})
    dsn = cfg.get("postgres_dsn", "postgresql://postgres:password@localhost:5432/football_rag")

    logger.info("Connecting to PostgreSQL …")
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur  = conn.cursor()

    try:
        if drop:
            logger.warning("Dropping existing tables …")
            cur.execute("DROP TABLE IF EXISTS matches CASCADE")
            cur.execute("DROP TABLE IF EXISTS teams CASCADE")
            conn.commit()

        # Create schema
        cur.execute(CREATE_TEAMS)
        for stmt in CREATE_MATCHES.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        conn.commit()
        logger.info("Schema ready.")

        # Load data
        df = pd.read_csv(CSV_PATH, low_memory=False)
        logger.info("CSV loaded: %d rows", len(df))

        tactics = {}
        if TACTICS_PATH.exists():
            with open(TACTICS_PATH) as f:
                tactics = json.load(f)
            logger.info("Tactics loaded for %d teams.", len(tactics))
        else:
            logger.warning("team_tactics.json not found — continuing without tactics.")

        insert_teams(cur, df, tactics)
        conn.commit()

        insert_matches(cur, df)
        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error("Error: %s", e)
        raise
    finally:
        cur.close()
        conn.close()

    logger.info("✅  PostgreSQL database build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build PostgreSQL football database")
    parser.add_argument("--drop", action="store_true", help="Drop tables before recreating")
    args = parser.parse_args()
    main(drop=args.drop)
