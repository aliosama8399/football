"""
PostgreSQL Database Builder
==============================
Creates and populates two tables in PostgreSQL:

  teams   : team name, league, rich aggregate stats, full tactical profile
  matches : every match row from processed_matches.csv (all rolling features)

New team columns vs. the old schema:
  win_rate, draw_rate, loss_rate
  avg_shots, avg_shots_against, avg_sot, avg_sot_against
  avg_corners, avg_fouls, avg_yellows
  clean_sheet_rate
  attack_headline, defense_headline
  strengths[], weaknesses[]

Usage:
    python rag/build_postgres_db.py
    python rag/build_postgres_db.py --drop   # drop + recreate tables
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
    name                 TEXT PRIMARY KEY,
    league               TEXT,
    total_matches        INTEGER,

    -- Goal scoring
    avg_goals_home       FLOAT,
    avg_goals_away       FLOAT,
    avg_xg               FLOAT,
    avg_xga              FLOAT,

    -- Shot & chance creation
    avg_shots            FLOAT,
    avg_shots_against    FLOAT,
    avg_sot              FLOAT,
    avg_sot_against      FLOAT,

    -- Set pieces & discipline
    avg_corners          FLOAT,
    avg_fouls            FLOAT,
    avg_yellows          FLOAT,

    -- Results
    win_rate             FLOAT,
    draw_rate            FLOAT,
    loss_rate            FLOAT,
    clean_sheet_rate     FLOAT,

    seasons              TEXT[],

    -- Tactical profile (full paragraphs)
    attack_tactic        TEXT,
    defense_tactic       TEXT,

    -- Tactical headlines (one-liners)
    attack_headline      TEXT,
    defense_headline     TEXT,

    -- Strength / weakness bullets
    strengths            TEXT[],
    weaknesses           TEXT[],

    created_at           TIMESTAMP DEFAULT NOW()
);
"""

CREATE_MATCHES = """
CREATE TABLE IF NOT EXISTS matches (
    id             SERIAL PRIMARY KEY,
    date           DATE,
    home_team      TEXT,
    away_team      TEXT,

    -- Full-time result
    home_goals     INTEGER,
    away_goals     INTEGER,
    result         TEXT,

    -- xG
    home_xg        FLOAT,
    away_xg        FLOAT,

    -- Shots
    home_shots     INTEGER,
    away_shots     INTEGER,
    home_sot       INTEGER,
    away_sot       INTEGER,

    -- Set pieces
    home_corners   INTEGER,
    away_corners   INTEGER,

    -- Discipline
    home_fouls     INTEGER,
    away_fouls     INTEGER,
    home_yellows   INTEGER,
    away_yellows   INTEGER,
    home_reds      INTEGER,
    away_reds      INTEGER,

    -- Half-time
    home_ht_goals  INTEGER,
    away_ht_goals  INTEGER,
    ht_result      TEXT,

    -- Context
    league         TEXT,
    season         TEXT,

    -- Rolling form (last-5)
    home_form_5           FLOAT,
    away_form_5           FLOAT,
    home_gf_5             FLOAT,
    away_gf_5             FLOAT,
    home_ga_5             FLOAT,
    away_ga_5             FLOAT,
    home_xg_5             FLOAT,
    away_xg_5             FLOAT,
    home_xga_5            FLOAT,
    away_xga_5            FLOAT,
    home_shots_5          FLOAT,
    away_shots_5          FLOAT,
    home_shots_against_5  FLOAT,
    away_shots_against_5  FLOAT,
    home_sot_5            FLOAT,
    away_sot_5            FLOAT,
    home_sot_against_5    FLOAT,
    away_sot_against_5    FLOAT,
    home_corners_5        FLOAT,
    away_corners_5        FLOAT,
    home_corners_against_5 FLOAT,
    away_corners_against_5 FLOAT,
    home_fouls_5          FLOAT,
    away_fouls_5          FLOAT,
    home_fouls_against_5  FLOAT,
    away_fouls_against_5  FLOAT,
    home_yellows_5        FLOAT,
    away_yellows_5        FLOAT,
    home_reds_5           FLOAT,
    away_reds_5           FLOAT,

    -- H2H
    h2h_matches    INTEGER,
    h2h_home_wins  INTEGER,
    h2h_away_wins  INTEGER,
    h2h_draws      INTEGER,
    h2h_home_goals FLOAT,
    h2h_away_goals FLOAT,

    -- Derived
    total_goals    INTEGER,
    goal_diff      INTEGER,
    over_2_5       INTEGER,
    btts           INTEGER,
    over_1_5       INTEGER,
    over_3_5       INTEGER,
    home_clean_sheet INTEGER,
    away_clean_sheet INTEGER,

    -- Referee
    ref_avg_yellows FLOAT,
    ref_avg_reds    FLOAT,
    ref_strictness  FLOAT,

    UNIQUE(date, home_team, away_team)
);

CREATE INDEX IF NOT EXISTS idx_matches_home   ON matches(home_team);
CREATE INDEX IF NOT EXISTS idx_matches_away   ON matches(away_team);
CREATE INDEX IF NOT EXISTS idx_matches_date   ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanmean(series: pd.Series) -> float:
    v = series.mean()
    return float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0


def build_team_aggregates(df: pd.DataFrame) -> dict:
    aggregates = {}
    all_teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())

    for team in all_teams:
        home_rows = df[df["HomeTeam"] == team]
        away_rows = df[df["AwayTeam"] == team]
        all_rows  = pd.concat([home_rows, away_rows])
        total     = len(home_rows) + len(away_rows)

        if total == 0:
            continue

        # League
        league = (
            home_rows["League"].mode()[0] if len(home_rows) > 0 else
            away_rows["League"].mode()[0] if len(away_rows) > 0 else "Unknown"
        )

        # Goals
        avg_gh  = _nanmean(home_rows["FTHG"]) if len(home_rows) > 0 else 0.0
        avg_ga  = _nanmean(away_rows["FTAG"]) if len(away_rows) > 0 else 0.0

        # xG (team's own xG when playing home or away)
        xg_home  = home_rows["Home_xG"] if "Home_xG" in home_rows.columns else pd.Series(dtype=float)
        xg_away  = away_rows["Away_xG"] if "Away_xG" in away_rows.columns else pd.Series(dtype=float)
        avg_xg   = _nanmean(pd.concat([xg_home, xg_away]))

        # xGA (opponent's xG against this team)
        xga_home = home_rows["Away_xG"] if "Away_xG" in home_rows.columns else pd.Series(dtype=float)
        xga_away = away_rows["Home_xG"] if "Home_xG" in away_rows.columns else pd.Series(dtype=float)
        avg_xga  = _nanmean(pd.concat([xga_home, xga_away]))

        # Shots
        sh_h  = home_rows["HS"] if "HS" in home_rows.columns else pd.Series(dtype=float)
        sh_a  = away_rows["AS"] if "AS" in away_rows.columns else pd.Series(dtype=float)
        avg_shots = _nanmean(pd.concat([sh_h, sh_a]))

        sha_h = home_rows["AS"] if "AS" in home_rows.columns else pd.Series(dtype=float)
        sha_a = away_rows["HS"] if "HS" in away_rows.columns else pd.Series(dtype=float)
        avg_shots_against = _nanmean(pd.concat([sha_h, sha_a]))

        sot_h  = home_rows["HST"] if "HST" in home_rows.columns else pd.Series(dtype=float)
        sot_a  = away_rows["AST"] if "AST" in away_rows.columns else pd.Series(dtype=float)
        avg_sot = _nanmean(pd.concat([sot_h, sot_a]))

        sota_h = home_rows["AST"] if "AST" in home_rows.columns else pd.Series(dtype=float)
        sota_a = away_rows["HST"] if "HST" in away_rows.columns else pd.Series(dtype=float)
        avg_sot_against = _nanmean(pd.concat([sota_h, sota_a]))

        # Corners
        c_h = home_rows["HC"] if "HC" in home_rows.columns else pd.Series(dtype=float)
        c_a = away_rows["AC"] if "AC" in away_rows.columns else pd.Series(dtype=float)
        avg_corners = _nanmean(pd.concat([c_h, c_a]))

        # Fouls
        f_h = home_rows["HF"] if "HF" in home_rows.columns else pd.Series(dtype=float)
        f_a = away_rows["AF"] if "AF" in away_rows.columns else pd.Series(dtype=float)
        avg_fouls = _nanmean(pd.concat([f_h, f_a]))

        # Yellows
        y_h = home_rows["HY"] if "HY" in home_rows.columns else pd.Series(dtype=float)
        y_a = away_rows["AY"] if "AY" in away_rows.columns else pd.Series(dtype=float)
        avg_yellows = _nanmean(pd.concat([y_h, y_a]))

        # Result rates
        home_wins  = len(home_rows[home_rows["FTR"] == "H"]) if "FTR" in home_rows.columns else 0
        away_wins  = len(away_rows[away_rows["FTR"] == "A"]) if "FTR" in away_rows.columns else 0
        home_draws = len(home_rows[home_rows["FTR"] == "D"]) if "FTR" in home_rows.columns else 0
        away_draws = len(away_rows[away_rows["FTR"] == "D"]) if "FTR" in away_rows.columns else 0
        home_loss  = len(home_rows[home_rows["FTR"] == "A"]) if "FTR" in home_rows.columns else 0
        away_loss  = len(away_rows[away_rows["FTR"] == "H"]) if "FTR" in away_rows.columns else 0

        win_rate  = (home_wins + away_wins)  / total
        draw_rate = (home_draws + away_draws) / total
        loss_rate = (home_loss + away_loss)  / total

        # Clean sheets
        cs_h = len(home_rows[home_rows["FTAG"] == 0]) if "FTAG" in home_rows.columns else 0
        cs_a = len(away_rows[away_rows["FTHG"] == 0]) if "FTHG" in away_rows.columns else 0
        clean_sheet_rate = (cs_h + cs_a) / total

        seasons = sorted(set(all_rows["Season"].astype(str).unique()))

        aggregates[team] = dict(
            league=league,
            total_matches=total,
            avg_goals_home=avg_gh,
            avg_goals_away=avg_ga,
            avg_xg=avg_xg,
            avg_xga=avg_xga,
            avg_shots=avg_shots,
            avg_shots_against=avg_shots_against,
            avg_sot=avg_sot,
            avg_sot_against=avg_sot_against,
            avg_corners=avg_corners,
            avg_fouls=avg_fouls,
            avg_yellows=avg_yellows,
            win_rate=win_rate,
            draw_rate=draw_rate,
            loss_rate=loss_rate,
            clean_sheet_rate=clean_sheet_rate,
            seasons=seasons,
        )

    return aggregates


# ─────────────────────────────────────────────────────────────────────────────
# Populate
# ─────────────────────────────────────────────────────────────────────────────

def insert_teams(cur, df: pd.DataFrame, tactics: dict):
    aggregates = build_team_aggregates(df)
    rows = []
    for team, agg in aggregates.items():
        t = tactics.get(team, {})
        rows.append((
            team,
            agg["league"],
            agg["total_matches"],
            agg["avg_goals_home"],
            agg["avg_goals_away"],
            agg["avg_xg"],
            agg["avg_xga"],
            agg["avg_shots"],
            agg["avg_shots_against"],
            agg["avg_sot"],
            agg["avg_sot_against"],
            agg["avg_corners"],
            agg["avg_fouls"],
            agg["avg_yellows"],
            agg["win_rate"],
            agg["draw_rate"],
            agg["loss_rate"],
            agg["clean_sheet_rate"],
            agg["seasons"],
            t.get("attack_tactic",    ""),
            t.get("defense_tactic",   ""),
            t.get("attack_headline",  ""),
            t.get("defense_headline", ""),
            t.get("strengths",  []),
            t.get("weaknesses", []),
        ))

    execute_values(cur,
        """
        INSERT INTO teams (
            name, league, total_matches,
            avg_goals_home, avg_goals_away, avg_xg, avg_xga,
            avg_shots, avg_shots_against, avg_sot, avg_sot_against,
            avg_corners, avg_fouls, avg_yellows,
            win_rate, draw_rate, loss_rate, clean_sheet_rate,
            seasons,
            attack_tactic, defense_tactic,
            attack_headline, defense_headline,
            strengths, weaknesses
        )
        VALUES %s
        ON CONFLICT (name) DO UPDATE SET
            league              = EXCLUDED.league,
            total_matches       = EXCLUDED.total_matches,
            avg_goals_home      = EXCLUDED.avg_goals_home,
            avg_goals_away      = EXCLUDED.avg_goals_away,
            avg_xg              = EXCLUDED.avg_xg,
            avg_xga             = EXCLUDED.avg_xga,
            avg_shots           = EXCLUDED.avg_shots,
            avg_shots_against   = EXCLUDED.avg_shots_against,
            avg_sot             = EXCLUDED.avg_sot,
            avg_sot_against     = EXCLUDED.avg_sot_against,
            avg_corners         = EXCLUDED.avg_corners,
            avg_fouls           = EXCLUDED.avg_fouls,
            avg_yellows         = EXCLUDED.avg_yellows,
            win_rate            = EXCLUDED.win_rate,
            draw_rate           = EXCLUDED.draw_rate,
            loss_rate           = EXCLUDED.loss_rate,
            clean_sheet_rate    = EXCLUDED.clean_sheet_rate,
            seasons             = EXCLUDED.seasons,
            attack_tactic       = EXCLUDED.attack_tactic,
            defense_tactic      = EXCLUDED.defense_tactic,
            attack_headline     = EXCLUDED.attack_headline,
            defense_headline    = EXCLUDED.defense_headline,
            strengths           = EXCLUDED.strengths,
            weaknesses          = EXCLUDED.weaknesses
        """,
        rows,
    )
    logger.info("Inserted/updated %d team rows.", len(rows))


def insert_matches(cur, df: pd.DataFrame, batch_size: int = 500):
    col_map = {
        "Date":     "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR":  "result",
        "HTHG": "home_ht_goals",
        "HTAG": "away_ht_goals",
        "HTR":  "ht_result",
        "Home_xG": "home_xg",
        "Away_xG": "away_xg",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_sot",
        "AST": "away_sot",
        "HC": "home_corners",
        "AC": "away_corners",
        "HF": "home_fouls",
        "AF": "away_fouls",
        "HY": "home_yellows",
        "AY": "away_yellows",
        "HR": "home_reds",
        "AR": "away_reds",
        "League": "league",
        "Season": "season",
        # Rolling form
        "HomeForm_5":           "home_form_5",
        "AwayForm_5":           "away_form_5",
        "HomeGF_5":             "home_gf_5",
        "AwayGF_5":             "away_gf_5",
        "HomeGA_5":             "home_ga_5",
        "AwayGA_5":             "away_ga_5",
        "HomexG_5":             "home_xg_5",
        "AwayxG_5":             "away_xg_5",
        "HomexGA_5":            "home_xga_5",
        "AwayxGA_5":            "away_xga_5",
        "HomeShots_5":          "home_shots_5",
        "AwayShots_5":          "away_shots_5",
        "HomeShotsAgainst_5":   "home_shots_against_5",
        "AwayShotsAgainst_5":   "away_shots_against_5",
        "HomeSOT_5":            "home_sot_5",
        "AwaySOT_5":            "away_sot_5",
        "HomeSOTAgainst_5":     "home_sot_against_5",
        "AwaySOTAgainst_5":     "away_sot_against_5",
        "HomeCorners_5":        "home_corners_5",
        "AwayCorners_5":        "away_corners_5",
        "HomeCornersAgainst_5": "home_corners_against_5",
        "AwayCornersAgainst_5": "away_corners_against_5",
        "HomeFouls_5":          "home_fouls_5",
        "AwayFouls_5":          "away_fouls_5",
        "HomeFoulsAgainst_5":   "home_fouls_against_5",
        "AwayFoulsAgainst_5":   "away_fouls_against_5",
        "HomeYellows_5":        "home_yellows_5",
        "AwayYellows_5":        "away_yellows_5",
        "HomeReds_5":           "home_reds_5",
        "AwayReds_5":           "away_reds_5",
        # H2H
        "H2H_Matches":   "h2h_matches",
        "H2H_HomeWins":  "h2h_home_wins",
        "H2H_AwayWins":  "h2h_away_wins",
        "H2H_Draws":     "h2h_draws",
        "H2H_HomeGoals": "h2h_home_goals",
        "H2H_AwayGoals": "h2h_away_goals",
        # Derived
        "TotalGoals":      "total_goals",
        "GoalDiff":        "goal_diff",
        "Over2.5":         "over_2_5",
        "BTTS":            "btts",
        "Over1.5":         "over_1_5",
        "Over3.5":         "over_3_5",
        "HomeCleanSheet":  "home_clean_sheet",
        "AwayCleanSheet":  "away_clean_sheet",
        # Referee
        "Ref_AvgYellows":  "ref_avg_yellows",
        "Ref_AvgReds":     "ref_avg_reds",
        "Ref_Strictness":  "ref_strictness",
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
        logger.info("CSV loaded: %d rows, %d columns", len(df), len(df.columns))

        # Load tactics
        tactics = {}
        if TACTICS_PATH.exists():
            with open(TACTICS_PATH) as f:
                tactics = json.load(f)
            logger.info("Tactics loaded for %d teams.", len(tactics))
        else:
            logger.warning(
                "team_tactics.json not found — run `python rag/extract_tactics.py` first."
            )

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
