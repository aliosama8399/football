"""
Shared team statistics — the single source of truth for team-level
aggregations used by BOTH the PostgreSQL builder (rag/build_postgres_db.py)
and the knowledge base (rag/knowledge_base/*).

Any change here automatically applies to every consumer, so the DB tables,
GraphQL numbers and KB answers can never drift.

Usage:
    from data.team_stats import build_team_aggregates, build_league_table
    agg = build_team_aggregates(df)              # {team: {league, avg_goals_home, ...}}
    tbl = build_league_table(df, league="Premier_League", season="2324")
"""

import numpy as np
import pandas as pd


def pyval(val):
    """Convert numpy scalars / NaN to plain Python types (JSON-safe)."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _nanmean(series: pd.Series) -> float:
    v = series.mean()
    return float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0


def build_team_aggregates(df: pd.DataFrame) -> dict:
    """Per-team aggregate stats (goals, xG, shots, results, clean sheets …).

    Keys match the `teams` table columns, so a KB profile built from the CSV
    has exactly the same shape as a profile fetched from PostgreSQL.
    """
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


def build_league_table(df: pd.DataFrame, league=None, season=None) -> list:
    """Standings table from match rows (3-1-0 points), JSON-safe.

    Args:
        df: processed matches DataFrame
        league: 'League' name ('Premier_League') or 'Div' code ('E0') or None
        season: 'Season' value ('2324') or None (all seasons)

    Returns:
        List of dicts sorted by points, goal difference, goals for:
        {position, team, played, wins, draws, losses,
         goals_for, goals_against, goal_diff, points}
    """
    mask = pd.Series(True, index=df.index)

    if league:
        l = str(league).strip().lower().replace(" ", "_")
        league_mask = pd.Series(False, index=df.index)
        if "League" in df.columns:
            league_mask |= (
                df["League"].astype(str).str.strip().str.lower()
                .str.replace(" ", "_").eq(l)
            )
        if "Div" in df.columns:
            league_mask |= (
                df["Div"].astype(str).str.strip().str.lower().eq(l)
            )
        mask &= league_mask

    if season:
        mask &= df["Season"].astype(str).eq(str(season).strip())

    sub = df[mask]
    if sub.empty:
        return []

    teams = set(sub["HomeTeam"].unique()) | set(sub["AwayTeam"].unique())
    rows = []
    for team in teams:
        home = sub[sub["HomeTeam"] == team]
        away = sub[sub["AwayTeam"] == team]
        wins   = len(home[home["FTR"] == "H"]) + len(away[away["FTR"] == "A"])
        draws  = len(home[home["FTR"] == "D"]) + len(away[away["FTR"] == "D"])
        losses = len(home[home["FTR"] == "A"]) + len(away[away["FTR"] == "H"])
        gf = int(home["FTHG"].sum() + away["FTAG"].sum())
        ga = int(home["FTAG"].sum() + away["FTHG"].sum())
        rows.append(dict(
            team=team,
            played=wins + draws + losses,
            wins=wins,
            draws=draws,
            losses=losses,
            goals_for=gf,
            goals_against=ga,
            goal_diff=gf - ga,
            points=3 * wins + draws,
        ))

    rows.sort(key=lambda r: (-r["points"], -r["goal_diff"], -r["goals_for"]))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    return rows
