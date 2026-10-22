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


def _wmean(series: pd.Series, weights: pd.Series) -> float:
    """Weighted mean robust to NaNs and empty series."""
    if series.empty or weights.empty:
        return 0.0
    valid = (~series.isna()) & (~weights.isna())
    if not valid.any():
        return 0.0
    w = weights[valid]
    v = series[valid]
    w_sum = float(w.sum())
    if w_sum <= 0:
        return 0.0
    return float((v * w).sum() / w_sum)


def compute_season_weights(df: pd.DataFrame) -> pd.Series:
    """
    Computes hybrid recency weights:
    - Last 2 seasons (e.g. 2324, 2425) get large weight (3.5x).
    - Previous season (e.g. 2223) gets medium weight (1.75x).
    - Older historical seasons get lower weight (0.75x).
    """
    if "Season" not in df.columns or df.empty:
        return pd.Series(1.0, index=df.index, dtype=float)

    # Sort numeric seasons ascending
    raw_seasons = df["Season"].dropna().unique()
    try:
        sorted_seasons = sorted(raw_seasons, key=lambda s: int(str(s).replace("-", "")))
    except Exception:
        sorted_seasons = sorted(raw_seasons)

    if len(sorted_seasons) <= 2:
        return pd.Series(1.0, index=df.index, dtype=float)

    latest_2 = set(sorted_seasons[-2:])
    prev_1 = set(sorted_seasons[-3:-2])

    def _weight(s):
        if s in latest_2:
            return 3.5
        elif s in prev_1:
            return 1.75
        return 0.75

    return df["Season"].map(_weight).fillna(1.0).astype(float)


def build_team_aggregates(df: pd.DataFrame, recency_weighted: bool = True) -> dict:
    """Per-team aggregate stats (goals, xG, shots, results, clean sheets …).

    When recency_weighted=True, matches from the most recent 2 seasons carry
    a 3.5x weight relative to older seasons to capture modern tactical reality
    without discarding historical sample size.
    """
    aggregates = {}
    all_teams = set(df["HomeTeam"].dropna().unique()) | set(df["AwayTeam"].dropna().unique())

    weights = compute_season_weights(df) if recency_weighted else pd.Series(1.0, index=df.index, dtype=float)

    for team in all_teams:
        h_idx = df.index[df["HomeTeam"] == team]
        a_idx = df.index[df["AwayTeam"] == team]
        home_rows = df.loc[h_idx]
        away_rows = df.loc[a_idx]
        all_rows  = pd.concat([home_rows, away_rows])
        total     = len(home_rows) + len(away_rows)

        if total == 0:
            continue

        h_w = weights.loc[h_idx]
        a_w = weights.loc[a_idx]
        all_w = pd.concat([h_w, a_w])

        # League
        league = (
            home_rows["League"].mode()[0] if len(home_rows) > 0 else
            away_rows["League"].mode()[0] if len(away_rows) > 0 else "Unknown"
        )

        # Goals Scored
        avg_gh = _wmean(home_rows["FTHG"], h_w) if len(home_rows) > 0 else 0.0
        avg_ga = _wmean(away_rows["FTAG"], a_w) if len(away_rows) > 0 else 0.0

        # Goals Conceded
        avg_ch = _wmean(home_rows["FTAG"], h_w) if len(home_rows) > 0 else 0.0
        avg_ca = _wmean(away_rows["FTHG"], a_w) if len(away_rows) > 0 else 0.0

        # Combined goals scored & conceded
        team_goals_series = pd.concat([home_rows["FTHG"], away_rows["FTAG"]])
        team_conceded_series = pd.concat([home_rows["FTAG"], away_rows["FTHG"]])
        avg_goals = _wmean(team_goals_series, all_w)
        avg_conceded = _wmean(team_conceded_series, all_w)

        # xG (team's own xG when playing home or away)
        xg_home = home_rows["Home_xG"] if "Home_xG" in home_rows.columns else pd.Series(dtype=float)
        xg_away = away_rows["Away_xG"] if "Away_xG" in away_rows.columns else pd.Series(dtype=float)
        avg_xg  = _wmean(pd.concat([xg_home, xg_away]), all_w)

        # xGA (opponent's xG against this team)
        xga_home = home_rows["Away_xG"] if "Away_xG" in home_rows.columns else pd.Series(dtype=float)
        xga_away = away_rows["Home_xG"] if "Home_xG" in away_rows.columns else pd.Series(dtype=float)
        avg_xga  = _wmean(pd.concat([xga_home, xga_away]), all_w)

        # Shots
        sh_h  = home_rows["HS"] if "HS" in home_rows.columns else pd.Series(dtype=float)
        sh_a  = away_rows["AS"] if "AS" in away_rows.columns else pd.Series(dtype=float)
        avg_shots = _wmean(pd.concat([sh_h, sh_a]), all_w)

        sha_h = home_rows["AS"] if "AS" in home_rows.columns else pd.Series(dtype=float)
        sha_a = away_rows["HS"] if "HS" in away_rows.columns else pd.Series(dtype=float)
        avg_shots_against = _wmean(pd.concat([sha_h, sha_a]), all_w)

        sot_h  = home_rows["HST"] if "HST" in home_rows.columns else pd.Series(dtype=float)
        sot_a  = away_rows["AST"] if "AST" in away_rows.columns else pd.Series(dtype=float)
        avg_sot = _wmean(pd.concat([sot_h, sot_a]), all_w)

        sota_h = home_rows["AST"] if "AST" in home_rows.columns else pd.Series(dtype=float)
        sota_a = away_rows["HST"] if "HST" in away_rows.columns else pd.Series(dtype=float)
        avg_sot_against = _wmean(pd.concat([sota_h, sota_a]), all_w)

        # Corners
        c_h = home_rows["HC"] if "HC" in home_rows.columns else pd.Series(dtype=float)
        c_a = away_rows["AC"] if "AC" in away_rows.columns else pd.Series(dtype=float)
        avg_corners = _wmean(pd.concat([c_h, c_a]), all_w)

        # Fouls
        f_h = home_rows["HF"] if "HF" in home_rows.columns else pd.Series(dtype=float)
        f_a = away_rows["AF"] if "AF" in away_rows.columns else pd.Series(dtype=float)
        avg_fouls = _wmean(pd.concat([f_h, f_a]), all_w)

        # Yellows
        y_h = home_rows["HY"] if "HY" in home_rows.columns else pd.Series(dtype=float)
        y_a = away_rows["AY"] if "AY" in away_rows.columns else pd.Series(dtype=float)
        avg_yellows = _wmean(pd.concat([y_h, y_a]), all_w)

        # Result rates (weighted)
        total_w = float(h_w.sum() + a_w.sum())
        if total_w > 0:
            h_wins = h_w[home_rows["FTR"] == "H"].sum() if "FTR" in home_rows.columns else 0.0
            a_wins = a_w[away_rows["FTR"] == "A"].sum() if "FTR" in away_rows.columns else 0.0
            h_draws = h_w[home_rows["FTR"] == "D"].sum() if "FTR" in home_rows.columns else 0.0
            a_draws = a_w[away_rows["FTR"] == "D"].sum() if "FTR" in away_rows.columns else 0.0
            h_loss = h_w[home_rows["FTR"] == "A"].sum() if "FTR" in home_rows.columns else 0.0
            a_loss = a_w[away_rows["FTR"] == "H"].sum() if "FTR" in away_rows.columns else 0.0

            win_rate  = float((h_wins + a_wins) / total_w)
            draw_rate = float((h_draws + a_draws) / total_w)
            loss_rate = float((h_loss + a_loss) / total_w)

            cs_h = h_w[home_rows["FTAG"] == 0].sum() if "FTAG" in home_rows.columns else 0.0
            cs_a = a_w[away_rows["FTHG"] == 0].sum() if "FTHG" in away_rows.columns else 0.0
            clean_sheet_rate = float((cs_h + cs_a) / total_w)
        else:
            win_rate = draw_rate = loss_rate = clean_sheet_rate = 0.0

        seasons = sorted(set(all_rows["Season"].astype(str).unique()))

        aggregates[team] = dict(
            league=league,
            total_matches=total,
            avg_goals=avg_goals,
            avg_conceded=avg_conceded,
            avg_goals_home=avg_gh,
            avg_goals_away=avg_ga,
            avg_conceded_home=avg_ch,
            avg_conceded_away=avg_ca,
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
