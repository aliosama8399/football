"""
Pre-warm squad caches for every league/team of a season.

The best-11 feature fetches squads lazily and caches them under
data/raw/squads_cache/. This script fills the cache for all teams of all
configured leagues (from the processed match dataset) so first-use
predictions are instant and never trigger a live scrape in the UI.

CLI:
    python -m data.collectors.prewarm_squads --season 2425
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prewarm_squads")

_PROCESSED = Path("data/processed/processed_matches.csv")


def season_teams(league_code: str, season: str) -> list:
    from data._config import get_leagues
    from data.team_registry import normalize_team_name

    name = get_leagues()[league_code]["name"]
    df = pd.read_csv(_PROCESSED)
    df = df[(df["League"] == name) & (df["Season"].astype(str) == str(season))]
    teams = set()
    for col in ("HomeTeam", "AwayTeam"):
        teams |= {normalize_team_name(str(t), league_code) for t in df[col].dropna()}
    return sorted(teams)


def prewarm(league_code: str, season: str) -> None:
    from data.best11 import _load_squad_cached

    teams = season_teams(league_code, season)
    logger.info("%s %s: %d teams", league_code, season, len(teams))
    ok = 0
    for team in teams:
        try:
            squad = _load_squad_cached(team, league_code, season, "all")
        except Exception as e:
            logger.warning("%s failed: %s", team, e)
            squad = None
        if squad:
            ok += 1
            logger.info("cached %s (%d players)", team, len(squad))
        else:
            logger.warning("no squad for %s", team)
    logger.info("%s done: %d/%d squads cached", league_code, ok, len(teams))


def main():
    parser = argparse.ArgumentParser(description="Pre-warm best-11 squad caches")
    parser.add_argument("--season", default="2425")
    parser.add_argument("--leagues", default="E0,SP1,D1,I1,F1")
    args = parser.parse_args()
    for lc in args.leagues.split(","):
        prewarm(lc.strip(), args.season)


if __name__ == "__main__":
    main()
