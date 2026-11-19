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
    from api.repositories.best11_repo import Best11Repository

    repo = Best11Repository()
    teams = season_teams(league_code, season)
    logger.info("%s %s: %d teams", league_code, season, len(teams))
    ok = 0
    for team in teams:
        try:
            squad = repo.load_squad(team, league_code, season, "all")
        except Exception as e:
            logger.warning("%s failed: %s", team, e)
            squad = None
        if squad:
            ok += 1
            logger.info("cached %s (%d players)", team, len(squad))
        else:
            logger.warning("no squad for %s", team)
    logger.info("%s done: %d/%d squads cached", league_code, ok, len(teams))


def prewarm_missing_all(league_codes: list[str], include_latest: bool = False) -> None:
    """Cache squads via fast ESPN JSON API for clubs across all seasons lacking a saved file."""
    import json
    from api.repositories.best11_repo import CACHE_DIR, Best11Repository
    from data._config import get_leagues
    from data.player_providers.espn import EspnPlayerProvider
    from data.team_totals import find_latest_team_season

    repo = Best11Repository()
    espn = EspnPlayerProvider(rate_limit_sec=0.15)
    df = pd.read_csv(_PROCESSED, usecols=["League", "Season", "HomeTeam", "AwayTeam"])
    leagues_cfg = get_leagues()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_cached = 0
    total_missing = 0
    for lc in league_codes:
        league_name = leagues_cfg[lc]["name"]
        sub = df[df["League"] == league_name]
        all_teams = sorted(
            {str(t) for col in ("HomeTeam", "AwayTeam") for t in sub[col].dropna()}
        )
        for team in all_teams:
            safe = team.replace(" ", "_")
            existing = list(CACHE_DIR.glob(f"all_{safe}_{lc}_*.json"))
            latest = find_latest_team_season(team, lc)
            target_season = latest or "2425"
            need_latest = bool(
                (include_latest or target_season == "2627")
                and not (CACHE_DIR / f"all_{safe}_{lc}_{target_season}.json").exists()
            )
            if existing and not need_latest:
                continue
            total_missing += 1
            try:
                squad = espn.fetch_team_squad(team, lc, target_season)
            except Exception as e:
                logger.warning("prewarm_missing %s (%s %s) failed: %s", team, lc, target_season, e)
                squad = []
            if squad:
                cache_path = repo._cache_path(team, lc, target_season, "all")  # pylint: disable=protected-access
                cache_path.write_text(
                    json.dumps([r.to_dict() for r in squad], ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                total_cached += 1
                logger.info("cached missing %s (%s %s) -> %d players", team, lc, target_season, len(squad))
            else:
                logger.warning("still no squad for %s (%s %s)", team, lc, target_season)
    logger.info("prewarm_missing_all complete: %d/%d resolved", total_cached, total_missing)


def main():
    parser = argparse.ArgumentParser(description="Pre-warm best-11 squad caches")
    parser.add_argument("--season", default="2425")
    parser.add_argument("--leagues", default="E0,SP1,D1,I1,F1")
    parser.add_argument(
        "--missing-all",
        action="store_true",
        help="Scan all seasons in processed_matches.csv and cache squads for every club missing a saved file",
    )
    parser.add_argument(
        "--include-latest",
        action="store_true",
        help="With --missing-all, also cache each club's latest active season even if an older season exists",
    )
    args = parser.parse_args()
    league_list = [lc.strip() for lc in args.leagues.split(",") if lc.strip()]
    if args.missing_all:
        prewarm_missing_all(league_list, include_latest=args.include_latest)
        return
    for lc in league_list:
        prewarm(lc, args.season)


if __name__ == "__main__":
    main()
