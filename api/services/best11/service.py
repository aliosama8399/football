"""Best11Service — domain orchestration for the best-11 feature.

Facade pattern: clients get one solve() entry point; the service
orchestrates the Best11Repository (collects all source data) with the
rating, formation and substitution strategies (the processing layer).
The repository is injected (Dependency Inversion): the service never
touches a data module directly.

Layers:
    api/routes/best11.py             → HTTP concerns
    api/services/best11_service.py   → application use-case (Best11ApiService)
    api/services/best11/service.py   → domain orchestration (this service)
    api/repositories/best11_repo.py  → collects data from sources
"""

import logging
from datetime import date
from typing import Dict, List, Optional

from data.player_ratings import MIN_MINUTES

from .strategies.formations import (AutoFormationStrategy,
                                    FixedFormationStrategy, _fill_lineup)
from .strategies.ratings import (H2HBlendDecorator, RatingStrategy,
                                 SeasonRatingStrategy,
                                 ThroughDateRatingStrategy)
from .strategies.substitutions import (RotationSubstitutionStrategy,
                                       SubstitutionStrategy)

logger = logging.getLogger("best11.service")


class Best11Service:
    """Orchestrates the repository and strategies into a lineup prediction."""

    def __init__(self, repository,
                 rating_strategy: Optional[RatingStrategy] = None,
                 substitution_strategy: Optional[SubstitutionStrategy] = None):
        self.repository = repository
        self.rating_strategy = rating_strategy
        self.substitution_strategy = substitution_strategy or RotationSubstitutionStrategy()

    def solve(self, team: str, league_code: str, season: str = "2425",
              formation: str = "auto", provider: str = "all",
              refresh: bool = False, as_of: Optional[str] = None,
              opponent: Optional[str] = None) -> Dict:
        """Return the best 11 for a team, season and formation.

        Tries the requested season first, then walks the team's active seasons
        (prioritizing seasons with cached squads on disk if live fetch fails)
        and automatically adapts the minimum-minutes threshold for early-season
        campaigns so promoted/relegated/current-season teams always resolve.
        """
        requested_season = season
        season_fallback_note = None

        candidate_seasons: List[str] = []
        if hasattr(self.repository, "find_team_seasons"):
            active_seasons = self.repository.find_team_seasons(team, league_code)
        elif hasattr(self.repository, "find_latest_season_for_team"):
            latest_s = self.repository.find_latest_season_for_team(team, league_code)
            active_seasons = [latest_s] if latest_s else []
        else:
            active_seasons = []

        cached_seasons = (
            self.repository.find_cached_seasons(team, league_code, provider)
            if hasattr(self.repository, "find_cached_seasons")
            else []
        )

        # Priority: requested season -> cached active seasons -> remaining active seasons (newest first)
        candidate_seasons.append(requested_season)
        for s in cached_seasons:
            if s not in candidate_seasons and (not active_seasons or s in active_seasons):
                candidate_seasons.append(s)
        for s in active_seasons:
            if s not in candidate_seasons:
                candidate_seasons.append(s)

        totals = None
        squad = None
        resolved_season = requested_season
        for cand_season in candidate_seasons:
            cand_totals = self.repository.load_totals(league_code, cand_season)
            if not cand_totals or team not in cand_totals:
                continue
            cand_squad = self.repository.load_squad(team, league_code, cand_season, provider, refresh)
            if cand_squad:
                totals = cand_totals
                squad = cand_squad
                resolved_season = cand_season
                break

        season = resolved_season
        if season != requested_season:
            logger.info(
                "%s fallback from %s %s to season %s",
                team, league_code, requested_season, season,
            )
            season_fallback_note = (
                f"{team} had no complete squad/match dataset in {league_code} {requested_season}; "
                f"showing season {season}"
            )

        if not totals or team not in totals:
            return {"team": team, "league_code": league_code, "season": season,
                    "formation": formation,
                    "error": f"no match data for {team} in {league_code} ({requested_season}) — "
                             f"run data/collect_all.py + data/preprocess.py first"}

        if not squad:
            return {"team": team, "league_code": league_code, "season": season,
                    "formation": formation, "error": f"no squad data found for {team} ({season})"}

        if as_of is None and season == _current_season_code():
            as_of = self._auto_as_of(league_code, season)  # cumulative through latest match

        rating_strategy = self.rating_strategy or (
            ThroughDateRatingStrategy(self.repository) if as_of
            else SeasonRatingStrategy(self.repository))
        outcome = rating_strategy.rate(squad, league_code, season, as_of)
        ratings, matched, used_through = (outcome.ratings, outcome.matched,
                                          outcome.used_through)

        season_stats = {
            rec.name: {f: round(getattr(rec, f) or 0.0, 2) for f in
                       ("goals", "assists", "xg", "xa", "shots")}
            for rec in squad}
        if used_through:
            # season block must match the through-date ratings: cumulative stats
            try:
                cum = self.repository.cumulative_stats(
                    league_code, season, as_of, team)
                for name, stats in cum.items():
                    season_stats[name] = {f: round(stats.get(f, 0.0) or 0.0, 2)
                                          for f in ("goals", "assists", "xg", "xa", "shots")}
            except Exception as e:
                logger.warning("cumulative season block failed: %s", e)

        h2h_stats, h2h_matches = {}, 0
        if opponent and opponent != team:
            try:
                h2h_stats, h2h_matches = self.repository.h2h_stats(
                    league_code, season, team, opponent)
            except Exception as e:
                logger.warning("h2h stats failed for %s vs %s: %s", team, opponent, e)
            if h2h_stats:
                H2HBlendDecorator(h2h_stats).enhance(ratings)

        eligible = [r for r in ratings if r.minutes >= MIN_MINUTES]
        adaptive_note = None
        if len(eligible) < 11:
            team_matches = getattr(totals.get(team), "matches", 0) or 0
            early_thr = max(60.0, min(MIN_MINUTES, team_matches * 30.0)) if team_matches else 90.0
            eligible = [r for r in ratings if r.minutes >= early_thr]
            if len(eligible) < 11:
                eligible = [r for r in ratings if r.minutes > 0]
            if len(eligible) < 11:
                eligible = list(ratings)
            adaptive_note = (
                f"early/compact season minutes filter applied ({len(eligible)}/{len(ratings)} players eligible)"
            )
        eligible_ids = {id(r) for r in eligible}
        bench = [r for r in ratings if id(r) not in eligible_ids]

        if formation == "auto":
            formation_strategy = AutoFormationStrategy()
        else:
            formation_strategy = FixedFormationStrategy(formation)
        auto = isinstance(formation_strategy, AutoFormationStrategy)
        slots, _ = formation_strategy.slots(eligible)
        if auto:
            formation = formation_strategy.last_choice
        lineup, notes = _fill_lineup(eligible, slots)
        if season_fallback_note:
            notes.insert(0, season_fallback_note)
        if adaptive_note:
            notes.append(adaptive_note)
        if auto:
            notes.append(f"formation auto-fit: {formation}")
        if opponent and h2h_matches:
            notes.append(f"ratings blended 70/30 with H2H vs {opponent} "
                         f"({h2h_matches} meetings this season)")

        lineup.sort(key=lambda e: e["slot"])
        for e in lineup:
            e["season"] = season_stats.get(e["name"], {})
            e["h2h"] = _h2h_entry(h2h_stats.get(e["name"], {}), h2h_matches)
        subs = self.substitution_strategy.suggest(lineup, eligible)
        captain = max(lineup, key=lambda e: e["rating"])["name"] if lineup else None
        if used_through:
            notes.append(f"ratings through {as_of}: per-match stats for {matched} "
                         f"players, team totals through matchday {as_of}")
        elif as_of:
            notes.append(f"as-of {as_of}: per-match data not collected for this "
                         f"season — full-season ratings used")
        return {
            "team": team,
            "league_code": league_code,
            "season": season,
            "formation": formation,
            "lineup": lineup,
            "captain": captain,
            "subs": subs,
            "bench": [{"name": b.name, "position": b.position, "rating": round(b.rating, 1),
                       "minutes": b.minutes} for b in sorted(bench, key=lambda r: -r.rating)],
            "notes": notes,
        }

    def _auto_as_of(self, league_code: str, season: str) -> Optional[str]:
        """Latest played match in the per-match feed for an in-progress season."""
        try:
            return self.repository.latest_match_date(league_code, season)
        except Exception as e:
            logger.warning("latest_match_date failed for %s %s: %s",
                           league_code, season, e)
            return None


def _h2h_entry(stats: Dict[str, float], matches: int) -> Dict:
    """H2H stat block for a lineup entry (empty when no data/appearance)."""
    if not stats or not stats.get("minutes"):
        return {}
    return {
        "matches": matches,
        "minutes": int(stats.get("minutes", 0)),
        "goals": int(stats.get("goals", 0)),
        "assists": int(stats.get("assists", 0)),
        "xg": round(stats.get("xg", 0.0), 2),
        "xa": round(stats.get("xa", 0.0), 2),
        "shots": int(stats.get("shots", 0)),
    }


def _current_season_code() -> str:
    """Calendar 'YYNN' season code for today (July 1 = new season start)."""
    today = date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"
