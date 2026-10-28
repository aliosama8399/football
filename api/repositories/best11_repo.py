"""Best11Repository — collects all best-11 data from its sources.

Repository that encapsulates collecting every piece of data the best-11
service needs, the same way TeamGraphRepository encapsulates queries to
the KG provider:

    squads          → fused player providers (FBRef + understat), cached
                      as JSON under data/raw/squads_cache/ (or synthesized
                      from per-match player feed if offline)
    team totals     → data/team_totals.py (processed_matches.csv)
    per-match form  → data/player_form.py (player_match feed):
                      cumulative through-date ratings, H2H stats,
                      cumulative season blocks, latest match date

This is a data-collection layer only — formations, rating strategies
and lineup decisions live in api/services/best11/ (the processing layer).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.player_form import (cumulative_player_stats, h2h_player_stats,
                              latest_match_date, load_match_player_stats,
                              rate_squad_as_of)
from data.player_providers.factory import get_player_provider
from data.player_providers.schema import PlayerRecord
from data.team_registry import normalize_team_name
from data.team_totals import find_latest_team_season, load_team_totals

logger = logging.getLogger("best11.repository")

CACHE_DIR = Path("data/raw/squads_cache")


class Best11Repository:
    """Collects best-11 data from sources behind a single interface."""

    def __init__(self, cache_dir: Path = CACHE_DIR,
                 provider: str = "all"):
        self.cache_dir = cache_dir
        self.provider = provider

    # ── Squads (provider fusion, cached on disk) ─────────────────────────────

    def _cache_path(self, team: str, league_code: str, season: str,
                    provider: str) -> Path:
        safe = team.replace(" ", "_")
        return self.cache_dir / f"{provider}_{safe}_{league_code}_{season}.json"

    def load_squad(self, team: str, league_code: str, season: str,
                   provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
        """Collect a team's squad for a season from the fused providers (with per-match CSV fallback)."""
        prov = provider or self.provider
        path = self._cache_path(team, league_code, season, prov)
        if not refresh and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = [PlayerRecord(**d) for d in raw]
            if any("pos_list" not in (r.extra or {}) for r in records):
                logger.info("squad cache %s lacks pos_list → refreshing", path.name)
                refresh = True
            else:
                return records

        squad: List[PlayerRecord] = []
        try:
            provider_obj = get_player_provider(prov)
            squad = provider_obj.fetch_team_squad(team, league_code, season)
        except Exception as e:
            logger.warning("get_player_provider(%s) failed for %s %s %s: %s",
                           prov, team, league_code, season, e)

        # Offline fallback: synthesize squad from data/raw/player_match/ if available
        if not squad:
            squad = self._synthesize_from_player_match(team, league_code, season)

        if squad:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([r.to_dict() for r in squad],
                                           ensure_ascii=False, indent=1),
                                encoding="utf-8")
                logger.info("cached %s → %s (%d players)", team, path, len(squad))
            except Exception as e:
                logger.warning("Could not write squad cache %s: %s", path, e)
        return squad

    def _synthesize_from_player_match(self, team: str, league_code: str, season: str) -> List[PlayerRecord]:
        """Build a PlayerRecord list from data/raw/player_match/player_match_{lc}_{season}.csv if present."""
        df = load_match_player_stats(league_code, season)
        if df.empty:
            return []
        canon_target = normalize_team_name(str(team), league_code)
        df = df[df["team"].map(lambda t: normalize_team_name(str(t), league_code)) == canon_target]
        if df.empty:
            return []

        from data.player_providers.understat import _coarse_position, _pos_list
        records: List[PlayerRecord] = []
        for pname, g in df.groupby("player_name"):
            pos_raw = str(g["position"].dropna().iloc[0]) if "position" in g.columns and not g["position"].dropna().empty else "M"
            coarse = _coarse_position(pos_raw) or "MF"
            records.append(PlayerRecord(
                name=str(pname),
                team=team,
                league=league_code,
                season=season,
                source="player_match_csv",
                position=coarse,
                minutes=float(g["minutes"].sum()),
                appearances=float(len(g)),
                goals=float(g["goals"].sum()),
                assists=float(g["assists"].sum()),
                shots=float(g["shots"].sum()),
                xg=float(g["xg"].sum()),
                xa=float(g["xa"].sum()),
                key_passes=float(g["key_passes"].sum()) if "key_passes" in g.columns else 0.0,
                yellow_cards=int(g["yellow_cards"].sum()) if "yellow_cards" in g.columns else 0,
                red_cards=int(g["red_cards"].sum()) if "red_cards" in g.columns else 0,
                extra={
                    "key_passes": float(g["key_passes"].sum()) if "key_passes" in g.columns else 0.0,
                    "pos_list": _pos_list(pos_raw, coarse),
                },
            ))
        return records

    # ── Team totals ──────────────────────────────────────────────────────────

    def load_totals(self, league_code: str, season: str,
                    as_of: Optional[str] = None) -> Dict:
        """Collect team-season totals from the processed match dataset."""
        return load_team_totals(league_code, season, as_of=as_of)

    def find_latest_season_for_team(self, team: str, league_code: str) -> Optional[str]:
        """Find the latest season in processed_matches.csv where `team` played in `league_code`."""
        return find_latest_team_season(team, league_code)

    # ── Per-match player form ────────────────────────────────────────────────

    def rate_squad_as_of(self, squad, as_of: str, league_code: str, season: str):
        """Collect cumulative through-date ratings for a squad."""
        return rate_squad_as_of(squad, as_of, league_code, season)

    def cumulative_stats(self, league_code: str, season: str, as_of: str,
                         team: str) -> Dict[str, Dict[str, float]]:
        """Collect per-player cumulative stat dicts through as_of."""
        return cumulative_player_stats(league_code, season, as_of, team)

    def h2h_stats(self, league_code: str, season: str, team: str,
                  opponent: str) -> Tuple[Dict[str, Dict[str, float]], int]:
        """Collect per-player stats in meetings vs opponent; (stats, n)."""
        return h2h_player_stats(league_code, season, team, opponent)

    def latest_match_date(self, league_code: str, season: str) -> Optional[str]:
        """Collect the most recent match date in the per-match feed (ISO)."""
        return latest_match_date(league_code, season)
