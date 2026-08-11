"""Best11Repository — collects all best-11 data from its sources.

Repository that encapsulates collecting every piece of data the best-11
service needs, the same way TeamGraphRepository encapsulates queries to
the KG provider:

    squads          → fused player providers (FBRef + understat), cached
                      as JSON under data/raw/squads_cache/ (FBRef needs
                      a real browser for Cloudflare; delete the cache or
                      pass refresh=True to re-fetch)
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
                              latest_match_date, rate_squad_as_of)
from data.player_providers.factory import get_player_provider
from data.player_providers.schema import PlayerRecord
from data.team_totals import load_team_totals

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
        """Collect a team's squad for a season from the fused providers."""
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
        provider_obj = get_player_provider(prov)
        squad = provider_obj.fetch_team_squad(team, league_code, season)
        if squad:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([r.to_dict() for r in squad],
                                       ensure_ascii=False, indent=1),
                            encoding="utf-8")
            logger.info("cached %s → %s", team, path)
        return squad

    # ── Team totals ──────────────────────────────────────────────────────────

    def load_totals(self, league_code: str, season: str,
                    as_of: Optional[str] = None) -> Dict:
        """Collect team-season totals from the processed match dataset."""
        return load_team_totals(league_code, season, as_of=as_of)

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
