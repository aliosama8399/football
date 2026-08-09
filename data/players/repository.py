"""Repository layer for the best-11 feature.

Repository pattern: every piece of data the best-11 domain needs —
squads (provider fusion + disk cache), team totals, and per-match
player-form stats — is accessed through a repository interface. The
domain service depends on these abstractions (Dependency Inversion), so
implementations can be swapped (files, DB, mocks) without touching the
service.

Repositories:
    SquadRepository      — fused provider squads, cached as JSON under
                           data/raw/squads_cache/ (FBRef needs a real
                           browser for Cloudflare; delete the cache or
                           pass refresh=True to re-fetch)
    TotalsRepository     — team totals from processed_matches.csv
    PlayerFormRepository — per-match player stats, H2H stats, cumulative
                           through-date stats, latest match date
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.player_form import (cumulative_player_stats, h2h_player_stats,
                              latest_match_date, rate_squad_as_of)
from data.player_providers.factory import get_player_provider
from data.player_providers.schema import PlayerRecord
from data.team_totals import load_team_totals

logger = logging.getLogger("best11.repository")

CACHE_DIR = Path("data/raw/squads_cache")


# ── Interfaces (what the domain depends on) ──────────────────────────────────


class SquadRepositoryABC(ABC):
    """Squad data access contract."""

    @abstractmethod
    def load_squad(self, team: str, league_code: str, season: str,
                   provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
        """Return a team's squad records for a season."""


class TotalsRepositoryABC(ABC):
    """Team totals data access contract."""

    @abstractmethod
    def load_totals(self, league_code: str, season: str,
                    as_of: Optional[str] = None) -> Dict:
        """Team-season totals; as_of truncates to matches up to that date."""


class PlayerFormRepositoryABC(ABC):
    """Per-match player form data access contract."""

    @abstractmethod
    def rate_squad_as_of(self, squad, as_of: str, league_code: str,
                         season: str):
        """Cumulative through-date ratings for a squad."""

    @abstractmethod
    def cumulative_stats(self, league_code: str, season: str, as_of: str,
                         team: str) -> Dict[str, Dict[str, float]]:
        """Per-player cumulative stat dicts through as_of."""

    @abstractmethod
    def h2h_stats(self, league_code: str, season: str, team: str,
                  opponent: str) -> Tuple[Dict[str, Dict[str, float]], int]:
        """Per-player stats in meetings vs opponent; returns (stats, n)."""

    @abstractmethod
    def latest_match_date(self, league_code: str, season: str) -> Optional[str]:
        """Most recent match date in the per-match feed (ISO)."""


# ── Concrete implementations ─────────────────────────────────────────────────


class SquadRepository(SquadRepositoryABC):
    """Loads squads, caching provider fetches on disk."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def _cache_path(self, team: str, league_code: str, season: str,
                    provider: str) -> Path:
        safe = team.replace(" ", "_")
        return self.cache_dir / f"{provider}_{safe}_{league_code}_{season}.json"

    def load_squad(self, team: str, league_code: str, season: str,
                   provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
        """Return the squad, cached on disk when possible."""
        path = self._cache_path(team, league_code, season, provider)
        if not refresh and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            records = [PlayerRecord(**d) for d in raw]
            if any("pos_list" not in (r.extra or {}) for r in records):
                logger.info("squad cache %s lacks pos_list → refreshing",
                            path.name)
                refresh = True
            else:
                return records
        provider_obj = get_player_provider(provider)
        squad = provider_obj.fetch_team_squad(team, league_code, season)
        if squad:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([r.to_dict() for r in squad],
                                       ensure_ascii=False, indent=1),
                            encoding="utf-8")
            logger.info("cached %s → %s", team, path)
        return squad


class TotalsRepository(TotalsRepositoryABC):
    """Team totals from the processed match dataset."""

    def load_totals(self, league_code: str, season: str,
                    as_of: Optional[str] = None) -> Dict:
        return load_team_totals(league_code, season, as_of=as_of)


class PlayerFormRepository(PlayerFormRepositoryABC):
    """Per-match player stats from the player_match feed."""

    def rate_squad_as_of(self, squad, as_of: str, league_code: str,
                         season: str):
        return rate_squad_as_of(squad, as_of, league_code, season)

    def cumulative_stats(self, league_code: str, season: str, as_of: str,
                         team: str) -> Dict[str, Dict[str, float]]:
        return cumulative_player_stats(league_code, season, as_of, team)

    def h2h_stats(self, league_code: str, season: str, team: str,
                  opponent: str) -> Tuple[Dict[str, Dict[str, float]], int]:
        return h2h_player_stats(league_code, season, team, opponent)

    def latest_match_date(self, league_code: str, season: str) -> Optional[str]:
        return latest_match_date(league_code, season)


# ── Process-wide singletons ──────────────────────────────────────────────────

_squad_repository: Optional[SquadRepositoryABC] = None
_totals_repository: Optional[TotalsRepositoryABC] = None
_player_form_repository: Optional[PlayerFormRepositoryABC] = None


def get_squad_repository() -> SquadRepositoryABC:
    global _squad_repository
    if _squad_repository is None:
        _squad_repository = SquadRepository()
    return _squad_repository


def get_totals_repository() -> TotalsRepositoryABC:
    global _totals_repository
    if _totals_repository is None:
        _totals_repository = TotalsRepository()
    return _totals_repository


def get_player_form_repository() -> PlayerFormRepositoryABC:
    global _player_form_repository
    if _player_form_repository is None:
        _player_form_repository = PlayerFormRepository()
    return _player_form_repository
