"""Best11Repository — data collection for the best-11 feature.

Repository that encapsulates collecting all best-11 data from its
sources: fused squad providers (FBRef + understat, cached on disk),
team totals (processed_matches.csv) and per-match player form (the
player_match feed). This provides a clean, SOLID data-access layer for
the backend service — mirrors TeamGraphRepository, which encapsulates
queries to the connected KG provider.

The domain Best11Service consumes this repository through its repository
interfaces (SquadRepositoryABC / TotalsRepositoryABC /
PlayerFormRepositoryABC), so the backend can swap the collection
implementation (files, DB, mocks) without touching the domain.
"""

from typing import Dict, List, Optional, Tuple

from data.player_providers.schema import PlayerRecord
from data.players.repository import (PlayerFormRepositoryABC,
                                     SquadRepositoryABC, TotalsRepositoryABC,
                                     get_player_form_repository,
                                     get_squad_repository,
                                     get_totals_repository)


class Best11Repository(SquadRepositoryABC, TotalsRepositoryABC,
                       PlayerFormRepositoryABC):
    """
    Collects all best-11 data from its sources behind one repository.

    Composes the domain data-access implementations (squad cache, team
    totals, per-match form); any of them can be swapped via the
    constructor for testing or for other sources.
    """

    def __init__(self,
                 squad_repository: Optional[SquadRepositoryABC] = None,
                 totals_repository: Optional[TotalsRepositoryABC] = None,
                 player_form_repository: Optional[PlayerFormRepositoryABC] = None):
        self.squad_repository = squad_repository or get_squad_repository()
        self.totals_repository = totals_repository or get_totals_repository()
        self.player_form_repository = player_form_repository or get_player_form_repository()

    # ── Squads (provider fusion, cached on disk) ─────────────────────────────

    def load_squad(self, team: str, league_code: str, season: str,
                   provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
        """Collect a team's squad for a season from the fused providers."""
        return self.squad_repository.load_squad(team, league_code, season,
                                                provider, refresh)

    # ── Team totals ──────────────────────────────────────────────────────────

    def load_totals(self, league_code: str, season: str,
                    as_of: Optional[str] = None) -> Dict:
        """Collect team-season totals from the processed match dataset."""
        return self.totals_repository.load_totals(league_code, season, as_of=as_of)

    # ── Per-match player form ────────────────────────────────────────────────

    def rate_squad_as_of(self, squad, as_of: str, league_code: str,
                         season: str):
        """Collect cumulative through-date ratings for a squad."""
        return self.player_form_repository.rate_squad_as_of(
            squad, as_of, league_code, season)

    def cumulative_stats(self, league_code: str, season: str, as_of: str,
                         team: str) -> Dict[str, Dict[str, float]]:
        """Collect per-player cumulative stat dicts through as_of."""
        return self.player_form_repository.cumulative_stats(
            league_code, season, as_of, team)

    def h2h_stats(self, league_code: str, season: str, team: str,
                  opponent: str) -> Tuple[Dict[str, Dict[str, float]], int]:
        """Collect per-player stats in meetings vs opponent; (stats, n)."""
        return self.player_form_repository.h2h_stats(
            league_code, season, team, opponent)

    def latest_match_date(self, league_code: str, season: str) -> Optional[str]:
        """Collect the most recent match date in the per-match feed (ISO)."""
        return self.player_form_repository.latest_match_date(league_code, season)
