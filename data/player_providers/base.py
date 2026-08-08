"""
BasePlayerProvider — common interface for player data sources.

Three implementations ship with the probe feature:
    fbref     : FBRef squad/player season stats scrape (richest: positions,
                GK stats, xG, tackles — everything the ratings need)
    fod       : football-data.org v4 API (squads, positions, ages — NO stats)
    understat : per-league playersData JSON (xG/xA etc., NO positions/GK)

The probe (data/collectors/player_probe.py) extracts the same sample teams
from every provider and produces a coverage report, so we can pick the best
provider or fuse several.

Provider selection mirrors get_llm_provider(): get_player_provider(name).
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from data.player_providers.schema import PlayerRecord


class BasePlayerProvider(ABC):
    provider_name: str = "base"

    @abstractmethod
    def fetch_team_squad(self, team: str, league_code: str, season: str) -> List[PlayerRecord]:
        """
        Fetch one team's squad + season stats.

        Args:
            team: canonical team name ('Arsenal', 'Bayern Munich', ...)
            league_code: data/config.yaml league key ('E0', 'SP1', 'D1', 'I1', 'F1')
            season: 4-char season ('2425')
        """
        raise NotImplementedError

    def capabilities(self) -> Dict[str, bool]:
        """Which PlayerRecord fields this provider can fill (for the probe report)."""
        return {}

    def describe(self) -> str:
        return f"{self.provider_name}: {self.capabilities()}"
