from typing import List, Optional
from rag.providers.base import BaseKGProvider

class TeamGraphRepository:
    """
    Repository that encapsulates queries to the active connected Knowledge Graph provider.
    This provides a clean, SOLID database abstraction layer for GraphQL resolvers.
    """
    def __init__(self, provider: BaseKGProvider):
        self.provider = provider

    def get_team_profile(self, team_name: str) -> dict:
        """Fetch general tactical profile and metrics for a team."""
        return self.provider.get_team_profile(team_name)

    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> List[dict]:
        """Fetch past head-to-head fixtures between two teams."""
        return self.provider.get_head_to_head(team_a, team_b, limit=limit)

    def get_recent_form(self, team_name: str, n: int = 5) -> List[dict]:
        """Fetch recent matches played by a team."""
        return self.provider.get_recent_form(team_name, n=n)

    def get_league_teams(self, league: str, season: Optional[str] = None) -> List[str]:
        """Fetch list of all teams belonging to a specific league."""
        return self.provider.get_league_teams(league, season=season)

    def get_match(self, home_team: str, away_team: str, date: Optional[str] = None) -> dict:
        """Fetch details of a single match query."""
        return self.provider.get_match(home_team, away_team, date=date)
