import strawberry
from typing import List, Optional
from api.graphql.types import TeamNode, MatchRelation
from api.graph_db import get_graph_db
from api.repositories.graph_repo import TeamGraphRepository

def _get_repo() -> TeamGraphRepository:
    """Helper to resolve TeamGraphRepository inside execution context."""
    return TeamGraphRepository(get_graph_db())

@strawberry.type
class Query:
    @strawberry.field
    def team_profile(self, name: str) -> Optional[TeamNode]:
        """Query tactical profile and performance averages for a team name."""
        repo = _get_repo()
        profile = repo.get_team_profile(name)
        if not profile:
            return None
            
        return TeamNode(
            name=profile.get("name") or name,
            league=profile.get("league", "Unknown"),
            total_matches=int(profile.get("total_matches", 0)),
            win_rate=float(profile.get("win_rate", 0.0)),
            draw_rate=float(profile.get("draw_rate", 0.0)),
            loss_rate=float(profile.get("loss_rate", 0.0)),
            clean_sheet_rate=float(profile.get("clean_sheet_rate", 0.0)),
            avg_goals_home=float(profile.get("avg_goals_home", 0.0)),
            avg_goals_away=float(profile.get("avg_goals_away", 0.0)),
            avg_xg=float(profile.get("avg_xg", 0.0)),
            avg_xga=float(profile.get("avg_xga", 0.0)),
            avg_shots=float(profile.get("avg_shots", 0.0)),
            avg_shots_against=float(profile.get("avg_shots_against", 0.0)),
            avg_sot=float(profile.get("avg_sot", 0.0)),
            avg_sot_against=float(profile.get("avg_sot_against", 0.0)),
            avg_corners=float(profile.get("avg_corners", 0.0)),
            avg_fouls=float(profile.get("avg_fouls", 0.0)),
            avg_yellows=float(profile.get("avg_yellows", 0.0)),
            attack_tactic=profile.get("attack_tactic"),
            defense_tactic=profile.get("defense_tactic"),
            attack_headline=profile.get("attack_headline"),
            defense_headline=profile.get("defense_headline"),
            strengths=profile.get("strengths") or [],
            weaknesses=profile.get("weaknesses") or []
        )

    @strawberry.field
    def head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> List[MatchRelation]:
        """Query head-to-head match history between two specific teams."""
        repo = _get_repo()
        matches = repo.get_head_to_head(team_a, team_b, limit=limit)
        results = []
        for m in matches:
            dt = m.get("date")
            results.append(
                MatchRelation(
                    date=str(dt) if dt else "",
                    home_team=m.get("home_team", team_a),
                    away_team=m.get("away_team", team_b),
                    home_goals=int(m.get("home_goals", 0)),
                    away_goals=int(m.get("away_goals", 0)),
                    result=m.get("result", "D"),
                    home_xg=float(m["home_xg"]) if m.get("home_xg") is not None else None,
                    away_xg=float(m["away_xg"]) if m.get("away_xg") is not None else None,
                    league=m.get("league", "Unknown"),
                    season=m.get("season", "Unknown")
                )
            )
        return results

    @strawberry.field
    def recent_form(self, team_name: str, limit: int = 5) -> List[MatchRelation]:
        """Query recent form matches for a team."""
        repo = _get_repo()
        matches = repo.get_recent_form(team_name, n=limit)
        results = []
        for m in matches:
            dt = m.get("date")
            results.append(
                MatchRelation(
                    date=str(dt) if dt else "",
                    home_team=m.get("home_team", ""),
                    away_team=m.get("away_team", ""),
                    home_goals=int(m.get("home_goals", 0)),
                    away_goals=int(m.get("away_goals", 0)),
                    result=m.get("result", "D"),
                    home_xg=float(m["home_xg"]) if m.get("home_xg") is not None else None,
                    away_xg=float(m["away_xg"]) if m.get("away_xg") is not None else None,
                    league=m.get("league", "Unknown"),
                    season=m.get("season", "Unknown")
                )
            )
        return results

    @strawberry.field
    def league_teams(self, league: str, season: Optional[str] = None) -> List[str]:
        """Query names of all teams competing in a league, optionally filtered by season."""
        repo = _get_repo()
        return repo.get_league_teams(league, season=season)
