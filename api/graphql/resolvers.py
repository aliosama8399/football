import strawberry
from typing import List, Optional
from fastapi import HTTPException
from api.graphql.types import (TeamNode, MatchRelation, Best11Entry,
                               Best11Sub, Best11Result, Best11Season,
                               Best11H2H, KBBundle, KBAnswer, KBSource)
from api.graph_db import get_graph_db
from api.repositories.graph_repo import TeamGraphRepository

def _get_repo() -> TeamGraphRepository:
    """Helper to resolve TeamGraphRepository inside execution context."""
    return TeamGraphRepository(get_graph_db())

def _get_kb():
    """Helper to resolve the global KnowledgeBase singleton inside execution context."""
    from api.dependencies import get_knowledge_base
    return get_knowledge_base()

def _kb_sources(srcs) -> List[KBSource]:
    return [KBSource(
        ref=s.get("ref", ""), title=s.get("title", ""), text=s.get("text", ""),
        source_type=s.get("source_type", ""), team=s.get("team"),
        league=s.get("league"), season=s.get("season"), doc_id=s.get("doc_id"),
    ) for s in srcs]

def _match_relation(m: dict) -> MatchRelation:
    dt = m.get("date")
    return MatchRelation(
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
        return [_match_relation(m) for m in matches]

    @strawberry.field
    def recent_form(self, team_name: str, limit: int = 5) -> List[MatchRelation]:
        """Query recent form matches for a team."""
        repo = _get_repo()
        matches = repo.get_recent_form(team_name, n=limit)
        return [_match_relation(m) for m in matches]

    @strawberry.field
    def matches_between(self, team_a: str, team_b: str,
                        league: Optional[str] = None,
                        season: Optional[str] = None,
                        limit: int = 10) -> List[MatchRelation]:
        """Matches between two teams, optionally scoped to one league-season
        (e.g. league='Premier_League', season='2425') — backed by the
        processed match dataset."""
        store = _get_kb()._store
        matches = store.head_to_head(team_a, team_b, limit=limit,
                                     league=league, season=season)
        return [_match_relation(m) for m in matches]

    @strawberry.field
    async def best11(self, team: str, league: str, season: str = "2425",
                     formation: str = "auto",
                     date: Optional[str] = None,
                     opponent: Optional[str] = None) -> Optional[Best11Result]:
        """Best XI for a team & season from team-share player ratings.

        formation: 'auto' (default) fits the strongest shape to the squad
        (4-3-3 / 4-2-3-1 / 4-4-2 / 3-5-2), or a specific formation.
        opponent: when set, ratings are blended 70/30 with the player's
        head-to-head performance against that opponent, so the XI is
        match-specific; each entry carries season and H2H stat blocks.
        league is the config league name (Premier_League, La_Liga, ...).
        date (ISO 'YYYY-MM-DD') switches to through-the-season ratings:
        per-match player stats and team totals cumulated up to that date,
        so the lineup for a specific fixture never uses future data.
        Squad fetch is cached on disk; first call for a team may scrape
        FBRef/understat, so this runs off the event loop.
        """
        from api.dependencies import get_best11_api_service
        try:
            result = await get_best11_api_service().recommend(
                team, league, season, formation, date, opponent)
        except HTTPException as e:
            return Best11Result(team=team, league_code="", season=season,
                                formation=formation, error=e.detail)
        except Exception as e:
            return Best11Result(team=team, league_code="", season=season,
                                formation=formation, error=str(e))
        return Best11Result(
            team=result.team,
            league_code=result.league_code,
            season=result.season,
            formation=result.formation,
            lineup=[Best11Entry(
                slot=e.slot, name=e.name,
                position=e.position, rating=float(e.rating),
                minutes=int(e.minutes or 0), flex=bool(e.flex),
                top_shares=e.top_shares or [],
                season=Best11Season(**e.season)
                if e.season else None,
                h2h=Best11H2H(**e.h2h) if e.h2h else None,
            ) for e in result.lineup],
            captain=result.captain,
            subs=[Best11Sub(
                slot=s.slot, out=s.out, in_=s.in_,
                rating_delta=float(s.rating_delta),
                reason=s.reason,
            ) for s in result.subs],
            bench=[{"name": b.name, "position": b.position,
                    "rating": b.rating, "minutes": b.minutes}
                   for b in result.bench],
            notes=result.notes,
            error=None,
        )

    @strawberry.field
    def league_teams(self, league: str, season: Optional[str] = None) -> List[str]:
        """Query names of all teams competing in a league, optionally filtered by season."""
        repo = _get_repo()
        return repo.get_league_teams(league, season=season)

    # ── Knowledge Base ────────────────────────────────────────────────────────

    @strawberry.field
    def kb_retrieve(self, question: str, prefer_prediction: bool = False) -> KBBundle:
        """Retrieve the KB context bundle for a question — no LLM involved."""
        bundle = _get_kb().retrieve(question, prefer_prediction=prefer_prediction)
        return KBBundle(
            question=bundle.question,
            intent=bundle.intent,
            teams=bundle.teams,
            league=bundle.league,
            season=bundle.season,
            facts=bundle.facts,
            tables=bundle.tables,
            vector_hits=bundle.vector_hits,
            sources=_kb_sources([s.to_json() for s in bundle.sources]),
        )

    @strawberry.field
    def kb_ask(self, question: str, llm_provider: Optional[str] = None,
               prefer_prediction: bool = False) -> KBAnswer:
        """Ask the KB. llm_provider=None → structured answer without any LLM."""
        answer = _get_kb().ask(question, llm_name=llm_provider,
                               prefer_prediction=prefer_prediction)
        return KBAnswer(
            content=answer.content,
            provider=answer.provider,
            error=answer.error,
            sources=_kb_sources([s.to_json() for s in answer.bundle.sources]),
        )
