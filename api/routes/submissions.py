from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any

from api.schemas import (
    MatchSubmissionCreate, MatchSubmissionResponse,
    TacticalAnalysisCreate, TacticalAnalysisResponse,
    TeamProfileEditCreate, FeedbackResponse
)
from api.dependencies import get_feedback_repo
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/submissions", tags=["User Submissions"])


@router.post("/match", response_model=MatchSubmissionResponse)
async def submit_match_record(
    payload: MatchSubmissionCreate,
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
):
    return await feedback_repo.create_match_submission(
        user_id=current_user.id,
        home_team=payload.home_team,
        away_team=payload.away_team,
        match_date=payload.match_date,
        league=payload.league,
        season=payload.season,
        home_goals=payload.home_goals,
        away_goals=payload.away_goals,
        home_ht_goals=payload.home_ht_goals,
        away_ht_goals=payload.away_ht_goals,
        home_xg=payload.home_xg,
        away_xg=payload.away_xg,
        home_shots=payload.home_shots,
        away_shots=payload.away_shots,
        home_sot=payload.home_sot,
        away_sot=payload.away_sot,
        home_corners=payload.home_corners,
        away_corners=payload.away_corners,
        home_fouls=payload.home_fouls,
        away_fouls=payload.away_fouls,
        home_yellows=payload.home_yellows,
        away_yellows=payload.away_yellows,
        home_reds=payload.home_reds,
        away_reds=payload.away_reds,
    )


@router.post("/tactical-analysis", response_model=TacticalAnalysisResponse)
async def submit_tactical_analysis(
    payload: TacticalAnalysisCreate,
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
):
    return await feedback_repo.create_tactical_analysis(
        user_id=current_user.id,
        home_team=payload.home_team,
        away_team=payload.away_team,
        match_date=payload.match_date,
        analysis_text=payload.analysis_text,
    )


@router.post("/team-profile", response_model=FeedbackResponse)
async def submit_team_profile_edit(
    payload: TeamProfileEditCreate,
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
):
    if not (payload.suggested_attack_tactic or payload.suggested_defense_tactic
            or payload.suggested_attack_headline or payload.suggested_defense_headline
            or payload.suggested_strengths or payload.suggested_weaknesses):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one tactical field to update."
        )
    return await feedback_repo.create_feedback(
        user_id=current_user.id,
        type="team_profile_edit",
        team_name=payload.team_name,
        suggested_attack_tactic=payload.suggested_attack_tactic,
        suggested_defense_tactic=payload.suggested_defense_tactic,
        suggested_attack_headline=payload.suggested_attack_headline,
        suggested_defense_headline=payload.suggested_defense_headline,
        suggested_strengths=payload.suggested_strengths,
        suggested_weaknesses=payload.suggested_weaknesses,
    )


@router.get("/mine")
async def list_my_submissions(
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo)
) -> List[Dict[str, Any]]:
    items = []

    matches = await feedback_repo.list_user_match_submissions(user_id=current_user.id)
    for m in matches:
        items.append({
            "id": m.id,
            "type": "match_submission",
            "status": m.status,
            "summary": f"{m.home_team} vs {m.away_team} — {m.match_date}",
            "details": {"league": m.league, "season": m.season},
            "created_at": str(m.created_at)
        })

    analyses = await feedback_repo.list_user_tactical_analyses(user_id=current_user.id)
    for a in analyses:
        items.append({
            "id": a.id,
            "type": "tactical_analysis",
            "status": a.status,
            "summary": f"{a.home_team} vs {a.away_team} — {a.match_date}",
            "details": {"analysis_text": a.analysis_text[:200]},
            "created_at": str(a.created_at)
        })

    feedbacks = await feedback_repo.list_user_feedbacks(user_id=current_user.id)
    for f in feedbacks:
        if f.type not in ("tactic_modification", "team_profile_edit", "prediction_override"):
            continue
        items.append({
            "id": f.id,
            "type": f.type,
            "status": f.status,
            "summary": f.team_name or (f"{f.home_team} vs {f.away_team}"),
            "details": {},
            "created_at": str(f.created_at)
        })

    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items