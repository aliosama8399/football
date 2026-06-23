from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import MatchPredictionRequest, MatchPredictionResponse
from api.dependencies import get_feedback_repo, get_async_rag
from api.auth import get_current_user
from api.database import User

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.post("", response_model=MatchPredictionResponse)
async def predict_match(
    payload: MatchPredictionRequest,
    current_user: User = Depends(get_current_user),
    feedback_repo = Depends(get_feedback_repo),
    rag_wrapper = Depends(get_async_rag)
):
    """
    Get a match prediction. Checks if a supervisor has overridden the prediction in PostgreSQL
    first. If not, runs the live GNN + LLM hybrid RAG pipeline.
    """
    home = payload.home_team
    away = payload.away_team
    date_val = payload.match_date

    # Validate team names exist in the available teams cache
    available_teams = rag_wrapper.get_available_teams()
    if home not in available_teams or away not in available_teams:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"One or both teams not found. Available teams: {', '.join(available_teams[:10])}..."
        )

    # 1. Query PostgreSQL for prediction override
    override = await feedback_repo.get_prediction_override(home_team=home, away_team=away, match_date=date_val)
    if override:
        return MatchPredictionResponse(
            home_team=override.home_team,
            away_team=override.away_team,
            match_date=override.match_date,
            predicted_result=override.predicted_result,
            predicted_home_goals=override.predicted_home_goals,
            predicted_away_goals=override.predicted_away_goals,
            tactical_analysis=override.tactical_analysis,
            source="override"
        )

    # 2. Run GNN model to get structured prediction outcome
    gnn_prediction = await rag_wrapper.get_gnn_prediction_structured(home, away)
    predicted_res = gnn_prediction["predicted_result"] if gnn_prediction else "D"

    # 3. Run LLM model to get full tactical analysis
    analysis_text = await rag_wrapper.predict_match(home, away)

    return MatchPredictionResponse(
        home_team=home,
        away_team=away,
        match_date=date_val,
        predicted_result=predicted_res,
        predicted_home_goals=None,
        predicted_away_goals=None,
        tactical_analysis=analysis_text,
        source="live_model"
    )
