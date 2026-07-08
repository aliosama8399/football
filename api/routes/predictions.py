import re
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import MatchPredictionRequest, MatchPredictionResponse
from api.dependencies import get_feedback_repo, get_async_rag
from api.auth import get_current_user
from api.database import User


def _strip_markdown(text: str) -> str:
    """Remove all markdown formatting from model responses for clean plain-text display."""
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'^[\-=]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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
        # Mock probabilities for override: 100% for the predicted result, 0% for others
        mock_probs = {"H": 0.0, "D": 0.0, "A": 0.0}
        if override.predicted_result in mock_probs:
            mock_probs[override.predicted_result] = 1.0
        else:
            mock_probs["D"] = 1.0 # fallback
        return MatchPredictionResponse(
            home_team=override.home_team,
            away_team=override.away_team,
            match_date=override.match_date,
            predicted_result=override.predicted_result,
            predicted_home_goals=override.predicted_home_goals,
            predicted_away_goals=override.predicted_away_goals,
            tactical_analysis=override.tactical_analysis,
            source="override",
            probabilities=mock_probs
        )

    # 2. Run GNN model to get structured prediction outcome
    gnn_prediction = await rag_wrapper.get_gnn_prediction_structured(home, away)
    predicted_res = gnn_prediction["predicted_result"] if gnn_prediction else "D"
    probs = gnn_prediction["probabilities"] if gnn_prediction else {"H": 0.33, "D": 0.34, "A": 0.33}

    # 3. Run LLM model to get full tactical analysis
    analysis_text = await rag_wrapper.predict_match(home, away)
    analysis_text = _strip_markdown(analysis_text)

    return MatchPredictionResponse(
        home_team=home,
        away_team=away,
        match_date=date_val,
        predicted_result=predicted_res,
        predicted_home_goals=None,
        predicted_away_goals=None,
        tactical_analysis=analysis_text,
        source="live_model",
        probabilities=probs
    )
