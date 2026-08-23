import logging

from fastapi import APIRouter, Depends

from api.schemas import MatchPredictionRequest, MatchPredictionResponse
from api.dependencies import get_prediction_service
from api.auth import get_current_user
from api.database import User
from api.utils import get_phase_logger

logger = get_phase_logger("api.routes.predictions")

router = APIRouter(prefix="/predictions", tags=["Predictions"])

@router.post("", response_model=MatchPredictionResponse)
async def predict_match(
    payload: MatchPredictionRequest,
    current_user: User = Depends(get_current_user),
    prediction_service = Depends(get_prediction_service)
):
    """
    Get a match prediction. Checks if a supervisor has overridden the prediction in PostgreSQL
    first. If not, runs the live GNN + LLM hybrid RAG pipeline.
    """
    logger.info("[route] POST /predictions %s vs %s (user=%s, date=%s)",
                payload.home_team, payload.away_team,
                getattr(current_user, "username", "?"), payload.match_date)
    result = await prediction_service.predict(payload.home_team, payload.away_team, payload.match_date)
    logger.info("[route] POST /predictions done -> source=%s verdict=%s probs=%s",
                result.source, result.predicted_result, result.probabilities)
    return result
