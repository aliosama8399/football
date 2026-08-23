import logging

from fastapi import APIRouter, Depends

from api.schemas import LivePredictionRequest, LivePredictionResponse
from api.dependencies import get_live_prediction_service
from api.auth import get_current_user
from api.database import User
from api.utils import get_phase_logger

logger = get_phase_logger("api.routes.live_predictions")

router = APIRouter(prefix="/predictions", tags=["Live Predictions"])

@router.post("/live", response_model=LivePredictionResponse)
async def predict_live_match(
    payload: LivePredictionRequest,
    current_user: User = Depends(get_current_user),
    live_prediction_service = Depends(get_live_prediction_service)
):
    """
    Realtime in-match prediction. Given the current minute, score and live
    stats, returns live H/D/A probabilities (blending the TEA-GNN pre-match
    prior with Poisson-conditioned remaining-time math), key drivers, and —
    when `explain` is true — a coach-actionable LLM narrative.

    The endpoint is stateless: at the final whistle submit the full result
    via the supervision "add new match" flow to feed retraining.
    """
    logger.info("[route] POST /predictions/live %s vs %s @%d' (%d-%d, explain=%s, user=%s)",
                payload.home_team, payload.away_team, payload.minute,
                payload.home_goals, payload.away_goals, payload.explain,
                getattr(current_user, "username", "?"))
    result = await live_prediction_service.predict(payload)
    logger.info("[route] POST /predictions/live done -> source=%s verdict=%s probs=%s",
                result.source, result.predicted_result, result.probabilities)
    return result
