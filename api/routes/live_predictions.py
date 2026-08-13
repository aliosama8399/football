from fastapi import APIRouter, Depends

from api.schemas import LivePredictionRequest, LivePredictionResponse
from api.dependencies import get_live_prediction_service
from api.auth import get_current_user
from api.database import User

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
    return await live_prediction_service.predict(payload)
