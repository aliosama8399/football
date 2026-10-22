import logging

from fastapi import APIRouter, Depends, BackgroundTasks

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
    background_tasks: BackgroundTasks,
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
    try:
        from monitoring.metrics_registry import api_predictions_total
        mode_label = getattr(result, "source", "hybrid") or "hybrid"
        api_predictions_total.labels(mode=mode_label).inc()
    except Exception as e:
        logger.debug("Failed updating Prometheus prediction counter: %s", e)

    # Online 5-dimensional drift evaluation in background task (non-blocking)
    try:
        from api.services.drift_service import drift_evaluator
        background_tasks.add_task(
            drift_evaluator.evaluate_prediction_turn,
            home_team=payload.home_team,
            away_team=payload.away_team,
            predicted_result=result.predicted_result,
            probabilities=result.probabilities,
            source=getattr(result, "source", "hybrid") or "hybrid",
        )
    except Exception as e:
        logger.debug("Failed scheduling prediction drift evaluation: %s", e)

    logger.info("[route] POST /predictions done -> source=%s verdict=%s probs=%s",
                result.source, result.predicted_result, result.probabilities)
    return result
