import asyncio
import logging

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status

from api.schemas import (
    MatchPredictionRequest,
    MatchPredictionResponse,
    PreMatchDossierRequest,
    PreMatchDossierResponse,
    TacticalSimulationRequest,
    TacticalSimulationResponse,
)
from api.dependencies import get_prediction_service
from api.auth import get_current_user
from api.database import User
from api.services.shadow_model_service import shadow_model_service
from api.services.tactical_simulator_service import tactical_simulator
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

    background_tasks.add_task(
        shadow_model_service.evaluate_shadow,
        home_team=payload.home_team,
        away_team=payload.away_team,
        champion_probs=result.probabilities,
        champion_result=result.predicted_result,
    )

    logger.info("[route] POST /predictions done -> source=%s verdict=%s probs=%s",
                result.source, result.predicted_result, result.probabilities)
    return result


def _require_slm_context(prediction_service):
    """Return active FootballRAGSystem and resolved SLM provider name or raise HTTP 503."""
    rag_sys = getattr(prediction_service.rag_wrapper, "rag", None)
    llm = getattr(rag_sys, "llm", None) if rag_sys is not None else None
    if rag_sys is None or llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fine-tuned SLM is not loaded; cannot generate dynamic tactical analysis.",
        )
    provider_name = str(
        getattr(llm, "provider_name", None)
        or getattr(llm, "_backend", None)
        or type(llm).__name__
    )
    return rag_sys, provider_name


@router.post("/simulate", response_model=TacticalSimulationResponse)
async def simulate_match_tactics(
    payload: TacticalSimulationRequest,
    current_user: User = Depends(get_current_user),
    prediction_service=Depends(get_prediction_service),
):
    """Run a counterfactual What-If tactical simulation synthesized dynamically by the fine-tuned SLM."""
    rag_sys, provider_name = _require_slm_context(prediction_service)
    baseline_probs = None
    home_prof = None
    away_prof = None
    kg_context = ""
    try:
        gnn_pred = await prediction_service.rag_wrapper.predict_structured(payload.home_team, payload.away_team)
        if gnn_pred and gnn_pred.get("probabilities"):
            baseline_probs = gnn_pred["probabilities"]
        home_prof = await prediction_service.rag_wrapper.get_team_profile(payload.home_team)
        away_prof = await prediction_service.rag_wrapper.get_team_profile(payload.away_team)
        kg_context = await asyncio.to_thread(
            rag_sys._get_kg_context,  # pylint: disable=protected-access
            [payload.home_team, payload.away_team],
            f"Tactical simulation {payload.home_team} vs {payload.away_team}",
        )
    except Exception as exc:
        logger.debug("Partial KG/GNN context in simulator (%s): %s", current_user.username, exc)

    def _slm_generate(prompt: str, system_prompt: str | None = None) -> str:
        res = rag_sys.generate_direct(prompt, system_prompt=system_prompt)
        return res["text"]

    try:
        return await asyncio.to_thread(
            tactical_simulator.simulate,
            payload,
            baseline_probs=baseline_probs,
            home_profile=home_prof,
            away_profile=away_prof,
            kg_context=kg_context,
            slm_generator=_slm_generate,
            llm_provider=provider_name,
        )
    except Exception as exc:
        logger.error("SLM tactical simulation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SLM simulation failed: {exc}",
        ) from exc


@router.post("/dossier", response_model=PreMatchDossierResponse)
async def generate_prematch_dossier(
    payload: PreMatchDossierRequest,
    current_user: User = Depends(get_current_user),
    prediction_service=Depends(get_prediction_service),
):
    """Generate a one-click executive pre-match scouting dossier authored dynamically by the fine-tuned SLM."""
    rag_sys, provider_name = _require_slm_context(prediction_service)
    baseline_probs = None
    home_prof = None
    away_prof = None
    kg_context = ""
    try:
        gnn_pred = await prediction_service.rag_wrapper.predict_structured(payload.home_team, payload.away_team)
        if gnn_pred and gnn_pred.get("probabilities"):
            baseline_probs = gnn_pred["probabilities"]
        home_prof = await prediction_service.rag_wrapper.get_team_profile(payload.home_team)
        away_prof = await prediction_service.rag_wrapper.get_team_profile(payload.away_team)
        kg_context = await asyncio.to_thread(
            rag_sys._get_kg_context,  # pylint: disable=protected-access
            [payload.home_team, payload.away_team],
            f"Pre-match dossier {payload.home_team} vs {payload.away_team}",
        )
    except Exception as exc:
        logger.debug("Partial KG/GNN context in dossier (%s): %s", current_user.username, exc)

    def _slm_generate(prompt: str, system_prompt: str | None = None) -> str:
        res = rag_sys.generate_direct(prompt, system_prompt=system_prompt)
        return res["text"]

    try:
        return await asyncio.to_thread(
            tactical_simulator.generate_dossier,
            payload,
            baseline_probs=baseline_probs,
            home_profile=home_prof,
            away_profile=away_prof,
            kg_context=kg_context,
            slm_generator=_slm_generate,
            llm_provider=provider_name,
        )
    except Exception as exc:
        logger.error("SLM dossier generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"SLM dossier generation failed: {exc}",
        ) from exc


@router.get("/shadow/metrics")
async def get_shadow_deployment_metrics(_current_user: User = Depends(get_current_user)):
    """Retrieve Champion (`@production`) vs. Challenger shadow deployment agreement metrics."""
    return shadow_model_service.get_shadow_summary()
