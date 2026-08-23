import time
from datetime import date
from typing import Optional

from fastapi import HTTPException, status

from api.schemas import MatchPredictionResponse
from api.utils import strip_markdown, extract_json_object, extract_team_analysis, get_phase_logger
from api.repositories.feedback_repo import FeedbackRepository
from api.async_rag import AsyncRAGWrapper

logger = get_phase_logger("api.services.prediction")


class PredictionService:
    """
    Business flow for match predictions:
    1. validate teams, 2. override check, 3. GNN structured result (None-safe),
    4. optional LLM narrative (stripped of markdown), 5. assemble response.
    """

    def __init__(self, feedback_repo: FeedbackRepository, rag_wrapper: AsyncRAGWrapper):
        self.feedback_repo = feedback_repo
        self.rag_wrapper = rag_wrapper

    async def predict(self, home_team: str, away_team: str, match_date: Optional[date]) -> MatchPredictionResponse:
        t0 = time.perf_counter()
        logger.info("[service] predict start: %s vs %s", home_team, away_team)

        available_teams = self.rag_wrapper.get_available_teams()
        if home_team not in available_teams or away_team not in available_teams:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"One or both teams not found. Available teams: {', '.join(available_teams[:10])}..."
            )

        override = await self.feedback_repo.get_prediction_override(
            home_team=home_team, away_team=away_team, match_date=match_date
        )
        if override:
            logger.info("[service] supervisor override found for %s vs %s -> %s",
                        home_team, away_team, override.predicted_result)
            mock_probs = {"H": 0.0, "D": 0.0, "A": 0.0}
            if override.predicted_result in mock_probs:
                mock_probs[override.predicted_result] = 1.0
            else:
                mock_probs["D"] = 1.0
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

        # ── Expert 1: GNN (ONNX) ────────────────────────────────────────────
        gnn_prediction = await self.rag_wrapper.predict_structured(home_team, away_team)
        predicted_res = gnn_prediction["predicted_result"] if gnn_prediction else "D"
        probs = gnn_prediction["probabilities"] if gnn_prediction else {"H": 0.33, "D": 0.34, "A": 0.33}
        logger.info("[service] GNN result: %s probs=%s", predicted_res, probs)

        # ── Expert 2: LLM narrative (ONNX GPU) ──────────────────────────────
        t_llm = time.perf_counter()
        logger.info("[service] requesting LLM narrative...")
        analysis_text = await self.rag_wrapper.predict_match(home_team, away_team)
        analysis_text = strip_markdown(analysis_text)
        logger.info("[service] LLM narrative done in %.1fs (%d chars)",
                    time.perf_counter() - t_llm, len(analysis_text or ""))

        # Structured breakdown: prefer real JSON from the narrative; for
        # free-text narratives, HEURISTICALLY extract per-team strengths /
        # weaknesses from the summary so the UI scouting cards stay dynamic.
        breakdown = extract_json_object(analysis_text) or {}
        heur = extract_team_analysis(analysis_text, home_team, away_team)
        merged = False
        for key in ("home_team_analysis", "away_team_analysis"):
            if not (isinstance(breakdown.get(key), dict) and
                    (breakdown[key].get("strengths") or breakdown[key].get("weaknesses"))):
                if heur[key]["strengths"] or heur[key]["weaknesses"]:
                    breakdown[key] = heur[key]
                    merged = True
        if not breakdown.get("prediction_verdict") and not any(
                k in breakdown for k in ("home_team_analysis", "away_team_analysis")):
            breakdown = {}  # nothing useful extracted
        logger.info("[service] predict done in %.1fs (source=live_model, "
                    "breakdown=%s heuristic_extract=%s)",
                    time.perf_counter() - t0, bool(breakdown), merged)

        return MatchPredictionResponse(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            predicted_result=predicted_res,
            predicted_home_goals=None,
            predicted_away_goals=None,
            tactical_analysis=analysis_text,
            source="live_model",
            probabilities=probs,
            analysis_breakdown=breakdown or None,
        )
