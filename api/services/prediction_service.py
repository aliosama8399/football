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
    4. optional LLM narrative (re-using pre-computed GNN prediction), 5. assemble response.
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

        # ── Expert 1: GNN (ONNX) — executed once and passed into Expert 2 ───
        gnn_prediction = await self.rag_wrapper.predict_structured(home_team, away_team)
        predicted_res = gnn_prediction["predicted_result"] if gnn_prediction else "D"
        probs = gnn_prediction["probabilities"] if gnn_prediction else {"H": 0.33, "D": 0.34, "A": 0.33}
        logger.info("[service] GNN result: %s probs=%s", predicted_res, probs)

        # ── Expert 2: LLM narrative (ONNX GPU / Ollama / Gemini) ────────────
        t_llm = time.perf_counter()
        logger.info("[service] requesting LLM narrative...")
        as_of_str = str(match_date) if match_date else None
        analysis_text = await self.rag_wrapper.predict_match(
            home_team,
            away_team,
            as_of=as_of_str,
            gnn_prediction=gnn_prediction,
        )
        analysis_text = strip_markdown(analysis_text)
        logger.info("[service] LLM narrative done in %.1fs (%d chars)",
                    time.perf_counter() - t_llm, len(analysis_text or ""))

        breakdown = extract_json_object(analysis_text) or {}
        heur = extract_team_analysis(analysis_text, home_team, away_team)

        if "home_team_analysis" not in breakdown or not isinstance(breakdown["home_team_analysis"], dict):
            breakdown["home_team_analysis"] = {"strengths": [], "weaknesses": []}
        if "away_team_analysis" not in breakdown or not isinstance(breakdown["away_team_analysis"], dict):
            breakdown["away_team_analysis"] = {"strengths": [], "weaknesses": []}

        for key in ("home_team_analysis", "away_team_analysis"):
            for bucket in ("strengths", "weaknesses"):
                if not breakdown[key].get(bucket) and heur.get(key, {}).get(bucket):
                    breakdown[key][bucket] = heur[key][bucket]

        home_prof = await self.rag_wrapper.get_team_profile(home_team) or {}
        away_prof = await self.rag_wrapper.get_team_profile(away_team) or {}

        if not breakdown["home_team_analysis"].get("strengths"):
            breakdown["home_team_analysis"]["strengths"] = list(home_prof.get("strengths") or [])
        if not breakdown["home_team_analysis"].get("weaknesses"):
            breakdown["home_team_analysis"]["weaknesses"] = list(home_prof.get("weaknesses") or [])

        if not breakdown["away_team_analysis"].get("strengths"):
            breakdown["away_team_analysis"]["strengths"] = list(away_prof.get("strengths") or [])
        if not breakdown["away_team_analysis"].get("weaknesses"):
            breakdown["away_team_analysis"]["weaknesses"] = list(away_prof.get("weaknesses") or [])

        if not breakdown["home_team_analysis"].get("strengths") or not breakdown["away_team_analysis"].get("strengths"):
            try:
                import os, json
                tactics_path = os.path.join("rag", "knowledge_base", "team_tactics.json")
                if os.path.exists(tactics_path):
                    with open(tactics_path, "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                        if not breakdown["home_team_analysis"].get("strengths"):
                            breakdown["home_team_analysis"]["strengths"] = list(tdata.get(home_team, {}).get("strengths", []))
                        if not breakdown["home_team_analysis"].get("weaknesses"):
                            breakdown["home_team_analysis"]["weaknesses"] = list(tdata.get(home_team, {}).get("weaknesses", []))
                        if not breakdown["away_team_analysis"].get("strengths"):
                            breakdown["away_team_analysis"]["strengths"] = list(tdata.get(away_team, {}).get("strengths", []))
                        if not breakdown["away_team_analysis"].get("weaknesses"):
                            breakdown["away_team_analysis"]["weaknesses"] = list(tdata.get(away_team, {}).get("weaknesses", []))
            except Exception:
                pass

        logger.info("[service] predict done in %.1fs (source=live_model, "
                    "home_strengths=%d, away_strengths=%d)",
                    time.perf_counter() - t0,
                    len(breakdown["home_team_analysis"].get("strengths", [])),
                    len(breakdown["away_team_analysis"].get("weaknesses", [])))

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
