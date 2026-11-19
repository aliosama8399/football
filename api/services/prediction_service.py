import json
import os
import time
from datetime import date
from typing import Optional

from fastapi import HTTPException, status

from api.async_rag import AsyncRAGWrapper
from api.repositories.feedback_repo import FeedbackRepository
from api.schemas import MatchPredictionResponse
from api.utils import (
    clean_slm_output,
    get_phase_logger,
    parse_slm_tactical_response,
    sanitize_slm_prose,
)

logger = get_phase_logger("api.services.prediction")


def _synthesize_fallback_summary(
    home_team: str,
    away_team: str,
    predicted_res: str,
    probs: dict,
    breakdown: dict,
) -> str:
    """Build clean tactical prose from GNN probabilities + team strengths if SLM gave zero prose."""
    outcome_label = {
        "H": f"{home_team} win",
        "D": "draw",
        "A": f"{away_team} win",
    }.get(predicted_res, predicted_res)
    parts = [
        f"TEA-GNN favors a {outcome_label} outcome "
        f"(Home {probs.get('H', 0.33):.1%}, Draw {probs.get('D', 0.34):.1%}, Away {probs.get('A', 0.33):.1%})."
    ]
    h_str = (breakdown.get("home_team_analysis", {}).get("strengths") or [])[:2]
    a_str = (breakdown.get("away_team_analysis", {}).get("strengths") or [])[:2]
    if h_str:
        parts.append(f"{home_team} key strengths include {'; '.join(h_str)}.")
    if a_str:
        parts.append(f"{away_team} key strengths include {'; '.join(a_str)}.")
    return sanitize_slm_prose(" ".join(parts))


def _backfill_team_profiles(
    breakdown: dict,
    home_team: str,
    away_team: str,
    home_prof: dict,
    away_prof: dict,
) -> None:
    """Populate any empty strengths/weaknesses bucket from KG profiles or team_tactics.json."""
    for side_key, prof in (("home_team_analysis", home_prof), ("away_team_analysis", away_prof)):
        block = breakdown[side_key]
        for bucket in ("strengths", "weaknesses"):
            if not block.get(bucket):
                block[bucket] = list(prof.get(bucket) or [])

    needs_disk = any(
        not breakdown[side_key].get(bucket)
        for side_key in ("home_team_analysis", "away_team_analysis")
        for bucket in ("strengths", "weaknesses")
    )
    if not needs_disk:
        return
    tactics_path = os.path.join("rag", "knowledge_base", "team_tactics.json")
    if not os.path.exists(tactics_path):
        return
    try:
        with open(tactics_path, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        for side_key, team in (("home_team_analysis", home_team), ("away_team_analysis", away_team)):
            team_entry = tdata.get(team, {})
            for bucket in ("strengths", "weaknesses"):
                if not breakdown[side_key].get(bucket):
                    breakdown[side_key][bucket] = list(team_entry.get(bucket, []))
    except Exception:
        pass


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

        # ── Expert 2: LLM narrative (HuggingFace / vLLM / ONNX / Ollama) ─────
        t_llm = time.perf_counter()
        logger.info("[service] requesting LLM narrative...")
        as_of_str = str(match_date) if match_date else None
        raw_analysis = await self.rag_wrapper.predict_match(
            home_team,
            away_team,
            as_of=as_of_str,
            gnn_prediction=gnn_prediction,
        )
        cleaned_raw = clean_slm_output(raw_analysis or "")
        logger.info("[service] LLM narrative done in %.1fs (%d chars)",
                    time.perf_counter() - t_llm, len(cleaned_raw))

        parsed = parse_slm_tactical_response(cleaned_raw, home_team, away_team)
        if parsed["is_fallback"]:
            err_hint = parsed["fallback_error"] or "SLM inference unavailable"
            analysis_text = (
                f"Expert 2 SLM inference unavailable ({err_hint}). "
                f"Showing Expert 1 (TEA-GNN) probabilities and Knowledge Graph tactical profiles."
            )
            breakdown = {
                "home_team_analysis": {"strengths": [], "weaknesses": []},
                "away_team_analysis": {"strengths": [], "weaknesses": []},
                "tactical_matchup_summary": analysis_text,
            }
        else:
            analysis_text = sanitize_slm_prose(parsed["tactical_matchup_summary"])
            breakdown = {
                "home_team_analysis": parsed["home_team_analysis"],
                "away_team_analysis": parsed["away_team_analysis"],
                "tactical_matchup_summary": analysis_text,
            }
            if parsed["prediction_verdict"]:
                breakdown["prediction_verdict"] = parsed["prediction_verdict"]
            if parsed["confidence_rating"]:
                breakdown["confidence_rating"] = parsed["confidence_rating"]

        home_prof = await self.rag_wrapper.get_team_profile(home_team) or {}
        away_prof = await self.rag_wrapper.get_team_profile(away_team) or {}
        _backfill_team_profiles(breakdown, home_team, away_team, home_prof, away_prof)

        if not analysis_text:
            analysis_text = _synthesize_fallback_summary(
                home_team, away_team, predicted_res, probs, breakdown
            )
            breakdown["tactical_matchup_summary"] = analysis_text

        logger.info(
            "[service] predict done in %.1fs (source=live_model, home_s=%d, home_w=%d, away_s=%d, away_w=%d)",
            time.perf_counter() - t0,
            len(breakdown["home_team_analysis"].get("strengths", [])),
            len(breakdown["home_team_analysis"].get("weaknesses", [])),
            len(breakdown["away_team_analysis"].get("strengths", [])),
            len(breakdown["away_team_analysis"].get("weaknesses", [])),
        )

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
