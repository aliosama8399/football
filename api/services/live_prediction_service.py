"""
Live Match Prediction Service
===============================
Business flow for in-match (realtime) predictions:

1. validate teams (same pool as pre-match predictions)
2. Expert 1 prior: TEA-GNN pre-match H/D/A probabilities (None-safe fallback)
3. calibrate pre-match expected-goal rates (independent Poisson inversion)
4. pull season-average baselines for both teams (KG team profiles)
5. compute live pace factors (stat pace + score momentum + red cards)
6. Poisson-condition the final result on the current score / remaining time
7. logistic-blend prior -> live probabilities by match minute
8. deterministic key drivers (always) + optional LLM narrative (explain=True)

The endpoint is stateless: nothing is persisted here. At the final whistle the
coach submits the full result through the existing supervision "add new match"
flow, which feeds the matches table used for retraining.
"""

import logging
import time
from typing import Optional

from fastapi import HTTPException, status

from api.schemas import LivePredictionRequest, LivePredictionResponse
from api.utils import extract_json_object, strip_markdown, get_phase_logger
from api.async_rag import AsyncRAGWrapper
from rag.live_adjustment import (
    rates_from_probs,
    conditional_remaining_probs,
    expected_final_score,
    compute_pace_factors,
    blend_probs,
    extract_drivers,
    TOTAL_MINUTES,
)

logger = get_phase_logger("api.services.live_prediction")

# stat key -> team-profile field holding the season average (per-90)
SEASON_AVG_FIELDS = {
    "xg": "avg_xg",
    "sot": "avg_sot",
    "shots": "avg_shots",
    "corners": "avg_corners",
    "fouls": "avg_fouls",
    "yellows": "avg_yellows",
}

# live request field per stat key (home side)
LIVE_STAT_FIELDS = {
    "xg": "home_xg",
    "shots": "home_shots",
    "sot": "home_sot",
    "corners": "home_corners",
    "fouls": "home_fouls",
    "yellows": "home_yellows",
}

UNIFORM_PROBS = {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}


def _argmax_result(probs: dict) -> str:
    return max(("H", "D", "A"), key=lambda k: probs.get(k, 0.0))


def _backfill_breakdown(breakdown: dict, drivers: list, payload: LivePredictionRequest) -> None:
    """
    Ensure the coach-advisor JSON always carries the sections the UI renders.
    When the LLM omits/empties a section (early stop, lazy output), backfill
    deterministically from the quantitative driver data so the card is never
    hollow. Marks backfilled items with reason prefix "(model data)".
    """
    analysis = breakdown.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        analysis = breakdown["analysis"] = {}

    if not (analysis.get("why") or []):
        why = [
            f"{d.get('side', 'team')} {d.get('label', 'stat')} running at x{d.get('pace', '?')} pace "
            f"(live {d.get('live')}/min vs season {d.get('season_avg')}/min)"
            for d in (drivers or [])[:3]
        ]
        analysis["why"] = why or ["Live stat pace vs season baseline is the primary signal."]

    if not analysis.get("how_outlook_changed"):
        d = payload.minute
        analysis["how_outlook_changed"] = (
            f"Live state at {d}' ({payload.home_goals}-{payload.away_goals}) shifts the "
            f"pre-match outlook toward the in-play probabilities shown above."
        )

    recs = analysis.get("coach_recommendations")
    if not isinstance(recs, list) or not recs:
        built = []
        for i, drv in enumerate((drivers or [])[:3]):
            side = "Home" if drv.get("side") == "home" else "Away"
            direction = "sustain" if drv.get("direction") == "over" else "correct"
            built.append({
                "priority": i + 1,
                "action": f"{direction} the {drv.get('label', 'stat')} trend on the {side.lower()} side",
                "reason": f"(model data) {side} {drv.get('label')} at x{drv.get('pace')} pace "
                          f"(live {drv.get('live')}/min vs season {drv.get('season_avg')}/min)",
            })
        if not built:
            built.append({
                "priority": 1,
                "action": "Maintain current shape; no dominant live deviation detected",
                "reason": "(model data) all tracked stat paces are near season baseline",
            })
        analysis["coach_recommendations"] = built


class LivePredictionService:
    def __init__(self, rag_wrapper: AsyncRAGWrapper):
        self.rag_wrapper = rag_wrapper

    async def predict(self, payload: LivePredictionRequest) -> LivePredictionResponse:
        t0 = time.perf_counter()
        logger.info("[service] live predict start: %s vs %s @%d' (%d-%d)",
                    payload.home_team, payload.away_team, payload.minute,
                    payload.home_goals, payload.away_goals)
        available_teams = self.rag_wrapper.get_available_teams()
        if payload.home_team not in available_teams or payload.away_team not in available_teams:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"One or both teams not found. Available teams: {', '.join(available_teams[:10])}..."
            )

        # ── 2. Expert 1 prior (TEA-GNN) ─────────────────────────────────────
        gnn = await self.rag_wrapper.predict_structured(payload.home_team, payload.away_team)
        pre_probs = gnn["probabilities"] if gnn and gnn.get("probabilities") else dict(UNIFORM_PROBS)
        pre_probs = {"H": pre_probs.get("H", 1/3), "D": pre_probs.get("D", 1/3), "A": pre_probs.get("A", 1/3)}
        logger.info("[service] GNN pre-match prior: %s",
                    {k: round(v, 3) for k, v in pre_probs.items()})

        # ── 3. Calibrate pre-match goal rates ───────────────────────────────
        rates = rates_from_probs(pre_probs)
        lmbda_h, lmbda_a = rates["home"], rates["away"]

        # ── 4. Season baselines ─────────────────────────────────────────────
        home_profile = await self.rag_wrapper.get_team_profile(payload.home_team) or {}
        away_profile = await self.rag_wrapper.get_team_profile(payload.away_team) or {}

        def _avgs(profile: dict) -> dict:
            # Keyed by profile field name (avg_xg, avg_shots, ...) — the
            # convention used by live_adjustment's pace/driver functions.
            return {field: profile.get(field) for field in SEASON_AVG_FIELDS.values()}

        home_avgs, away_avgs = _avgs(home_profile), _avgs(away_profile)

        # ── 5. Live pace factors ────────────────────────────────────────────
        home_live = {key: getattr(payload, f"home_{key}" if key != "xg" else "home_xg") for key in LIVE_STAT_FIELDS}
        away_live = {key: getattr(payload, f"away_{key}" if key != "xg" else "away_xg") for key in LIVE_STAT_FIELDS}

        pace_h = compute_pace_factors(
            minute=payload.minute, live_stats=home_live, season_avgs=home_avgs,
            goals=payload.home_goals, lmbda_pre=lmbda_h, reds=payload.home_reds or 0,
        )
        pace_a = compute_pace_factors(
            minute=payload.minute, live_stats=away_live, season_avgs=away_avgs,
            goals=payload.away_goals, lmbda_pre=lmbda_a, reds=payload.away_reds or 0,
        )

        # ── 6. Remaining-time Poisson conditioning ──────────────────────────
        rem = max(0.0, (TOTAL_MINUTES - payload.minute) / TOTAL_MINUTES)
        live_probs = conditional_remaining_probs(
            lmbda_h * rem * pace_h["pace"],
            lmbda_a * rem * pace_a["pace"],
            payload.home_goals,
            payload.away_goals,
        )

        # ── 7. Blend prior -> live by minute ────────────────────────────────
        blended = blend_probs(pre_probs, live_probs, payload.minute)

        expected_score = expected_final_score(
            lmbda_h, lmbda_a, payload.minute,
            payload.home_goals, payload.away_goals,
            pace_h["pace"], pace_a["pace"],
        )

        # ── 8. Drivers + optional LLM narrative ─────────────────────────────
        drivers = extract_drivers(
            minute=payload.minute,
            home_stats=home_live, away_stats=away_live,
            home_avgs=home_avgs, away_avgs=away_avgs,
            home_reds=payload.home_reds or 0, away_reds=payload.away_reds or 0,
            home_goals=payload.home_goals, away_goals=payload.away_goals,
            lmbda_h=lmbda_h, lmbda_a=lmbda_a,
        )

        analysis_text = None
        breakdown = None
        source = "live_model"
        if payload.explain:
            live_context = {
                "minute": payload.minute,
                "home_goals": payload.home_goals,
                "away_goals": payload.away_goals,
                "live_probs": {k: round(v, 4) for k, v in live_probs.items()},
                "pre_probs": {k: round(v, 4) for k, v in pre_probs.items()},
                "expected_score": expected_score,
                "drivers": drivers,
                "pace": {"home": pace_h["pace"], "away": pace_a["pace"]},
            }
            try:
                t_llm = time.perf_counter()
                logger.info("[service] requesting LLM live narrative...")
                raw = await self.rag_wrapper.predict_live_match(
                    payload.home_team, payload.away_team, live_context
                )
                breakdown = extract_json_object(raw)
                analysis_text = strip_markdown(raw)
                if breakdown is None and analysis_text:
                    # truncated-JSON recovery also runs on the cleaned text
                    breakdown = extract_json_object(analysis_text)
                if breakdown:
                    _backfill_breakdown(breakdown, drivers, payload)
                source = "live_model+llm"
                logger.info("[service] LLM live narrative done in %.1fs (%d chars, breakdown=%s)",
                            time.perf_counter() - t_llm, len(analysis_text or ""),
                            breakdown is not None)
            except Exception as e:
                logger.error("Live LLM narrative failed: %s", e)
                analysis_text = None

        delta = {k: round(blended.get(k, 0.0) - pre_probs.get(k, 0.0), 4) for k in ("H", "D", "A")}
        logger.info("[service] live predict done in %.1fs (source=%s verdict=%s blended=%s)",
                    time.perf_counter() - t0, source, _argmax_result(blended),
                    {k: round(v, 3) for k, v in blended.items()})

        return LivePredictionResponse(
            home_team=payload.home_team,
            away_team=payload.away_team,
            minute=payload.minute,
            home_goals=payload.home_goals,
            away_goals=payload.away_goals,
            predicted_result=_argmax_result(blended),
            probabilities={k: round(v, 4) for k, v in blended.items()},
            pre_match_probabilities={k: round(v, 4) for k, v in pre_probs.items()},
            delta=delta,
            expected_final_score=expected_score,
            key_drivers=drivers,
            tactical_analysis=analysis_text,
            tactical_breakdown=breakdown,
            explain=bool(payload.explain),
            source=source,
        )
