"""
Live Match Prediction Service
===============================
Business flow for in-match (realtime) predictions:

1. validate teams (same pool as pre-match predictions)
2. Expert 1 prior: TEA-GNN pre-match H/D/A probabilities (None-safe fallback)
3. calibrate pre-match expected-goal rates (Bivariate Poisson inversion: lambda_H, lambda_A, lambda_3)
4. pull season-average baselines for both teams (KG team profiles)
5. compute live pace factors (stat pace + score momentum + timestamped red-card decay)
6. Bivariate Poisson-condition the final result on the current score / remaining time
7. logistic-blend prior -> live probabilities by match minute
8. deterministic key drivers (always) + optional LLM narrative (explain=True)
"""

import logging
import math
import time
from typing import Dict, List, Optional

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

SEASON_AVG_FIELDS = {
    "xg": "avg_xg",
    "sot": "avg_sot",
    "shots": "avg_shots",
    "corners": "avg_corners",
    "fouls": "avg_fouls",
    "yellows": "avg_yellows",
}

LIVE_STAT_FIELDS = {
    "xg": "home_xg",
    "shots": "home_shots",
    "sot": "home_sot",
    "corners": "home_corners",
    "fouls": "home_fouls",
    "yellows": "home_yellows",
}

UNIFORM_PROBS = {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}

_PLACEHOLDER_STRINGS = {
    "bullet 1", "bullet 2", "bullet 3",
    "reason 1", "reason 2", "reason 3",
    "specific action", "grounded reason",
}


def _argmax_result(probs: dict) -> str:
    return max(("H", "D", "A"), key=lambda k: probs.get(k, 0.0))


def _is_placeholder(text: str) -> bool:
    if not text or not isinstance(text, str):
        return True
    cleaned = text.strip().lower()
    if cleaned in _PLACEHOLDER_STRINGS:
        return True
    if cleaned.startswith("bullet ") and len(cleaned) <= 12:
        return True
    if "no dominant live deviation detected" in cleaned:
        return True
    return False


def _build_dynamic_breakdown(
    breakdown: Optional[dict],
    drivers: List[dict],
    payload: LivePredictionRequest,
    pre_probs: Dict[str, float],
    blended: Dict[str, float],
    expected_score: Dict[str, float],
    lmbda_h: float,
    lmbda_a: float,
    home_profile: dict,
    away_profile: dict,
) -> dict:
    """
    Ensure the coach-advisor JSON always carries dynamic, match-specific sections
    grounded in the live score, clock, Bivariate Poisson shifts, and team tactical profiles.
    Replaces any literal template placeholders (e.g. 'bullet 1') emitted by small SLMs.
    """
    if not isinstance(breakdown, dict):
        breakdown = {}

    breakdown.setdefault("match_state", {
        "minute": payload.minute,
        "score": f"{payload.home_goals}-{payload.away_goals}",
    })

    analysis = breakdown.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        analysis = breakdown["analysis"] = {}

    h_team = payload.home_team
    a_team = payload.away_team
    rem_min = max(0, TOTAL_MINUTES - payload.minute)
    exp_h_now = lmbda_h * (payload.minute / TOTAL_MINUTES)
    exp_a_now = lmbda_a * (payload.minute / TOTAL_MINUTES)

    h_atk = home_profile.get("attack_tactic") or home_profile.get("attack_headline") or "progressive build-up"
    h_def = home_profile.get("defense_tactic") or home_profile.get("defense_headline") or "compact block"
    a_atk = away_profile.get("attack_tactic") or away_profile.get("attack_headline") or "direct transitions"
    a_def = away_profile.get("defense_tactic") or away_profile.get("defense_headline") or "low-block organization"

    # 1. Who controls now
    who_now = analysis.get("who_controls_now")
    if not who_now or _is_placeholder(str(who_now)):
        if payload.home_goals > payload.away_goals:
            leader, trailer = h_team, a_team
            analysis["who_controls_now"] = (
                f"{leader} holds a {payload.home_goals}-{payload.away_goals} lead at {payload.minute}' "
                f"({blended['H']*100:.1f}% win probability), forcing {trailer} to open up over the final {rem_min} minutes."
            )
        elif payload.away_goals > payload.home_goals:
            leader, trailer = a_team, h_team
            analysis["who_controls_now"] = (
                f"{leader} leads {payload.away_goals}-{payload.home_goals} away at {payload.minute}' "
                f"({blended['A']*100:.1f}% win probability), putting pressure on {trailer}'s {h_atk}."
            )
        else:
            fav = h_team if pre_probs["H"] >= pre_probs["A"] else a_team
            und = a_team if fav == h_team else h_team
            analysis["who_controls_now"] = (
                f"Deadlocked at {payload.home_goals}-{payload.away_goals} after {payload.minute} minutes: "
                f"{und} is neutralizing pre-match favorite {fav} (draw probability now {blended['D']*100:.1f}% vs {pre_probs['D']*100:.1f}% pre-match)."
            )

    # 2. Live Signals — Why (filter out 'bullet 1' etc.)
    raw_why = analysis.get("why")
    valid_why = [
        str(b).strip() for b in (raw_why if isinstance(raw_why, list) else [])
        if not _is_placeholder(str(b))
    ]
    if not valid_why:
        built_why = []
        for d in (drivers or [])[:3]:
            side_team = h_team if d.get("side") == "home" else a_team
            built_why.append(
                f"{side_team} — {d.get('label', 'STAT')}: x{d.get('pace', 1.0)} pace "
                f"(live {d.get('live', 0)}/min vs baseline {d.get('season_avg', 0)}/min)"
            )
        if len(built_why) < 3:
            built_why.append(
                f"Clock compression ({rem_min}' remaining): expected final score is now "
                f"{h_team} {expected_score['home']:.2f} - {expected_score['away']:.2f} {a_team}."
            )
        if len(built_why) < 3:
            built_why.append(
                f"Pre-match xG expectation by {payload.minute}' was {h_team} {exp_h_now:.2f} vs "
                f"{a_team} {exp_a_now:.2f} (actual score {payload.home_goals}-{payload.away_goals})."
            )
        analysis["why"] = built_why[:3]
    else:
        analysis["why"] = valid_why[:4]

    # 3. How the outlook changed
    how_changed = analysis.get("how_outlook_changed")
    if not how_changed or _is_placeholder(str(how_changed)):
        dh = (blended["H"] - pre_probs["H"]) * 100
        dd = (blended["D"] - pre_probs["D"]) * 100
        da = (blended["A"] - pre_probs["A"]) * 100
        analysis["how_outlook_changed"] = (
            f"At {payload.minute}' ({payload.home_goals}-{payload.away_goals}), Bivariate Poisson conditioning "
            f"shifts {h_team} win from {pre_probs['H']*100:.1f}% to {blended['H']*100:.1f}% ({dh:+.1f} pp), "
            f"Draw from {pre_probs['D']*100:.1f}% to {blended['D']*100:.1f}% ({dd:+.1f} pp), and "
            f"{a_team} win from {pre_probs['A']*100:.1f}% to {blended['A']*100:.1f}% ({da:+.1f} pp)."
        )

    # 4. Coach recommendations (always 3 match-specific tactical actions)
    raw_recs = analysis.get("coach_recommendations")
    valid_recs = []
    if isinstance(raw_recs, list):
        for r in raw_recs:
            if isinstance(r, dict):
                act = str(r.get("action", "")).strip()
                rsn = str(r.get("reason", "")).strip()
                if not _is_placeholder(act) and not _is_placeholder(rsn):
                    valid_recs.append({
                        "priority": int(r.get("priority", len(valid_recs) + 1)),
                        "action": act,
                        "reason": rsn,
                    })

    if not valid_recs:
        built_recs = []
        # P1: Attacking / Scoreline urgency adjustment
        if payload.home_goals == payload.away_goals:
            fav_team = h_team if pre_probs["H"] >= pre_probs["A"] else a_team
            dog_team = a_team if fav_team == h_team else h_team
            fav_exp = exp_h_now if fav_team == h_team else exp_a_now
            fav_goals = payload.home_goals if fav_team == h_team else payload.away_goals
            built_recs.append({
                "priority": 1,
                "action": f"{fav_team}: Increase box occupation & half-space overloads; introduce an attacking midfielder/winger before {min(85, payload.minute + 10)}'",
                "reason": f"{fav_team} has {fav_goals} goals vs {fav_exp:.2f} expected by {payload.minute}' against {dog_team}'s defensive block ({rem_min}' left).",
            })
            built_recs.append({
                "priority": 2,
                "action": f"{dog_team}: Refresh midfield legs to protect central zones and trigger fast counter-attacks into wide channels",
                "reason": f"Draw probability has risen to {blended['D']*100:.1f}% (+{(blended['D']-pre_probs['D'])*100:.1f} pp vs pre-match); discipline preserves the point.",
            })
        elif payload.home_goals > payload.away_goals:
            built_recs.append({
                "priority": 1,
                "action": f"{a_team}: Shift to a higher pressing line and commit full-backs forward to create overloads",
                "reason": f"Trailing {payload.home_goals}-{payload.away_goals} at {payload.minute}' with only {rem_min}' remaining (expected final score {expected_score['home']:.2f}-{expected_score['away']:.2f}).",
            })
            built_recs.append({
                "priority": 2,
                "action": f"{h_team}: Compact midfield spacing ({h_def}) and exploit transition space behind {a_team}'s advancing line",
                "reason": f"{h_team} win probability is {blended['H']*100:.1f}%; game-state favors controlled possession and counter-punching.",
            })
        else:
            built_recs.append({
                "priority": 1,
                "action": f"{h_team}: Accelerate tempo via {h_atk} and add a second striker/attacking runner in the box",
                "reason": f"Trailing {payload.home_goals}-{payload.away_goals} at home at {payload.minute}' ({rem_min}' remaining to recover {exp_h_now:.2f} expected goal baseline).",
            })
            built_recs.append({
                "priority": 2,
                "action": f"{a_team}: Reinforce {a_def} with fresh defensive legs and manage set-piece restarts",
                "reason": f"{a_team} away win probability is {blended['A']*100:.1f}% (+{(blended['A']-pre_probs['A'])*100:.1f} pp over pre-match prior).",
            })

        # P3: Top live driver or red-card / set-piece specific action
        if (payload.home_reds or 0) > 0 or (payload.away_reds or 0) > 0:
            red_side = h_team if (payload.home_reds or 0) > 0 else a_team
            adv_side = a_team if red_side == h_team else h_team
            built_recs.append({
                "priority": 3,
                "action": f"{red_side}: Reorganize into a compact 4-4-1/5-3-1 low block; {adv_side}: stretch pitch width to tire the 10-man unit",
                "reason": f"Red-card attrition factor active at {payload.minute}', reducing {red_side}'s remaining attacking rate.",
            })
        elif drivers:
            top_d = drivers[0]
            side_team = h_team if top_d.get("side") == "home" else a_team
            built_recs.append({
                "priority": 3,
                "action": f"{side_team}: Calibrate final-third shot selection and set-piece delivery over the final {rem_min} minutes",
                "reason": f"Primary live signal: {side_team} {top_d.get('label')} running at x{top_d.get('pace')} pace ({top_d.get('live')}/min vs {top_d.get('season_avg')}/min baseline).",
            })
        else:
            built_recs.append({
                "priority": 3,
                "action": f"Target set-pieces and high-value central entries ({h_team} avg {home_profile.get('avg_sot', 4.5):.1f} SOT/gm vs {a_team} {away_profile.get('avg_sot', 3.5):.1f} SOT/gm)",
                "reason": f"With {rem_min}' left, remaining expected goals are {max(0.0, expected_score['home']-payload.home_goals):.2f} ({h_team}) vs {max(0.0, expected_score['away']-payload.away_goals):.2f} ({a_team}).",
            })

        analysis["coach_recommendations"] = built_recs[:3]
    else:
        analysis["coach_recommendations"] = valid_recs[:3]

    return breakdown


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

        # ── 3. Calibrate pre-match Bivariate Poisson goal rates ─────────────
        rates = rates_from_probs(pre_probs)
        lmbda_h, lmbda_a, lmbda_cov = rates["home"], rates["away"], rates.get("cov", 0.08)

        # ── 4. Season baselines ─────────────────────────────────────────────
        home_profile = await self.rag_wrapper.get_team_profile(payload.home_team) or {}
        away_profile = await self.rag_wrapper.get_team_profile(payload.away_team) or {}

        def _avgs(profile: dict) -> dict:
            return {field: profile.get(field) for field in SEASON_AVG_FIELDS.values()}

        home_avgs, away_avgs = _avgs(home_profile), _avgs(away_profile)

        # ── 5. Live pace factors + timestamped red-card attrition ───────────
        home_live = {key: getattr(payload, f"home_{key}" if key != "xg" else "home_xg") for key in LIVE_STAT_FIELDS}
        away_live = {key: getattr(payload, f"away_{key}" if key != "xg" else "away_xg") for key in LIVE_STAT_FIELDS}

        pace_h = compute_pace_factors(
            minute=payload.minute,
            live_stats=home_live,
            season_avgs=home_avgs,
            goals=payload.home_goals,
            lmbda_pre=lmbda_h,
            reds=payload.home_reds or 0,
            red_card_minutes=payload.home_red_minutes,
        )
        pace_a = compute_pace_factors(
            minute=payload.minute,
            live_stats=away_live,
            season_avgs=away_avgs,
            goals=payload.away_goals,
            lmbda_pre=lmbda_a,
            reds=payload.away_reds or 0,
            red_card_minutes=payload.away_red_minutes,
        )

        # ── 6. Remaining-time Bivariate Poisson conditioning ────────────────
        rem = max(0.0, (TOTAL_MINUTES - payload.minute) / TOTAL_MINUTES)
        shared_pace = math.sqrt(max(0.1, pace_h["pace"] * pace_a["pace"]))
        live_probs = conditional_remaining_probs(
            lmbda_h * rem * pace_h["pace"],
            lmbda_a * rem * pace_a["pace"],
            payload.home_goals,
            payload.away_goals,
            lmbda_3_rem=lmbda_cov * rem * shared_pace,
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
            home_red_minutes=payload.home_red_minutes,
            away_red_minutes=payload.away_red_minutes,
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
                "blended_probs": {k: round(v, 4) for k, v in blended.items()},
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
                    breakdown = extract_json_object(analysis_text)
                source = "live_model+llm"
                logger.info("[service] LLM live narrative done in %.1fs (%d chars, breakdown=%s)",
                            time.perf_counter() - t_llm, len(analysis_text or ""),
                            breakdown is not None)
            except Exception as e:
                logger.error("Live LLM narrative failed: %s", e)
                analysis_text = None

        # Always guarantee a rich, placeholder-free tactical breakdown grounded in live math & team profiles
        breakdown = _build_dynamic_breakdown(
            breakdown=breakdown,
            drivers=drivers,
            payload=payload,
            pre_probs=pre_probs,
            blended=blended,
            expected_score=expected_score,
            lmbda_h=lmbda_h,
            lmbda_a=lmbda_a,
            home_profile=home_profile,
            away_profile=away_profile,
        )

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
