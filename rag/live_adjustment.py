"""
Live In-Match Prediction Adjustment (Bivariate Poisson + Timestamped Attrition)
================================================================================
Pure-math layer that turns a PRE-MATCH prediction (e.g. TEA-GNN probabilities)
into a LIVE prediction given the current match state (minute, score, live stats).

Implements Specification Section 6.3:
  1. Bivariate Poisson model (lambda_H, lambda_A, lambda_3):
     G_H = U_1 + U_3, G_A = U_2 + U_3 where U_1 ~ Poisson(lambda_1),
     U_2 ~ Poisson(lambda_2), U_3 ~ Poisson(lambda_3) (shared game-intensity
     covariance component), evaluated via vectorized 2D NumPy grids.
  2. Live pace factors + Spec Section 6.3.2 timestamped red-card attrition:
     Phi_card(T, t) = prod_{k in Reds(T)} [1 - gamma_red * ((90 - t_k) / 90)]
     with gamma_red = 0.22.
  3. Remaining-time Bivariate Poisson conditioning: P(final result | current score).
  4. Logistic blend between pre-match prior and live-adjusted probabilities.
"""

import functools
import math
from typing import Dict, List, Optional

import numpy as np

TOTAL_MINUTES = 90
MAX_GOALS = 10  # per side truncation for the Poisson tables
GAMMA_RED = 0.22  # Spec Section 6.3.2 red-card attrition coefficient

FACTORIALS = np.array(
    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800],
    dtype=np.float64,
)


# ── Vectorized Bivariate Poisson Outcome Math ────────────────────────────────

def poisson_pmf(lmbda: float, k: int) -> float:
    """P(X = k) for X ~ Poisson(lmbda)."""
    if lmbda <= 0.0:
        return 1.0 if k == 0 else 0.0
    fact = float(FACTORIALS[k]) if k < len(FACTORIALS) else float(math.factorial(k))
    return math.exp(-lmbda) * (lmbda ** k) / fact


def _poisson_pmf_vec(lmbda: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Vectorized Poisson PMF array for k = 0..max_goals."""
    lmbda = max(0.0, float(lmbda))
    ks = np.arange(max_goals + 1, dtype=np.float64)
    if max_goals < len(FACTORIALS):
        facts = FACTORIALS[: max_goals + 1]
    else:
        facts = np.array([math.factorial(int(k)) for k in ks], dtype=np.float64)
    if lmbda == 0.0:
        out = np.zeros(max_goals + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    return np.exp(-lmbda) * np.power(lmbda, ks) / facts


def bivariate_score_grid(
    lmbda_h: float,
    lmbda_a: float,
    lmbda_3: float = 0.08,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Compute the (max_goals+1, max_goals+1) joint probability matrix P(G_H = i, G_A = j)
    under the Bivariate Poisson distribution:
      G_H = U_1 + U_3,  G_A = U_2 + U_3
      U_1 ~ Poisson(max(0.01, lmbda_h - lmbda_3))
      U_2 ~ Poisson(max(0.01, lmbda_a - lmbda_3))
      U_3 ~ Poisson(max(0.0, min(lmbda_3, min(lmbda_h, lmbda_a) * 0.6)))
    """
    cov = max(0.0, min(float(lmbda_3), min(float(lmbda_h), float(lmbda_a)) * 0.6))
    l1 = max(1e-4, float(lmbda_h) - cov) if lmbda_h > 0 else 0.0
    l2 = max(1e-4, float(lmbda_a) - cov) if lmbda_a > 0 else 0.0

    p1 = _poisson_pmf_vec(l1, max_goals)
    p2 = _poisson_pmf_vec(l2, max_goals)
    indep_grid = np.outer(p1, p2)

    if cov <= 1e-6:
        total = indep_grid.sum()
        return indep_grid / total if total > 0 else indep_grid

    p3 = _poisson_pmf_vec(cov, max_goals)
    joint = np.zeros((max_goals + 1, max_goals + 1), dtype=np.float64)
    for u3 in range(max_goals + 1):
        w = p3[u3]
        if w < 1e-12:
            continue
        span = max_goals + 1 - u3
        joint[u3:, u3:] += w * indep_grid[:span, :span]

    total = joint.sum()
    return joint / total if total > 0 else joint


def outcome_probs(
    lmbda_h: float,
    lmbda_a: float,
    max_goals: int = MAX_GOALS,
    lmbda_3: float = 0.08,
) -> Dict[str, float]:
    """
    Full-time H/D/A probabilities under the Bivariate Poisson distribution.
    Returns {"H": ..., "D": ..., "A": ...} (sums to 1.0).
    """
    grid = bivariate_score_grid(lmbda_h, lmbda_a, lmbda_3=lmbda_3, max_goals=max_goals)
    draw = float(np.trace(grid))
    home = float(np.tril(grid, k=-1).sum())
    away = float(np.triu(grid, k=1).sum())
    total = home + draw + away
    return {"H": home / total, "D": draw / total, "A": away / total}


@functools.lru_cache(maxsize=2048)
def _rates_from_probs_cached(
    t_h: float,
    t_d: float,
    t_a: float,
    lmbda_h0: float,
    lmbda_a0: float,
) -> tuple[float, float, float]:
    """
    Internal memoized solver for Bivariate Poisson goal rates (lambda_H, lambda_A, lambda_3).
    Optimizes marginal rates and shared covariance lambda_3 to match pre-match H/D/A prior.
    """
    cov0 = 0.08

    def loss(lh: float, la: float, cov: float) -> float:
        o = outcome_probs(lh, la, lmbda_3=cov)
        return (o["H"] - t_h) ** 2 + (o["D"] - t_d) ** 2 + (o["A"] - t_a) ** 2

    best = (lmbda_h0, lmbda_a0, cov0)
    best_loss = loss(lmbda_h0, lmbda_a0, cov0)
    step = 0.25
    lh, la, cov = lmbda_h0, lmbda_a0, cov0

    for _ in range(60):
        improved = False
        candidates = (
            (lh + step, la, cov),
            (lh - step, la, cov),
            (lh, la + step, cov),
            (lh, la - step, cov),
            (lh, la, cov + step * 0.3),
            (lh, la, cov - step * 0.3),
        )
        for nlh, nla, ncov in candidates:
            nlh = max(0.05, nlh)
            nla = max(0.05, nla)
            ncov = max(0.0, min(0.35, ncov, min(nlh, nla) * 0.5))
            l = loss(nlh, nla, ncov)
            if l < best_loss:
                best_loss = l
                best = (nlh, nla, ncov)
                improved = True
        if not improved:
            step /= 2.0
            if step < 1e-4:
                break
            continue
        lh, la, cov = best
        if step > 0.02:
            step = max(0.02, step * 0.7)

    return (round(best[0], 4), round(best[1], 4), round(best[2], 4))


def rates_from_probs(
    probs: Dict[str, float],
    lmbda_h0: float = 1.4,
    lmbda_a0: float = 1.1,
) -> Dict[str, float]:
    """
    Invert the Bivariate Poisson outcome model: find (home, away, cov) whose outcome
    probabilities best match the given pre-match H/D/A probs.
    """
    t_h = round(float(probs.get("H", 1 / 3)), 4)
    t_d = round(float(probs.get("D", 1 / 3)), 4)
    t_a = round(float(probs.get("A", 1 / 3)), 4)
    lh, la, cov = _rates_from_probs_cached(t_h, t_d, t_a, lmbda_h0, lmbda_a0)
    return {"home": lh, "away": la, "cov": cov}


def conditional_remaining_probs(
    lmbda_h_rem: float,
    lmbda_a_rem: float,
    home_goals: int,
    away_goals: int,
    max_goals: int = MAX_GOALS,
    lmbda_3_rem: float = 0.0,
) -> Dict[str, float]:
    """
    P(final result | current score (home_goals, away_goals)) when the remaining
    goals follow BivariatePoisson(lmbda_h_rem, lmbda_a_rem, lmbda_3_rem).
    Evaluated over a vectorized 2D NumPy grid.
    """
    grid = bivariate_score_grid(
        lmbda_h_rem, lmbda_a_rem, lmbda_3=lmbda_3_rem, max_goals=max_goals
    )
    idx = np.arange(max_goals + 1)
    final_h = home_goals + idx[:, None]
    final_a = away_goals + idx[None, :]

    home = float(grid[final_h > final_a].sum())
    draw = float(grid[final_h == final_a].sum())
    away = float(grid[final_h < final_a].sum())
    total = home + draw + away
    return {"H": home / total, "D": draw / total, "A": away / total}


def expected_final_score(
    lmbda_h: float, lmbda_a: float, minute: int,
    home_goals: int, away_goals: int,
    pace_h: float = 1.0, pace_a: float = 1.0,
) -> Dict[str, float]:
    """Expected final score line: current score + remaining expected goals."""
    rem = max(0.0, (TOTAL_MINUTES - minute) / TOTAL_MINUTES)
    return {
        "home": round(home_goals + lmbda_h * rem * pace_h, 2),
        "away": round(away_goals + lmbda_a * rem * pace_a, 2),
    }


# ── Live pace factors & Timestamped Red-Card Decay (Spec Section 6.3.2) ──────

PACE_STAT_WEIGHTS = {
    "xg":      (0.35, "avg_xg"),
    "sot":     (0.25, "avg_sot"),
    "shots":   (0.20, "avg_shots"),
    "corners": (0.10, "avg_corners"),
    "fouls":   (0.05, "avg_fouls"),
    "yellows": (0.05, "avg_yellows"),
}

PACE_FLOOR, PACE_CEIL = 0.25, 3.0


def _safe_ratio(numer: float, denom: float, floor: float, ceil: float) -> float:
    if denom <= 0:
        return 1.0
    return max(floor, min(ceil, numer / denom))


def compute_red_card_attrition(
    reds: int,
    minute: int,
    red_card_minutes: Optional[List[int]] = None,
    gamma_red: float = GAMMA_RED,
) -> float:
    """
    Specification Section 6.3.2 Timestamped Red-Card Decay:
      Phi_card(T, t) = prod_{k in Reds(T)} [1 - gamma_red * ((90 - t_k) / 90)]
    where gamma_red = 0.22 and t_k is the dismissal minute of red card k.
    If explicit dismissal minutes are not provided, defaults to t_k = minute / 2.
    """
    if reds <= 0 and not red_card_minutes:
        return 1.0

    if red_card_minutes:
        dismissal_times = [max(0, min(TOTAL_MINUTES, int(tk))) for tk in red_card_minutes]
    else:
        est_tk = max(1, int(minute // 2))
        dismissal_times = [est_tk] * int(reds)

    phi = 1.0
    for tk in dismissal_times:
        rem_fraction = max(0.0, (TOTAL_MINUTES - tk) / float(TOTAL_MINUTES))
        phi *= (1.0 - gamma_red * rem_fraction)

    return max(0.45, round(phi, 4))


def compute_pace_factors(
    minute: int,
    live_stats: Dict[str, Optional[float]],
    season_avgs: Dict[str, Optional[float]],
    goals: int,
    lmbda_pre: float,
    reds: int = 0,
    red_card_minutes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Combine stat-based pace + score momentum + timestamped red-card attrition
    into ONE multiplier for a team's remaining expected goals.
    """
    if minute <= 0:
        return {"pace": 1.0, "components": {"note": "match not started"}}

    components: Dict[str, float] = {}
    stat_terms, stat_weight_sum = [], 0.0

    for key, (weight, avg_field) in PACE_STAT_WEIGHTS.items():
        live = live_stats.get(key)
        avg = season_avgs.get(avg_field)
        if live is None or avg is None or avg <= 0:
            continue
        live_rate = live / minute
        season_rate = avg / TOTAL_MINUTES
        ratio = _safe_ratio(live_rate, season_rate, PACE_FLOOR, PACE_CEIL)
        components[key] = round(ratio, 3)
        stat_terms.append(weight * ratio)
        stat_weight_sum += weight

    stat_pace = sum(stat_terms) / stat_weight_sum if stat_weight_sum > 0 else 1.0
    stat_pace = max(PACE_FLOOR, min(PACE_CEIL, stat_pace))
    components["stat_pace"] = round(stat_pace, 3)

    expected_rate = lmbda_pre / TOTAL_MINUTES
    observed_rate = goals / minute
    raw_momentum = _safe_ratio(observed_rate, expected_rate, PACE_FLOOR, PACE_CEIL) if expected_rate > 0 else 1.0
    momentum = 1.0 + 0.5 * (raw_momentum - 1.0)
    components["goal_momentum"] = round(momentum, 3)

    w = min(1.0, minute / 45.0)
    base = 1.0 + w * (0.6 * (stat_pace - 1.0) + 0.4 * (momentum - 1.0))

    if reds > 0 or red_card_minutes:
        penalty = compute_red_card_attrition(
            reds=reds,
            minute=minute,
            red_card_minutes=red_card_minutes,
        )
        base *= penalty
        components["red_card"] = round(penalty, 3)

    pace = max(PACE_FLOOR, min(PACE_CEIL, base))
    return {"pace": round(pace, 3), "components": components}


def blend_probs(pre: Dict[str, float], live: Dict[str, float], minute: int) -> Dict[str, float]:
    """
    Logistic blend: weight(live) = sigmoid((minute - 45) / 12).
    ~0 at kickoff, 0.5 at HT, ~0.98 at 90'.
    """
    w = 1.0 / (1.0 + math.exp(-(minute - 45) / 12.0))
    out = {
        k: (1 - w) * pre.get(k, 0.0) + w * live.get(k, 0.0)
        for k in ("H", "D", "A")
    }
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


# ── Driver extraction ────────────────────────────────────────────────────────

def extract_drivers(
    minute: int,
    home_stats: Dict[str, Optional[float]], away_stats: Dict[str, Optional[float]],
    home_avgs: Dict[str, Optional[float]], away_avgs: Dict[str, Optional[float]],
    home_reds: int = 0, away_reds: int = 0,
    home_goals: int = 0, away_goals: int = 0,
    lmbda_h: float = 1.4, lmbda_a: float = 1.1,
    home_red_minutes: Optional[List[int]] = None,
    away_red_minutes: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Ranked list of the biggest live-state deviations vs season baselines,
    for grounding the LLM narrative and for the deterministic response.
    Even when optional live box-score stats (xG, shots, corners) are omitted,
    extracts dynamic drivers from goal conversion vs pre-match expected goals
    (lambda_H / lambda_A), defensive resistance, and clock compression.
    """
    drivers: List[Dict] = []
    eff_min = max(1, minute)

    def _per_team(side: str, live_stats, avgs, lmbda, reds, red_mins, goals, opp_goals, opp_lmbda):
        for key, (_, avg_field) in PACE_STAT_WEIGHTS.items():
            live = live_stats.get(key)
            avg = avgs.get(avg_field)
            if live is None or avg is None or avg <= 0:
                continue
            live_rate = live / eff_min
            season_rate = avg / TOTAL_MINUTES
            ratio = _safe_ratio(live_rate, season_rate, PACE_FLOOR, PACE_CEIL)
            deviation = abs(ratio - 1.0)
            drivers.append({
                "stat": key,
                "side": side,
                "label": key.upper(),
                "live": round(live_rate, 3),
                "season_avg": round(season_rate, 3),
                "pace": round(ratio, 2),
                "deviation": round(deviation, 3),
                "direction": "over" if ratio > 1.05 else ("under" if ratio < 0.95 else "neutral"),
            })

        # Goal conversion pace vs pre-match expected goal rate (always available)
        if minute > 0 and lmbda > 0:
            live_goal_rate = goals / eff_min
            exp_goal_rate = lmbda / TOTAL_MINUTES
            exp_goals_by_now = lmbda * (eff_min / TOTAL_MINUTES)
            # Smoothed ratio so 0 goals at 60' against a 1.5 expected goal rate reflects real under-conversion
            smoothed_ratio = _safe_ratio(
                (goals + 0.25) / eff_min,
                (exp_goals_by_now + 0.25) / eff_min,
                PACE_FLOOR,
                PACE_CEIL,
            )
            dev = abs(smoothed_ratio - 1.0)
            if dev >= 0.08 or not drivers:
                drivers.append({
                    "stat": "goal_conversion",
                    "side": side,
                    "label": f"GOALS ({goals} vs {exp_goals_by_now:.2f} exp by {minute}')",
                    "live": round(live_goal_rate, 3),
                    "season_avg": round(exp_goal_rate, 3),
                    "pace": round(smoothed_ratio, 2),
                    "deviation": round(dev, 3),
                    "direction": "over" if smoothed_ratio > 1.05 else ("under" if smoothed_ratio < 0.95 else "neutral"),
                })

        # Defensive suppression / resistance when holding an opponent below expected goals
        if minute >= 20 and opp_lmbda > 0:
            exp_opp_by_now = opp_lmbda * (eff_min / TOTAL_MINUTES)
            if opp_goals < exp_opp_by_now - 0.35:
                suppression_ratio = _safe_ratio(
                    (exp_opp_by_now + 0.25),
                    (opp_goals + 0.25),
                    1.0,
                    PACE_CEIL,
                )
                drivers.append({
                    "stat": "defensive_resistance",
                    "side": side,
                    "label": f"DEFENSIVE HOLD ({opp_goals} conceded vs {exp_opp_by_now:.2f} exp)",
                    "live": round(opp_goals / eff_min, 3),
                    "season_avg": round(opp_lmbda / TOTAL_MINUTES, 3),
                    "pace": round(suppression_ratio, 2),
                    "deviation": round(abs(suppression_ratio - 1.0) * 0.85, 3),
                    "direction": "over",
                })

        if reds > 0 or red_mins:
            phi = compute_red_card_attrition(reds=reds, minute=minute, red_card_minutes=red_mins)
            drivers.append({
                "stat": "red_card",
                "side": side,
                "label": "RED CARD ATTRITION",
                "live": float(reds or len(red_mins or [])),
                "season_avg": 0.0,
                "pace": round(phi, 2),
                "deviation": 1.2,
                "direction": "under",
            })

    _per_team("home", home_stats, home_avgs, lmbda_h, home_reds, home_red_minutes, home_goals, away_goals, lmbda_a)
    _per_team("away", away_stats, away_avgs, lmbda_a, away_reds, away_red_minutes, away_goals, home_goals, lmbda_h)

    drivers.sort(key=lambda d: d["deviation"], reverse=True)
    return drivers[:10]
