"""
Live In-Match Prediction Adjustment
====================================
Pure-math layer that turns a PRE-MATCH prediction (e.g. TEA-GNN probabilities)
into a LIVE prediction given the current match state (minute, score, live stats).

Pipeline:
  1. pre-match probs (H/D/A) -> expected-goals rates (lambda_home, lambda_away)
     by inverting an independent-Poisson model (least squares).
  2. Live pace factors: each provided live stat (shots, SOT, xG, corners,
     fouls, yellows) is compared to the team's season-average per-minute rate;
     red cards and score momentum (observed vs expected scoring rate) also
     adjust the remaining-time goal rates.
  3. Remaining-time Poisson conditioning: P(final result | current score)
     computed from the remaining expected goals over the remaining minutes.
  4. Logistic blend between the pre-match prior and the live-adjusted probs
     (the live state matters more as the match progresses).

All functions are deterministic and unit-testable; nothing here touches IO.
"""

import functools
import math
from typing import Dict, List, Optional

TOTAL_MINUTES = 90
MAX_GOALS = 10  # per side truncation for the Poisson tables (plenty of headroom)

# Precomputed factorials 0..10 for high-frequency Poisson calculations
FACTORIALS = (1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800)


# ── Poisson outcome math ─────────────────────────────────────────────────────

def poisson_pmf(lmbda: float, k: int) -> float:
    """P(X = k) for X ~ Poisson(lmbda)."""
    fact = FACTORIALS[k] if k < len(FACTORIALS) else math.factorial(k)
    return math.exp(-lmbda) * (lmbda ** k) / fact


def outcome_probs(lmbda_h: float, lmbda_a: float, max_goals: int = MAX_GOALS) -> Dict[str, float]:
    """
    Full-time H/D/A probabilities under independent Poissons.
    Returns {"H": ..., "D": ..., "A": ...} (sums to 1.0).
    """
    ph = [poisson_pmf(lmbda_h, i) for i in range(max_goals + 1)]
    pa = [poisson_pmf(lmbda_a, j) for j in range(max_goals + 1)]

    home = draw = away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    total = home + draw + away
    return {"H": home / total, "D": draw / total, "A": away / total}


@functools.lru_cache(maxsize=2048)
def _rates_from_probs_cached(t_h: float, t_d: float, t_a: float,
                             lmbda_h0: float, lmbda_a0: float) -> tuple[float, float]:
    """Internal memoized solver for independent Poisson goal rates."""
    def loss(lh: float, la: float) -> float:
        o = outcome_probs(lh, la)
        return ((o["H"] - t_h) ** 2 + (o["D"] - t_d) ** 2 + (o["A"] - t_a) ** 2)

    best = None
    best_loss = float("inf")
    step = 0.25
    lh, la = lmbda_h0, lmbda_a0
    for _ in range(60):  # hill-descent with shrinking step
        improved = False
        for dlh, dla in ((step, 0), (-step, 0), (0, step), (0, -step)):
            nlh, nla = max(0.01, lh + dlh), max(0.01, la + dla)
            l = loss(nlh, nla)
            if l < best_loss:
                best_loss, best = l, (nlh, nla)
        if best is None:
            step /= 2
            if step < 1e-4:
                break
            continue
        lh, la = best
        if step > 0.02:
            step = max(0.02, step * 0.7)

    if best is None:
        best = (lmbda_h0, lmbda_a0)
    return (round(best[0], 4), round(best[1], 4))


def rates_from_probs(probs: Dict[str, float],
                     lmbda_h0: float = 1.4, lmbda_a0: float = 1.1) -> Dict[str, float]:
    """
    Invert the Poisson outcome model: find (lambda_h, lambda_a) whose outcome
    probabilities best match the given pre-match H/D/A probs.

    Memoized via LRU cache on rounded probabilities to avoid redundant optimizations.
    """
    t_h = round(float(probs.get("H", 1 / 3)), 4)
    t_d = round(float(probs.get("D", 1 / 3)), 4)
    t_a = round(float(probs.get("A", 1 / 3)), 4)
    lh, la = _rates_from_probs_cached(t_h, t_d, t_a, lmbda_h0, lmbda_a0)
    return {"home": lh, "away": la}


def conditional_remaining_probs(
    lmbda_h_rem: float,
    lmbda_a_rem: float,
    home_goals: int,
    away_goals: int,
    max_goals: int = MAX_GOALS,
) -> Dict[str, float]:
    """
    P(final result | current score (home_goals, away_goals)) when the remaining
    goals are Poisson(lmbda_h_rem) x Poisson(lmbda_a_rem).
    """
    ph = [poisson_pmf(lmbda_h_rem, i) for i in range(max_goals + 1)]
    pa = [poisson_pmf(lmbda_a_rem, j) for j in range(max_goals + 1)]

    home = draw = away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if home_goals + i > away_goals + j:
                home += p
            elif home_goals + i == away_goals + j:
                draw += p
            else:
                away += p
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


# ── Live pace factors ────────────────────────────────────────────────────────

# Live stat key -> (season-average field in team profile, weight in pace blend)
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


def compute_pace_factors(
    minute: int,
    live_stats: Dict[str, Optional[float]],
    season_avgs: Dict[str, Optional[float]],
    goals: int,
    lmbda_pre: float,
    reds: int = 0,
) -> Dict[str, float]:
    """
    Combine stat-based pace + score momentum + red cards into ONE multiplier
    for a team's remaining expected goals.

    live_stats:   {"xg": .., "shots": .., "sot": .., "corners": .., "fouls": .., "yellows": ..}
                  values are cumulative counts so far; None = not provided.
    season_avgs:  per-90 averages from the team profile (avg_xg, avg_shots, ...).
    goals:        goals scored so far (drives the momentum term).
    lmbda_pre:    this team's pre-match expected goals (full match).
    reds:         red cards so far (negative adjustment).

    Returns a dict with 'pace' (combined multiplier) and 'components' breakdown.
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

    # Score momentum: observed scoring rate vs pre-match expectation, shrunk.
    expected_rate = lmbda_pre / TOTAL_MINUTES
    observed_rate = goals / minute
    raw_momentum = _safe_ratio(observed_rate, expected_rate, PACE_FLOOR, PACE_CEIL) if expected_rate > 0 else 1.0
    momentum = 1.0 + 0.5 * (raw_momentum - 1.0)  # 50% shrinkage
    components["goal_momentum"] = round(momentum, 3)

    # Confidence ramps with time played (early minutes are noisy).
    w = min(1.0, minute / 45.0)
    base = 1.0 + w * (0.6 * (stat_pace - 1.0) + 0.4 * (momentum - 1.0))

    # Red-card penalty: a red costs roughly a quarter of remaining attacking output.
    if reds > 0:
        penalty = max(0.5, 1.0 - 0.25 * reds)
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
) -> List[Dict]:
    """
    Ranked list of the biggest live-state deviations vs season baselines,
    for grounding the LLM narrative and for the deterministic response.

    Each driver: {"stat", "side", "label", "live", "season_avg", "pace", "direction"}
    direction = "over" | "under" | "neutral"
    """
    drivers: List[Dict] = []

    def _per_team(side: str, live_stats, avgs, lmbda, reds, goals):
        for key, (_, avg_field) in PACE_STAT_WEIGHTS.items():
            live = live_stats.get(key)
            avg = avgs.get(avg_field)
            if live is None or avg is None or avg <= 0:
                continue
            live_rate = live / minute
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
        if reds > 0:
            drivers.append({
                "stat": "red_card",
                "side": side,
                "label": "RED CARD",
                "live": float(reds),
                "season_avg": 0.0,
                "pace": round(max(0.5, 1.0 - 0.25 * reds), 2),
                "deviation": 1.0,
                "direction": "over",
            })

    _per_team("home", home_stats, home_avgs, lmbda_h, home_reds, home_goals)
    _per_team("away", away_stats, away_avgs, lmbda_a, away_reds, away_goals)

    drivers.sort(key=lambda d: d["deviation"], reverse=True)
    return drivers[:10]
