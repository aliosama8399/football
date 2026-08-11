"""Scoring strategies for the scouting feature.

Strategy pattern: a `ScoringStrategy` ranks PlayerRecord candidates by
position. Every implementation scores a pool of records and attaches the
breakdown (per-metric normalised contribution) for transparency.

Two-stage scoring
-----------------
1. **Base score (0..100)** — position-specific weighted sum of per-90
   metrics, every one of which lives in the fused squad cache (FBRef +
   understat). No API call is needed.
2. **Style fit bonus** — when the coach supplied `my_team` we receive a
   style template (the coach's own per-90 average at that position) and
   add a 30% weighted fit term to the base. Cosine similarity is used so
   the bonus is scale-invariant (a kid for a tiny role gets the same fit
   credit as a star if their *shape* matches).

The score is `0.7 * base + 0.3 * fit_norm * 100` when a style template
is supplied, otherwise just `base`. The breakdown reports both the
base components and a single `style_fit` cell (0..1).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional, Tuple


@dataclass
class ScoredCandidate:
    record_score: float
    breakdown: Dict[str, float]
    fit: Optional[float] = None


# Metrics used to compare a candidate's play style to the coach's squad.
# All are present in the fused squad cache (no API-Football needed).
STYLE_METRICS: Tuple[str, ...] = (
    "goals", "assists", "xg", "xa", "shots", "key_passes",
    "tackles", "interceptions", "blocks",
)


def _per_90(value: Optional[float], minutes: Optional[float]) -> Optional[float]:
    if value is None or not minutes or minutes <= 0:
        return None
    return float(value) * 90.0 / float(minutes)


def _safe(v):
    return None if v is None else float(v)


def _norm_pool(values: List[Optional[float]]) -> Dict[int, float]:
    """Min-max normalise within the pool, None → 0."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {i: 0.0 for i, v in enumerate(values)}
    lo, hi = min(valid), max(valid)
    spread = hi - lo
    out: Dict[int, float] = {}
    for i, v in enumerate(values):
        if v is None:
            out[i] = 0.0
        elif spread == 0:
            out[i] = 1.0          # single data point → the pool's best
        else:
            out[i] = (v - lo) / spread
    return out


def _style_vector(rec) -> List[float]:
    v = []
    for m in STYLE_METRICS:
        raw = getattr(rec, m, None)
        p90 = _per_90(_safe(raw), _safe(getattr(rec, "minutes", None)))
        v.append(p90 if p90 is not None else 0.0)
    return v


def _cosine(a: List[float], b: List[float]) -> float:
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class ScoringStrategy(ABC):
    """Score a candidate pool by position, optionally blending a style fit."""

    # Subclasses declare WEIGHTS = {metric: weight, ...} with sum == 1.0
    WEIGHTS: Dict[str, float] = {}
    # Fit blend weight (overrides nothing when no template is supplied)
    FIT_WEIGHT: float = 0.30

    @abstractmethod
    def _metrics(self, records) -> Dict[str, List[Optional[float]]]: ...

    def score(self, records, style: Optional[Dict] = None) -> List[ScoredCandidate]:
        """Score the pool. `style` is a dict metric → per-90 (reference).
        When None, only the base score is computed.
        """
        metrics = self._metrics(records)
        normed = {k: _norm_pool(v) for k, v in metrics.items()}
        n = len(records)
        # Base fraction 0..1 (normalised weighted sum). Pre-compute per row.
        base_frac: List[float] = []
        for i in range(n):
            bf = sum(self.WEIGHTS.get(k, 0) * normed[k][i]
                     for k in self.WEIGHTS)
            base_frac.append(bf)
        # Cosine similarity of each candidate's style vector vs the coach's
        # per-90 reference (when supplied).
        fit_vals: List[Optional[float]] = [None] * n
        if style:
            ref_vec = [style.get(m) if style.get(m) is not None else 0.0
                       for m in STYLE_METRICS]
            for i, rec in enumerate(records):
                fit_vals[i] = _cosine(_style_vector(rec), ref_vec)
        # Assemble breakdowns + final scores.
        out: List[ScoredCandidate] = []
        for i in range(n):
            bf = base_frac[i]
            cell = {k: round(self.WEIGHTS.get(k, 0) * normed[k][i], 3)
                    for k in self.WEIGHTS}
            fit = fit_vals[i]
            if fit is None:
                # No coach-style template → base score alone (rescaled 0..100,
                # so the missing-fit 30% bucket is folded into production).
                score = round(bf / max(sum(self.WEIGHTS.values()), 1e-9) * 100, 1)
                cell["style_fit"] = 0.0
                out.append(ScoredCandidate(score, cell, None))
            else:
                # 70% base (of the weighted production) + 30% fit.
                base_norm = bf / max(sum(self.WEIGHTS.values()), 1e-9)
                score = round((1 - self.FIT_WEIGHT) * base_norm * 100
                              + self.FIT_WEIGHT * fit * 100, 1)
                cell["style_fit"] = round(fit, 3)
                out.append(ScoredCandidate(score, cell, fit))
        return out


class ForwardScoring(ScoringStrategy):
    """Forwards — output (goals/assists/xg/xa/shots) + creator."""

    WEIGHTS = {
        "xg_per90":      0.20,
        "xa_per90":      0.15,
        "goals_per90":   0.20,
        "assists_per90":  0.15,
        "shots_per90":   0.10,
        "key_passes_per90": 0.10,
        "minutes":       0.10,
    }

    def _metrics(self, records):
        return {
            "xg_per90":     [_per_90(_safe(r.xg),     _safe(r.minutes)) for r in records],
            "xa_per90":     [_per_90(_safe(r.xa),     _safe(r.minutes)) for r in records],
            "goals_per90":  [_per_90(_safe(r.goals),  _safe(r.minutes)) for r in records],
            "assists_per90":[_per_90(_safe(r.assists),_safe(r.minutes)) for r in records],
            "shots_per90":  [_per_90(_safe(r.shots),  _safe(r.minutes)) for r in records],
            "key_passes_per90": [_per_90(_safe(r.key_passes), _safe(r.minutes)) for r in records],
            "minutes":      [_safe(r.minutes) for r in records],
        }


class MidfielderScoring(ScoringStrategy):
    """Midfielders — creation (xa/key passes) + progression + minutes."""

    WEIGHTS = {
        "xa_per90":      0.20,
        "assists_per90": 0.15,
        "key_passes_per90": 0.20,
        "goals_per90":   0.10,
        "xg_per90":      0.05,
        "tackles_per90": 0.10,
        "interceptions_per90": 0.05,
        "minutes":       0.15,
    }

    def _metrics(self, records):
        return {
            "xa_per90":      [_per_90(_safe(r.xa),     _safe(r.minutes)) for r in records],
            "assists_per90": [_per_90(_safe(r.assists),_safe(r.minutes)) for r in records],
            "key_passes_per90": [_per_90(_safe(r.key_passes), _safe(r.minutes)) for r in records],
            "goals_per90":   [_per_90(_safe(r.goals),  _safe(r.minutes)) for r in records],
            "xg_per90":      [_per_90(_safe(r.xg),      _safe(r.minutes)) for r in records],
            "tackles_per90": [_per_90(_safe(r.tackles), _safe(r.minutes)) for r in records],
            "interceptions_per90": [_per_90(_safe(r.interceptions), _safe(r.minutes)) for r in records],
            "minutes":       [_safe(r.minutes) for r in records],
        }


class DefenderScoring(ScoringStrategy):
    """Defenders — duels/wins (tackles/ints/blocks) + discipline + minutes."""

    WEIGHTS = {
        "tackles_per90":       0.20,
        "interceptions_per90": 0.20,
        "blocks":              0.10,
        "minutes":             0.15,
        "progressive_passes":  0.10,
        "yellow_cards":        0.05,   # inverted below
        "key_passes_per90":    0.10,
        "shots_per90":         0.10,
    }

    def _metrics(self, records):
        yellow = [None if r.yellow_cards is None else -float(r.yellow_cards)
                  for r in records]
        return {
            "tackles_per90":       [_per_90(_safe(r.tackles), _safe(r.minutes)) for r in records],
            "interceptions_per90": [_per_90(_safe(r.interceptions), _safe(r.minutes)) for r in records],
            "blocks":              [_safe(r.blocks) for r in records],
            "minutes":             [_safe(r.minutes) for r in records],
            "progressive_passes":  [_safe(r.progressive_passes) for r in records],
            "yellow_cards":        yellow,
            "key_passes_per90":    [_per_90(_safe(r.key_passes), _safe(r.minutes)) for r in records],
            "shots_per90":         [_per_90(_safe(r.shots), _safe(r.minutes)) for r in records],
        }


class GoalkeeperScoring(ScoringStrategy):
    """Goalkeepers — saves & goals conceded + playing time."""

    WEIGHTS = {
        "saves":           0.30,
        "goals_conceded":  0.10,   # inverted below
        "appearances":     0.15,
        "minutes":         0.15,
        "clean_sheets":    0.30,
    }

    def _metrics(self, records):
        conceded = [None if r.goals_conceded is None else -float(r.goals_conceded)
                    for r in records]
        return {
            "saves":          [_safe(r.saves) for r in records],
            "goals_conceded": conceded,
            "appearances":    [_safe(r.appearances) for r in records],
            "minutes":        [_safe(r.minutes) for r in records],
            "clean_sheets":   [_safe(r.clean_sheets) for r in records],
        }


def make_scoring_strategy(position: str) -> ScoringStrategy:
    p = position.upper()
    if p == "FW":
        return ForwardScoring()
    if p == "MF":
        return MidfielderScoring()
    if p == "DF":
        return DefenderScoring()
    if p == "GK":
        return GoalkeeperScoring()
    raise ValueError(f"Unknown position '{position}'. Supported: GK | DF | MF | FW")
