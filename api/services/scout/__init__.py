"""api/services/scout — the scouting processing layer.

Strategy pattern: ScoringStrategy ranks candidates by position.
Facade pattern: ScoutService orchestrates the repository (collector)
and the strategies into a scouting report.
"""
from .service import ScoutService
from .strategies.scoring import make_scoring_strategy

__all__ = ["ScoutService", "make_scoring_strategy"]
