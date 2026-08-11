"""Strategy package for the scouting feature."""
from .scoring import (DefenderScoring, ForwardScoring, GoalkeeperScoring,
                      MidfielderScoring, ScoredCandidate, ScoringStrategy,
                      make_scoring_strategy)

__all__ = [
    "DefenderScoring", "ForwardScoring", "GoalkeeperScoring",
    "MidfielderScoring", "ScoredCandidate", "ScoringStrategy",
    "make_scoring_strategy",
]
