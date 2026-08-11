"""Strategy package for the best-11 feature (processing layer).

Strategies are interchangeable, so rating mode, formation choice and
substitution logic can be composed, tested and extended independently
(see ratings.py, formations.py, substitutions.py).
"""

from .formations import (FORMATIONS, AutoFormationStrategy,
                         FixedFormationStrategy, FormationStrategy,
                         make_formation_strategy)
from .ratings import (H2HBlendDecorator, RatingEnhancer, RatingOutcome,
                      RatingStrategy, SeasonRatingStrategy,
                      ThroughDateRatingStrategy, make_rating_strategy)
from .substitutions import RotationSubstitutionStrategy, SubstitutionStrategy

__all__ = [
    "FORMATIONS", "AutoFormationStrategy", "FixedFormationStrategy",
    "FormationStrategy", "make_formation_strategy",
    "H2HBlendDecorator", "RatingEnhancer", "RatingOutcome",
    "RatingStrategy", "SeasonRatingStrategy", "ThroughDateRatingStrategy",
    "make_rating_strategy",
    "RotationSubstitutionStrategy", "SubstitutionStrategy",
]
