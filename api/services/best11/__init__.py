"""api/services/best11 — the best-11 processing layer.

Strategy pattern: rating modes, formations and substitution rules.
Facade pattern: Best11Service orchestrates the repository (collector)
and the strategies into a lineup prediction. This package is the
single processing entry point for the best-11 feature.
"""

from .service import Best11Service
from .strategies.formations import FORMATIONS

__all__ = ["Best11Service", "FORMATIONS"]
