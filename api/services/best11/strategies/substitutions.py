"""Substitution strategies for the best-11 feature (processing layer).

Strategy pattern: SubstitutionStrategy.suggest() turns the XI + all
eligible players into like-for-like substitution suggestions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from data.player_ratings import PlayerRating


class SubstitutionStrategy(ABC):
    """Base class for bench/replacement suggestions."""

    @abstractmethod
    def suggest(self, lineup: List[Dict],
                eligible: List[PlayerRating]) -> List[Dict]:
        """Return substitution dicts (slot, out, in, rating_delta, reason)."""


class RotationSubstitutionStrategy(SubstitutionStrategy):
    """3 like-for-like substitutions for the weakest starters.

    Candidates: non-XI players at/above the minute floor, same position
    slot (or a listed secondary position). GKs never sub for outfield
    slots and vice versa; slots without a candidate are skipped.
    """

    WEAKEST_COUNT = 3

    def suggest(self, lineup: List[Dict],
                eligible: List[PlayerRating]) -> List[Dict]:
        in_xi = {e["name"] for e in lineup}
        weak_starters = sorted(lineup, key=lambda e: e["rating"])[:self.WEAKEST_COUNT]

        def candidates_for(slot: str) -> List[PlayerRating]:
            cands = []
            for r in eligible:
                if r.name in in_xi:
                    continue
                if slot == "GK":
                    if r.position == "GK":
                        cands.append(r)
                elif r.position != "GK" and (r.position == slot
                                             or slot in (r.pos_list or [])):
                    cands.append(r)
            cands.sort(key=lambda r: r.rating, reverse=True)
            return cands

        subs: List[Dict] = []
        for starter in weak_starters:
            cands = candidates_for(starter["slot"])
            if not cands:
                continue
            sub = cands[0]
            delta = round(sub.rating - starter["rating"], 1)
            if delta >= 0:
                reason = (f"Replaces {starter['name']} — +{delta} rating, "
                          f"fresher ({int(starter['minutes'])} vs "
                          f"{int(sub.minutes)} min this season)")
            else:
                reason = (f"Replaces {starter['name']} — rotation option "
                          f"({int(starter['minutes'])} vs {int(sub.minutes)} min "
                          f"this season)")
            subs.append({
                "slot": starter["slot"],
                "out": starter["name"],
                "in": sub.name,
                "rating_delta": delta,
                "reason": reason,
            })
            in_xi.add(sub.name)
        return subs
