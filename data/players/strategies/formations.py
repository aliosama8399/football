"""Formation strategies for the best-11 feature.

Strategy pattern: every formation mode implements FormationStrategy and
returns (slots, notes). AutoFormationStrategy scores every shape as
summed XI rating − 12 per flex pick; FixedFormationStrategy validates a
named shape. _fill_lineup / _entry are shared helpers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from data.player_ratings import PlayerRating

FORMATIONS: Dict[str, Dict[str, int]] = {
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-2-3-1": {"GK": 1, "DF": 4, "MF": 2, "FW": 4},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
}

FLEX_PENALTY = 12.0
DEFAULT_FORMATION = "4-3-3"


def _entry(r: PlayerRating, slot: str, flex: bool) -> Dict:
    top_shares = {k: round(v, 3) for k, v in sorted(
        r.shares.items(), key=lambda kv: -(kv[1] or 0))[:5] if v is not None}
    return {
        "slot": slot,
        "name": r.name,
        "position": r.position,
        "rating": round(r.rating, 1),
        "minutes": r.minutes,
        "flex": flex,
        "top_shares": top_shares,
    }


def _fill_lineup(eligible: List[PlayerRating], slots: Dict[str, int]):
    """Fill a formation from eligible players; flex-picks cover gaps."""
    lineup: List[Dict] = []
    notes: List[str] = []
    used: set = set()
    for bucket, need in sorted(slots.items(), key=lambda kv: kv[0] != "GK"):
        order = [r for r in eligible if r.position == bucket and id(r) not in used]
        order.sort(key=lambda r: r.rating, reverse=True)
        for i in range(need):
            if i < len(order):
                r = order[i]
                used.add(id(r))
                lineup.append(_entry(r, bucket, flex=False))
            else:
                rest = [r for r in eligible
                        if id(r) not in used and r.position != "GK"]
                rest.sort(key=lambda r: (bucket in (r.pos_list or []), r.rating),
                          reverse=True)
                if rest:
                    r = rest[0]
                    used.add(id(r))
                    lineup.append(_entry(r, bucket, flex=True))
                    fit = "natural fit" if bucket in (r.pos_list or []) else ""
                    notes.append(f"{bucket}: promoted {r.name} (flex pick, "
                                 f"natural {r.position}{', ' + fit if fit else ''})")
                else:
                    notes.append(f"{bucket}: no eligible player left")
    return lineup, notes


class FormationStrategy(ABC):
    """Base class for selecting a formation shape."""

    @abstractmethod
    def slots(self, eligible: List[PlayerRating]) -> Tuple[Dict[str, int], List[str]]:
        """Return (slot counts per position bucket, notes)."""


class FixedFormationStrategy(FormationStrategy):
    """Use a named formation; raises on unknown shapes."""

    def __init__(self, name: str):
        if name not in FORMATIONS:
            raise ValueError(f"Unknown formation '{name}'. "
                             f"Supported: {sorted(FORMATIONS)} or 'auto'")
        self.name = name

    def slots(self, eligible: List[PlayerRating]) -> Tuple[Dict[str, int], List[str]]:
        return dict(FORMATIONS[self.name]), []


class AutoFormationStrategy(FormationStrategy):
    """Pick the shape that yields the strongest natural XI.

    Score = summed XI rating − 12 per flex (out-of-position) pick, so a
    team with 3 elite CBs gets 3-5-2 while a team stacked in attack
    stays in a front-heavy shape. Defaults to 4-3-3 if nothing fields
    a full XI.
    """

    def __init__(self, default: str = DEFAULT_FORMATION):
        self.default = default
        self.last_choice: str = default

    def slots(self, eligible: List[PlayerRating]) -> Tuple[Dict[str, int], List[str]]:
        best_name, best_score = self.default, None
        for name, counts in FORMATIONS.items():
            lineup, _ = _fill_lineup(eligible, counts)
            if len(lineup) < 11:
                continue
            total = sum(e["rating"] for e in lineup)
            flexes = sum(1 for e in lineup if e["flex"])
            score = total - FLEX_PENALTY * flexes
            if best_score is None or score > best_score:
                best_name, best_score = name, score
        self.last_choice = best_name
        return dict(FORMATIONS[best_name]), []


def make_formation_strategy(name: str = "auto") -> FormationStrategy:
    """Strategy factory: 'auto' picks a shape, anything else is fixed."""
    if name == "auto":
        return AutoFormationStrategy()
    return FixedFormationStrategy(name)
