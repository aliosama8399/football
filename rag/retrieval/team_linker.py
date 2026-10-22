"""
Smart Team Linker (Phase 2)
============================
Entity linking for football queries: identifies team mentions and normalizes
them to canonical names using data/team_registry.py, substring matching,
and fuzzy token similarity.

Usage:
    from rag.retrieval.team_linker import link_teams
    teams = link_teams("How does Man City attack vs Wolves?")
    # -> ["Manchester City", "Wolverhampton Wanderers"]
"""

import re
import difflib
from typing import Optional
from data.team_registry import TEAM_REGISTRY

# Build a flattened alias -> canonical map (lowercase for fast lookup)
_ALIAS_MAP: dict[str, str] = {}
_ALL_CANONICAL: set[str] = set()

for league_code, teams in TEAM_REGISTRY.items():
    for alias, canonical in teams.items():
        _ALIAS_MAP[alias.lower()] = canonical
        _ALL_CANONICAL.add(canonical)

_CANONICAL_LIST = sorted(list(_ALL_CANONICAL), key=len, reverse=True)


def link_teams(query: str, known_teams: Optional[list[str]] = None, max_teams: int = 2) -> list[str]:
    """
    Extract up to `max_teams` canonical team names from a free-text question.
    """
    found: list[str] = []
    q_lower = query.lower()

    # 1. Exact alias match from registry (prioritize longer aliases first)
    sorted_aliases = sorted(_ALIAS_MAP.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        # Match as word boundary if short, or plain substring if >= 5 chars
        pattern = r"\b" + re.escape(alias) + r"\b" if len(alias) <= 6 else re.escape(alias)
        if re.search(pattern, q_lower):
            canonical = _ALIAS_MAP[alias]
            if canonical not in found:
                found.append(canonical)
            if len(found) >= max_teams:
                return found

    # 2. Match against provided known_teams or canonical names list
    candidates = known_teams or _CANONICAL_LIST
    for team in sorted(candidates, key=len, reverse=True):
        if team.lower() in q_lower and team not in found:
            found.append(team)
        if len(found) >= max_teams:
            return found

    # 3. Fuzzy fallback using difflib for words in query
    if not found:
        tokens = re.findall(r"\b[A-Za-z]{4,}\b", query)
        for tok in tokens:
            matches = difflib.get_close_matches(tok, _CANONICAL_LIST, n=1, cutoff=0.82)
            if matches and matches[0] not in found:
                found.append(matches[0])
            if len(found) >= max_teams:
                return found

    return found
