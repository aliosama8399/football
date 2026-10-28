"""
UnderstatPlayerProvider — per-league playersData JSON (xG/xA rich, thin elsewhere).

Provides goals/assists/xG/xA/shots per player per season plus a rough
position — works via pure HTTP JSON requests without any browser dependency.

Season format in: '2425'  →  Understat year '2024' (start of season).
"""

import json
import logging
from typing import Dict, List, Optional

import requests

from data.player_providers.base import BasePlayerProvider
from data.player_providers.schema import PlayerRecord
from data.team_registry import normalize_team_name

logger = logging.getLogger(__name__)

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
}

# League pages no longer embed playersData; the JS fetches it via this endpoint.
_LEAGUE_DATA_URL = "https://understat.com/getLeagueData/{slug}/{year}/"

# data/config.yaml league code → understat slug (as used in URL path)
_LEAGUE_SLUGS = {
    "E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga",
    "I1": "Serie_A", "F1": "Ligue_1",
}

_POS_MAP = {
    "G": "GK",
    "D": "DF", "DC": "DF", "DL": "DF", "DR": "DF",
    "M": "MF", "MC": "MF", "ML": "MF", "MR": "MF", "DM": "MF", "AM": "MF",
    "F": "FW", "FC": "FW", "FL": "FW", "FR": "FW", "ST": "FW", "SS": "FW",
}


def _coarse_position(pos: Optional[str]) -> Optional[str]:
    """Understat positions look like 'F M', 'D C', 'G', 'M' → GK|DF|MF|FW."""
    if not pos:
        return None
    primary = str(pos).strip().upper()[:1]
    return _POS_MAP.get(primary)


def _pos_list(pos: Optional[str], primary_bucket: Optional[str]) -> List[str]:
    if not pos:
        return [primary_bucket] if primary_bucket else []
    buckets = []
    for tok in str(pos).strip().upper().split():
        b = _POS_MAP.get(tok[:1])
        if b and b not in buckets:
            buckets.append(b)
    if not buckets and primary_bucket:
        buckets.append(primary_bucket)
    return buckets


class UnderstatPlayerProvider(BasePlayerProvider):
    provider_name = "understat"

    def __init__(self, rate_limit_sec: float = 1.5, timeout: int = 30):
        self._delay = rate_limit_sec
        self._timeout = timeout
        self._last_fetch = 0.0
        self._cache: Dict[str, List[dict]] = {}

    def capabilities(self) -> Dict[str, bool]:
        return dict(position=True, age=False, minutes=True, goals=True,
                    assists=True, xg=True, xa=True, shots=True,
                    shots_on_target=False, tackles=False, saves=False,
                    clean_sheets=False, cards=True)

    def _get(self, url: str) -> str:
        import time
        elapsed = time.time() - self._last_fetch
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        headers = dict(_UA, Referer="https://understat.com/",
                       **{"X-Requested-With": "XMLHttpRequest"})
        r = requests.get(url, headers=headers, timeout=self._timeout)
        r.raise_for_status()
        self._last_fetch = time.time()
        return r.text

    def _league_data(self, league_code: str, season: str) -> List[dict]:
        """Cache + parse the league JSON (getLeagueData: teams+players+dates)."""
        key = f"{league_code}:{season}"
        if key in self._cache:
            return self._cache[key]
        slug = _LEAGUE_SLUGS.get(league_code)
        if not slug:
            raise ValueError(f"Understat league code not mapped: {league_code}")
        year = f"20{season[:2]}"
        data = json.loads(self._get(_LEAGUE_DATA_URL.format(slug=slug, year=year)))
        rows = data.get("players", [])
        self._cache[key] = rows
        logger.info("Understat %s %s: %d player rows", league_code, season, len(rows))
        return rows

    def fetch_team_squad(self, team: str, league_code: str, season: str) -> List[PlayerRecord]:
        rows = self._league_data(league_code, season)
        want_canon = normalize_team_name(team, league_code).lower()
        want_raw = team.lower()
        records = []
        for p in rows:
            # Understat team_title can be a comma-separated list if player transferred mid-season
            raw_titles = [t.strip() for t in str(p.get("team_title", "")).split(",") if t.strip()]
            matched_team = False
            for t_title in raw_titles:
                t_canon = normalize_team_name(t_title, league_code).lower()
                if t_canon == want_canon or t_title.lower() == want_raw:
                    matched_team = True
                    break
            if not matched_team:
                continue
            coarse_pos = _coarse_position(p.get("position"))
            records.append(PlayerRecord(
                name=str(p.get("player_name", "")).strip(),
                team=team,
                league=_LEAGUE_SLUGS.get(league_code, league_code),
                season=season,
                source=self.provider_name,
                position=coarse_pos,
                minutes=_num(p.get("time")),
                appearances=_num(p.get("games")),
                goals=_num(p.get("goals")),
                assists=_num(p.get("assists")),
                xg=_num(p.get("xG")),
                xa=_num(p.get("xA")),
                shots=_num(p.get("shots")),
                yellow_cards=int(_num(p.get("yellow_card"), 0)),
                red_cards=int(_num(p.get("red_card"), 0)),
                extra={"understat_id": p.get("id"),
                       "key_passes": _num(p.get("key_passes")),
                       "npxg": _num(p.get("npxG")),
                       "xg_chain": _num(p.get("xGChain")),
                       "xg_buildup": _num(p.get("xg_buildup") or p.get("xGBuildup")),
                       "pos_list": _pos_list(p.get("position"), coarse_pos)},
            ))
        logger.info("Understat %s (%s): %d players", team, season, len(records))
        return records


def _num(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default
