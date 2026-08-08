"""
FodPlayerProvider — football-data.org v4 REST API.

Rich squads (positions, ages, nationalities, shirt numbers) but NO per-player
season stats on the free tier. Good for roster completeness and identity;
stat columns stay empty (the probe report shows this trade-off).

Auth: X-Auth-Token, read from env FOD_API_KEY (or FOOTBALL_DATA_API_KEY).
"""

import logging
import os
from typing import Dict, List, Optional

import requests

from data.player_providers.base import BasePlayerProvider
from data.player_providers.schema import PlayerRecord

logger = logging.getLogger(__name__)

_BASE = "https://api.football-data.org/v4"

# data/config.yaml league code → football-data.org competition code
_LEAGUE_CODES = {"E0": "PL", "SP1": "PD", "D1": "BL1", "I1": "SA", "F1": "FL1"}

_POS_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DF",
    "Midfielder": "MF",
    "Attacker": "FW",
}


class FodPlayerProvider(BasePlayerProvider):
    provider_name = "fod"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = (api_key or os.getenv("FOD_API_KEY")
                        or os.getenv("FOOTBALL_DATA_API_KEY") or "").strip()
        self._timeout = timeout
        if self.api_key:
            logger.info("football-data.org key configured (%s...)", self.api_key[:4])
        else:
            logger.warning("No FOD_API_KEY set — football-data.org provider will return []")

    def capabilities(self) -> Dict[str, bool]:
        return dict(position=True, age=True, nationality=True,
                    minutes=False, goals=False, xg=False, saves=False)

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self.api_key:
            return None
        r = requests.get(
            f"{_BASE}{path}",
            headers={"X-Auth-Token": self.api_key},
            params=params or {},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()

    def fetch_team_squad(self, team: str, league_code: str, season: str) -> List[PlayerRecord]:
        comp = _LEAGUE_CODES.get(league_code)
        if not comp:
            raise ValueError(f"football-data.org league code not mapped: {league_code}")
        # v4 competition teams endpoint returns the squad inline.
        data = self._get(f"/competitions/{comp}/teams", params={"season": "2024"})
        if not data:
            return []
        want = team.lower()
        squad = None
        for t in data.get("teams", []):
            if t.get("name", "").lower() == want or t.get("shortName", "").lower() == want:
                squad = t.get("squad", [])
                break
        if squad is None:
            logger.warning("football-data.org: team '%s' not found in %s", team, comp)
            return []

        records = []
        for p in squad:
            if not p.get("name"):
                continue
            records.append(PlayerRecord(
                name=p["name"],
                team=team,
                league=_LEAGUE_CODES.get(league_code, league_code),
                season=season,
                source=self.provider_name,
                position=_POS_MAP.get(p.get("position", "")),
                age=_parse_age(p.get("dateOfBirth")),
                nationality=p.get("nationality"),
                extra={"shirt_number": p.get("shirtNumber"), "id": p.get("id")},
            ))
        logger.info("football-data.org %s: %d players (stats unavailable on free tier)",
                    team, len(records))
        return records


def _parse_age(dob: Optional[str]) -> Optional[float]:
    """Approximate age in years from an ISO date of birth."""
    if not dob:
        return None
    try:
        from datetime import date, datetime
        y = datetime.strptime(dob[:10], "%Y-%m-%d").year
        return (date.today().year - y) + (date.today().month - int(dob[5:7])) / 12
    except (ValueError, IndexError):
        return None
