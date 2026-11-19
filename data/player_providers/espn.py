"""EspnPlayerProvider — public ESPN Soccer JSON API (rosters & season stats).

Covers all 5 top European leagues plus their 2nd/3rd divisions without API keys
or Cloudflare challenges, filling gaps for newly promoted, relegated, or
historical clubs absent from Understat/FBRef caches.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import requests

from data._config import get_leagues
from data.player_providers.base import BasePlayerProvider
from data.player_providers.schema import PlayerRecord
from data.team_registry import normalize_team_name

logger = logging.getLogger(__name__)

_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

_LEAGUE_SLUGS: Dict[str, Tuple[str, ...]] = {
    "E0": ("eng.1", "eng.2", "eng.3"),
    "SP1": ("esp.1", "esp.2"),
    "D1": ("ger.1", "ger.2"),
    "I1": ("ita.1", "ita.2"),
    "F1": ("fra.1", "fra.2"),
}

_POS_MAP = {
    "G": "GK",
    "GK": "GK",
    "D": "DF",
    "DF": "DF",
    "M": "MF",
    "MF": "MF",
    "F": "FW",
    "FW": "FW",
}

_CLUB_NOISE = {
    "fc", "afc", "cf", "sc", "as", "ac", "rc", "cd", "ud", "sd", "us",
    "tsg", "tsv", "vfb", "vfl", "sv", "fco", "aj", "ogc", "stade", "real",
    "city", "united", "town", "hotspur", "wanderers", "albion", "calcio",
    "spvgg", "1.", "1846", "98", "05", "04", "07", "1913", "93",
}

_ESPN_ALIASES = {
    "hamburger sv": "hamburg sv",
    "hamburg sv": "hamburger sv",
    "rasenballsport leipzig": "rb leipzig",
    "rb leipzig": "rasenballsport leipzig",
    "deportivo la coruna": "deportivo",
    "santander": "racing santander",
    "racing santander": "santander",
    "oviedo": "real oviedo",
    "inter": "internazionale",
    "internazionale": "inter",
    "roma": "as roma",
    "verona": "hellas verona",
    "parma calcio 1913": "parma",
    "rennes": "stade rennais",
    "reims": "stade de reims",
    "paris saint germain": "paris saint-germain",
    "borussia m.gladbach": "borussia monchengladbach",
    "m'gladbach": "borussia monchengladbach",
}


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in norm if not unicodedata.combining(ch)).strip().lower()


def _clean_tokens(text: str) -> str:
    raw = _strip_accents(text).replace("-", " ").replace(".", " ")
    tokens = [tok for tok in re.split(r"\s+", raw) if tok and tok not in _CLUB_NOISE]
    return " ".join(tokens)


def _season_year(season: str) -> int:
    s = str(season or "").strip()
    if len(s) == 4 and s.startswith("20"):
        return int(s)
    if len(s) == 4 and s[:2].isdigit():
        return 2000 + int(s[:2])
    if len(s) == 6 and s[:4].isdigit():
        return int(s[:4])
    return 0


def _num(val, default: Optional[float] = 0.0) -> Optional[float]:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_stats(athlete: dict) -> Dict[str, float]:
    out: Dict[str, float] = {}
    splits = (athlete.get("statistics") or {}).get("splits") or {}
    for cat in splits.get("categories") or []:
        for st in cat.get("stats") or []:
            name = st.get("name")
            if name:
                out[name] = float(_num(st.get("value"), 0.0) or 0.0)
    return out


class EspnPlayerProvider(BasePlayerProvider):
    """Fetches team squads and season statistics via ESPN's public soccer JSON API."""

    provider_name = "espn"

    def __init__(self, rate_limit_sec: float = 0.25, timeout: int = 12):
        self._delay = rate_limit_sec
        self._timeout = timeout
        self._last_fetch = 0.0
        self._lock = threading.Lock()
        self._team_list_cache: Dict[str, List[dict]] = {}
        self._league_names = {
            code: info["name"] for code, info in get_leagues().items()
        }

    def capabilities(self) -> Dict[str, bool]:
        return dict(
            position=True,
            age=True,
            nationality=True,
            minutes=True,
            goals=True,
            xg=False,
            xa=False,
            shots=True,
            shots_on_target=True,
            tackles=False,
            saves=True,
            clean_sheets=False,
            goals_conceded=True,
            cards=True,
        )

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        with self._lock:
            elapsed = time.time() - self._last_fetch
            if elapsed < self._delay:
                time.sleep(self._delay - elapsed)
            try:
                resp = requests.get(
                    url,
                    params=params or {},
                    headers={"User-Agent": "Mozilla/5.0 (FootballAnalytics/2.0)"},
                    timeout=self._timeout,
                )
                self._last_fetch = time.time()
                if resp.status_code != 200:
                    return None
                return resp.json()
            except Exception as exc:
                self._last_fetch = time.time()
                logger.warning("ESPN request failed (%s): %s", url, exc)
                return None

    def _league_teams(self, slug: str) -> List[dict]:
        if slug in self._team_list_cache:
            return self._team_list_cache[slug]
        data = self._get_json(f"{_BASE_URL}/{slug}/teams")
        if not data:
            self._team_list_cache[slug] = []
            return []
        teams: List[dict] = []
        for sport in data.get("sports") or []:
            for league in sport.get("leagues") or []:
                for item in league.get("teams") or []:
                    tm = item.get("team") or {}
                    if tm.get("id") and (tm.get("displayName") or tm.get("name")):
                        teams.append(tm)
        self._team_list_cache[slug] = teams
        return teams

    def _match_team_entry(
        self, team: str, league_code: str, teams: List[dict]
    ) -> Optional[dict]:
        canon_want = normalize_team_name(team, league_code)
        want_variants = {
            _strip_accents(team),
            _strip_accents(canon_want),
            _ESPN_ALIASES.get(_strip_accents(team), ""),
            _ESPN_ALIASES.get(_strip_accents(canon_want), ""),
        }
        want_variants.discard("")
        want_tokens = {_clean_tokens(w) for w in want_variants if _clean_tokens(w)}

        # Pass 1: exact normalized match
        for tm in teams:
            candidates = {
                _strip_accents(tm.get("displayName", "")),
                _strip_accents(tm.get("shortDisplayName", "")),
                _strip_accents(tm.get("name", "")),
                _strip_accents(
                    normalize_team_name(tm.get("displayName", ""), league_code)
                ),
            }
            if want_variants & candidates:
                return tm

        # Pass 2: club prefix/suffix stripped match
        for tm in teams:
            cand_tokens = {
                _clean_tokens(tm.get("displayName", "")),
                _clean_tokens(tm.get("shortDisplayName", "")),
                _clean_tokens(tm.get("name", "")),
            }
            cand_tokens.discard("")
            if want_tokens & cand_tokens:
                return tm

        # Pass 3: high-confidence fuzzy match
        best_tm = None
        best_ratio = 0.0
        for tm in teams:
            disp = _clean_tokens(tm.get("displayName", ""))
            for wt in want_tokens:
                ratio = difflib.SequenceMatcher(None, wt, disp).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_tm = tm
        return best_tm if best_ratio >= 0.78 else None

    def _fetch_roster(self, slug: str, team_id: str, season: str) -> List[dict]:
        year = _season_year(season)
        url = f"{_BASE_URL}/{slug}/teams/{team_id}/roster"
        if year >= 2022:
            data = self._get_json(url, params={"season": year})
            athletes = (data or {}).get("athletes") or []
            if athletes:
                return athletes
        # Fallback to current club roster when requested historical season is unindexed
        data = self._get_json(url)
        return (data or {}).get("athletes") or []

    def fetch_team_squad(
        self, team: str, league_code: str, season: str
    ) -> List[PlayerRecord]:
        slugs = _LEAGUE_SLUGS.get(league_code, ())
        if not slugs:
            return []

        matched_slug = None
        matched_team = None
        for slug in slugs:
            teams = self._league_teams(slug)
            tm = self._match_team_entry(team, league_code, teams)
            if tm:
                matched_slug = slug
                matched_team = tm
                break

        if not matched_slug or not matched_team:
            return []

        athletes = self._fetch_roster(matched_slug, str(matched_team["id"]), season)
        if not athletes:
            return []

        league_name = self._league_names.get(league_code, league_code)
        canon_team = normalize_team_name(team, league_code)
        records: List[PlayerRecord] = []

        for p in athletes:
            name = str(p.get("displayName") or p.get("fullName") or "").strip()
            if not name:
                continue
            pos_abbr = str((p.get("position") or {}).get("abbreviation") or "").strip().upper()
            coarse_pos = _POS_MAP.get(pos_abbr, "MF")
            stats = _extract_stats(p)

            apps = stats.get("appearances", 0.0)
            sub_ins = min(apps, stats.get("subIns", 0.0))
            starts = max(0.0, apps - sub_ins)
            est_minutes = round(starts * 82.0 + sub_ins * 22.0, 1) if apps > 0 else 0.0

            jersey_raw = p.get("jersey")
            shirt_num: Optional[int] = None
            if jersey_raw is not None and str(jersey_raw).strip().isdigit():
                shirt_num = int(str(jersey_raw).strip())

            nationality = p.get("citizenship") or (p.get("flag") or {}).get("alt")
            goals = stats.get("totalGoals") if stats else None
            assists = stats.get("goalAssists") if stats else None
            shots = stats.get("totalShots") if stats else None
            sot = stats.get("shotsOnTarget") if stats else None
            saves = stats.get("saves") if (stats and coarse_pos == "GK") else None
            conceded = stats.get("goalsConceded") if (stats and coarse_pos == "GK") else None
            fouls = int(stats["foulsCommitted"]) if "foulsCommitted" in stats else None
            yellows = int(stats.get("yellowCards", 0.0)) if stats else None
            reds = int(stats.get("redCards", 0.0)) if stats else None

            records.append(
                PlayerRecord(
                    name=name,
                    team=canon_team,
                    league=league_name,
                    season=str(season),
                    source=self.provider_name,
                    position=coarse_pos,
                    age=_num(p.get("age"), default=None),
                    nationality=str(nationality).strip() if nationality else None,
                    appearances=int(apps) if stats else None,
                    minutes=est_minutes if stats else 0.0,
                    goals=goals,
                    assists=assists,
                    shots=shots,
                    shots_on_target=sot,
                    saves=saves,
                    goals_conceded=conceded,
                    yellow_cards=yellows,
                    red_cards=reds,
                    fouls_committed=fouls,
                    shirt_number=shirt_num,
                    extra={
                        "espn_id": p.get("id"),
                        "espn_slug": matched_slug,
                        "pos_list": [coarse_pos],
                        "minutes_estimated": True,
                    },
                )
            )

        logger.info(
            "ESPN %s (%s via %s %s): %d players",
            canon_team,
            season,
            matched_slug,
            matched_team.get("displayName"),
            len(records),
        )
        return records
