"""ApiFootballPlayerProvider — API-Football v3 (free direct host).

Used by the scouting feature for per-player season stats and transfer
info. Free plan rate limit is 10 requests/minute, paginated 20/page;
every call is therefore cached on disk under data/raw/scout_cache/ so
repeated scouting queries stay well under quota.

Auth: header `x-apisports-key: <key>` (also accepts `x-rapidapi-key`).
The key is read from Settings (env `API_FOOTBALL_KEY`).

Endpoints used (confirmed live against the v3 host):
    /teams?league={id}&season={YYYY}     → all teams in a league-season
    /players?team={id}&season={YYYY}     → one team's full squad with season
                                            stats (paginated, 20/page)
    /transfers?team={id}                  → per-player transfer history

The rich `statistics` block (games, goals, shots, passes, tackles,
duels, dribbles, fouls, cards, penalty) is normalised into PlayerRecord
so the scouting service stays provider-agnostic.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from data.player_providers.base import BasePlayerProvider
from data.player_providers.schema import PlayerRecord

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/raw/scout_cache")
_CACHE_TTL_SECONDS = 24 * 3600  # 24h on disk

# data/config.yaml league code → API-Football league id (verified live)
_LEAGUE_IDS = {"E0": 39, "SP1": 140, "D1": 78, "I1": 135, "F1": 61}

_POS_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DF",
    "Midfielder": "MF",
    "Attacker": "FW",
}

# league code → API-Football host league "name" appears as "Premier League"
# but PlayerRecord.league keeps the project's snake_case ("Premier_League")
_LEAGUE_NAME = {
    "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga",
    "I1": "Serie A", "F1": "Ligue 1",
}


def _snake_league(name: str) -> str:
    return name.replace(" ", "_") if name else ""


class ApiFootballPlayerProvider(BasePlayerProvider):
    """API-Football v3 player/team/transfer provider with disk-cached GET."""

    provider_name = "api_football"

    PAGE_SIZE = 20  # API-Football pages are 20 items each (server-side fixed)

    def __init__(self, api_key: Optional[str] = None,
                 host: Optional[str] = None,
                 cache_dir: Path = _CACHE_DIR,
                 timeout: int = 60):
        self.api_key = (api_key or "").strip()
        self.host = (host or "v3.football.api-sports.io").strip()
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.base = f"https://{self.host}"
        self._rate_lock = threading.Lock()
        self._last_request_ts = 0.0
        # polite minimum spacing between live calls (free plan = 10/min).
        self._min_spacing_seconds = 6.0
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY not set — ApiFootballPlayerProvider "
                           "will return [] / {} (scouting stats unavailable)")

    # ── Capabilities for the probe report ────────────────────────────────────

    def capabilities(self) -> Dict[str, bool]:
        return dict(position=True, age=True, nationality=True,
                    minutes=True, goals=True, xg=False, saves=True,
                    dribbles=True, transfers=True)

    # ── Disk-cached GET with rate-limit spacing + pagination ──────────────────

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str, refresh: bool) -> Optional[dict]:
        path = self._cache_path(key)
        if refresh or not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > _CACHE_TTL_SECONDS:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("scout cache read failed (%s): %s", path.name, e)
            return None

    def _write_cache(self, key: str, payload: dict) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path(key).write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:
            logger.warning("scout cache write failed (%s): %s", key, e)

    def _throttle(self) -> None:
        """Be polite: never more than 1 live call per ~6s (≤10/min)."""
        with self._rate_lock:
            dt = time.time() - self._last_request_ts
            if dt < self._min_spacing_seconds:
                time.sleep(self._min_spacing_seconds - dt)
            self._last_request_ts = time.time()

    def _request(self, path: str, params: dict) -> Optional[dict]:
        if not self.api_key:
            return None
        self._throttle()
        url = f"{self.base}{path}"
        try:
            r = requests.get(url, headers={"x-apisports-key": self.api_key},
                             params=params, timeout=self.timeout)
            if r.status_code == 429:
                logger.warning("API-Football rate-limited (429) — backing off")
                time.sleep(60.0)
                r = requests.get(url, headers={"x-apisports-key": self.api_key},
                                 params=params, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning("API-Football %s -> %s: %s", path,
                               r.status_code, r.text[:200])
                return None
            return r.json()
        except Exception as e:
            logger.warning("API-Football request failed (%s %s): %s", path, params, e)
            return None

    def _request_all_pages(self, path: str, params: dict, cache_key: str,
                           refresh: bool, max_pages: int = 200) -> List[dict]:
        """Fetch every page of a paginated endpoint and concatenate; cache
        the assembled list when complete so later reads cost zero API calls.

        Page-aware: only sends the `page` parameter when the first response
        reports `paging.total > 1` (some endpoints like /teams return
        everything on a single page AND reject the page param).
        """
        cached = self._read_cache(cache_key, refresh)
        if cached is not None and isinstance(cached, list):
            return cached
        merged: List[dict] = []
        params = dict(params)
        # Page 1 — without an explicit page param (some endpoints reject it).
        data = self._request(path, params)
        if not data:
            return []
        chunk = data.get("response")
        if not isinstance(chunk, list):
            return []
        merged.extend(chunk)
        if data.get("errors"):
            logger.warning("API-Football %s errors: %s — not caching", path, data.get("errors"))
            return merged
        paging = data.get("paging") or {}
        total_pages = int(paging.get("total", 1))
        # Page 2.. if the endpoint reports more pages.
        for page in range(2, min(total_pages, max_pages) + 1):
            params["page"] = page
            data = self._request(path, params)
            if not data:
                break
            chunk = data.get("response")
            if not isinstance(chunk, list):
                break
            merged.extend(chunk)
            if data.get("errors"):
                break
        self._write_cache(cache_key, merged)
        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_team_squad(self, team: str, league_code: str,
                         season: str) -> List[PlayerRecord]:
        """Fall back to the league-wide list filtered to this team. Not the
        recommended path (paginates the whole league); prefer fetch_team_players
        with a known team_id. Kept for BasePlayerProvider conformance."""
        league_id = _LEAGUE_IDS.get(league_code)
        if not league_id:
            return []
        season_year = self._season_year(season)
        team_id = self.lookup_team_id(team, league_code, season_year)
        if not team_id:
            return []
        rows = self.fetch_team_players(team_id, season_year, refresh=False)
        return self._normalise(rows, league_code, season, team)

    def fetch_league_teams(self, league_code: str, season: str,
                          refresh: bool = False) -> List[dict]:
        """All teams in a league-season as [{id, name, code?}], cached."""
        league_id = _LEAGUE_IDS.get(league_code)
        if not league_id:
            return []
        season_year = self._season_year(season)
        rows = self._request_all_pages(
            "/teams", {"league": league_id, "season": season_year},
            cache_key=f"teams_{league_code}_{season}", refresh=refresh,
        )
        # each row: {"team": {...}, "venue": {...}} — flatten to {"id","name"}
        out = []
        for row in rows or []:
            t = row.get("team") or {}
            if t.get("id") and t.get("name"):
                out.append({"id": t["id"], "name": t["name"]})
        return out

    def fetch_team_players(self, team_id: int, season_year: int,
                          refresh: bool = False) -> List[dict]:
        """One team's full squad with season stats — paginated & cached.
        Returns the raw API-Football `response` list of player objects.
        Refuses team_id=None to prevent cross-team bleed / league-defaults.
        """
        if not team_id:
            logger.warning("fetch_team_players called without team_id — skipping")
            return []
        return self._request_all_pages(
            "/players", {"team": team_id, "season": season_year},
            cache_key=f"players_team_{team_id}_{season_year}",
            refresh=refresh, max_pages=20,
        )

    def fetch_team_transfers(self, team_id: int, refresh: bool = False) -> List[dict]:
        """Per-player transfer history for a team — cached 24h."""
        return self._request_all_pages(
            "/transfers", {"team": team_id},
            cache_key=f"transfers_team_{team_id}", refresh=refresh,
        )

    def fetch_player_transfers(self, player_id: int, refresh: bool = False) -> List[dict]:
        """Transfer history for one player — cached 24h."""
        return self._request_all_pages(
            "/transfers", {"player": player_id},
            cache_key=f"transfers_player_{player_id}", refresh=refresh,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def lookup_team_id(self, team_name: str, league_code: str,
                     season_year: int) -> Optional[int]:
        teams = self.fetch_league_teams(league_code, str(season_year))
        want = team_name.lower()
        for t in teams:
            if t["name"].lower() == want or _snake_league(t["name"]).lower() == want:
                return t["id"]
        # fuzzy suffix match — "Real Madrid" vs "Real Madrid CF"
        for t in teams:
            if t["name"].lower().startswith(want):
                return t["id"]
        return None

    @staticmethod
    def _season_year(season) -> int:
        """Return the starting calendar year the API expects (YYYY).

        Accepts any of:
            '2425'   (YYNN football-data.co.uk style  → 2024)
            '2024'   (full calendar year                 → 2024)
            '202425' (full YYNN with century             → 2024)
        """
        s = str(season).strip()
        # Full 4-digit calendar year ("2024", "2023") — never the YYNN form,
        # since YYNN starts with 2 digits < 50 ("24…") not "20…".
        if len(s) == 4 and s.startswith("20"):
            return int(s)
        # YYNN football-data form — first 2 digits are the start YY.
        if len(s) == 4:
            return 2000 + int(s[:2])
        if len(s) == 6:  # "202425"
            return int(s[:4])
        # Fallback: cast.
        return int(s or 0)

    def _normalise(self, raw_players: List[dict], league_code: str,
                  season: str, team_name_hint: Optional[str] = None) -> List[PlayerRecord]:
        league_id = _LEAGUE_IDS.get(league_code)
        league_name = _LEAGUE_NAME.get(league_code, league_code)
        season_year = self._season_year(season)
        records: List[PlayerRecord] = []
        for row in raw_players:
            pl = row.get("player") or {}
            name = pl.get("name")
            if not name:
                continue
            statistics = row.get("statistics") or []
            # keep only stints in the target league-season for this team
            stint = None
            for st in statistics:
                lg = st.get("league") or {}
                tm = st.get("team") or {}
                if (lg.get("id") == league_id
                        and int(lg.get("season") or 0) == season_year
                        and (team_name_hint is None
                             or (tm.get("name") or "").lower() == team_name_hint.lower())):
                    stint = st
                    break
            if stint is None and statistics:
                # fall back to first stint in the right league-season
                for st in statistics:
                    lg = st.get("league") or {}
                    if lg.get("id") == league_id and int(lg.get("season") or 0) == season_year:
                        stint = st
                        break
            if stint is None and statistics:
                stint = statistics[0]
            games = (stint or {}).get("games") or {}
            goals = (stint or {}).get("goals") or {}
            shots = (stint or {}).get("shots") or {}
            passes = (stint or {}).get("passes") or {}
            tackles = (stint or {}).get("tackles") or {}
            cards = (stint or {}).get("cards") or {}
            duels = (stint or {}).get("duels") or {}
            dribbles = (stint or {}).get("dribbles") or {}
            fouls = (stint or {}).get("fouls") or {}
            penalty = (stint or {}).get("penalty") or {}
            position = _POS_MAP.get(games.get("position"))
            team = ((stint or {}).get("team") or {}).get("name") or team_name_hint
            rating_val = None
            r_str = games.get("rating")
            if r_str:
                try:
                    rating_val = int(round(float(r_str)))
                except (TypeError, ValueError):
                    rating_val = None
            records.append(PlayerRecord(
                name=name,
                team=team or "",
                league=_snake_league(league_name),
                season=season,
                source=self.provider_name,
                position=position,
                age=pl.get("age"),
                nationality=pl.get("nationality"),
                appearances=games.get("appearences"),
                minutes=games.get("minutes"),
                goals=goals.get("total"),
                assists=goals.get("assists"),
                shots=shots.get("total"),
                shots_on_target=shots.get("on"),
                tackles=tackles.get("total"),
                interceptions=tackles.get("interceptions"),
                blocks=tackles.get("blocks"),
                saves=goals.get("saves"),
                clean_sheets=None,
                goals_conceded=goals.get("conceded"),
                key_passes=passes.get("key"),
                yellow_cards=cards.get("yellow"),
                red_cards=cards.get("red"),
                rating=rating_val,
                dribbles_success=dribbles.get("success"),
                duels_won=duels.get("won"),
                fouls_committed=fouls.get("committed"),
                penalties_won=penalty.get("won"),
                penalties_scored=penalty.get("scored"),
                shirt_number=games.get("number"),
                extra={"api_id": pl.get("id"),
                       "height": pl.get("height"),
                       "weight": pl.get("weight"),
                       "photo": pl.get("photo"),
                       "team_api_id": ((stint or {}).get("team") or {}).get("id")},
            ))
        return records
