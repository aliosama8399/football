"""ScoutRepository — data collection for the player-scouting feature.

Composes two repositories/providers behind one repository:

    Best11Repository           — fused squad providers (FBRef + understat),
                                 cached on disk under data/raw/squads_cache/.
                                 Used for the *wide* candidate pool: identity
                                 (name, age, position, nationality).

    ApiFootballPlayerProvider  — API-Football v3 (10 req/min). Used to enrich
                                 only the shortlisted unique teams with full
                                 per-player season stats (goals/assists/shots/
                                 rating/saves/...), plus transfer info for the
                                 final top-N. Cached 24h under
                                 data/raw/scout_cache/.

Quota-smart flow: identity pre-filter cuts the pool to tens of
youth/position matches across 5 leagues, so we only spend API-Football
calls on those teams — usually 1-3 per scout query.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from data._config import get_leagues as _gl
from data.player_providers.api_football import ApiFootballPlayerProvider
from data.player_providers.schema import PlayerRecord

from api.repositories.best11_repo import Best11Repository

logger = logging.getLogger("best11.scout")

# Dynamically derived from data/config.yaml league codes.
SCOUT_LEAGUES: Tuple[str, ...] = tuple(_gl().keys()) or ("E0", "SP1", "D1", "I1", "F1")


def _normalize(name: str) -> str:
    """Diacritic-insensitive match key for squad↔api-fusion."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def _season_year(season: str) -> int:
    s = str(season).strip()
    if len(s) == 4 and s.startswith("20"):
        return int(s)
    if len(s) == 4:
        return 2000 + int(s[:2])
    return int(s or 0)


class ScoutRepository:
    """Collects all data the scouting service needs."""

    def __init__(self,
                 squads: Optional[Best11Repository] = None,
                 stats_provider: Optional[ApiFootballPlayerProvider] = None):
        # API-Football keys come from the app settings (.env) so the
        # provider works identically in the server and in scripts.
        from api.config import settings
        self.squads = squads or Best11Repository()
        self.stats_provider = stats_provider or ApiFootballPlayerProvider(
            api_key=settings.api_football_key or None,
            host=settings.api_football_host or None)
        # Per-(league, season) cache of team lists — read processed_matches.csv
        # only once and slice it in memory across the 5 leagues of a query.
        self._team_cache: Dict[Tuple[str, str], List[str]] = {}
        self._tom_df = None
        self._tom_path: Optional[str] = None

    # ── Pool: identity + age + position from the cached squad providers ────

    def identity_pool(self, league_codes: Tuple[str, ...], season: str,
                      position: str, youth: bool,
                      exclude_team: Optional[str] = None) -> List[PlayerRecord]:
        """Wide pool filtered by position + age ceiling (youth ≤ 19).

        Reuses Best11Repository.load_squad — the fused-provider squad
        cache (data/raw/squads_cache/), so the identity pass is free,
        works offline and yields full season stats (xg/xa/goals/assists/
        tackles/...) already fused from FBRef + understat.
        
        Loads squad data concurrently across teams for fast response times.
        """
        from concurrent.futures import ThreadPoolExecutor

        pool: List[PlayerRecord] = []
        targets = []
        for lc in league_codes:
            teams = self._teams_in_league(lc, season)
            for team in teams:
                targets.append((team, lc, season))

        def _fetch_squad(target):
            t, lc, s = target
            try:
                return self.squads.load_squad(t, lc, s)
            except Exception as e:
                logger.warning("identity fetch failed for %s %s: %s", t, lc, e)
                return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            squad_results = list(executor.map(_fetch_squad, targets))

        for squad in squad_results:
            for rec in squad:
                if rec.position != position:
                    continue
                if youth and (rec.age is None or rec.age > 19):
                    continue
                if exclude_team and _normalize(rec.team) == _normalize(exclude_team):
                    continue
                pool.append(rec)
        return pool

    # ── Coach-style template: aggregate per-90 of the coach's position-mates
    def style_reference(self, team: str, league_code: str, position: str,
                        season: str) -> Dict[str, Optional[float]]:
        """Aggregate per-90 style metrics for the coach's own players at the
        target position — the reference profile the candidates are compared
        against to compute the fit bonus.

        Returns a dict of metric → per-90 float (or None when no data),
        using only fields already in the fused squad cache (no extra API
        calls). When the requested team produces no squad rows, returns an
        empty dict — the service then falls back to a neutral (no-fit) score.
        """
        team_name = _resolve_team_name(team, league_code, season,
                                       self._teams_in_league(league_code, season))
        if not team_name:
            return {}
        try:
            squad = self.squads.load_squad(team_name, league_code, season)
        except Exception as e:
            logger.info("style_reference: squad load failed for %s %s: %s",
                        team_name, league_code, e)
            return {}
        own = [r for r in squad if r.position == position and r.minutes]
        if not own:
            return {}
        ref: Dict[str, Optional[float]] = {}
        for m in STYLE_METRICS:
            vals = [_per_90(_g(r, m), r.minutes) for r in own]
            vals = [v for v in vals if v is not None]
            ref[m] = sum(vals) / len(vals) if vals else None
        return ref

    # ── Stats enrichment: API-Football only for teams in the shortlist ───────

    def _lookup_cached_team_id(self, team_name: str, league_code: str, year: int) -> Optional[int]:
        """Resolve API-Football team ID purely from disk cache without network calls."""
        rows = self.stats_provider.read_cached_only(f"teams_{league_code}_{year}")
        if not rows:
            return None
        want = _normalize(team_name)
        for row in rows:
            tm = row.get("team") or {}
            tname = tm.get("name") or ""
            if _normalize(tname) == want or _normalize(tname.replace(" ", "_")) == want:
                return tm.get("id")
        for row in rows:
            tm = row.get("team") or {}
            tname = tm.get("name") or ""
            if want and _normalize(tname).startswith(want):
                return tm.get("id")
        return None

    def enrich_with_stats(self, candidates: List[PlayerRecord],
                          season: str,
                          refresh: bool = False,
                          max_live_seconds: float = 5.0) -> List[PlayerRecord]:
        """Enrich shortlist candidates with API-Football stats (disk cache first, bounded live HTTP)."""
        year = _season_year(season)
        deadline = time.monotonic() + max_live_seconds
        by_team: Dict[str, List[PlayerRecord]] = defaultdict(list)
        for rec in candidates:
            by_team[_normalize(rec.team)].append(rec)

        for recs in by_team.values():
            rec = recs[0]
            team_name = rec.team
            league_code = self._to_league_code(rec.league) or rec.league

            team_id = self._lookup_cached_team_id(team_name, league_code, year)
            raw = (
                self.stats_provider.read_cached_only(f"players_team_{team_id}_{year}")
                if (team_id and not refresh)
                else None
            )

            if raw is None and refresh and self.stats_provider.api_key:
                if time.monotonic() >= deadline:
                    logger.info("enrichment time budget reached — using fused squad cache for remaining teams")
                    break
                team_id = team_id or self.stats_provider.lookup_team_id(team_name, league_code, year)
                if team_id:
                    raw = self.stats_provider.fetch_team_players(
                        team_id, year, refresh=True, max_pages=2)

            if not raw:
                continue
            api_records = self.stats_provider._normalise(
                raw, league_code, season, team_name)
            api_by_name = {_normalize(r.name): r for r in api_records}
            for candidate in recs:
                api_rec = api_by_name.get(_normalize(candidate.name))
                if not api_rec:
                    cn = _normalize(candidate.name)
                    for k, v in api_by_name.items():
                        if cn and (cn in k or k in cn):
                            api_rec = v
                            break
                if api_rec:
                    _fuse(candidate, api_rec)
        return candidates

    # ── Transfers for the final top-N (cheap: ≤ top transfers) ──────────────

    def attach_transfers(self, candidates: List[PlayerRecord],
                         refresh: bool = False,
                         max_live_seconds: float = 3.0) -> Dict[str, Dict]:
        """Attach cached or bounded-live transfer history for shortlisted top-N candidates."""
        out: Dict[str, Dict] = {}
        deadline = time.monotonic() + max_live_seconds
        seen_teams: set = set()
        for rec in candidates:
            tid = (rec.extra or {}).get("team_api_id") if rec.extra else None
            if not tid or tid in seen_teams:
                continue
            seen_teams.add(tid)
            transfers = (
                self.stats_provider.read_cached_only(f"transfers_team_{tid}")
                if not refresh
                else None
            )
            if transfers is None and refresh and self.stats_provider.api_key:
                if time.monotonic() >= deadline:
                    logger.info("transfers time budget reached — skipping remaining transfer lookups")
                    break
                transfers = self.stats_provider.fetch_team_transfers(tid, refresh=True)
            for entry in transfers or []:
                pl = entry.get("player") or {}
                name = pl.get("name")
                if not name:
                    continue
                transfers_list = entry.get("transfers") or []
                latest = transfers_list[0] if transfers_list else {}
                teams = latest.get("teams") or {}
                out[_normalize(name)] = {
                    "date": latest.get("date"),
                    "type": latest.get("type"),
                    "from_team": (teams.get("out") or {}).get("name"),
                    "to_team": (teams.get("in") or {}).get("name"),
                }
        return out

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _teams_in_league(self, league_code: str, season: str) -> List[str]:
        """Read teams competing in this league-season from the processed
        match dataset (no API call). Cached per (league, season); the CSV
        is parsed once on first call and slice-filtered in memory."""
        key = (league_code, str(season))
        if key in self._team_cache:
            return self._team_cache[key]
        name = _league_name(league_code)
        df = self._matches_df()
        teams: set = set()
        if df is not None:
            sub = df[(df["League"] == name) &
                     (df["Season"].astype(str) == str(season))]
            from data.team_registry import normalize_team_name
            for col in ("HomeTeam", "AwayTeam"):
                teams |= {normalize_team_name(str(t), league_code)
                          for t in sub[col].dropna()}
        ordered = sorted(teams)
        self._team_cache[key] = ordered
        return ordered

    def _matches_df(self):
        """Load data/processed/processed_matches.csv once, cache the frame
        across the 5-league loop of a single scout query."""
        from pathlib import Path
        csv = Path("data/processed/processed_matches.csv")
        if not csv.exists():
            return None
        path = str(csv)
        if self._tom_df is None or self._tom_path != path:
            import pandas as pd
            self._tom_df = pd.read_csv(csv, usecols=["League", "Season",
                                                      "HomeTeam", "AwayTeam"])
            self._tom_path = path
        return self._tom_df

    @staticmethod
    def _to_league_code(league_snake: str) -> Optional[str]:
        for code, name in _LEAGUE_NAMES.items():
            if name.lower() == str(league_snake).lower():
                return code
        return None


# league-code → project league name lookup
_LEAGUE_NAMES = {code: info["name"] for code, info in _gl().items()}


def _league_name(league_code: str) -> str:
    return _LEAGUE_NAMES.get(league_code, league_code)


def _per_90(value, minutes):
    if value is None or not minutes or minutes <= 0:
        return None
    return float(value) * 90.0 / float(minutes)


def _g(rec, field):
    """Pull an attribute off a PlayerRecord (used for the style template)."""
    v = getattr(rec, field, None)
    return None if v is None else float(v)


# Metrics the style template / distance score uses — all already in the
# fused squad cache (FBRef + understat), so computing fit never triggers
# an API-Football call.
STYLE_METRICS: Tuple[str, ...] = (
    "goals", "assists", "xg", "xa", "shots", "key_passes",
    "tackles", "interceptions", "blocks",
)


def _resolve_team_name(team: str, league_code: str, season: str,
                       candidates: List[str]) -> Optional[str]:
    """Case/diacritic/underscore-insensitive team-name match against the
    team list of the league-season. Returns the canonical registry name
    or None."""
    norm = _normalize(team)
    for c in candidates:
        if _normalize(c) == norm:
            return c
    # loose replace match ("barcelona" matches "FC Barcelona" after _/space strip)
    needle = norm.replace(" ", "")
    for c in candidates:
        if _normalize(c).replace(" ", "") == needle:
            return c
    return None


def _fuse(primary: PlayerRecord, other: PlayerRecord) -> PlayerRecord:
    """Fill missing fields on primary from other (fod identity → api stats)."""
    for fname in ("appearances", "minutes", "goals", "assists", "shots",
                  "shots_on_target", "tackles", "interceptions", "blocks",
                  "saves", "goals_conceded",
                  "key_passes", "yellow_cards", "red_cards",
                  "rating", "dribbles_success", "duels_won",
                  "fouls_committed", "penalties_won", "penalties_scored",
                  "shirt_number"):
        if getattr(primary, fname, None) is None and getattr(other, fname, None) is not None:
            setattr(primary, fname, getattr(other, fname))
    pe = dict(primary.extra or {})
    oe = dict(other.extra or {})
    for k, v in oe.items():
        if pe.get(k) is None:
            pe[k] = v
    primary.extra = pe
    return primary


_scout_repo: Optional[ScoutRepository] = None


def get_scout_repository() -> ScoutRepository:
    global _scout_repo
    if _scout_repo is None:
        _scout_repo = ScoutRepository()
    return _scout_repo
