"""
TeamProfileStore — lazy team info/stats with PostgreSQL primary source.

Primary source: TeamGraphRepository over the active KG provider — the exact
repository behind the GraphQL resolvers, so KB numbers are identical to
GraphQL numbers by construction (no HTTP, no drift).

Fallback: when the DB is unavailable (or the team missing) and
kb_csv_fallback is enabled, profiles are assembled from the shared CSV
aggregates (data.team_stats) + rag/knowledge_base/team_tactics.json with the
same key shape as a Postgres row.

Loading is per question, per team — nothing is bulk-loaded. Profiles are
cached in-process with a TTL (kb_team_cache_ttl).
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

from rag.knowledge_base.config import kb_settings

logger = logging.getLogger(__name__)


class TeamProfileStore:
    def __init__(self, provider=None, csv_store=None,
                 cache_ttl: Optional[int] = None,
                 csv_fallback: Optional[bool] = None):
        cfg = kb_settings()
        self._provider = provider
        self._csv = csv_store
        self._repo = None
        self._source = "none"
        self._cache_ttl = cfg["kb_team_cache_ttl"] if cache_ttl is None else cache_ttl
        self._csv_fallback = cfg["kb_csv_fallback"] if csv_fallback is None else csv_fallback
        self._cache = {}
        self._lock = threading.Lock()
        self._tactics = None

    # ─────────────────────────────────────────────────────────────────────────
    # Repository bootstrap (lazy, tolerant)
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_repo(self):
        """Build TeamGraphRepository once; None when the KG provider is missing."""
        if self._repo is not None:
            return self._repo
        try:
            if self._provider is None:
                from api.graph_db import get_graph_db
                from api.repositories.graph_repo import TeamGraphRepository
                self._repo = TeamGraphRepository(get_graph_db())
            else:
                from api.repositories.graph_repo import TeamGraphRepository
                self._repo = TeamGraphRepository(self._provider)
            self._source = "postgres"
            logger.info("TeamProfileStore using PostgreSQL via TeamGraphRepository")
        except Exception as e:
            self._repo = None
            self._source = "csv-fallback"
            logger.warning("TeamProfileStore: KG provider unavailable (%s) — using CSV fallback", e)
        return self._repo

    @property
    def source(self) -> str:
        """'postgres' | 'csv-fallback' | 'none' — where profiles currently come from."""
        return self._source

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_profile(self, team_name: str) -> Optional[dict]:
        name = self._canonical_name(team_name)
        if name is None:
            return None

        cached = self._cache_get(name)
        if cached:
            self._source = cached[1]
            return cached[0]

        repo = self._ensure_repo()
        if repo is not None:
            try:
                profile = repo.get_team_profile(name)
                if profile:
                    self._cache_put(name, profile, "postgres")
                    return profile
            except Exception as e:
                logger.warning("Team profile query failed for '%s' (%s) — CSV fallback", name, e)

        if self._csv_fallback and self._csv is not None:
            profile = self._csv_profile(name)
            if profile:
                self._cache_put(name, profile, "csv-fallback")
                return profile

        return None

    def get_head_to_head(self, team_a: str, team_b: str, limit: int = 10) -> List[dict]:
        a = self._canonical_name(team_a)
        b = self._canonical_name(team_b)
        if a is None or b is None or a == b:
            return []
        repo = self._ensure_repo()
        if repo is not None:
            try:
                # The underlying provider only matches home_team=a AND away_team=b,
                # so query both orientations and merge (dedupe by date).
                rows = repo.get_head_to_head(a, b, limit=limit)
                rows += repo.get_head_to_head(b, a, limit=limit)
                seen, merged = set(), []
                for r in rows:
                    key = str(r.get("date", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(self._json_safe_row(r))
                merged.sort(key=lambda r: str(r.get("date", "")), reverse=True)
                if merged:
                    return merged[:limit]
            except Exception as e:
                logger.warning("H2H query failed (%s) — CSV fallback", e)
        if self._csv is not None:
            return self._csv.head_to_head(a, b, limit=limit)
        return []

    def get_recent_form(self, team_name: str, n: int = 5) -> List[dict]:
        name = self._canonical_name(team_name)
        if name is None:
            return []
        repo = self._ensure_repo()
        if repo is not None:
            try:
                rows = repo.get_recent_form(name, n=n)
                if rows:
                    return [self._json_safe_row(r) for r in rows]
            except Exception as e:
                logger.warning("Form query failed (%s) — CSV fallback", e)
        if self._csv is not None:
            return self._csv.recent_form(name, n=n)
        return []

    def get_league_teams(self, league: str, season: Optional[str] = None) -> List[str]:
        repo = self._ensure_repo()
        if repo is not None:
            try:
                teams = repo.get_league_teams(league, season=season)
                if teams:
                    return teams
            except Exception as e:
                logger.warning("League teams query failed (%s) — CSV fallback", e)
        if self._csv is not None:
            rows = self._csv.league_table(league=league, season=season)
            return [r["team"] for r in rows]
        return []

    def status(self) -> dict:
        return dict(
            source=self._source,
            cached_teams=len(self._cache),
            cache_ttl=self._cache_ttl,
            csv_fallback=self._csv_fallback,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────────────

    def _canonical_name(self, team_name) -> Optional[str]:
        """Registry/fuzzy-canonicalized team name (CSV-side), or raw input."""
        if not team_name:
            return None
        if self._csv is not None:
            resolved = self._csv.resolve_team(team_name)
            if resolved is not None:
                return resolved
        return str(team_name).strip()

    @staticmethod
    def _json_safe_row(row: dict) -> dict:
        """Convert provider rows (dates, numpy scalars) to JSON-safe types."""
        from data.team_stats import pyval
        out = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            else:
                v = pyval(v)
            out[k] = v
        return out

    def _cache_get(self, name):
        with self._lock:
            entry = self._cache.get(name)
            if entry and entry[0] > time.time():
                return entry[1:]
            return None

    def _cache_put(self, name, profile, source):
        with self._lock:
            self._cache[name] = (time.time() + self._cache_ttl, profile, source)

    def _tactics_data(self) -> dict:
        if self._tactics is None:
            path = Path(kb_settings()["kb_tactics_path"])
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self._tactics = json.load(f)
                except Exception as e:
                    logger.warning("Could not load %s: %s", path, e)
                    self._tactics = {}
            else:
                self._tactics = {}
        return self._tactics

    def _csv_profile(self, team: str) -> Optional[dict]:
        agg = self._csv.aggregates_for(team) if self._csv is not None else None
        if not agg:
            return None
        t = self._tactics_data().get(team, {})
        profile = dict(agg)
        profile["name"] = team
        profile["attack_tactic"] = t.get("attack_tactic")
        profile["defense_tactic"] = t.get("defense_tactic")
        profile["attack_headline"] = t.get("attack_headline")
        profile["defense_headline"] = t.get("defense_headline")
        profile["strengths"] = t.get("strengths", [])
        profile["weaknesses"] = t.get("weaknesses", [])
        return profile
