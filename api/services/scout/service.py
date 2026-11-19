"""ScoutService — domain orchestration for the scouting feature.

Facade pattern: clients get one scout() entry point; the service
orchestrates the ScoutRepository (collects data from sources) and the
position-specific ScoringStrategy (the processing layer).

Layers:
    api/routes/scout.py              → HTTP concerns
    api/services/scout_service.py    → application use-case (ScoutApiService)
    api/services/scout/service.py    → domain orchestration (this service)
    api/repositories/scout_repo.py   → data collection from sources
    data/                            → raw collection & scraping only
"""

import logging
from typing import Dict, List, Optional, Tuple

from data.player_providers.schema import PlayerRecord

from .strategies.scoring import (
    MIN_MINUTES_SENIOR,
    MIN_MINUTES_YOUTH,
    ScoringStrategy,
    make_scoring_strategy,
    STYLE_METRICS,
)

logger = logging.getLogger("scout.service")


class ScoutService:
    """Orchestrates the ScoutRepository and scoring strategies."""

    # Shortlist multiplier — we score-from-cache, take this × top of the best
    # candidates, and only then spend API-Football calls on the unique teams
    # in that top slice. Keeps a big pool cheap while still letting late
    # movers re-rank after the rating/transfer enrichment.
    SHORTLIST_FACTOR = 2

    def __init__(self, repository, scoring_strategy_factory=None):
        self.repository = repository
        self._strategy_factory = scoring_strategy_factory or make_scoring_strategy

    def scout(self,
              league_codes: Tuple[str, ...],
              season: str,
              position: str,
              top: int = 5,
              youth: bool = False,
              team_needing: Optional[str] = None,
              refresh: bool = False) -> Dict:
        """Return a scout report: top-N candidates ranked by position and
        — when `team_needing` (the coach's club) is supplied — by fit to
        that club's playing style.

        Pipeline (cheap before expensive):
            1. identity pool          — fused squad cache (no API call)
            2. style template          — coach's own per-position per-90
                                         average (cache only) — optional
            3. score-from-cache        — weighted cache-only stats + fit
                                         bonus; pick a shortlist of
                                         top × SHORTLIST_FACTOR candidates
            4. enrich top slice        — API-Football stats (rating +
                                         secondary metrics) for the
                                         unique teams of the shortlist only
            5. re-score                — final rank, take top-N
            6. attach transfers        — /transfers?team= per shortlist team
        """
        top = max(1, min(int(top), 20))
        shortlist_size = max(top, top * self.SHORTLIST_FACTOR)

        # 1. identity pool — fused squad cache (no network)
        raw_candidates = self.repository.identity_pool(
            league_codes, season, position, youth=youth,
            exclude_team=team_needing)
        if not raw_candidates:
            return {
                "season": season, "position": position, "youth": youth,
                "leagues": list(league_codes), "pool_size": 0, "top": top,
                "candidates": [], "notes": [
                    "no candidates matched the position/age filter"],
            }

        candidates, applied_min, dropped_low_mins = self._apply_minutes_gate(
            raw_candidates, youth=youth, top=top)

        # 2. style template — your own club's per-position per-90 average.
        style: Optional[Dict[str, Optional[float]]] = None
        style_team: Optional[str] = None
        if team_needing:
            style_team, style_lc = self._resolve_coach_team(
                team_needing, league_codes, season)
            if style_team:
                style = self.repository.style_reference(
                    style_team, style_lc, position, season)
                if not style:
                    logger.info("style_reference empty for %s %s — "
                                "fit disabled", style_team, style_lc)
        strategy = self._strategy_factory(position)

        # 3. score-from-cache (uses xg/xa/goals/assists/shots/key_passes/
        # tackles/interceptions/blocks — all in the fused squad cache).
        scored = strategy.score(candidates, style=style)
        ranked = sorted(zip(candidates, scored),
                        key=lambda kv: kv[1].record_score, reverse=True)
        shortlist_recs = [c for c, _ in ranked[:shortlist_size]]

        # 4. enrich the shortlist — uses 0ms disk cache by default; live HTTP only if refresh=True.
        self.repository.enrich_with_stats(shortlist_recs, season, refresh=refresh)

        # 5. re-score across the full gated pool so P5..P95 normalization stays globally consistent.
        full_scored = strategy.score(candidates, style=style)
        full_scores_by_id = {id(r): s for r, s in zip(candidates, full_scored)}
        final_ranked = sorted(
            shortlist_recs,
            key=lambda r: full_scores_by_id[id(r)].record_score,
            reverse=True)[:top]

        # 6. transfers — cached 24h on disk; live HTTP only if refresh=True.
        transfers_map = self.repository.attach_transfers(final_ranked, refresh=refresh)

        out_candidates = []
        for rank, rec in enumerate(final_ranked, start=1):
            sc = full_scores_by_id[id(rec)]
            xfer = transfers_map.get(_normalize(rec), {})
            out_candidates.append({
                "rank": rank,
                "name": rec.name,
                "team": rec.team,
                "league": rec.league,
                "position": rec.position,
                "age": rec.age,
                "nationality": rec.nationality,
                "shirt_number": rec.shirt_number,
                "stats": {
                    "appearances": rec.appearances,
                    "minutes": rec.minutes,
                    "goals": rec.goals,
                    "assists": rec.assists,
                    "shots": rec.shots,
                    "shots_on_target": rec.shots_on_target,
                    "saves": rec.saves,
                    "goals_conceded": rec.goals_conceded,
                    "key_passes": rec.key_passes,
                    "tackles": rec.tackles,
                    "interceptions": rec.interceptions,
                    "blocks": rec.blocks,
                    "duels_won": rec.duels_won,
                    "dribbles_success": rec.dribbles_success,
                    "yellow_cards": rec.yellow_cards,
                    "red_cards": rec.red_cards,
                    "rating": rec.rating,
                },
                "transfer": xfer or None,
                "score": sc.record_score,
                "score_breakdown": sc.breakdown,
            })
        notes = [
            f"pool: {len(candidates)} {position} candidates across "
            f"{len(league_codes)} league(s)",
            f"minutes gate: >={int(applied_min)} min (filtered {dropped_low_mins} low-minute cameo(s))",
            f"youth filter (≤19): {youth}",
        ]
        if style and any(v is not None for v in style.values()):
            notes.append(f"style fit on: weighted 70/30 against "
                         f"{style_team}'s {position} per-90 profile")
        else:
            notes.append("style fit: off (no my_team supplied or empty profile)")
        if self.repository.stats_provider.api_key:
            notes.append(f"enrichment: API-Football for "
                         f"{len({r.team for r in shortlist_recs})} shortlist team(s)")
        else:
            notes.append("API_FOOTBALL_KEY missing — rating/transfer "
                         "skipped (set in .env to enable)")
        return {
            "season": season,
            "position": position,
            "youth": youth,
            "leagues": list(league_codes),
            "pool_size": len(candidates),
            "scanned": len(shortlist_recs),
            "top": top,
            "style_team": style_team,
            "candidates": out_candidates,
            "notes": notes,
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_minutes_gate(
        candidates: List[PlayerRecord],
        youth: bool,
        top: int,
    ) -> Tuple[List[PlayerRecord], float, int]:
        """Exclude tiny-sample cameo appearances while gracefully relaxing only on tiny pools."""
        thresholds = (
            (MIN_MINUTES_YOUTH, 180.0, 90.0)
            if youth
            else (MIN_MINUTES_SENIOR, 270.0, 90.0)
        )
        min_keep = min(len(candidates), max(top * 2, 10))
        for thr in thresholds:
            filtered = [r for r in candidates if float(r.minutes or 0.0) >= thr]
            if len(filtered) >= min_keep:
                return filtered, thr, len(candidates) - len(filtered)
        for thr in thresholds:
            filtered = [r for r in candidates if float(r.minutes or 0.0) >= thr]
            if len(filtered) >= min(top, len(candidates)):
                return filtered, thr, len(candidates) - len(filtered)
        non_zero = [r for r in candidates if float(r.minutes or 0.0) >= 90.0]
        if non_zero:
            return non_zero, 90.0, len(candidates) - len(non_zero)
        return candidates, 0.0, 0

    def _resolve_coach_team(self, team: str,
                            league_codes: Tuple[str, ...],
                            season: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a coach-supplied team name to its canonical registry name
        and league code by scanning the requested leagues' team lists."""
        from api.repositories.scout_repo import _resolve_team_name
        try:
            for lc in league_codes:
                teams = self.repository._teams_in_league(lc, season)
                canon = _resolve_team_name(team, lc, season, teams)
                if canon:
                    return canon, lc
        except Exception as e:
            logger.info("coach team lookup failed for '%s': %s", team, e)
            return None, None
        if len(league_codes) == 1:
            logger.info("coach team '%s' not found in %s %s",
                        team, league_codes[0], season)
        return None, None


def _normalize(rec: PlayerRecord) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", str(rec.name))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()
