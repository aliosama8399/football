"""
Best11 — lineup recommendation for a team & season.

Pipeline:
    squad fetch (fused providers, cached on disk) →
    team-share ratings (data/player_ratings.py) →
    position-aware formation fill.

Squad caching: FBRef requires a real browser (Cloudflare), so fetched
squads are cached as JSON under data/raw/squads_cache/ and reused.
Delete the cache or pass --refresh to re-fetch.

CLI:
    python -m data.best11 "Arsenal" "E0" --formation 4-3-3
    python -m data.best11 "Inter" "I1" --formation 3-5-2 --refresh

Lineup slots follow coarse position buckets (GK/DF/MF/FW), so tactical
variants like 4-2-3-1 map to MF=2 / FW=4 (the 3 AMs + ST share the FW
bucket).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.player_providers.factory import get_player_provider
from data.player_providers.schema import PlayerRecord
from data.player_form import h2h_player_stats, rate_squad_as_of
from data.player_ratings import (MIN_MINUTES, PlayerRating, rate_squad)
from data.team_totals import load_team_totals

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("best11")

_CACHE_DIR = Path("data/raw/squads_cache")

_FORMATIONS: Dict[str, Dict[str, int]] = {
    "4-3-3": {"GK": 1, "DF": 4, "MF": 3, "FW": 3},
    "4-4-2": {"GK": 1, "DF": 4, "MF": 4, "FW": 2},
    "4-2-3-1": {"GK": 1, "DF": 4, "MF": 2, "FW": 4},
    "3-5-2": {"GK": 1, "DF": 3, "MF": 5, "FW": 2},
}


def _cache_path(team: str, league_code: str, season: str, provider: str) -> Path:
    safe = team.replace(" ", "_")
    return _CACHE_DIR / f"{provider}_{safe}_{league_code}_{season}.json"


def _load_squad_cached(team: str, league_code: str, season: str,
                       provider: str = "all", refresh: bool = False) -> List[PlayerRecord]:
    path = _cache_path(team, league_code, season, provider)
    if not refresh and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = [PlayerRecord(**d) for d in raw]
        if any("pos_list" not in (r.extra or {}) for r in records):
            logger.info("squad cache %s lacks pos_list → refreshing", path.name)
            refresh = True
        else:
            return records
    provider_obj = get_player_provider(provider)
    squad = provider_obj.fetch_team_squad(team, league_code, season)
    if squad:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([r.to_dict() for r in squad],
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("cached %s → %s", team, path)
    return squad


def solve_best11(team: str, league_code: str, season: str = "2425",
                 formation: str = "auto", provider: str = "all",
                 refresh: bool = False, as_of: Optional[str] = None,
                 opponent: Optional[str] = None) -> Dict:
    """Return the best 11 for a team, season and formation.

    formation "auto" (default) picks the formation that fits the squad
    best: highest summed XI rating minus a penalty per out-of-position
    (flex) pick, among 4-3-3 / 4-2-3-1 / 4-4-2 / 3-5-2.

    opponent switches the XI to match-specific: players with meaningful
    head-to-head minutes get a 70/30 season↔H2H blended rating, so the
    lineup can differ per opponent. Each lineup entry carries season and
    H2H stat blocks for verification.

    as_of (ISO date) switches to through-the-season ratings: per-match
    player stats and team totals are cumulated up to that date, so a
    prediction for a specific fixture never uses future data.

    Output dict: formation, lineup (slot, name, position, rating, minutes,
    shares, season, h2h), captain, subs, bench, notes.
    """
    if formation != "auto" and formation not in _FORMATIONS:
        raise ValueError(f"Unknown formation '{formation}'. "
                         f"Supported: {sorted(_FORMATIONS)} or 'auto'")
    squad = _load_squad_cached(team, league_code, season, provider, refresh)
    if not squad:
        return {"team": team, "formation": formation, "error": "no squad data"}

    totals = load_team_totals(league_code, season)
    if as_of:
        try:
            ratings, matched, used_through = rate_squad_as_of(
                squad, as_of, league_code, season)
        except Exception as e:
            logger.warning("as-of rating failed (%s); falling back: %s", as_of, e)
            ratings, matched, used_through = rate_squad(squad, totals), 0, False
    else:
        ratings, matched, used_through = rate_squad(squad, totals), 0, False

    season_stats = {
        rec.name: {f: round(getattr(rec, f) or 0.0, 2) for f in
                   ("goals", "assists", "xg", "xa", "shots")}
        for rec in squad}

    h2h_stats, h2h_matches = {}, 0
    if opponent and opponent != team:
        try:
            h2h_stats, h2h_matches = h2h_player_stats(
                league_code, season, team, opponent)
        except Exception as e:
            logger.warning("h2h stats failed for %s vs %s: %s", team, opponent, e)
        if h2h_stats:
            _blend_h2h(ratings, h2h_stats)

    eligible = [r for r in ratings if r.minutes >= MIN_MINUTES]
    bench = [r for r in ratings if r.minutes < MIN_MINUTES]

    auto = formation == "auto"
    if auto:
        formation = _auto_formation(eligible)
    slots = dict(_FORMATIONS[formation])
    lineup, notes = _fill_lineup(eligible, slots)
    if auto:
        notes.append(f"formation auto-fit: {formation}")
    if opponent and h2h_matches:
        notes.append(f"ratings blended 70/30 with H2H vs {opponent} "
                     f"({h2h_matches} meetings this season)")

    lineup.sort(key=lambda e: e["slot"])
    for e in lineup:
        e["season"] = season_stats.get(e["name"], {})
        e["h2h"] = _h2h_entry(h2h_stats.get(e["name"], {}), h2h_matches)
    subs = _suggest_subs(lineup, eligible)
    captain = max(lineup, key=lambda e: e["rating"])["name"] if lineup else None
    if used_through:
        notes.append(f"ratings through {as_of}: per-match stats for {matched} "
                     f"players, team totals through matchday {as_of}")
    elif as_of:
        notes.append(f"as-of {as_of}: per-match data not collected for this "
                     f"season — full-season ratings used")
    return {
        "team": team,
        "league_code": league_code,
        "season": season,
        "formation": formation,
        "lineup": lineup,
        "captain": captain,
        "subs": subs,
        "bench": [{"name": b.name, "position": b.position, "rating": round(b.rating, 1),
                   "minutes": b.minutes} for b in sorted(bench, key=lambda r: -r.rating)],
        "notes": notes,
    }


def _h2h_entry(stats: Dict[str, float], matches: int) -> Dict:
    """H2H stat block for a lineup entry (empty when no data/appearance)."""
    if not stats or not stats.get("minutes"):
        return {}
    return {
        "matches": matches,
        "minutes": int(stats.get("minutes", 0)),
        "goals": int(stats.get("goals", 0)),
        "assists": int(stats.get("assists", 0)),
        "xg": round(stats.get("xg", 0.0), 2),
        "xa": round(stats.get("xa", 0.0), 2),
        "shots": int(stats.get("shots", 0)),
    }


def _blend_h2h(ratings: List[PlayerRating], h2h_stats: Dict[str, Dict[str, float]]):
    """Blend season ratings with head-to-head performance (boost-only).

    H2H raw score uses the same share scheme as season ratings
    (xg .40, goals .30, shots .15, xa .15) over the meetings between the
    two teams, mapped to 0-100 on a fixed scale (raw .33 ≈ 100). Final
    rating = max(season, 0.7 × season + 0.3 × H2H rating), so players
    who produced against the opponent get a boost (cap +30 pts) and
    everyone else keeps their season rating — GKs, quiet defenders and
    bit-part players are never punished.
    """
    H2H_WEIGHTS = (("xg", 0.40), ("goals", 0.30), ("shots", 0.15), ("xa", 0.15))
    totals = {}
    for v in h2h_stats.values():
        for f, _ in H2H_WEIGHTS:
            totals[f] = totals.get(f, 0.0) + (v.get(f) or 0.0)

    boosted: Dict[int, float] = {}
    for r in ratings:
        v = h2h_stats.get(r.name)
        if not v or (v.get("minutes") or 0) < 90:
            continue
        score, n = 0.0, 0.0
        for f, w in H2H_WEIGHTS:
            t = totals.get(f) or 0.0
            p = v.get(f) or 0.0
            if t > 0:
                score += min(p / t, 2.0) * w
                n += w
        raw = (score / n) if n else 0.0
        if raw < 0.03:  # no meaningful H2H output → keep season rating
            continue
        h2h_rating = min(100.0, raw * 300.0)
        boosted[id(r)] = max(r.rating, 0.7 * r.rating + 0.3 * h2h_rating)

    for r in ratings:
        if id(r) in boosted:
            r.rating = boosted[id(r)]


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


def _auto_formation(eligible: List[PlayerRating]) -> str:
    """Pick the formation that yields the strongest natural XI.

    Score = summed XI rating − 12 per flex (out-of-position) pick, so a
    team with 3 elite CBs gets 3-5-2 while a team stacked in attack
    stays in a front-heavy shape. Defaults to 4-3-3 if nothing fields
    a full XI.
    """
    best_name, best_score = "4-3-3", None
    for name, slots in _FORMATIONS.items():
        lineup, _ = _fill_lineup(eligible, slots)
        if len(lineup) < 11:
            continue
        total = sum(e["rating"] for e in lineup)
        flexes = sum(1 for e in lineup if e["flex"])
        score = total - 12.0 * flexes
        if best_score is None or score > best_score:
            best_name, best_score = name, score
    return best_name


def _suggest_subs(lineup: List[Dict], eligible: List[PlayerRating]) -> List[Dict]:
    """3 like-for-like substitutions for the weakest starters.

    Candidates: non-XI players at/above the minute floor, same position
    slot (or a listed secondary position). GKs never sub for outfield
    slots and vice versa; slots without a candidate are skipped.
    """
    in_xi = {e["name"] for e in lineup}
    weak_starters = sorted(lineup, key=lambda e: e["rating"])[:3]

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


def _print(result: Dict) -> None:
    if result.get("error"):
        print("ERROR:", result["error"])
        return
    print(f"\nBest XI — {result['team']} ({result['league_code']}) "
          f"{result['season']}, {result['formation']}\n")
    print(f"{'SLOT':<5} {'PLAYER':<28} {'POS':<4} {'RATING':<8} {'MIN':<7} KEY SHARES")
    for e in result["lineup"]:
        shares = ", ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in e["top_shares"].items()[:3])
        flex = "  (flex)" if e["flex"] else ""
        print(f"{e['slot']:<5} {e['name']:<28} {str(e['position']):<4} "
              f"{e['rating']:<8} {int(e['minutes']):<7} {shares}{flex}")
    if result["bench"]:
        top = result["bench"][:3]
        print("\nBench (top 3, below minute floor): "
              + ", ".join(f"{b['name']} ({b['rating']})" for b in top))
    for n in result["notes"]:
        print("note:", n)


def main():
    parser = argparse.ArgumentParser(description="Best-11 lineup recommendation")
    parser.add_argument("team")
    parser.add_argument("league_code", help="E0 | SP1 | D1 | I1 | F1")
    parser.add_argument("--season", default="2425")
    parser.add_argument("--formation", default="auto",
                        choices=["auto"] + sorted(_FORMATIONS),
                        help="auto picks the best-fit shape")
    parser.add_argument("--provider", default="all")
    parser.add_argument("--refresh", action="store_true", help="re-fetch squads")
    parser.add_argument("--json", action="store_true", help="dump JSON to stdout")
    args = parser.parse_args()

    result = solve_best11(args.team, args.league_code, args.season,
                          args.formation, args.provider, args.refresh)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        _print(result)
        out = Path("data/raw/best11") / f"{args.team.replace(' ', '_')}_{args.season}_{args.formation}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\nSaved:", out)


if __name__ == "__main__":
    main()
