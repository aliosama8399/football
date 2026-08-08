"""
Validation for the team-share ratings + best-11 pipeline.

Checks per sample team:
  1. Share closure: Σ player xG share ≈ 1 (±15%), same for goals/shots/SoT
     — validates that player stats and the match-dataset totals measure
     the same thing.
  2. Best-11 integrity: 11 players, GK present, slots match formation,
     minute floor enforced.
  3. Top-3 per position printed for eyeballing (star FWs should rank top).

Usage (football env, from project root):
    python data/validate_ratings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.best11 import solve_best11
from data.player_ratings import rate_squad
from data.team_totals import load_team_totals

SAMPLE_TEAMS = [
    ("Arsenal", "E0"),
    ("Barcelona", "SP1"),
    ("Bayern Munich", "D1"),
    ("Inter", "I1"),
    ("Paris Saint Germain", "F1"),
]
SEASON = "2425"
SHARE_FIELDS = {"xg_share": "xG", "goals_share": "goals", "shots_share": "shots",
                "sot_share": "SoT"}
TOL = 0.15


def main():
    failures = 0
    for team, league_code in SAMPLE_TEAMS:
        print("=" * 72)
        print(f"{team} ({league_code}) {SEASON}")
        result = solve_best11(team, league_code, SEASON)
        if result.get("error"):
            print("  ERROR:", result["error"])
            failures += 1
            continue

        squad = _squad_from_result(result, team, league_code)
        totals = load_team_totals(league_code, SEASON)
        ratings = rate_squad(squad, totals)

        for share_field, label in SHARE_FIELDS.items():
            vals = [r.shares.get(share_field) for r in ratings
                    if r.shares.get(share_field) is not None]
            if not vals:
                print(f"  share {label}: NO DATA (skip)")
                continue
            total = sum(vals)
            ok = abs(total - 1.0) <= TOL
            failures += 0 if ok else 1
            print(f"  share {label}: Σ = {total:.2f} over {len(vals)} players "
                  f"{'OK' if ok else 'FAIL'}")

        for bucket in ("GK", "DF", "MF", "FW"):
            group = sorted([r for r in ratings if r.position == bucket],
                           key=lambda r: -r.rating)
            if not group:
                continue
            top = group[:3]
            print(f"  top {bucket}: " + " | ".join(
                f"{r.name} {r.rating:.0f}" for r in top))

        lineup = result["lineup"]
        from data.best11 import _FORMATIONS
        need = _FORMATIONS[result["formation"]]
        counts = {}
        for e in lineup:
            counts[e["slot"]] = counts.get(e["slot"], 0) + 1
        ok = (len(lineup) == 11 and any(e["slot"] == "GK" for e in lineup)
              and all(counts.get(s, 0) == n for s, n in need.items()))
        failures += 0 if ok else 1
        print(f"  best11: {len(lineup)} players, slots {counts} "
              f"{'OK' if ok else 'FAIL'}")
        print("  XI: " + ", ".join(f"{e['slot']} {e['name']}" for e in lineup))
        for note in result.get("notes", []):
            print("  note:", note)

    print("=" * 72)
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


def _squad_from_result(result, team, league_code):
    from data.player_providers.factory import get_player_provider
    from data.player_providers.schema import PlayerRecord
    import json
    path = Path("data/raw/squads_cache") / f"all_{team.replace(' ', '_')}_{league_code}_{SEASON}.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [PlayerRecord(**d) for d in raw]
    return get_player_provider("all").fetch_team_squad(team, league_code, SEASON)


if __name__ == "__main__":
    main()
