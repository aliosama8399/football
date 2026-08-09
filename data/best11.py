"""
Best11 — lineup recommendation for a team & season (CLI facade).

The feature logic lives in data/players/, structured with design
patterns:
    SquadRepository   (Repository — squad cache + provider fetching)
    RatingStrategy    (Strategy   — season / through-date / H2H blend)
    FormationStrategy (Strategy   — auto-fit or fixed shapes)
    SubstitutionStrategy (Strategy — like-for-like bench suggestions)
    Best11Service     (Facade     — single solve() entry point)

This module keeps the legacy CLI surface (main / _print) and re-exports
solve_best11 / _load_squad_cached so existing callers (GraphQL
resolvers, prewarm) keep working unchanged.

CLI:
    python -m data.best11 "Arsenal" "E0" --formation 4-3-3
    python -m data.best11 "Inter" "I1" --formation 3-5-2 --refresh

Lineup slots follow coarse position buckets (GK/DF/MF/FW), so tactical
variants like 4-2-3-1 map to MF=2 / FW=4 (the 3 AMs + ST share the FW
bucket).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.players import (FORMATIONS as _FORMATIONS,  # noqa: E402
                          solve_best11, _load_squad_cached)

__all__ = ["solve_best11", "_load_squad_cached", "_FORMATIONS", "main"]


def _print(result: dict) -> None:
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
