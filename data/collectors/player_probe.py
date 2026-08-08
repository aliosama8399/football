"""
Player Provider Probe — extract a small sample from every provider and
compare coverage so we can pick the best source (or fuse several).

Sample: one flagship team per league, season 2024/25 ('2425').

Usage:
    python data/collectors/player_probe.py            # all providers
    python data/collectors/player_probe.py fbref      # single provider

Output:
    data/raw/player_probe/{provider}/{team}.json      (PlayerRecord dumps)
    data/raw/player_probe/comparison_report.md        (coverage table)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.player_providers.factory import get_player_provider, list_providers
from data.player_providers.schema import PlayerRecord

# canonical team names + league codes (must match data/config.yaml league keys)
SAMPLE_TEAMS = [
    ("Arsenal", "E0"),
    ("Barcelona", "SP1"),
    ("Bayern Munich", "D1"),
    ("Inter", "I1"),
    ("Paris Saint Germain", "F1"),
]
SEASON = "2425"

# Which fields the report tracks (sorted groups: identity / attack / defense / gk)
TRACKED_FIELDS = [
    "position", "age", "nationality",
    "appearances", "minutes",
    "goals", "assists", "xg", "xa", "shots", "shots_on_target",
    "tackles", "interceptions",
    "saves", "clean_sheets", "goals_conceded",
    "yellow_cards", "red_cards",
]


def main():
    parser = argparse.ArgumentParser(description="Player provider coverage probe")
    parser.add_argument("providers", nargs="*",
                        help="provider names (default: all of %s)" % list_providers())
    args = parser.parse_args()

    providers = args.providers or list_providers()
    out_dir = Path("data/raw/player_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Player Provider Probe — season 2425",
        "",
        "Sample teams: " + ", ".join(f"{t} ({l})" for t, l in SAMPLE_TEAMS),
        "",
        f"Coverage = % of tracked fields populated per team (avg across teams).",
        "",
        "| Provider | teams | avg players/team | identity | attack | defense | GK | all fields |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name in providers:
        try:
            provider = get_player_provider(name)
        except Exception as e:
            report_lines.append(f"| {name} | FAILED: {e} |")
            print(f"[{name}] unavailable: {e}")
            continue

        counts = {f: 0 for f in TRACKED_FIELDS}
        totals = {f: 0 for f in TRACKED_FIELDS}
        teams_ok = 0
        players_total = 0

        if hasattr(provider, "fetch_squads"):
            try:
                results = provider.fetch_squads(SAMPLE_TEAMS, SEASON)
            except Exception as e:
                print(f"[{name}] batch fetch ERROR {e}")
                results = {}
            squad_iter = [(team, results.get(team, [])) for team, _ in SAMPLE_TEAMS]
        else:
            squad_iter = []
            for team, league_code in SAMPLE_TEAMS:
                try:
                    squad = provider.fetch_team_squad(team, league_code, SEASON)
                except Exception as e:
                    print(f"[{name}] {team}: ERROR {e}")
                    squad = []
                squad_iter.append((team, squad))

        for team, squad in squad_iter:
            dump_dir = out_dir / name
            dump_dir.mkdir(exist_ok=True)
            (dump_dir / f"{team}.json").write_text(
                json.dumps([r.to_dict() for r in squad], indent=1, ensure_ascii=False),
                encoding="utf-8",
            )
            if squad:
                teams_ok += 1
            players_total += len(squad)
            for rec in squad:
                for f in TRACKED_FIELDS:
                    if getattr(rec, f) is not None:
                        totals[f] += 1
            print(f"[{name}] {team}: {len(squad)} players")
        base = players_total if players_total else 1
        cells = []
        for f in TRACKED_FIELDS:
            cells.append((f, 100.0 * totals[f] / base))
        groups = {
            "identity": ["position", "age", "nationality"],
            "attack": ["goals", "assists", "xg", "xa", "shots", "shots_on_target"],
            "defense": ["tackles", "interceptions"],
            "GK": ["saves", "clean_sheets", "goals_conceded"],
        }
        all_pct = 100.0 * sum(totals.values()) / (base * len(TRACKED_FIELDS))
        row = "| %s | %d/%d | %.0f |" % (name, teams_ok, len(SAMPLE_TEAMS),
                                         players_total / max(teams_ok, 1))
        for g in ("identity", "attack", "defense", "GK"):
            gsum = sum(totals[f] for f in groups[g])
            gbase = base * len(groups[g])
            row += " %.0f%% |" % (100.0 * gsum / gbase)
        row += " %.0f%% |" % all_pct
        report_lines.append(row)

    report_lines += [
        "",
        "Interpretation (2025+ FBRef moved xG/xAG behind the Stathead paywall):",
        "  - fbref: positions/age/goals/shots/tackles/GK saves — best standalone,",
        "    but xG/xA columns are empty on the free tier.",
        "  - understat: xG/xA rich, no GK/defensive stats, rough positions.",
        "  - fod: complete rosters (age/nationality/positions) but zero stats",
        "    (needs FOD_API_KEY).",
        "  - all: fuses by normalized name — understat fills xG/xA, fbref fills",
        "    defense/GK; the recommended provider for the best-11 feature.",
    ]
    report_path = out_dir / "comparison_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\nReport written to", report_path)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
