"""
Canonical Understat xG data collector via the soccerdata library.

Covers all 5 leagues defined in data/config.yaml. This is the recommended
path; the 2-league custom scrapers (understat_collector.py, understat_simple.py)
are kept as optional fallbacks only.
"""

import soccerdata as sd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config


def main():
    cfg = load_config()
    soccerdata_ids = [info["soccerdata_id"] for info in cfg["leagues"].values()]
    seasons = cfg["soccerdata_seasons"]

    print(f"Fetching Understat Match Data for {len(soccerdata_ids)} leagues...")
    print(f"  Leagues: {soccerdata_ids}")
    print(f"  Seasons ({len(seasons)}): {seasons[0]} .. {seasons[-1]}")

    u = sd.Understat(soccerdata_ids, seasons=seasons)
    df = u.read_team_match_stats()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "understat_xg_data.csv"

    df.reset_index().to_csv(out_path, index=False)
    print(f"Saved {len(df)} records to {out_path}")


if __name__ == "__main__":
    main()

