"""
Main Data Collection Script
Runs all collectors (football-data.co.uk, Understat) using config from data/config.yaml.

Seasons count, leagues list, and scraper routing are driven by data/_config.py.
No hard-coded values — edit data/config.yaml to change scope.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "collectors"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for data._config

import pandas as pd
from datetime import datetime
from data._config import load_config


def run_all_collectors(incremental: bool = True, force: bool = False):
    """Run all data collectors: football-data.co.uk + canonical Understat (soccerdata)."""
    cfg = load_config()
    league_names = [v["name"] for v in cfg["leagues"].values()]
    seasons_list = cfg["seasons"]

    print("=" * 70)
    print("FOOTBALL DATA COLLECTION PIPELINE")
    print(f"Leagues ({len(league_names)}): {', '.join(league_names)}")
    print(f"Seasons ({len(seasons_list)}): {seasons_list[0]} .. {seasons_list[-1]}")
    print(f"Mode: {'FULL FORCE RE-DOWNLOAD' if force else ('INCREMENTAL' if incremental else 'FULL')}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: Download from football-data.co.uk (5 leagues, N seasons per config)
    print("\n" + "=" * 70)
    print("STEP 1: Downloading from football-data.co.uk")
    print("=" * 70)

    from collectors.football_data_uk import download_all_data
    fduk_df = download_all_data(incremental=incremental and not force, force=force)

    # Step 2: Collect Understat xG data via soccerdata (canonical 5-league path)
    print("\n" + "=" * 70)
    print("STEP 2: Collecting Understat xG data (canonical — soccerdata)")
    print("=" * 70)

    from collectors.understat_scraper import main as run_understat_scraper
    run_understat_scraper(incremental=incremental and not force, force=force)

    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print("\nFiles saved to data/raw/:")
    print("  - football_data_uk_combined.csv")
    print("  - understat_xg_data.csv       (soccerdata-based, 5 leagues)")
    print("\nRun data/preprocess.py next to generate ML-ready features.")

    return fduk_df


def collect_football_data_uk(incremental: bool = True, force: bool = False):
    """Collect only football-data.co.uk data (5 leagues, N seasons per config)."""
    from collectors.football_data_uk import download_all_data
    return download_all_data(incremental=incremental and not force, force=force)


def collect_understat_data(incremental: bool = True, force: bool = False):
    """Collect only Understat xG data via the canonical soccerdata scraper."""
    from collectors.understat_scraper import main as run_understat_scraper
    run_understat_scraper(incremental=incremental and not force, force=force)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all football data collectors")
    parser.add_argument("--force", action="store_true", help="Force full multi-season re-download")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental mode")
    args = parser.parse_args()
    run_all_collectors(incremental=not args.no_incremental and not args.force, force=args.force)
