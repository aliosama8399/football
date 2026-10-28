"""
Canonical Understat xG data collector via the soccerdata library.

Covers all 5 leagues defined in data/config.yaml. This is the recommended
path; the 2-league custom scrapers (understat_collector.py, understat_simple.py)
are kept as optional fallbacks only.
"""

import argparse
import soccerdata as sd
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config


def _normalize_season_code(val) -> str:
    """Normalize soccerdata season identifiers (e.g. '2526', 2526, '2025/2026') to 4-char 'YYZZ'."""
    s = str(val).strip()
    if len(s) == 4 and s.isdigit():
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:  # e.g. 20252026
        return digits[2:4] + digits[6:8]
    if len(digits) == 4:
        return digits
    return s


def main(seasons_list=None, incremental: bool = True, force: bool = False):
    cfg = load_config()
    soccerdata_ids = [info["soccerdata_id"] for info in cfg["leagues"].values()]

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "understat_xg_data.csv"

    old_df = None
    existing_seasons: set[str] = set()
    if out_path.exists() and not force:
        try:
            old_df = pd.read_csv(out_path, low_memory=False)
            if "season" in old_df.columns:
                existing_seasons = {
                    _normalize_season_code(v)
                    for v in old_df["season"].dropna().unique()
                }
        except Exception as e:
            print(f"  [Warn] Could not inspect existing {out_path}: {e}")

    if seasons_list is not None:
        seasons = [str(s) for s in seasons_list]
    elif incremental and not force and existing_seasons:
        all_configured = [str(s) for s in cfg["soccerdata_seasons"]]
        missing_historical = [s for s in all_configured if s not in existing_seasons]
        active_season = str(cfg.get("active_soccerdata_season", all_configured[-1]))
        latest_completed = str(all_configured[-1])

        # Always refresh the latest/active season plus any missing historical seasons
        seasons = list(dict.fromkeys(missing_historical + [latest_completed, active_season]))
        print(
            f"  [Incremental] Found {len(existing_seasons)} cached seasons in {out_path}. "
            f"Fetching only: {seasons}"
        )
    else:
        seasons = [str(s) for s in cfg["soccerdata_seasons"]]

    print(f"Fetching Understat Match Data for {len(soccerdata_ids)} leagues...")
    print(f"  Leagues: {soccerdata_ids}")
    print(f"  Seasons ({len(seasons)}): {seasons[0]} .. {seasons[-1]}")

    df = None
    try:
        u = sd.Understat(soccerdata_ids, seasons=seasons)
        df = u.read_team_match_stats()
    except Exception as e:
        print(f"  [Warn] Primary Understat fetch raised: {e}")
        # Fallback: if active future season isn't published on Understat yet, retry without it
        active_s = str(cfg.get("active_soccerdata_season", ""))
        fallback_seasons = [s for s in seasons if s != active_s]
        if fallback_seasons and fallback_seasons != seasons:
            print(f"  [Fallback] Retrying Understat fetch without unstarted season {active_s}: {fallback_seasons}")
            try:
                u = sd.Understat(soccerdata_ids, seasons=fallback_seasons)
                df = u.read_team_match_stats()
            except Exception as e2:
                print(f"  ERR Failed fetching Understat stats on fallback: {e2}")
                return
        else:
            print(f"  ERR Failed fetching Understat stats: {e}")
            return

    if df is None or len(df) == 0:
        print("  [Warn] No new Understat records returned.")
        return

    new_df = df.reset_index()
    if old_df is not None:
        try:
            combined = pd.concat([old_df, new_df], ignore_index=True)
            # Deduplicate by league, season, date, teams if present
            dedup_cols = [c for c in ['league', 'season', 'date', 'home_team', 'away_team'] if c in combined.columns]
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
            combined.to_csv(out_path, index=False)
            print(f"[OK] Saved {len(combined)} records (deduplicated) to {out_path}")
            return
        except Exception as e:
            print(f"  [Warn] Failed appending to existing understat file: {e}")

    new_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(new_df)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Understat xG collector via soccerdata")
    parser.add_argument("--seasons", nargs="+", default=None, help="Specific season codes (e.g. 2425 2526)")
    parser.add_argument("--force", action="store_true", help="Force full multi-season re-download")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental season filtering")
    args = parser.parse_args()
    main(
        seasons_list=args.seasons,
        incremental=not args.no_incremental and not args.force,
        force=args.force,
    )
