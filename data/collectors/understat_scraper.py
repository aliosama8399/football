"""
Canonical Understat xG data collector (soccerdata + Direct HTTP JSON Fallback).

Covers all 5 leagues defined in data/config.yaml.
- Primary path: `soccerdata.Understat` when `soccerdata` is installed.
- Automatic fallback: Direct Understat JSON API (`https://understat.com/getLeagueData/{slug}/{year}/`)
  using standard `requests`, so incremental scraping works inside Docker containers
  even when `soccerdata` is not installed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config

try:
    import soccerdata as sd
    _HAS_SOCCERDATA = True
except ImportError:
    sd = None
    _HAS_SOCCERDATA = False

_UNDERSTAT_API_SLUGS = {
    "ENG-Premier League": "EPL",
    "ESP-La Liga": "La_liga",
    "GER-Bundesliga": "Bundesliga",
    "ITA-Serie A": "Serie_A",
    "FRA-Ligue 1": "Ligue_1",
}

_UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://understat.com/",
    "X-Requested-With": "XMLHttpRequest",
}


def _normalize_season_code(val) -> str:
    """Normalize soccerdata season identifiers (e.g. '2526', 2526, '2025/2026', 2526.0) to 4-char 'YYZZ'."""
    if isinstance(val, (float, int)) and not pd.isna(val):
        val = int(val)
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) == 4 and s.isdigit():
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 8:  # e.g. 20252026
        return digits[2:4] + digits[6:8]
    if len(digits) == 4:
        return digits
    return s


def _fetch_understat_http_json(soccerdata_ids: list[str], seasons: list[str]) -> pd.DataFrame:
    """
    Dependency-free HTTP JSON fetcher for Understat match xG across all 5 leagues.
    Used automatically inside Docker containers or when `soccerdata` is unavailable.
    """
    rows = []
    for sd_league in soccerdata_ids:
        slug = _UNDERSTAT_API_SLUGS.get(sd_league)
        if not slug:
            continue
        for s_code in seasons:
            norm_s = _normalize_season_code(s_code)
            year = f"20{norm_s[:2]}" if len(norm_s) == 4 else str(s_code)
            url = f"https://understat.com/getLeagueData/{slug}/{year}/"
            try:
                resp = requests.get(url, headers=_UA_HEADERS, timeout=8)
                if resp.status_code != 200:
                    print(f"  [HTTP Fallback] {sd_league} {norm_s} ({url}) returned status {resp.status_code}")
                    continue
                payload = json.loads(resp.text)
                dates = payload.get("dates", [])
                completed_count = 0
                for m in dates:
                    if not m.get("isResult"):
                        continue
                    h = m.get("h") or {}
                    a = m.get("a") or {}
                    goals = m.get("goals") or {}
                    xg = m.get("xG") or {}
                    dt_str = str(m.get("datetime", "")).strip()
                    h_title = str(h.get("title", "")).strip()
                    a_title = str(a.get("title", "")).strip()
                    if not dt_str or not h_title or not a_title:
                        continue
                    rows.append({
                        "league": sd_league,
                        "season": norm_s,
                        "game": f"{dt_str[:10]} {h_title}-{a_title}",
                        "game_id": m.get("id"),
                        "date": dt_str,
                        "home_team_id": h.get("id"),
                        "away_team_id": a.get("id"),
                        "home_team": h_title,
                        "away_team": a_title,
                        "home_team_code": h.get("short_title", ""),
                        "away_team_code": a.get("short_title", ""),
                        "home_goals": int(goals.get("h", 0) or 0),
                        "away_goals": int(goals.get("a", 0) or 0),
                        "home_xg": float(xg.get("h", 0.0) or 0.0),
                        "away_xg": float(xg.get("a", 0.0) or 0.0),
                    })
                    completed_count += 1
                print(f"  [HTTP JSON] {sd_league} ({norm_s}): {completed_count} completed matches")
                time.sleep(0.8)
            except Exception as e:
                print(f"  [Warn] Direct HTTP Understat fetch failed for {sd_league} {norm_s}: {e}")
    return pd.DataFrame(rows)


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
        missing_historical = [
            s for s in all_configured
            if _normalize_season_code(s) not in existing_seasons
        ]
        active_season = str(cfg.get("active_soccerdata_season", all_configured[-1]))

        # In incremental mode, only fetch active season plus any genuinely missing historical seasons
        seasons = list(dict.fromkeys(missing_historical + [active_season]))
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
    if _HAS_SOCCERDATA:
        try:
            u = sd.Understat(soccerdata_ids, seasons=seasons)
            df = u.read_team_match_stats()
            if df is not None and len(df) > 0:
                df = df.reset_index()
        except Exception as e:
            print(f"  [Warn] soccerdata.Understat raised: {e}; falling back to direct HTTP JSON.")
            df = None

    if df is None or len(df) == 0:
        print("  Using direct Understat HTTP JSON API collector...")
        df = _fetch_understat_http_json(soccerdata_ids, seasons)

    if df is None or len(df) == 0:
        print("  [Warn] No new Understat records returned.")
        return

    new_df = df.copy()
    if old_df is not None:
        try:
            combined = pd.concat([old_df, new_df], ignore_index=True)
            dedup_cols = [c for c in ["league", "season", "date", "home_team", "away_team"] if c in combined.columns]
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            combined.to_csv(out_path, index=False)
            print(f"[OK] Saved {len(combined)} records (deduplicated) to {out_path}")
            return
        except Exception as e:
            print(f"  [Warn] Failed appending to existing understat file: {e}")

    new_df.to_csv(out_path, index=False)
    print(f"[OK] Saved {len(new_df)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Understat xG collector (soccerdata + HTTP JSON fallback)")
    parser.add_argument("--seasons", nargs="+", default=None, help="Specific season codes (e.g. 2425 2526)")
    parser.add_argument("--force", action="store_true", help="Force full multi-season re-download")
    parser.add_argument("--no-incremental", action="store_true", help="Disable incremental season filtering")
    args = parser.parse_args()
    main(
        seasons_list=args.seasons,
        incremental=not args.no_incremental and not args.force,
        force=args.force,
    )
