"""
Configuration loader for the football data pipeline.
=====================================================
Single entry point: load_config() reads data/config.yaml (with mtime caching)
and dynamically computes seasons from start_year up to the current active season,
derived automatically from the system date.

ZERO MANUAL CODE OR CONFIG EDITS NEEDED WHEN A NEW SEASON STARTS.
Every collector, graph builder, preprocessor, and pipeline script pulls
their active and historical seasons from here.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_CACHE = {"mtime": None, "data": None, "cached_date": None}


def get_active_season_info(ref_date: datetime | None = None) -> dict:
    """
    Determine current active season and date parameters.
    Football season begins in July (month >= 7).
    """
    now = ref_date or datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    end_year = start_year + 1

    fduk_code = f"{str(start_year)[-2:]}{str(end_year)[-2:]}"
    soccerdata_code = f"{start_year}-{end_year}"

    return {
        "current_start_year": start_year,
        "current_end_year": end_year,
        "active_season": fduk_code,
        "active_soccerdata_season": soccerdata_code,
        "active_understat_year": start_year,
    }


def generate_dynamic_seasons(start_year: int = 2015, ref_date: datetime | None = None) -> dict:
    """
    Generates lists of all seasons from start_year through current active season.
    """
    info = get_active_season_info(ref_date)
    curr_start = info["current_start_year"]

    fduk_seasons = []
    understat_years = []
    soccerdata_seasons = []

    for y in range(start_year, curr_start + 1):
        fduk_seasons.append(f"{str(y)[-2:]}{str(y + 1)[-2:]}")
        understat_years.append(y)
        soccerdata_seasons.append(f"{y}-{y + 1}")

    latest_completed = fduk_seasons[-2] if len(fduk_seasons) > 1 else fduk_seasons[-1]

    return {
        "seasons": fduk_seasons,
        "understat_years": understat_years,
        "soccerdata_seasons": soccerdata_seasons,
        "active_season": info["active_season"],
        "latest_completed_season": latest_completed,
        "current_start_year": curr_start,
    }


def load_config(force_reload: bool = False) -> dict:
    """
    Load config.yaml with dynamic season generation and mtime caching.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {_CONFIG_PATH}. "
            "Run from project root or create data/config.yaml."
        )

    current_mtime = _CONFIG_PATH.stat().st_mtime
    today_str = datetime.now().strftime("%Y-%m-%d")

    if (
        not force_reload
        and _CACHE["mtime"] == current_mtime
        and _CACHE["cached_date"] == today_str
        and _CACHE["data"] is not None
    ):
        return _CACHE["data"]

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    auto_seasons = cfg.get("auto_seasons", True)
    start_year = int(cfg.get("start_year", 2015))

    if auto_seasons:
        dyn = generate_dynamic_seasons(start_year=start_year)
        cfg["seasons"] = dyn["seasons"]
        cfg["understat_years"] = dyn["understat_years"]
        cfg["soccerdata_seasons"] = dyn["soccerdata_seasons"]
        cfg["active_season"] = dyn["active_season"]
        cfg["latest_completed_season"] = dyn["latest_completed_season"]
        cfg["current_start_year"] = dyn["current_start_year"]
    else:
        info = get_active_season_info()
        cfg.setdefault("active_season", info["active_season"])
        cfg.setdefault("latest_completed_season", cfg.get("seasons", ["2526"])[-2])

    _CACHE["mtime"] = current_mtime
    _CACHE["cached_date"] = today_str
    _CACHE["data"] = cfg
    return cfg


def get_seasons() -> list[str]:
    """Shortcut: return dynamically computed football-data.co.uk season list."""
    return load_config()["seasons"]


def get_active_season() -> str:
    """Shortcut: return current active football season code (e.g. '2627')."""
    return load_config()["active_season"]


def get_latest_completed_season() -> str:
    """Shortcut: return latest finished season code (e.g. '2526')."""
    return load_config()["latest_completed_season"]


def resolve_default_player_season() -> str:
    """
    Return the latest completed season code that has cached player squads or match rows,
    falling back gracefully to the newest season in `data/raw/squads_cache/` if the
    calendar's latest completed season has not yet been scraped.
    """
    target = get_latest_completed_season()
    squads_dir = _CONFIG_PATH.parent / "raw" / "squads_cache"
    if squads_dir.exists():
        if list(squads_dir.glob(f"*_{target}.json")):
            return target
        cached_seasons = sorted(
            {
                p.stem.rsplit("_", 1)[-1]
                for p in squads_dir.glob("*.json")
                if "_" in p.stem and p.stem.rsplit("_", 1)[-1].isdigit()
            }
        )
        if cached_seasons:
            return cached_seasons[-1]
    return target


def get_leagues() -> dict:
    """Shortcut: return the leagues dict keyed by fduk_code."""
    return load_config()["leagues"]


def get_setting(key: str, default=None):
    """Shortcut: return a single top-level setting by key."""
    return load_config().get(key, default)


if __name__ == "__main__":
    cfg = load_config(force_reload=True)
    print(f"Active season (auto-detected): {cfg['active_season']}")
    print(f"Latest completed season:      {cfg['latest_completed_season']}")
    print(f"Default player season:        {resolve_default_player_season()}")
    print(f"Dynamic seasons ({len(cfg['seasons'])}): {cfg['seasons']}")
    print(f"Understat years ({len(cfg['understat_years'])}): {cfg['understat_years']}")
    print(f"Soccerdata seasons ({len(cfg['soccerdata_seasons'])}): {cfg['soccerdata_seasons'][0]} .. {cfg['soccerdata_seasons'][-1]}")
    print(f"Leagues: {list(cfg['leagues'].keys())}")
