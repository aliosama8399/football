"""
Configuration loader for the football data pipeline.

Single entry point: load_config() reads data/config.yaml (with mtime caching)
and returns a plain dict. Every collector and preprocess.py should pull their
seasons/leagues/paths from here instead of hard-coding constants.

Usage:
    from data._config import load_config
    cfg = load_config()
    seasons = cfg['seasons']               # ['1516', ..., '2425']
    leagues = cfg['leagues']               # {E0: {name, fduk_code, ...}, ...}
"""

from pathlib import Path
import os
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_CACHE = {"mtime": None, "data": None}


def load_config(force_reload: bool = False) -> dict:
    """Load config.yaml with mtime caching.

    Re-reads the file only if its mtime changed since last call, so calling
    load_config() many times in one process is essentially free.

    Args:
        force_reload: bypass the cache and re-read from disk

    Returns:
        dict with keys: seasons, understat_years, soccerdata_seasons,
                        leagues, rolling_window, h2h_window, weather_*,
                        paths, chunk_size, dtypes
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {_CONFIG_PATH}. "
            "Run from project root or create data/config.yaml."
        )

    current_mtime = _CONFIG_PATH.stat().st_mtime
    if not force_reload and _CACHE["mtime"] == current_mtime and _CACHE["data"] is not None:
        return _CACHE["data"]

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _CACHE["mtime"] = current_mtime
    _CACHE["data"] = cfg
    return cfg


def get_seasons() -> list:
    """Shortcut: return the football-data.co.uk-format seasons list."""
    return load_config()["seasons"]


def get_leagues() -> dict:
    """Shortcut: return the leagues dict keyed by fduk_code."""
    return load_config()["leagues"]


def get_setting(key: str, default=None):
    """Shortcut: return a single top-level setting by key."""
    return load_config().get(key, default)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Seasons ({len(cfg['seasons'])}): {cfg['seasons']}")
    print(f"Understat years ({len(cfg['understat_years'])}): {cfg['understat_years']}")
    print(f"Leagues ({len(cfg['leagues'])}):")
    for code, info in cfg["leagues"].items():
        print(f"  {code}: {info['name']}  (fduk={info['fduk_code']}, "
              f"understat={info['understat_slug']}, soccerdata={info['soccerdata_id']})")
    print(f"Rolling window: {cfg['rolling_window']}")
    print(f"H2H window: {cfg['h2h_window']}")
    print(f"Weather rate limit: {cfg['weather_rate_limit_sec']}s/call")
    print(f"Chunk size: {cfg['chunk_size']}")
    print(f"Paths: {cfg['paths']}")
    print(f"Dtypes: {list(cfg['dtypes'].keys())}")
