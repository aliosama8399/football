"""
Weather Data Collector
Gets historical weather data for football matches using Open-Meteo API (free, no key required).

Stadium coordinates are loaded from data/stadium_coords.yaml (5 leagues, ~150 stadiums).

Retry logic: up to 3 attempts with exponential backoff (1s/2s/4s) on
502 Bad Gateway, connection timeouts, and transient failures.

JSON cache: keyed by (lat, lon, date) — written to data/raw/weather_cache.json,
reused automatically on re-runs. Delete the cache file to force a full re-fetch.
"""

import json
import pandas as pd
import requests
import yaml
from pathlib import Path
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config
from data.team_registry import normalize_team_name


# Load stadium coordinates from YAML (5 leagues)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "stadium_coords.yaml"
_STADIUM_CACHE = None

# Weather result cache — key = f"{lat}_{lon}_{date_str}", value = {t,p,r,ws,h}
_CACHE_PATH = Path(__file__).resolve().parent.parent / "raw" / "weather_cache.json"
_CACHE = None
_STATS = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0}


def _load_cache():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                _CACHE = json.load(f)
        except Exception:
            _CACHE = {}
    else:
        _CACHE = {}
    return _CACHE


def _save_cache():
    global _CACHE
    if _CACHE is not None:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_CACHE, f)
        except Exception:
            pass


def _load_stadium_coords():
    """Load and flatten the stadium coords YAML into a single {team: {...}} dict."""
    global _STADIUM_CACHE
    if _STADIUM_CACHE is not None:
        return _STADIUM_CACHE

    if not _CONFIG_PATH.exists():
        print(f"  ERR stadium_coords.yaml not found at {_CONFIG_PATH}")
        return {}

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    flat = {}
    for league_name, teams in (yaml_data or {}).items():
        if teams:
            flat.update(teams)
    _STADIUM_CACHE = flat
    return flat


def get_stadium_coords(team: str) -> dict:
    """
    Get stadium coords for a team. Tries the registry canonical name first,
    then a normalization pass through the team_registry for known aliases.
    """
    coords = _load_stadium_coords()
    if team in coords:
        return coords[team]
    canonical = normalize_team_name(team)
    if canonical != team and canonical in coords:
        return coords[canonical]
    return {}


def get_historical_weather(lat: float, lon: float, date: str, hour: int = 15) -> dict:
    """
    Get historical weather from Open-Meteo Archive API with retry+cache.

    Cache key: "{lat}_{lon}_{date}" — first hit avoids API entirely.
    Retry: up to 3 attempts with exponential backoff (1s, 2s, 4s) on
    502 Bad Gateway, ConnectTimeout, and other transient HTTP errors.
    """
    cache_key = f"{lat}_{lon}_{date}"

    # Hit the on-disk cache first — instant return for previously-fetched data
    cache = _load_cache()
    if cache_key in cache:
        _STATS["cache_hits"] += 1
        return cache[cache_key]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,precipitation,rain,wind_speed_10m,relative_humidity_2m",
        "timezone": "auto",
    }

    fallback = {
        "temperature": None, "precipitation": None, "rain": None,
        "wind_speed": None, "humidity": None,
    }

    RETRY_BACKOFF = [1.0, 2.0, 4.0]
    MAX_RETRIES = len(RETRY_BACKOFF)

    for attempt in range(MAX_RETRIES + 1):
        try:
            _STATS["calls"] += 1
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'hourly' in data and data['hourly']['time']:
                result = {
                    'temperature': data['hourly']['temperature_2m'][hour],
                    'precipitation': data['hourly']['precipitation'][hour],
                    'rain': data['hourly']['rain'][hour],
                    'wind_speed': data['hourly']['wind_speed_10m'][hour],
                    'humidity': data['hourly']['relative_humidity_2m'][hour],
                }
                cache[cache_key] = result
                global _CACHE
                _CACHE = cache
                return result

            # Valid HTTP but no hourly data — don't retry
            _STATS["failures"] += 1
            return fallback

        except Exception as e:
            err_msg = str(e)

            # Only retry on transient errors
            is_transient = any(t in err_msg.lower() for t in 
                               ('502', 'timeout', 'timed', 'connection', 'proxy', 
                                'reset', 'refused', 'unreachable'))
            if is_transient and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                _STATS["retries"] += 1
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for {date}: "
                      f"(sleep {wait}s) -> {err_msg[:100]}")
                time.sleep(wait)
                continue

            print(f"Weather API error for {date}: {e}")
            _STATS["failures"] += 1
            return fallback

    _STATS["failures"] += 1
    return fallback


def add_weather_to_matches(matches_df: pd.DataFrame, date_col: str = 'Date',
                           home_team_col: str = 'HomeTeam') -> pd.DataFrame:
    """
    Add weather data (temperature, precipitation, rain, wind_speed, humidity)
    AND stadium metadata (stadium_lat, stadium_lon, stadium_name) to a matches
    DataFrame. Uses the Open-Meteo Archive API (free, no key required).

    Rate-limit sleep per request is loaded from data/config.yaml
    (weather_rate_limit_sec). With ~17,000 matches × 0.2s ≈ 1 hour total.

    Retries 502/timeout errors up to 3× with backoff.
    Cache survives across runs via data/raw/weather_cache.json.

    Args:
        matches_df: DataFrame with match data
        date_col: Name of the date column
        home_team_col: Name of the home team column

    Returns:
        DataFrame with 8 new columns: temperature, precipitation, rain,
        wind_speed, humidity, stadium_lat, stadium_lon, stadium_name
    """
    global _STATS
    _STATS = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0}

    cfg = load_config()
    rate_limit = cfg.get('weather_rate_limit_sec', 0.2)
    weather_hour = cfg.get('weather_hour', 15)

    est_min = max(1, int(len(matches_df) * rate_limit / 60))
    print(f"Adding weather data to {len(matches_df)} matches "
          f"(rate limit: {rate_limit}s/call, est. ~{est_min} min "
          f"+ retry overhead)...")

    weather_data = []
    stadium_data = []
    save_every = int(max(1, 300 / rate_limit))

    for idx, row in matches_df.iterrows():
        home_team = row[home_team_col]
        match_date = row[date_col]

        # Parse date
        if pd.isna(match_date):
            date_str = None
        elif isinstance(match_date, str):
            try:
                for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y']:
                    try:
                        dt = datetime.strptime(match_date, fmt)
                        date_str = dt.strftime('%Y-%m-%d')
                        break
                    except:
                        continue
                else:
                    date_str = None
            except:
                date_str = None
        elif isinstance(match_date, pd.Timestamp):
            date_str = match_date.strftime('%Y-%m-%d')
        else:
            date_str = None

        coords = get_stadium_coords(home_team)

        # Stadium metadata (always returned even if weather fails)
        if coords:
            stadium_data.append({
                'stadium_lat': coords.get('lat'),
                'stadium_lon': coords.get('lon'),
                'stadium_name': coords.get('stadium'),
            })
        else:
            stadium_data.append({
                'stadium_lat': None, 'stadium_lon': None, 'stadium_name': None,
            })

        if coords and date_str:
            weather = get_historical_weather(coords['lat'], coords['lon'],
                                              date_str, hour=weather_hour)
        else:
            weather = {
                'temperature': None, 'precipitation': None, 'rain': None,
                'wind_speed': None, 'humidity': None,
            }
        weather_data.append(weather)

        # Progress indicator every 200 matches
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(matches_df)} matches "
                  f"(calls={_STATS['calls']}, cache={_STATS['cache_hits']}, "
                  f"retries={_STATS['retries']}, fails={_STATS['failures']})")

        # Persist cache to disk periodically so partial runs survive crashes
        rate_limit_over = rate_limit 

        # Persist cache to disk every save_every rows + at the end
        if (idx + 1) % save_every == 0:
            _save_cache()

        time.sleep(rate_limit)

    # Final cache save
    _save_cache()

    weather_df = pd.DataFrame(weather_data)
    stadium_df = pd.DataFrame(stadium_data)
    result_df = pd.concat([matches_df.reset_index(drop=True),
                           weather_df, stadium_df], axis=1)

    n_with_weather = result_df['temperature'].notna().sum()
    n_stadiums = result_df['stadium_name'].notna().sum()
    print(f"OK Added weather+stadium data to {len(result_df)} matches")
    print(f"  Weather: {n_with_weather}/{len(result_df)} real values, "
          f"{len(result_df)-n_with_weather} missing (no coords/API-fail)")
    print(f"  Stadiums: {n_stadiums}/{len(result_df)} resolved")
    print(f"  Stats: {_STATS['calls']} API calls, {_STATS['cache_hits']} cache "
          f"hits, {_STATS['retries']} retries, {_STATS['failures']} failures")
    if _STATS['retries'] > 0:
        recovered = _STATS['retries'] - _STATS['failures']
        print(f"  Retries recovered ~{max(0,recovered)} matches from transient errors")
    return result_df


def get_weather_for_date_location(team: str, date: str) -> dict:
    """Get weather for a specific team and date (team name resolved via registry)."""
    coords = get_stadium_coords(team)
    if coords:
        return get_historical_weather(coords['lat'], coords['lon'], date)
    return {}


if __name__ == "__main__":
    print("Testing Open-Meteo Weather API + cache...")

    test_team = "Liverpool"
    test_date = "2024-01-01"

    print(f"\nWeather for {test_team} on {test_date}:")
    weather = get_weather_for_date_location(test_team, test_date)
    for key, value in weather.items():
        print(f"  {key}: {value}")

    test_teams_5_leagues = [
        ('Man City', 'EPL alias'),
        ('Dortmund', 'Bundesliga alias'),
        ('Paris SG', 'Ligue 1 alias'),
        ('Ath Madrid', 'La Liga alias'),
        ('Milan', 'Serie A alias'),
    ]
    print("\nStadium coord lookup (5 leagues, via registry):")
    for raw_name, label in test_teams_5_leagues:
        coords = get_stadium_coords(raw_name)
        print(f"  {label:20s}  {raw_name:15s} -> "
              f"lat={coords.get('lat')}, lon={coords.get('lon')}, "
              f"stadium={coords.get('stadium')}")

    print(f"\nTotal stadiums loaded: {len(_load_stadium_coords())}")
    cache = _load_cache()
    print(f"Cache entries: {len(cache)}")
    print(f"Cache file: {_CACHE_PATH}")