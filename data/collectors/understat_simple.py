"""
Simplified Understat xG Data Collector (sync, OPTIONAL FALLBACK).
Uses synchronous requests with robust JSON extraction.

This is the 2-league custom sync scraper (EPL + La Liga). The canonical
path is understat_scraper.py (soccerdata-based, covers all 5 leagues).

Leagues and years are loaded from data/config.yaml via data._config.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config


def _get_league_years():
    """Read leagues/years from config. This scraper only supports EPL+La Liga."""
    cfg = load_config()
    # understat URL convention uses 'EPL' and 'La_liga' — distinct from the slug
    supported = {"epl": ("EPL", "https://understat.com/league/EPL"),
                 "la_liga": ("La_liga", "https://understat.com/league/La_liga")}
    leagues = {}
    for code, info in cfg["leagues"].items():
        slug = info["understat_slug"]
        if slug in supported:
            leagues[supported[slug][0]] = supported[slug][1]
    years = cfg["understat_years"]
    return leagues, years


def fetch_page(url: str) -> str:
    """Fetch a page with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def extract_json_from_script(html: str, variable_name: str) -> list:
    """
    Extract JSON data from JavaScript variable in the page.
    Understat stores data as JavaScript variables.
    """
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.find_all('script')
    
    for script in scripts:
        if script.string and variable_name in str(script.string):
            # Try to find the JSON.parse pattern
            pattern = rf"var\s+{variable_name}\s*=\s*JSON\.parse\('(.+?)'\)"
            match = re.search(pattern, str(script.string), re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Decode unicode escapes
                try:
                    json_str = json_str.encode('utf-8').decode('unicode_escape')
                    return json.loads(json_str)
                except Exception as e:
                    print(f"  JSON decode error: {e}")
    
    return []


def get_league_matches(league: str, year: int) -> list:
    """
    Get all match data for a league and season
    """
    leagues_dict, _ = _get_league_years()
    url = f"{leagues_dict[league]}/{year}"
    print(f"Fetching: {url}")
    
    try:
        html = fetch_page(url)
        
        # Extract datesData which contains match information
        matches_data = extract_json_from_script(html, 'datesData')
        
        if not matches_data:
            print(f"  Warning: No match data found for {league} {year}")
            return []
        
        processed = []
        for match in matches_data:
            try:
                # Handle both dict and other formats
                h_data = match.get('h', {})
                a_data = match.get('a', {})
                goals = match.get('goals', {})
                xg = match.get('xG', {})
                
                processed.append({
                    'match_id': match.get('id', ''),
                    'date': match.get('datetime', ''),
                    'home_team': h_data.get('title', '') if isinstance(h_data, dict) else '',
                    'away_team': a_data.get('title', '') if isinstance(a_data, dict) else '',
                    'home_goals': int(goals.get('h', 0)) if isinstance(goals, dict) else 0,
                    'away_goals': int(goals.get('a', 0)) if isinstance(goals, dict) else 0,
                    'home_xg': float(xg.get('h', 0)) if isinstance(xg, dict) else 0.0,
                    'away_xg': float(xg.get('a', 0)) if isinstance(xg, dict) else 0.0,
                    'is_result': match.get('isResult', False)
                })
            except Exception as e:
                continue
        
        # Filter only completed matches
        completed = [m for m in processed if m.get('is_result', False)]
        print(f"  ✓ Found {len(completed)} completed matches")
        return completed
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def collect_xg_data(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Collect all xG data for the leagues/years in data/config.yaml.
    This scraper only supports EPL + La Liga (custom sync fallback).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    leagues_dict, years = _get_league_years()
    all_matches = []

    for league in leagues_dict:
        league_name = 'Premier_League' if league == 'EPL' else 'La_Liga'

        for year in years:
            matches = get_league_matches(league, year)
            
            for match in matches:
                match['league'] = league_name
                match['season'] = f"{year}{str(year+1)[-2:]}"
            
            all_matches.extend(matches)
            
            # Be respectful to the server
            time.sleep(2)
    
    if not all_matches:
        print("\nNo matches collected. Using fallback: your existing understat data")
        # Check if user has existing understat data
        existing_files = [
            "understat.com.csv",
            "understat_per_game.csv"
        ]
        for f in existing_files:
            path = Path(output_dir).parent / f
            if path.exists():
                print(f"  ✓ Found existing file: {f}")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_matches)
    
    # Save to CSV
    output_file = output_path / "understat_xg_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} matches to {output_file}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Understat xG Data Collector")
    print("Leagues: Premier League, La Liga")
    print("Seasons: 2022-23, 2023-24, 2024-25")
    print("=" * 60)
    print()
    
    df = collect_xg_data()
    
    if not df.empty:
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(f"\nTotal matches: {len(df)}")
        print(f"\nMatches per league:")
        print(df['league'].value_counts())
        print(f"\nSample data:")
        print(df[['home_team', 'away_team', 'home_xg', 'away_xg', 'home_goals', 'away_goals']].head(5))
