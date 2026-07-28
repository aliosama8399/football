"""
Understat xG Data Collector (custom async scraper — OPTIONAL FALLBACK).

This is the 2-league custom async scraper (EPL + La Liga). It is kept as a
fallback; the canonical path is understat_scraper.py (soccerdata-based,
covers all 5 leagues).

Seasons and leagues are loaded from data/config.yaml. Because this scraper
is EPL/La-Liga-only by nature, only those leagues will be returned even if
config.yaml lists 5 — the others are filtered out.
"""

import asyncio
import pandas as pd
from pathlib import Path
import json
import aiohttp
from bs4 import BeautifulSoup
import re
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config


def _get_league_years():
    """Read leagues/years from config. This scraper only supports EPL+La Liga."""
    cfg = load_config()
    # Map understat slug -> friendly name; only epl + la_liga are supported
    supported_slugs = {"epl": "Premier_League", "la_liga": "La_Liga"}
    leagues = {}
    for code, info in cfg["leagues"].items():
        slug = info["understat_slug"]
        if slug in supported_slugs:
            leagues[slug] = supported_slugs[slug]
    years = cfg["understat_years"]
    return leagues, years


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch a page with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    async with session.get(url, headers=headers) as response:
        return await response.text()


def extract_json_data(html: str, var_name: str) -> list:
    """Extract JSON data from script tags in Understat HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.find_all('script')
    
    for script in scripts:
        if script.string and var_name in script.string:
            # Find the JSON data
            pattern = rf"{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
            match = re.search(pattern, script.string)
            if match:
                json_str = match.group(1)
                # Decode unicode escapes
                json_str = json_str.encode().decode('unicode_escape')
                return json.loads(json_str)
    
    return []


async def get_league_matches(session: aiohttp.ClientSession, league: str, year: int) -> list:
    """
    Get all match data for a league and season from Understat
    
    Args:
        session: aiohttp session
        league: 'epl' or 'la_liga'
        year: Season start year (e.g., 2023 for 2023-24)
    
    Returns:
        List of match dictionaries with xG data
    """
    url = f"https://understat.com/league/{league}/{year}"
    print(f"Fetching: {url}")
    
    try:
        html = await fetch_page(session, url)
        matches = extract_json_data(html, 'datesData')
        
        processed_matches = []
        for match in matches:
            processed_matches.append({
                'match_id': match.get('id'),
                'date': match.get('datetime'),
                'home_team': match.get('h', {}).get('title', ''),
                'away_team': match.get('a', {}).get('title', ''),
                'home_goals': int(match.get('goals', {}).get('h', 0)),
                'away_goals': int(match.get('goals', {}).get('a', 0)),
                'home_xg': float(match.get('xG', {}).get('h', 0)),
                'away_xg': float(match.get('xG', {}).get('a', 0)),
                'is_result': match.get('isResult', False)
            })
        
        # Filter only completed matches
        completed = [m for m in processed_matches if m['is_result']]
        print(f"  ✓ Found {len(completed)} completed matches")
        return completed
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


async def get_team_stats(session: aiohttp.ClientSession, league: str, year: int) -> pd.DataFrame:
    """Get team-level statistics from Understat"""
    url = f"https://understat.com/league/{league}/{year}"
    
    try:
        html = await fetch_page(session, url)
        teams_data = extract_json_data(html, 'teamsData')
        
        team_stats = []
        for team_id, team_info in teams_data.items():
            history = team_info.get('history', [])
            
            # Calculate season totals
            total_xg = sum(float(h.get('xG', 0)) for h in history)
            total_xga = sum(float(h.get('xGA', 0)) for h in history)
            total_npxg = sum(float(h.get('npxG', 0)) for h in history)
            total_npxga = sum(float(h.get('npxGA', 0)) for h in history)
            total_goals = sum(int(h.get('scored', 0)) for h in history)
            total_conceded = sum(int(h.get('missed', 0)) for h in history)
            matches = len(history)
            
            team_stats.append({
                'team': team_info.get('title', ''),
                'matches': matches,
                'goals': total_goals,
                'conceded': total_conceded,
                'xg': round(total_xg, 2),
                'xga': round(total_xga, 2),
                'npxg': round(total_npxg, 2),
                'npxga': round(total_npxga, 2),
                'xg_per_match': round(total_xg / matches, 2) if matches > 0 else 0,
                'xga_per_match': round(total_xga / matches, 2) if matches > 0 else 0,
                'xg_diff': round(total_xg - total_xga, 2)
            })
        
        return pd.DataFrame(team_stats)
        
    except Exception as e:
        print(f"Error getting team stats: {e}")
        return pd.DataFrame()


async def collect_all_data(output_dir: str = "data/raw") -> tuple:
    """
    Collect all xG data for specified leagues and seasons
    
    Returns:
        Tuple of (matches_df, team_stats_df)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_matches = []
    all_team_stats = []
    
    async with aiohttp.ClientSession() as session:
        leagues, years = _get_league_years()
        for league_code, league_name in leagues.items():
            for year in years:
                # Get match data
                matches = await get_league_matches(session, league_code, year)
                
                for match in matches:
                    match['league'] = league_name
                    match['season'] = f"{year}{str(year+1)[-2:]}"
                
                all_matches.extend(matches)
                
                # Get team stats
                team_stats = await get_team_stats(session, league_code, year)
                if not team_stats.empty:
                    team_stats['league'] = league_name
                    team_stats['season'] = f"{year}{str(year+1)[-2:]}"
                    all_team_stats.append(team_stats)
                
                # Be respectful to the server
                await asyncio.sleep(2)
    
    # Create DataFrames
    matches_df = pd.DataFrame(all_matches)
    team_stats_df = pd.concat(all_team_stats, ignore_index=True) if all_team_stats else pd.DataFrame()
    
    # Save to CSV
    if not matches_df.empty:
        matches_path = output_path / "understat_matches.csv"
        matches_df.to_csv(matches_path, index=False)
        print(f"\n✓ Saved {len(matches_df)} matches to {matches_path}")
    
    if not team_stats_df.empty:
        stats_path = output_path / "understat_team_stats.csv"
        team_stats_df.to_csv(stats_path, index=False)
        print(f"✓ Saved team stats to {stats_path}")
    
    return matches_df, team_stats_df


def run_collector():
    """Run the async collector"""
    print("=" * 60)
    print("Understat xG Data Collector")
    print("Leagues: Premier League, La Liga")
    print("Seasons: 2022-23, 2023-24, 2024-25")
    print("=" * 60)
    print()
    
    matches_df, team_stats_df = asyncio.run(collect_all_data())
    
    if not matches_df.empty:
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(f"Total matches: {len(matches_df)}")
        print(f"\nMatches per league:")
        print(matches_df['league'].value_counts())
        print(f"\nSample xG values:")
        print(matches_df[['home_team', 'away_team', 'home_xg', 'away_xg', 'home_goals', 'away_goals']].head(10))
    
    return matches_df, team_stats_df


if __name__ == "__main__":
    run_collector()
