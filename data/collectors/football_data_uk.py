"""
Football-Data.co.uk Data Collector
Downloads historical match data with betting odds for 5 leagues x 10 seasons.

Seasons and leagues are loaded from data/config.yaml via data._config.load_config().
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data._config import load_config


def download_season(league_code: str, season: str) -> pd.DataFrame:
    """
    Download historical match data from football-data.co.uk
    
    Args:
        league_code: E0 (Premier League), SP1 (La Liga)
        season: e.g., '2324' for 2023-24
    
    Returns:
        DataFrame with match data
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
    print(f"Downloading: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text), encoding='utf-8', on_bad_lines='skip')
        print(f"  OK Downloaded {len(df)} matches")
        return df
    except Exception as e:
        print(f"  ERR Downloading {league_code} {season}: {e}")
        return pd.DataFrame()


def download_all_data(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download all match data for the leagues and seasons defined in data/config.yaml

    Args:
        output_dir: Directory to save the CSV files
         
    Returns:
        Combined DataFrame with all matches
    """
    cfg = load_config()
    leagues_cfg = cfg["leagues"]
    seasons = cfg["seasons"]
    leagues = {code: info["name"] for code, info in leagues_cfg.items()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_matches = []
    
    for league_code, league_name in leagues.items():
        for season in seasons:
            df = download_season(league_code, season)
            
            if not df.empty:
                # Add metadata columns
                df['League'] = league_name
                df['Season'] = season
                
                # Save individual file
                filename = f"{league_name}_{season}.csv"
                df.to_csv(output_path / filename, index=False)
                print(f"  Saved to {filename}")
                
                all_matches.append(df)
            
            # Be nice to the server
            time.sleep(1)
    
    if all_matches:
        # Combine all data
        combined_df = pd.concat(all_matches, ignore_index=True)
        
        # Save combined file
        combined_path = output_path / "football_data_uk_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✓ Combined data saved to {combined_path}")
        print(f"  Total matches: {len(combined_df)}")
        
        return combined_df
    
    return pd.DataFrame()


def get_column_descriptions() -> dict:
    """Return descriptions of key columns in the dataset"""
    return {
        # Match info
        'Div': 'League Division',
        'Date': 'Match date',
        'Time': 'Match time',
        'HomeTeam': 'Home team',
        'AwayTeam': 'Away team',
        
        # Results
        'FTHG': 'Full-time home goals',
        'FTAG': 'Full-time away goals',
        'FTR': 'Full-time result (H=Home Win, D=Draw, A=Away Win)',
        'HTHG': 'Half-time home goals',
        'HTAG': 'Half-time away goals',
        'HTR': 'Half-time result',
        
        # Match statistics
        'HS': 'Home team shots',
        'AS': 'Away team shots',
        'HST': 'Home team shots on target',
        'AST': 'Away team shots on target',
        'HF': 'Home team fouls',
        'AF': 'Away team fouls',
        'HC': 'Home team corners',
        'AC': 'Away team corners',
        'HY': 'Home team yellow cards',
        'AY': 'Away team yellow cards',
        'HR': 'Home team red cards',
        'AR': 'Away team red cards',
        
        # Betting odds (various bookmakers)
        'B365H': 'Bet365 home win odds',
        'B365D': 'Bet365 draw odds',
        'B365A': 'Bet365 away win odds',
        'BWH': 'BetandWin home win odds',
        'BWD': 'BetandWin draw odds',
        'BWA': 'BetandWin away win odds',
    }


if __name__ == "__main__":
    cfg = load_config()
    print("=" * 60)
    print("Football-Data.co.uk Data Collector")
    print(f"Leagues: {len(cfg['leagues'])}  ({', '.join(v['name'] for v in cfg['leagues'].values())})")
    print(f"Seasons: {len(cfg['seasons'])}  ({cfg['seasons'][0]} .. {cfg['seasons'][-1]})")
    print("=" * 60)
    print()
    
    df = download_all_data()
    
    if not df.empty:
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(f"Total matches: {len(df)}")
        print(f"\nMatches per league:")
        print(df['League'].value_counts())
        print(f"\nMatches per season:")
        print(df['Season'].value_counts())
        print(f"\nColumns available: {df.columns.tolist()}")
