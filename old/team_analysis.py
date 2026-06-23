import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_team_strengths(match_df: pd.DataFrame, team_name: str) -> Dict[str, List[str]]:
    """
    Analyze a team's strengths based on their statistics
    """
    # Get team's matches
    team_matches = match_df[match_df['team'] == team_name].sort_values('date')
    recent_matches = team_matches.tail(10)  # Last 10 matches for recent form
    
    strengths = []
    weaknesses = []
    
    # Attacking Analysis
    avg_scored = recent_matches['scored'].mean()
    avg_xG = recent_matches['xG'].mean()
    league_avg_scored = match_df.groupby('date')['scored'].mean().mean()
    
    if avg_scored > league_avg_scored * 1.2:
        strengths.append("Strong attacking output")
    elif avg_scored < league_avg_scored * 0.8:
        weaknesses.append("Struggles to score goals")
        
    # Finishing Efficiency
    if avg_scored > avg_xG * 1.2:
        strengths.append("Clinical finishing")
    elif avg_scored < avg_xG * 0.8:
        weaknesses.append("Poor finishing efficiency")
    
    # Defensive Analysis
    avg_conceded = recent_matches['missed'].mean()
    avg_xGA = recent_matches['xGA'].mean()
    league_avg_conceded = match_df.groupby('date')['missed'].mean().mean()
    
    if avg_conceded < league_avg_conceded * 0.8:
        strengths.append("Solid defense")
    elif avg_conceded > league_avg_conceded * 1.2:
        weaknesses.append("Vulnerable defense")
        
    # Pressing Analysis
    avg_ppda = recent_matches['ppda_coef'].mean()
    league_avg_ppda = match_df['ppda_coef'].mean()
    
    if avg_ppda < league_avg_ppda * 0.8:
        strengths.append("High pressing intensity")
    elif avg_ppda > league_avg_ppda * 1.2:
        weaknesses.append("Low pressing intensity")
        
    # Deep Completions
    avg_deep = recent_matches['deep'].mean()
    league_avg_deep = match_df['deep'].mean()
    
    if avg_deep > league_avg_deep * 1.2:
        strengths.append("Good at creating clear chances")
    elif avg_deep < league_avg_deep * 0.8:
        weaknesses.append("Struggles to create clear chances")
        
    return {
        'strengths': strengths,
        'weaknesses': weaknesses
    }

def get_tactical_insights(home_stats: Dict, away_stats: Dict) -> List[str]:
    """
    Generate tactical insights based on team statistics
    """
    insights = []
    
    # Pressing match-up
    home_press = home_stats['last_5_ppda_coef']
    away_press = away_stats['last_5_ppda_coef']
    
    if home_press < away_press * 0.8:
        insights.append("Home team likely to dominate possession through pressing")
    elif away_press < home_press * 0.8:
        insights.append("Away team likely to control game through pressing")
        
    # Counter-attack potential
    if home_stats['last_5_deep'] > away_stats['last_5_deep_allowed'] * 1.5:
        insights.append("Home team has good counter-attacking potential")
    if away_stats['last_5_deep'] > home_stats['last_5_deep_allowed'] * 1.5:
        insights.append("Away team has good counter-attacking potential")
        
    # Defensive solidity
    if home_stats['last_5_xGA'] < 1.0:
        insights.append("Home team has strong defensive record")
    if away_stats['last_5_xGA'] < 1.0:
        insights.append("Away team has strong defensive record")
        
    return insights

def get_key_matchup_areas(home_analysis: Dict[str, List[str]], 
                         away_analysis: Dict[str, List[str]]) -> List[str]:
    """
    Identify key areas of the game based on team strengths and weaknesses
    """
    key_areas = []
    
    # Check for attacking vs defensive matchups
    if "Strong attacking output" in home_analysis['strengths'] and "Vulnerable defense" in away_analysis['weaknesses']:
        key_areas.append("Home team's attack vs Away team's defense could be decisive")
    
    if "Strong attacking output" in away_analysis['strengths'] and "Vulnerable defense" in home_analysis['weaknesses']:
        key_areas.append("Away team's attack vs Home team's defense could be crucial")
        
    # Check for pressing battles
    if "High pressing intensity" in home_analysis['strengths'] and "Low pressing intensity" in away_analysis['weaknesses']:
        key_areas.append("Home team's pressing could disrupt away team's buildup")
    
    if "High pressing intensity" in away_analysis['strengths'] and "Low pressing intensity" in home_analysis['weaknesses']:
        key_areas.append("Away team's pressing could force home team errors")
        
    return key_areas
