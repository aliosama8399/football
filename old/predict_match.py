import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from team_analysis import analyze_team_strengths, get_tactical_insights, get_key_matchup_areas

def get_team_recent_stats(match_df, team_name, last_n_matches=5):
    """Get recent statistics for a team from their last N matches"""
    # Sort matches by date
    match_df = match_df.sort_values('date')
    team_matches = match_df[match_df['team'] == team_name]
    
    if len(team_matches) == 0:
        raise ValueError(f"No data found for team: {team_name}")
    
    # Get most recent matches
    recent_matches = team_matches.tail(last_n_matches)
    
    # Calculate rolling stats
    stats = {
        'last_5_xG': recent_matches['xG'].mean(),
        'last_5_xGA': recent_matches['xGA'].mean(),
        'last_5_scored': recent_matches['scored'].mean(),
        'last_5_missed': recent_matches['missed'].mean(),
        'last_5_deep': recent_matches['deep'].mean() if 'deep' in recent_matches.columns else 0,
        'last_5_deep_allowed': recent_matches['deep_allowed'].mean() if 'deep_allowed' in recent_matches.columns else 0,
        'last_5_ppda_coef': recent_matches['ppda_coef'].mean(),
        'last_5_oppda_coef': recent_matches['oppda_coef'].mean(),
    }
    
    # Calculate form (points from last 5 matches)
    stats['form'] = (
        recent_matches['wins'].sum() * 3 + 
        recent_matches['draws'].sum()
    ) / last_n_matches
    
    # Get current season stats
    current_season = team_matches['season'].iloc[-1]
    season_matches = team_matches[team_matches['season'] == current_season]
    
    cumulative_stats = {
        'matches_played': len(season_matches),
        'season_xG': season_matches['xG'].mean(),
        'season_xGA': season_matches['xGA'].mean(),
        'season_scored': season_matches['scored'].mean(),
        'season_missed': season_matches['missed'].mean(),
        'season_deep': season_matches['deep'].mean() if 'deep' in season_matches.columns else 0,
        'season_deep_allowed': season_matches['deep_allowed'].mean() if 'deep_allowed' in season_matches.columns else 0,
        'season_points': (
            season_matches['wins'].sum() * 3 + 
            season_matches['draws'].sum()
        )
    }
    
    stats.update(cumulative_stats)
    
    return stats

def prepare_match_features(home_team_stats, away_team_stats, league):
    """Prepare feature vector for prediction"""
    # Load models and preprocessing objects
    models = joblib.load('match_prediction_models.joblib')
    
    # Initialize feature dictionary
    feature_dict = {
        # Home team features
        'league_encoded': 0,  # Will be encoded properly in training
        'h_a_encoded': 1,  # 1 for home
        'rolling_xG': home_team_stats['last_5_xG'],
        'rolling_xGA': home_team_stats['last_5_xGA'],
        'rolling_scored': home_team_stats['last_5_scored'],
        'rolling_missed': home_team_stats['last_5_missed'],
        'rolling_deep': home_team_stats['last_5_deep'],
        'rolling_deep_allowed': home_team_stats['last_5_deep_allowed'],
        'rolling_ppda_coef': home_team_stats['last_5_ppda_coef'],
        'rolling_oppda_coef': home_team_stats['last_5_oppda_coef'],
        'goal_difference': home_team_stats['last_5_scored'] - home_team_stats['last_5_missed'],
        'xG_difference': home_team_stats['last_5_xG'] - home_team_stats['last_5_xGA']
    }
    
    # Calculate the rest of the features exactly as they appear in the model
    feature_dict.update({
        'cum_xG': home_team_stats['season_xG'],
        'cum_xGA': home_team_stats['season_xGA'],
        'cum_scored': home_team_stats['season_scored'],
        'cum_missed': home_team_stats['season_missed'],
        'cum_deep': home_team_stats['season_deep'],
        'cum_deep_allowed': home_team_stats['season_deep_allowed'],
        'season_progress': min(1.0, home_team_stats['matches_played'] / 38),
        'form_score': home_team_stats['form']
    })
    
    # Create numpy array with features in the correct order
    features = []
    for feature_name in models['all_features']:
        features.append(feature_dict.get(feature_name, 0.0))  # Use 0.0 as default if feature not found
    
    # Convert to numpy array
    X = np.array(features).reshape(1, -1)
    
    return X

def predict_upcoming_match(home_team, away_team, league, historical_data_path="understat_per_game.csv"):
    """Predict the result of an upcoming match"""
    try:
        # Load historical data
        match_df = pd.read_csv(historical_data_path)
        match_df['date'] = pd.to_datetime(match_df['date'])
        
        # Add season column if not exists
        if 'season' not in match_df.columns:
            match_df['season'] = match_df['date'].dt.year
            match_df.loc[match_df['date'].dt.month < 7, 'season'] -= 1
        
     
        # Get team stats
        print(f"\nGetting stats for {home_team}...")
        home_stats = get_team_recent_stats(match_df, home_team)
        print(f"Getting stats for {away_team}...")
        away_stats = get_team_recent_stats(match_df, away_team)
        
        # Prepare features
        print("\nPreparing match features...")
        X = prepare_match_features(home_stats, away_stats, league)
        
        # Load models
        print("Loading prediction models...")
        models = joblib.load('match_prediction_models.joblib')
        
        # Make prediction with full model
        print("Making predictions...")
        full_pred = models['full_model'].predict(X)[0]
        
        # Round to get final score
        home_score = int(round(full_pred[0]))
        away_score = int(round(full_pred[1]))
        
        # Calculate win probability based on predicted goals
        home_win_prob = float(full_pred[0] > full_pred[1])
        draw_prob = float(full_pred[0] == full_pred[1])
        away_win_prob = float(full_pred[0] < full_pred[1])
        
        # Get team analysis
        print("\nAnalyzing team strengths and weaknesses...")
        home_analysis = analyze_team_strengths(match_df, home_team)
        away_analysis = analyze_team_strengths(match_df, away_team)
        
        # Get tactical insights
        tactical_insights = get_tactical_insights(home_stats, away_stats)
        
        # Get key matchup areas
        key_matchups = get_key_matchup_areas(home_analysis, away_analysis)
        
        return {
            'predicted_score': f"{home_score}-{away_score}",
            'home_goals_prediction': home_score,
            'away_goals_prediction': away_score,
            'expected_goals': {
                'home_xG': float(full_pred[0]),
                'away_xG': float(full_pred[1])
            },
            'win_probabilities': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'team_analysis': {
                'home': home_analysis,
                'away': away_analysis
            },
            'tactical_insights': tactical_insights,
            'key_matchups': key_matchups,
            'team_stats': {
                'home': home_stats,
                'away': away_stats
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    home_team = input("Enter home team name: ")
    away_team = input("Enter away team name: ")
    league = input("Enter league name (e.g., EPL, La_liga, Bundesliga, Serie_A, Ligue_1): ")
    
    prediction = predict_upcoming_match(home_team, away_team, league)
    
    if 'error' in prediction:
        print(f"Error: {prediction['error']}")
    else:
        print(f"\nPredicted Score: {prediction['predicted_score']}")
        print(f"Expected Goals: Home {prediction['expected_goals']['home_xG']:.2f} - {prediction['expected_goals']['away_xG']:.2f} Away")
        
        print("\nWin Probabilities:")
        print(f"Home Win: {prediction['win_probabilities']['home_win']:.1%}")
        print(f"Draw: {prediction['win_probabilities']['draw']:.1%}")
        print(f"Away Win: {prediction['win_probabilities']['away_win']:.1%}")
        
        print(f"\n{home_team} Analysis:")
        print("Strengths:")
        for strength in prediction['team_analysis']['home']['strengths']:
            print(f"- {strength}")
        print("\nWeaknesses:")
        for weakness in prediction['team_analysis']['home']['weaknesses']:
            print(f"- {weakness}")
            
        print(f"\n{away_team} Analysis:")
        print("Strengths:")
        for strength in prediction['team_analysis']['away']['strengths']:
            print(f"- {strength}")
        print("\nWeaknesses:")
        for weakness in prediction['team_analysis']['away']['weaknesses']:
            print(f"- {weakness}")
        
        print("\nTactical Insights:")
        for insight in prediction['tactical_insights']:
            print(f"- {insight}")
            
        print("\nKey Matchups:")
        for matchup in prediction['key_matchups']:
            print(f"- {matchup}")
            
        print("\nDetailed Team Form (Last 5 matches):")
        print(f"{home_team}:")
        print(f"Goals scored per game: {prediction['team_stats']['home']['last_5_scored']:.2f}")
        print(f"Goals conceded per game: {prediction['team_stats']['home']['last_5_missed']:.2f}")
        print(f"Form score (out of 3): {prediction['team_stats']['home']['form']:.2f}")
        
        print(f"\n{away_team}:")
        print(f"Goals scored per game: {prediction['team_stats']['away']['last_5_scored']:.2f}")
        print(f"Goals conceded per game: {prediction['team_stats']['away']['last_5_missed']:.2f}")
        print(f"Form score (out of 3): {prediction['team_stats']['away']['form']:.2f}")
