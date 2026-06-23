import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data():
    # Load the datasets with correct paths
    season_df = pd.read_csv("understat.com.csv")
    match_df = pd.read_csv("understat_per_game.csv")

    # Convert 'date' to datetime objects
    match_df["date"] = pd.to_datetime(match_df["date"])

    # Create a season identifier for merging
    match_df['season'] = match_df['date'].dt.year
    match_df.loc[match_df['date'].dt.month < 7, 'season'] -= 1
    
    # Sort matches by date for each team
    match_df = match_df.sort_values(by=["team", "date"])
    
    # Prepare season data for merging
    season_df = season_df.rename(columns={
        'year': 'season',
        'team': 'team',
        'xG': 'season_xG',
        'xGA': 'season_xGA',
        'npxG': 'season_npxG',
        'npxGA': 'season_npxGA',
        'deep': 'season_deep',
        'deep_allowed': 'season_deep_allowed',
        'scored': 'season_scored',
        'missed': 'season_missed',
        'xpts': 'season_xpts',
        'wins': 'season_wins',
        'draws': 'season_draws',
        'loses': 'season_loses',
        'pts': 'season_pts',
        'ppda_coef': 'season_ppda_coef',
        'oppda_coef': 'season_oppda_coef'
    })

    # Merge season stats with match data
    match_df = pd.merge(
        match_df,
        season_df,
        on=['team', 'season', 'league'],
        how='left'
    )

    # Calculate season progress (what % of season completed)
    match_df['matches_played'] = match_df.groupby(['team', 'season']).cumcount() + 1
    match_df['season_progress'] = match_df['matches_played'] / 38  # Most leagues have 38 matches

    # Feature Engineering
    rolling_window = 5  # Consider the last 5 games

    # Calculate rolling features for each team
    match_features = [
        "xG", "xGA", "scored", "missed", "deep", "deep_allowed",
        "ppda_coef", "oppda_coef", "wins", "draws", "loses"
    ]
    
    # Calculate rolling averages for recent form
    for feature in match_features:
        match_df[f"last_{rolling_window}_{feature}"] = match_df.groupby(["team", "season"])[feature].transform(
            lambda x: x.rolling(window=rolling_window, min_periods=1, closed='left').mean()
        )
    
    # Calculate cumulative statistics for the season
    for feature in match_features:
        match_df[f"cum_{feature}"] = match_df.groupby(["team", "season"])[feature].transform(
            lambda x: x.expanding().mean()
        )

    # Create form indicators
    match_df["recent_form"] = (
        match_df["last_5_wins"] * 3 + 
        match_df["last_5_draws"] * 1
    ) / rolling_window

    # Create match-specific features
    match_df["goal_difference"] = match_df["scored"] - match_df["missed"]
    match_df["xG_difference"] = match_df["xG"] - match_df["xGA"]
    match_df["form_vs_season"] = match_df["recent_form"] - (match_df["season_pts"] / match_df["matches_played"])
    
    # Create home/away specific features
    match_df["is_home"] = (match_df["h_a"] == "h").astype(int)
    home_games = match_df[match_df["h_a"] == "h"].groupby(["team", "season"])
    away_games = match_df[match_df["h_a"] == "a"].groupby(["team", "season"])
    
    # Calculate home/away performance
    match_df["home_goals_avg"] = home_games["scored"].transform("mean")
    match_df["away_goals_avg"] = away_games["scored"].transform("mean")
    match_df["home_conceded_avg"] = home_games["missed"].transform("mean")
    match_df["away_conceded_avg"] = away_games["missed"].transform("mean")

    # Handle missing values before encoding
    numeric_columns = match_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        match_df[col] = match_df[col].fillna(match_df[col].mean())

    # Encode categorical variables
    label_encoders = {}
    for cat_col in ["league", "h_a", "result"]:
        label_encoders[cat_col] = LabelEncoder()
        match_df[f"{cat_col}_encoded"] = label_encoders[cat_col].fit_transform(match_df[cat_col])

    # Ensure all target variables are properly set
    match_df["scored"] = match_df["scored"].astype(float)
    match_df["missed"] = match_df["missed"].astype(float)
    match_df["xG"] = match_df["xG"].astype(float)
    match_df["xGA"] = match_df["xGA"].astype(float)

    # Create feature matrix
    recent_form_features = [f"last_{rolling_window}_{f}" for f in match_features]
    cumulative_features = [f"cum_{f}" for f in match_features]
    season_features = [col for col in match_df.columns if col.startswith('season_')]
    match_specific_features = [
        "goal_difference", "xG_difference", "form_vs_season",
        "is_home", "home_goals_avg", "away_goals_avg",
        "home_conceded_avg", "away_conceded_avg",
        "matches_played", "season_progress"
    ]
    encoded_features = ["league_encoded", "h_a_encoded"]
    
    feature_columns = (
        recent_form_features + 
        cumulative_features + 
        season_features + 
        match_specific_features + 
        encoded_features
    )
    
    available_features = [f for f in feature_columns if f in match_df.columns]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(match_df[available_features])
    
    # Prepare targets
    y_scores = match_df[['scored', 'missed']].values
    y_expected_goals = match_df[['xG', 'xGA']].values
    y_results = match_df['result_encoded'].values

    # Save preprocessed data
    preprocessed_df = pd.DataFrame(X, columns=available_features)
    preprocessed_df["result_encoded"] = y_results
    
    # Add target variables separately
    preprocessed_df["scored"] = match_df["scored"]
    preprocessed_df["missed"] = match_df["missed"]
    preprocessed_df["xG"] = match_df["xG"]
    preprocessed_df["xGA"] = match_df["xGA"]
    
    # Save to CSV without absolute path
    preprocessed_df.to_csv("preprocessed_match_data.csv", index=False)

    # Return preprocessed data and metadata
    y_scores = match_df[["scored", "missed"]].values
    y_expected_goals = match_df[["xG", "xGA"]].values
    y_results = match_df["result_encoded"].values

    return X, y_scores, y_expected_goals, y_results, available_features, label_encoders, scaler

if __name__ == "__main__":
    X, y_scores, y_expected_goals, y_results, features, encoders, scaler = preprocess_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(features)}")
    print("Features used:", features)


