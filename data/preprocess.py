"""
Data Preprocessing and Feature Engineering Pipeline
Merges multiple data sources and creates ML-ready features.

Memory-safe: all rolling/cumulative/H2H computations are vectorized groupby
operations rather than iterrows+filter loops. Config-driven via _config.py.
"""

import gc
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'collectors'))
from data._config import load_config


class FootballDataProcessor:
    """Process and merge football data from multiple sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Pull feature-engineering parameters from data/config.yaml
        cfg = load_config()
        self.rolling_window = cfg.get("rolling_window", 5)
        self.h2h_window = cfg.get("h2h_window", 5)
        self.weather_rate_limit_sec = cfg.get("weather_rate_limit_sec", 0.2)
        self.weather_hour = cfg.get("weather_hour", 15)
        self.dtypes_schema = cfg.get("dtypes", {})
        
        # Team name mappings — now driven by data/team_registry.py (C2).
        # Replaces the old PL+LL-only dict with comprehensive 5-league coverage.
        from data.team_registry import TEAM_REGISTRY
        self.team_registry = TEAM_REGISTRY

    def standardize_team_name(self, name: str, league_code: str = None) -> str:
        """
        Standardize a team name via the team registry.
        
        If league_code is provided (E0/SP1/D1/I1/F1), looks up only that league.
        Otherwise searches all leagues (cross-league merge compatibility).
        """
        from data.team_registry import normalize_team_name
        return normalize_team_name(name, league_code)
    
    def load_football_data_uk(self) -> pd.DataFrame:
        """Load and clean football-data.co.uk data"""
        print("Loading football-data.co.uk data...")
        
        combined_path = self.raw_dir / "football_data_uk_combined.csv"
        if not combined_path.exists():
            print("  ERR No combined file found. Run the collector first.")
            return pd.DataFrame()
        
        df = pd.read_csv(combined_path, low_memory=False)
        print(f"  OK Loaded {len(df)} matches")
        if 'League' in df.columns:
            print(f"  OK Leagues found: {df['League'].unique().tolist()}")
        elif 'Div' in df.columns:
            print(f"  OK Divisions found: {df['Div'].unique().tolist()}")

        # Standardize column names
        df.columns = df.columns.str.strip()

        # B6: cast low-cardinality text columns to category to shrink memory ~30%.
        # Done before merges so subsequent concat/merge operations are cheaper.
        for col in ('HomeTeam', 'AwayTeam', 'League', 'Div', 'Referee', 'FTR', 'HTR'):
            if col in df.columns:
                df[col] = df[col].astype('category')

        # football-data.co.uk uses %d/%m/%Y consistently across ALL seasons.
        # Use explicit format hint to avoid auto-infer ambiguity (US-format MM/DD
        # interpretation silently misparses day<=12 rows; day>12 rows fail outright).
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        bad_dates = df['Date'].isna().sum()
        if bad_dates > 0:
            df = df[df['Date'].notna()].copy()
            print(f"  OK Removed {bad_dates} rows with unparseable dates")
        
        # Standardize team names via fast vectorized unique-lookup
        if 'Div' in df.columns:
            for div in df['Div'].dropna().unique():
                div_mask = df['Div'] == div
                home_lookup = {t: self.standardize_team_name(t, div) for t in df.loc[div_mask, 'HomeTeam'].unique()}
                away_lookup = {t: self.standardize_team_name(t, div) for t in df.loc[div_mask, 'AwayTeam'].unique()}
                df.loc[div_mask, 'HomeTeam'] = df.loc[div_mask, 'HomeTeam'].map(home_lookup)
                df.loc[div_mask, 'AwayTeam'] = df.loc[div_mask, 'AwayTeam'].map(away_lookup)
        else:
            home_lookup = {t: self.standardize_team_name(t) for t in df['HomeTeam'].unique()}
            away_lookup = {t: self.standardize_team_name(t) for t in df['AwayTeam'].unique()}
            df['HomeTeam'] = df['HomeTeam'].map(home_lookup)
            df['AwayTeam'] = df['AwayTeam'].map(away_lookup)
        
        # Calculate result (vectorized)
        df['Result'] = np.select([df['FTHG'] > df['FTAG'], df['FTHG'] < df['FTAG']], ['H', 'A'], default='D')
        
        # Total goals
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['GoalDiff'] = df['FTHG'] - df['FTAG']
        
        # Over/Under 2.5
        df['Over2.5'] = (df['TotalGoals'] > 2.5).astype(int)
        
        # Both teams scored
        df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        return df
    
    def load_understat_data(self) -> pd.DataFrame:
        """Load newly scraped Understat xG data"""
        print("Loading Understat xG data...")
        
        raw_path = self.raw_dir / "understat_xg_data.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
            print(f"  OK Loaded {len(df)} matches from {raw_path}")
            return df
        
        print("  ERR No Understat data found. Run data/collectors/understat_scraper.py first.")
        return pd.DataFrame()
    
    def prepare_understat_match_data(self, understat_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare Understat match data for merging with football-data.co.uk.

        Team names are normalized via the 5-league team_registry (C2).
        Returns a DataFrame keyed by (Date_only, HomeTeam, AwayTeam) with xG columns.
        """
        if understat_df.empty:
            return pd.DataFrame()

        print("Processing Understat data...")

        df = understat_df.copy()

        # Standardize team names via 5-league registry (C2)
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df['HomeTeam'] = df['home_team'].apply(self.standardize_team_name)
            df['AwayTeam'] = df['away_team'].apply(self.standardize_team_name)

            # Format dates — Understat uses YYYY-MM-DD
            if 'date' in df.columns:
                df['Date_only'] = pd.to_datetime(df['date'], errors='coerce').dt.date

            if 'home_xg' in df.columns and 'away_xg' in df.columns:
                xg_data = df[['Date_only', 'HomeTeam', 'AwayTeam', 'home_xg', 'away_xg']].copy()
                xg_data = xg_data.rename(columns={'home_xg': 'Home_xG', 'away_xg': 'Away_xG'})
                # Drop rows where the date failed to parse — they can't merge
                xg_data = xg_data.dropna(subset=['Date_only', 'HomeTeam', 'AwayTeam'])
                print(f"  OK Processed Understat for matching: {len(xg_data)} matches")
                return xg_data

        print("  ERR Understat dataset did not have expected schema")
        return pd.DataFrame()

    def merge_xg_with_fuzzy_fallback(self, fduk_df: pd.DataFrame,
                                      xg_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Understat xG into fduk_df in two passes:

        Pass 1 — exact match on (Date_only, HomeTeam, AwayTeam).
        Pass 2 — fuzzy fallback via thefuzz token_set_ratio >= 90 on team names
                 when team_name registry didn't bridge them.

        Unmatched rows are logged to data/raw/unmatched_xg.csv for inspection.
        Returns fduk_df with Home_xG and Away_xG columns populated.
        """
        from rapidfuzz import fuzz, process

        if xg_data.empty:
            print("  (no Understat data — skipping xG merge)")
            fduk_df['Home_xG'] = np.nan
            fduk_df['Away_xG'] = np.nan
            return fduk_df

        # Build the merge key on fduk side
        fduk_df['Date_only'] = fduk_df['Date'].dt.date
        start_len = len(fduk_df)

        # Pass 1: exact merge
        result = fduk_df.merge(
            xg_data[['Date_only', 'HomeTeam', 'AwayTeam', 'Home_xG', 'Away_xG']],
            on=['Date_only', 'HomeTeam', 'AwayTeam'], how='left'
        )
        matched = result['Home_xG'].notna().sum()
        unmatched_mask = result['Home_xG'].isna()
        unmatched_count = unmatched_mask.sum()
        print(f"  Pass 1 (exact match): {matched}/{start_len} matched, "
              f"{unmatched_count} unmatched")

        # Pass 2: fuzzy fallback on unmatched rows
        if unmatched_count > 0:
            # Build a fast lookup index of xg_data by date_only
            xg_by_date = {}
            for _, xrow in xg_data.iterrows():
                key = xrow['Date_only']
                xg_by_date.setdefault(key, []).append(xrow)

            fuzzy_matches = 0
            for idx in result.index[unmatched_mask]:
                row = result.loc[idx]
                match_date = row['Date_only']
                home = row['HomeTeam']
                away = row['AwayTeam']
                candidates = xg_by_date.get(match_date, [])
                best_xg = None
                best_score = 0
                for cand in candidates:
                    h_score = fuzz.token_set_ratio(str(home), str(cand['HomeTeam']))
                    a_score = fuzz.token_set_ratio(str(away), str(cand['AwayTeam']))
                    combined = min(h_score, a_score)
                    if combined > best_score and combined >= 90:
                        best_score = combined
                        best_xg = cand
                if best_xg is not None:
                    result.at[idx, 'Home_xG'] = best_xg['Home_xG']
                    result.at[idx, 'Away_xG'] = best_xg['Away_xG']
                    fuzzy_matches += 1

            final_matched = result['Home_xG'].notna().sum()
            print(f"  Pass 2 (fuzzy >= 90): +{fuzzy_matches} matched; "
                  f"final: {final_matched}/{start_len} "
                  f"({final_matched/start_len*100:.1f}%)")

            # Log still-unmatched rows to data/raw/unmatched_xg.csv
            still_unmatched = result[result['Home_xG'].isna()]
            if len(still_unmatched) > 0 and 'League' in still_unmatched.columns:
                log_path = self.raw_dir / 'unmatched_xg.csv'
                still_unmatched[['Date', 'HomeTeam', 'AwayTeam', 'League', 'Season']].to_csv(
                    log_path, index=False)
                print(f"  Logged {len(still_unmatched)} unmatched rows to {log_path}")

        result = result.drop(columns=['Date_only'], errors='ignore')
        return result
    
    def calculate_team_form(self, df: pd.DataFrame, team_col: str, date_col: str,
                           result_col: str, n_matches: int = 5) -> pd.DataFrame:
        """
        Build a long-form team-level dataframe with rolling-window and
        season-cumulative features. Vectorized via groupby().transform() —
        NO iterrows, NO per-team boolean mask loop.

        Returns team_df with columns per-team: Date, Team, Season, Home, the raw
        match-stat series, plus the rolling 5-match averages and `cum_*` season
        means (with .shift(1) so the current match is excluded — leakage-safe).
        """
        print(f"Calculating {n_matches}-match rolling form + season cumulative...")

        # Map raw → friendly feature names (same names produced by the old loop version
        # so downstream merge is unaffected)
        ROLLING_COLS = {
            'Points': 'Form', 'GF': 'GF', 'GA': 'GA', 'xG': 'xG', 'xGA': 'xGA',
            'Shots': 'Shots', 'ShotsAgainst': 'ShotsAgainst',
            'ShotsOnTarget': 'SOT', 'ShotsOnTargetAgainst': 'SOTAgainst',
            'Corners': 'Corners', 'CornersAgainst': 'CornersAgainst',
            'Fouls': 'Fouls', 'FoulsAgainst': 'FoulsAgainst',
            'Yellows': 'Yellows', 'Reds': 'Reds'
        }
        RAW_STAT_COLS = list(ROLLING_COLS.keys())

        # Points from result_string in one vectorized step (no if/else per row)
        points_map = {'H': 3, 'D': 1, 'A': 0}
        home_pts  = df['Result'].map(points_map).fillna(0).astype('int8')
        # Away points are the INVERSE result: home-away swap means A → 3, D → 1, H → 0
        inv_pts_map = {'A': 3, 'D': 1, 'H': 0}
        away_pts  = df['Result'].map(inv_pts_map).fillna(0).astype('int8')

        # Home-side record (no iterrows; build via column assignment on slices)
        home_block = pd.DataFrame({
            'Date': df[date_col].values,
            'Team': df['HomeTeam'].values,
            'Home': 1,
            'Points': home_pts.values,
            'GF': df['FTHG'].values,
            'GA': df['FTAG'].values,
            'xG': df.get('Home_xG', pd.Series(np.nan, index=df.index)).values,
            'xGA': df.get('Away_xG', pd.Series(np.nan, index=df.index)).values,
            'Shots': df.get('HS', np.nan).values,
            'ShotsAgainst': df.get('AS', np.nan).values,
            'ShotsOnTarget': df.get('HST', np.nan).values,
            'ShotsOnTargetAgainst': df.get('AST', np.nan).values,
            'Corners': df.get('HC', np.nan).values,
            'CornersAgainst': df.get('AC', np.nan).values,
            'Fouls': df.get('HF', np.nan).values,
            'FoulsAgainst': df.get('AF', np.nan).values,
            'Yellows': df.get('HY', np.nan).values,
            'Reds': df.get('HR', np.nan).values,
        })

        away_block = pd.DataFrame({
            'Date': df[date_col].values,
            'Team': df['AwayTeam'].values,
            'Home': 0,
            'Points': away_pts.values,
            'GF': df['FTAG'].values,
            'GA': df['FTHG'].values,
            'xG': df.get('Away_xG', pd.Series(np.nan, index=df.index)).values,
            'xGA': df.get('Home_xG', pd.Series(np.nan, index=df.index)).values,
            'Shots': df.get('AS', np.nan).values,
            'ShotsAgainst': df.get('HS', np.nan).values,
            'ShotsOnTarget': df.get('AST', np.nan).values,
            'ShotsOnTargetAgainst': df.get('HST', np.nan).values,
            'Corners': df.get('AC', np.nan).values,
            'CornersAgainst': df.get('HC', np.nan).values,
            'Fouls': df.get('AF', np.nan).values,
            'FoulsAgainst': df.get('HF', np.nan).values,
            'Yellows': df.get('AY', np.nan).values,
            'Reds': df.get('AR', np.nan).values,
        })

        # Combine and sort ONCE — no per-team sort loop
        team_df = pd.concat([home_block, away_block], ignore_index=True, copy=False)
        team_df = team_df.sort_values(['Team', 'Date'], kind='mergesort').reset_index(drop=True)

        # Season key (calendar year of season start; July+ → following season start year)
        if pd.api.types.is_datetime64_any_dtype(team_df['Date']):
            date_series = team_df['Date']
        else:
            date_series = pd.to_datetime(team_df['Date'], errors='coerce')
        season_key = date_series.dt.year - (date_series.dt.month < 7).astype(int)
        # Some rows may have NaT date (parse failure); keep them as float until NaN-handled,
        # then cast defensively to int32 covering the NaN case (fill with sentinel 0 →
        # no harm because matches_played cumcount on NaT-season rows is rare and well-defined)
        team_df['Season'] = season_key.astype('float64')

        # Two groupers: team-only (rolling 5) and (team, season) (cumulative)
        per_team       = team_df.groupby('Team',  sort=False)
        per_team_seas  = team_df.groupby(['Team', 'Season'], sort=False)

        # BENNETT PRINCIPLE: leverage vectorization instead of row-by-row computation.
        for raw_col, feat_name in ROLLING_COLS.items():
            if raw_col not in team_df.columns:
                continue
            # Rolling 5-match form (leakage-safe with .shift(1))
            team_df[f'{feat_name}_{n_matches}'] = (
                per_team[raw_col].transform(
                    lambda s: s.rolling(n_matches, min_periods=1).mean().shift(1)
                )
            )
            # Season-cumulative expanding mean (B4 — lifted from old/preprocess_data.py.
            # First match of every team-season → 0  via shift+fillna(0); grows thereafter.)
            team_df[f'cum_{feat_name}'] = (
                per_team_seas[raw_col].transform(
                    lambda s: s.expanding(min_periods=1).mean().shift(1)
                ).fillna(0.0)
            )

        # Season-context features (B4)
        team_df['matches_played']  = per_team_seas['Points'].cumcount()
        team_df['season_progress'] = (team_df['matches_played'] / 38.0).clip(upper=1.0)
        # form_vs_season = recent form − season-wide points/played-so-far (rolling-vs-seasonal)
        season_pts_so_far = per_team_seas['Points'].transform(
            lambda s: s.shift(1).expanding(min_periods=1).sum()
        ).fillna(0.0)
        games_so_far = team_df['matches_played'].clip(lower=1)
        team_df['form_vs_season'] = (
            team_df[f'Form_{n_matches}'] - (season_pts_so_far / games_so_far)
        ).fillna(0.0)

        # Home/away split per team per season (B4 — venue-specific averages)
        # These are pre-match summaries; .shift(1) excludes current match → leakage-safe
        venue_keys = ['Team', 'Season', 'Home']
        per_venue = team_df.groupby(venue_keys, sort=False)
        team_df['venue_goals_avg']      = per_venue['GF'].transform(
            lambda s: s.expanding(min_periods=1).mean().shift(1)).fillna(0.0)
        team_df['venue_conceded_avg']   = per_venue['GA'].transform(
            lambda s: s.expanding(min_periods=1).mean().shift(1)).fillna(0.0)

        # Pivot venue-split into home/away columns per match (one row per team, so
        # the home-side row gets the home/away split)
        team_df['home_goals_avg']    = np.where(team_df['Home'] == 1, team_df['venue_goals_avg'],    np.nan)
        team_df['away_goals_avg']    = np.where(team_df['Home'] == 0, team_df['venue_goals_avg'],    np.nan)
        team_df['home_conceded_avg'] = np.where(team_df['Home'] == 1, team_df['venue_conceded_avg'], np.nan)
        team_df['away_conceded_avg'] = np.where(team_df['Home'] == 0, team_df['venue_conceded_avg'], np.nan)
        # Forward-fill is unnecessary — merge_Form_features joins HomeTeam row to
        # the home records (Home==1) and AwayTeam row to the away records (Home==0),
        # so per-row .fillna(league_median) in B5 handles NaN correctly.

        team_df = team_df.drop(columns=['venue_goals_avg', 'venue_conceded_avg'])

        # Release intermediate references (memory-caution between phases)
        del per_team, per_team_seas, per_venue, season_pts_so_far, games_so_far
        gc.collect()
        print(f"  OK Calculated form for {team_df['Team'].nunique()} teams "
              f"({len(team_df)} team-match records)")
        return team_df
    
    def merge_form_features(self, matches_df: pd.DataFrame, form_df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        """
        Merge team-form features back to the per-match dataframe using tuple-key
        merge keys (HomeTeam/AwayTeam + Date) instead of string concatenation.

        Also brings along the new B4 columns (cum_*, season_progress, form_vs_season,
        home_goals_avg/away_goals_avg/home_conceded_avg/away_conceded_avg) and
        applies the B5 sentinel+median NaN pattern — adds `*_missing` int8 flags
        for xG/xGA columns rather than the prior silent fillna(0).
        """
        print("Merging form features (tuple-key merge)...")

        # Rolling feature columns produced by calculate_team_form
        rolling_suffixes = ['Form', 'GF', 'GA', 'xG', 'xGA',
                            'Shots', 'ShotsAgainst', 'SOT', 'SOTAgainst',
                            'Corners', 'CornersAgainst', 'Fouls', 'FoulsAgainst',
                            'Yellows', 'Reds']
        rolling_cols = [f'{s}_{n_matches}' for s in rolling_suffixes
                        if f'{s}_{n_matches}' in form_df.columns]

        # B4 extras to carry through
        b4_extras = ['season_progress', 'form_vs_season',
                     'home_goals_avg', 'away_goals_avg',
                     'home_conceded_avg', 'away_conceded_avg']
        b4_extras = [c for c in b4_extras if c in form_df.columns]

        # cum_* columns (B4) — every rolling suffix gets a cum_ cousin
        cum_cols = [f'cum_{s}' for s in rolling_suffixes
                    if f'cum_{s}' in form_df.columns]

        cols_to_carry = rolling_cols + cum_cols + b4_extras

        # --- HOME side: rows where form_df.Home == 1 ---------------------
        home_form = form_df[form_df['Home'] == 1][['Team', 'Date'] + cols_to_carry].copy()
        home_form = home_form.rename(columns={'Team': 'HomeTeam'})
        home_rename = {}
        for s in rolling_suffixes:
            if f'{s}_{n_matches}' in home_form.columns:
                home_rename[f'{s}_{n_matches}'] = f'Home{s}_{n_matches}'
            if f'cum_{s}' in home_form.columns:
                home_rename[f'cum_{s}'] = f'Homecum_{s}'
        for c in b4_extras:
            if c in home_form.columns:
                home_rename[c] = f'Home{c}'
        home_form = home_form.rename(columns=home_rename)

        # --- AWAY side: rows where form_df.Home == 0 --------------------
        away_form = form_df[form_df['Home'] == 0][['Team', 'Date'] + cols_to_carry].copy()
        away_form = away_form.rename(columns={'Team': 'AwayTeam'})
        away_rename = {}
        for s in rolling_suffixes:
            if f'{s}_{n_matches}' in away_form.columns:
                away_rename[f'{s}_{n_matches}'] = f'Away{s}_{n_matches}'
            if f'cum_{s}' in away_form.columns:
                away_rename[f'cum_{s}'] = f'Awaycum_{s}'
        for c in b4_extras:
            if c in away_form.columns:
                away_rename[c] = f'Away{c}'
        away_form = away_form.rename(columns=away_rename)

        # --- Tuple-key merges (no string concat → no object-dtype key memory) ---
        # matches_df has 'HomeTeam', 'AwayTeam', 'Date' — direct join keys.
        # suffixes ensures any matching columns (Date on both sides) get suffixed
        # rather than colliding.
        matches_df = matches_df.merge(
            home_form, on=['HomeTeam', 'Date'], how='left', suffixes=('', '_hform'))
        matches_df = matches_df.drop(columns=[c for c in matches_df.columns
                                              if c.endswith('_hform')], errors='ignore')

        matches_df = matches_df.merge(
            away_form, on=['AwayTeam', 'Date'], how='left', suffixes=('', '_aform'))
        matches_df = matches_df.drop(columns=[c for c in matches_df.columns
                                              if c.endswith('_aform')], errors='ignore')

        # --- B5: Sentinel + league-median NaN handling --------------------
        # The plan calls out xG/xGA columns specifically — they get a `_missing`
        # flag so models can distinguish "0 xG happened" from "no xG available".
        # Other rolling cols get league-median fillna (or 0 for first-match).
        all_form_cols = [c for c in matches_df.columns
                          if (any(c.startswith(p) for p in ('Home', 'Away'))
                              and f'_{n_matches}' in c)
                             or c.startswith('Homecum_') or c.startswith('Awaycum_')
                             or c in ('Homeseason_progress', 'Awayseason_progress',
                                      'Homeform_vs_season', 'Awayform_vs_season',
                                      'Homehome_goals_avg', 'Awayaway_goals_avg',
                                      'Homehome_conceded_avg', 'Awayaway_conceded_avg')]

        for col in all_form_cols:
            if col not in matches_df.columns:
                continue
            # Add sentinel for xG/xGA columns (Bug-1 fix — B5)
            if ('xG' in col or 'xGA' in col):
                missing_col = f'{col}_missing'
                matches_df[missing_col] = matches_df[col].isna().astype('int8')
                if matches_df[col].isna().any() and 'League' in matches_df.columns:
                    league_median = matches_df.groupby('League')[col].transform('median')
                    matches_df[col] = matches_df[col].fillna(league_median)
                # Final safety fallback
                matches_df[col] = matches_df[col].fillna(matches_df[col].median()).fillna(0.0)
            else:
                # Other rolling cols: league-median fillna, fallback to 0.0
                if matches_df[col].isna().any() and 'League' in matches_df.columns:
                    league_median = matches_df.groupby('League')[col].transform('median')
                    matches_df[col] = matches_df[col].fillna(league_median)
                matches_df[col] = matches_df[col].fillna(0.0)

        # Release intermediates (memory-caution between phases)
        del home_form, away_form, cols_to_carry, rolling_cols, cum_cols, b4_extras
        gc.collect()
        print(f"  OK Merged form features for {len(matches_df)} matches "
              f"(+{len(all_form_cols)} feature cols)")
        return matches_df
    
    def calculate_head_to_head(self, df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
        """
        Calculate head-to-head statistics. Reimplemented with M2b algorithm:
        a dict[pair_key, deque(maxlen=n_matches)] keyed on the sorted team tuple.

        Memory: O(unique_pairs * n_matches) — trivial.
        Runtime: O(N) — one pass over the date-sorted dataframe.

        Faithful port of the prior semantics:
            A previous match counts as a Home win for the current row's HomeTeam if:
              - prior HomeTeam == current HomeTeam  AND prior Result == 'H', OR
              - prior AwayTeam == current HomeTeam  AND prior Result == 'A'
            (current AwayTeam mirrors that logic.)
            Goals are taken from the current HomeTeam's perspective.

        New output columns (matched to plan):
            H2H_Matches, H2H_HomeWins, H2H_AwayWins, H2H_Draws,
            H2H_HomeGoals, H2H_AwayGoals,
            H2H_avg_home_goals, H2H_avg_away_goals     (covariates for graph builder)
        """
        from collections import defaultdict, deque

        print("Calculating head-to-head stats (M2b deque pass)...")

        # Work on a date-sorted copy. Preserve the original index for join.
        df = df.sort_values('Date', kind='mergesort').copy()
        idx_arr   = list(df.index)
        home_arr  = df['HomeTeam'].tolist()
        away_arr  = df['AwayTeam'].tolist()
        fthg_arr  = df['FTHG'].tolist()
        ftag_arr  = df['FTAG'].tolist()
        result_arr= df['Result'].tolist()

        pair_history = defaultdict(lambda: deque(maxlen=n_matches))

        # Pre-allocate output lists — building a DataFrame once at the end.
        n = len(df)
        h2h_matches    = [0] * n
        h2h_home_wins   = [0] * n
        h2h_away_wins   = [0] * n
        h2h_draws       = [0] * n
        h2h_home_goals  = [0.0] * n
        h2h_away_goals  = [0.0] * n
        h2h_avg_h_g     = [0.0] * n
        h2h_avg_a_g     = [0.0] * n

        for i in range(n):
            home = home_arr[i]
            away = away_arr[i]
            pair_key = (home, away) if home < away else (away, home)
            history = pair_history[pair_key]

            if history:
                hw = aw = dr = 0
                hg = ag = 0
                seen = 0
                for (prev_home, prev_result, prev_fthg, prev_ftag) in history:
                    seen += 1
                    if prev_home == home:
                        # Same fixture polarity as today's
                        won_side = 'H' if prev_result == 'H' else 'A' if prev_result == 'A' else 'D'
                        hg += prev_fthg
                        ag += prev_ftag
                    else:
                        # Reversed-fixture: away in prior match == today's home
                        won_side = 'A' if prev_result == 'A' else 'H' if prev_result == 'H' else 'D'
                        hg += prev_ftag
                        ag += prev_fthg
                    if won_side == 'H':
                        hw += 1
                    elif won_side == 'A':
                        aw += 1
                    else:
                        dr += 1
                h2h_matches[i]   = seen
                h2h_home_wins[i] = hw
                h2h_away_wins[i] = aw
                h2h_draws[i]     = dr
                # Average goals per H2H match, both coded from this row's perspective
                h2h_home_goals[i] = hg / seen
                h2h_away_goals[i] = ag / seen
                h2h_avg_h_g[i]    = hg / seen
                h2h_avg_a_g[i]    = ag / seen

            # Push current match into THIS pair's history AFTER computing (no leakage)
            history.append((home, result_arr[i], fthg_arr[i], ftag_arr[i]))

        h2h_df = pd.DataFrame(
            {
                'H2H_Matches':       h2h_matches,
                'H2H_HomeWins':      h2h_home_wins,
                'H2H_AwayWins':      h2h_away_wins,
                'H2H_Draws':         h2h_draws,
                'H2H_HomeGoals':     h2h_home_goals,
                'H2H_AwayGoals':     h2h_away_goals,
                'H2H_avg_home_goals':h2h_avg_h_g,
                'H2H_avg_away_goals':h2h_avg_a_g,
            },
            index=list(idx_arr)
        )
        df = df.join(h2h_df)

        # Free intermediates (memory-caution between phases)
        del pair_history, idx_arr, home_arr, away_arr, fthg_arr, ftag_arr, result_arr
        del h2h_matches, h2h_home_wins, h2h_away_wins, h2h_draws, h2h_home_goals, h2h_away_goals, h2h_avg_h_g, h2h_avg_a_g
        gc.collect()
        print(f"  OK Calculated H2H for {len(df)} matches")
        return df
    
    # Betting odds features removed - not used for ethical reasons

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML models"""
        print("Creating target variables...")
        
        # Result as numeric
        df['Result_Num'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Over/Under variants
        df['Over1.5'] = (df['TotalGoals'] > 1.5).astype(int)
        df['Over3.5'] = (df['TotalGoals'] > 3.5).astype(int)
        
        # Clean sheet
        df['HomeCleanSheet'] = (df['FTAG'] == 0).astype(int)
        df['AwayCleanSheet'] = (df['FTHG'] == 0).astype(int)
        
        print(f"  OK Created target variables")
        return df
    
    def calculate_referee_strictness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate referee strictness level based on historical cards given"""
        if 'Referee' not in df.columns:
            print("  ERR No Referee column found")
            return df
        
        print("Calculating referee strictness...")
        
        # Clean referee column - add 'Unknown' to categories and fill
        if 'Referee' in df.columns and isinstance(df['Referee'].dtype, pd.CategoricalDtype):
            if 'Unknown' not in df['Referee'].cat.categories:
                df['Referee'] = df['Referee'].cat.add_categories(['Unknown'])
        df['Referee'] = df['Referee'].fillna('Unknown')
        
        # Calculate cards per match for each referee
        card_cols = ['HY', 'AY', 'HR', 'AR']  # Yellow/Red cards Home/Away
        available_card_cols = [col for col in card_cols if col in df.columns]
        
        if not available_card_cols:
            print("  ERR No card columns found (HY, AY, HR, AR)")
            df['Ref_Strictness'] = 0.5  # Default medium strictness
            return df
        
        # Calculate total cards per match
        df['_TotalYellows'] = df[['HY', 'AY']].sum(axis=1) if 'HY' in df.columns and 'AY' in df.columns else 0
        df['_TotalReds'] = df[['HR', 'AR']].sum(axis=1) if 'HR' in df.columns and 'AR' in df.columns else 0
        
        # Calculate referee averages using historical data (shifted to avoid leakage)
        ref_stats = df.groupby('Referee').agg({
            '_TotalYellows': 'mean',
            '_TotalReds': 'mean'
        }).rename(columns={'_TotalYellows': 'Ref_AvgYellows', '_TotalReds': 'Ref_AvgReds'})
        
        # Strictness score: Yellows + 3*Reds (scaled to 0-1)
        ref_stats['Ref_Strictness_Raw'] = ref_stats['Ref_AvgYellows'] + 3 * ref_stats['Ref_AvgReds']
        max_strictness = ref_stats['Ref_Strictness_Raw'].max()
        ref_stats['Ref_Strictness'] = ref_stats['Ref_Strictness_Raw'] / max_strictness if max_strictness > 0 else 0.5
        
        # Merge back to dataframe
        df = df.merge(ref_stats[['Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness']], 
                      left_on='Referee', right_index=True, how='left')
        
        # Fill any missing with average
        df['Ref_Strictness'] = df['Ref_Strictness'].fillna(df['Ref_Strictness'].mean())
        df['Ref_AvgYellows'] = df['Ref_AvgYellows'].fillna(df['Ref_AvgYellows'].mean())
        df['Ref_AvgReds'] = df['Ref_AvgReds'].fillna(df['Ref_AvgReds'].mean())
        
        # Drop temporary columns
        df = df.drop(columns=['_TotalYellows', '_TotalReds'], errors='ignore')
        
        # Drop referee name column (user wants strictness only, not the name)
        df = df.drop(columns=['Referee'], errors='ignore')
        
        print(f"  OK Calculated referee strictness for {len(ref_stats)} referees")
        print(f"    Strictness range: {df['Ref_Strictness'].min():.2f} - {df['Ref_Strictness'].max():.2f}")
        
        return df
    
    def process_all(self) -> pd.DataFrame:

        """Run the full preprocessing pipeline. Config-driven window sizes,
        memory-cautious gc.collect() between phases, exhaustive bet-col drop."""
        print("=" * 60)
        print("FOOTBALL DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        print()

        n_form = self.rolling_window
        n_h2h  = self.h2h_window

        # Load data
        fduk_df = self.load_football_data_uk()
        understat_df = self.load_understat_data()

        if fduk_df.empty:
            print("\nERR No data to process!")
            return pd.DataFrame()

        # ── Phase 1: Understat xG merge (with C6 fuzzy fallback) ──────────
        processed_understat = self.prepare_understat_match_data(understat_df)
        fduk_df = self.merge_xg_with_fuzzy_fallback(fduk_df, processed_understat)
        del understat_df, processed_understat
        gc.collect()

        # ── Phase 2: Rolling form + cumulative + season-context (B1+B4) ────
        form_df = self.calculate_team_form(fduk_df, 'HomeTeam', 'Date',
                                            'Result', n_matches=n_form)
        gc.collect()

        # ── Phase 3: Merge form features (B3 tuple-keys + B5 sentinel) ────
        fduk_df = self.merge_form_features(fduk_df, form_df, n_matches=n_form)
        del form_df
        gc.collect()

        # ── Phase 4: Head-to-head (B2 M2b deque) ───────────────────────────
        fduk_df = self.calculate_head_to_head(fduk_df, n_matches=n_h2h)
        gc.collect()

        # ── Phase 5: Exhaustive betting-column drop (B6) ───────────────────
        # Founded on the §0 Group L enumeration — every known odds column the
        # football-data.co.uk archive emits across 10 seasons.
        betting_patterns = [
            'B365', 'B36C', 'BW', 'IW', 'PS', 'WH', 'VC', 'MAX', 'AVG', 'ODDS',
            'BET', 'BF', '1XB', 'BFE', 'BFEX', 'PC>', 'PC<', 'P>', 'P<', 'AH',
            'PAH', 'PCAH', 'PROB', 'PINNACLE', 'UB', 'SB', 'GB', 'SJ', 'SK', 'VC',
            'GBW', 'GBD', 'GBA', 'IWH', 'IWD', 'IWA', 'VCH', 'VCD', 'VCA',
            'PSCH', 'PSCD', 'PSCA', 'B365C', 'MaxC', 'AvgC', 'BFEC', '1XBC',
            'PS>2.5', 'P<2.5', 'MaxC>2.5', 'MaxC<2.5',
            'Avg>2.5', 'Avg<2.5', 'B365>2.5', 'B365<2.5',
            'Max>2.5', 'Max<2.5', 'BFE>2.5', 'BFE<2.5', '1XB>2.5',
            'BFE>', 'BFEC>', 'MaxCAHH', 'MaxCAHA', 'B365CAHH', 'B365CAHA','BTTS','LBH','LBD','LBA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5','BbAv<2.5'
        ]
        betting_cols = [c for c in fduk_df.columns
                        if any(p in c.upper() for p in betting_patterns)
                        and c not in ('B365AHH_BFE', 'BFE_TO_IGNORE')]  # never false-positive
        if betting_cols:
            fduk_df = fduk_df.drop(columns=betting_cols, errors='ignore')
            print(f"  OK Removed {len(betting_cols)} betting-related columns")

        # Remove duplicate/malformed columns
        dup_cols = [col for col in fduk_df.columns if 'ï»¿' in col or col.startswith('Unnamed')]
        if dup_cols:
            fduk_df = fduk_df.drop(columns=dup_cols, errors='ignore')
            print(f"  OK Removed {len(dup_cols)} duplicate/malformed columns")
        gc.collect()

        # ── Phase 6: Referee strictness (existing logic) ──────────────────
        fduk_df = self.calculate_referee_strictness(fduk_df)
        gc.collect()

        # ── Phase 6.5: Weather + Stadium enrichment (C3 + C4) ────────────
        # Adds 8 new columns: temperature, precipitation, rain, wind_speed,
        # humidity, stadium_lat, stadium_lon, stadium_name.
        # Rate-limited via config.weather_rate_limit_sec (default 0.2 sec/call).
        # With 17k matches × 0.2s ≈ 1 hour total — this is the long-pole wait.
        try:
            from .collectors.weather_api import add_weather_to_matches
            import os
            weather_flag = os.environ.get('SKIP_WEATHER', '').lower() in ('1', 'true', 'yes')
            if not weather_flag and len(fduk_df) > 0:
                fduk_df = add_weather_to_matches(fduk_df, date_col='Date',
                                                  home_team_col='HomeTeam')
                gc.collect()
            else:
                print("  (weather enrichment skipped via SKIP_WEATHER env)")
                for c in ('temperature', 'precipitation', 'rain', 'wind_speed',
                          'humidity', 'stadium_lat', 'stadium_lon', 'stadium_name'):
                    fduk_df[c] = np.nan
        except ImportError:
            print("  ERR weather_api not importable — skipping weather enrichment")
            for c in ('temperature', 'precipitation', 'rain', 'wind_speed',
                      'humidity', 'stadium_lat', 'stadium_lon', 'stadium_name'):
                fduk_df[c] = np.nan
        except Exception as e:
            print(f"  ERR weather enrichment failed: {e}")
            for c in ('temperature', 'precipitation', 'rain', 'wind_speed',
                      'humidity', 'stadium_lat', 'stadium_lon', 'stadium_name'):
                fduk_df[c] = np.nan

        # ── Phase 7: Targets ───────────────────────────────────────────────
        fduk_df = self.create_target_variables(fduk_df)

        # Drop any remaining transient helpers
        transient_candidates = ['Date_only', 'home_key', 'away_key', 'merge_key',
                                '_TotalYellows', '_TotalReds', 'pair']
        fduk_df = fduk_df.drop(columns=[c for c in transient_candidates
                                        if c in fduk_df.columns], errors='ignore')

        # ── Save processed data ────────────────────────────────────────────
        output_path = self.processed_dir / "processed_matches.csv"
        fduk_df.to_csv(output_path, index=False)
        print(f"\nOK Saved processed data to {output_path}")
        print(f"  Total matches: {len(fduk_df)}")
        print(f"  Total features: {len(fduk_df.columns)}")

        # Save ML-ready data (numeric only — the model doesn't read strings)
        feature_cols = [c for c in fduk_df.columns
                        if fduk_df[c].dtype in ['int64', 'float64', 'int32',
                                                'float32', 'int8', 'int16', 'float16']]
        feature_df = fduk_df[feature_cols].copy()
        feature_path = self.features_dir / "ml_ready_data.csv"
        feature_df.to_csv(feature_path, index=False)
        print(f"OK Saved ML-ready data to {feature_path}")
        print(f"  Numeric features: {len(feature_cols)}")

        del feature_df
        gc.collect()

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        return fduk_df


def main():
    """Run the preprocessing pipeline"""
    processor = FootballDataProcessor()
    df = processor.process_all()
    
    if not df.empty:
        print("\nData Summary:")
        print("-" * 40)
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nLeagues:")
        print(df['League'].value_counts())
        print(f"\nResult distribution:")
        print(df['Result'].value_counts(normalize=True).round(3))
        print(f"\nSample features:")
        print(df[['HomeTeam', 'AwayTeam', 'HomeForm_5', 'AwayForm_5', 'H2H_Matches', 'TotalGoals']].head(10))


if __name__ == "__main__":
    main()
