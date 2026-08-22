"""
Graph Builder for Football Match Prediction
============================================
Converts processed match data into PyTorch Geometric graph format.

DESIGN:
- Nodes: Teams (147 unique across 5 leagues) with rolling + cumulative stat features
- Historical edges: Past matches form the graph structure, carrying
  match-stat features (shots, corners, fouls, cards — NOT goals/result)
- Prediction: For each match to predict, we use the graph state BEFORE
  that match. The prediction edge has NO edge features (match hasn't happened).
  Instead, we use home/away node embeddings + H2H features from graph structure.

TEA-GNN EXTRAS:
- edge_time: FloatTensor [num_edges], years-ago per match (for learned temporal decay)
- league_id: LongTensor [num_nodes], 0..4 per team (for cross-league context pooling)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FootballGraphBuilder:
    """Build temporal graphs from football match data."""
    
    # League → integer id (TEA-GNN cross-league context needs this)
    LEAGUE_TO_ID = {
        'Premier_League': 0,
        'La_Liga': 1,
        'Serie_A': 2,
        'Bundesliga': 3,
        'Ligue_1': 4,
    }
    
    # Node features: rolling averages + cumulative averages per team.
    # Read from df as 'Home{suffix}' and 'Away{suffix}' columns.
    NODE_FEATURE_SUFFIXES = [
        # Rolling 5-match window (15)
        'Form_5', 'GF_5', 'GA_5', 'xG_5', 'xGA_5',
        'Shots_5', 'ShotsAgainst_5', 'SOT_5', 'SOTAgainst_5',
        'Corners_5', 'CornersAgainst_5', 'Fouls_5', 'FoulsAgainst_5',
        'Yellows_5', 'Reds_5',
        # Cumulative season-to-date (15)
        'cum_Form', 'cum_GF', 'cum_GA', 'cum_xG', 'cum_xGA',
        'cum_Shots', 'cum_ShotsAgainst', 'cum_SOT', 'cum_SOTAgainst',
        'cum_Corners', 'cum_CornersAgainst', 'cum_Fouls', 'cum_FoulsAgainst',
        'cum_Yellows', 'cum_Reds',
        # Season context (2)
        'season_progress', 'form_vs_season',
    ]  # 32 features per team
    
    # Edge features for HISTORICAL matches (no goals — they'd leak the result)
    HIST_EDGE_FEATURE_COLS = [
        'HS', 'AS',          # Shots
        'HST', 'AST',        # Shots on target
        'HC', 'AC',          # Corners
        'HF', 'AF',          # Fouls  
        'HY', 'AY',          # Yellow cards
        'HR', 'AR',          # Red cards
    ]  # 12 features (kept unchanged — no leakage)
    
    # Tabular features for Hybrid model (per-edge, pre-match, no leakage).
    # Mirrors rolling + cumulative node features + H2H + referee + weather + stadium.
    TABULAR_FEATURES = [
        # Rolling 5 (Home + Away = 30)
        'HomeForm_5', 'HomeGF_5', 'HomeGA_5', 'HomexG_5', 'HomexGA_5',
        'HomeShots_5', 'HomeShotsAgainst_5', 'HomeSOT_5', 'HomeSOTAgainst_5',
        'HomeCorners_5', 'HomeCornersAgainst_5', 'HomeFouls_5', 'HomeFoulsAgainst_5',
        'HomeYellows_5', 'HomeReds_5',
        'AwayForm_5', 'AwayGF_5', 'AwayGA_5', 'AwayxG_5', 'AwayxGA_5',
        'AwayShots_5', 'AwayShotsAgainst_5', 'AwaySOT_5', 'AwaySOTAgainst_5',
        'AwayCorners_5', 'AwayCornersAgainst_5', 'AwayFouls_5', 'AwayFoulsAgainst_5',
        'AwayYellows_5', 'AwayReds_5',
        # Cumulative season-to-date (Home + Away = 30)
        'Homecum_Form', 'Homecum_GF', 'Homecum_GA', 'Homecum_xG', 'Homecum_xGA',
        'Homecum_Shots', 'Homecum_ShotsAgainst', 'Homecum_SOT', 'Homecum_SOTAgainst',
        'Homecum_Corners', 'Homecum_CornersAgainst', 'Homecum_Fouls', 'Homecum_FoulsAgainst',
        'Homecum_Yellows', 'Homecum_Reds',
        'Awaycum_Form', 'Awaycum_GF', 'Awaycum_GA', 'Awaycum_xG', 'Awaycum_xGA',
        'Awaycum_Shots', 'Awaycum_ShotsAgainst', 'Awaycum_SOT', 'Awaycum_SOTAgainst',
        'Awaycum_Corners', 'Awaycum_CornersAgainst', 'Awaycum_Fouls', 'Awaycum_FoulsAgainst',
        'Awaycum_Yellows', 'Awaycum_Reds',
        # Season context (Home + Away = 4)
        'Homeseason_progress', 'Homeform_vs_season',
        'Awayseason_progress', 'Awayform_vs_season',
        # H2H (6)
        'H2H_Matches', 'H2H_HomeWins', 'H2H_AwayWins', 'H2H_Draws',
        'H2H_HomeGoals', 'H2H_AwayGoals',
        # Referee (3)
        'Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness',
        # Weather (5)
        'temperature', 'precipitation', 'rain', 'wind_speed', 'humidity',
        # Stadium location (2) — numeric only (stadium_name is a string, excluded)
        'stadium_lat', 'stadium_lon',
    ]  # 80 features per match edge
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent / "processed" / "processed_matches.csv"
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Build team → index mapping
        all_teams = sorted(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))
        self.team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
        self.idx_to_team = {idx: team for team, idx in self.team_to_idx.items()}
        self.num_teams = len(all_teams)
        
        # Per-team league id via majority vote (for TEA-GNN cross-league context)
        self.team_to_league_idx = self._compute_team_league_ids()
        
        # Encode target: A=0, D=1, H=2
        self.label_map = {'A': 0, 'D': 1, 'H': 2}
        self.class_names = ['A', 'D', 'H']
        
        print(f"[OK] Loaded {len(self.df)} matches, {self.num_teams} teams, {self.df['League'].nunique()} leagues")
    
    def _compute_team_league_ids(self) -> torch.LongTensor:
        """Assign each team its league id via majority vote across all its matches."""
        league_idx = torch.zeros(self.num_teams, dtype=torch.long)
        for team, idx in self.team_to_idx.items():
            home_leagues = self.df.loc[self.df['HomeTeam'] == team, 'League']
            away_leagues = self.df.loc[self.df['AwayTeam'] == team, 'League']
            all_leagues = pd.concat([home_leagues, away_leagues])
            if len(all_leagues) == 0:
                continue
            top_league = all_leagues.value_counts().idxmax()
            league_idx[idx] = self.LEAGUE_TO_ID.get(top_league, 0)
        return league_idx
    
    def _get_node_features(self, row, side='Home'):
        """Extract node features for a team from a match row."""
        features = []
        for suffix in self.NODE_FEATURE_SUFFIXES:
            col = f'{side}{suffix}'
            val = row.get(col, 0.0)
            features.append(float(val) if not pd.isna(val) else 0.0)
        return features
    
    def _get_hist_edge_features(self, row):
        """Extract edge features from a historical match (no goals/result!)."""
        features = []
        for col in self.HIST_EDGE_FEATURE_COLS:
            val = row.get(col, 0.0)
            features.append(float(val) if not pd.isna(val) else 0.0)
        return features
    
    def _get_tabular_features(self, row):
        """Extract tabular features for a match edge (for Hybrid model)."""
        features = []
        for col in self.TABULAR_FEATURES:
            val = row.get(col, 0.0)
            features.append(float(val) if not pd.isna(val) else 0.0)
        return features
    
    def _compute_latest_node_features(self, df_subset: pd.DataFrame) -> torch.FloatTensor:
        """Vectorized extraction of latest rolling + cumulative node features per team."""
        home_cols = [f'Home{s}' for s in self.NODE_FEATURE_SUFFIXES]
        away_cols = [f'Away{s}' for s in self.NODE_FEATURE_SUFFIXES]
        
        # Available columns with fallback to 0.0
        for c in home_cols + away_cols:
            if c not in df_subset.columns:
                df_subset[c] = 0.0

        node_feats = np.zeros((self.num_teams, len(self.NODE_FEATURE_SUFFIXES)), dtype=np.float32)
        if len(df_subset) == 0:
            return torch.tensor(node_feats, dtype=torch.float)

        # For each team, locate their latest row (either home or away)
        for team, idx in self.team_to_idx.items():
            matches = df_subset[(df_subset['HomeTeam'] == team) | (df_subset['AwayTeam'] == team)]
            if len(matches) == 0:
                continue
            last_match = matches.iloc[-1]
            if last_match['HomeTeam'] == team:
                vals = last_match[home_cols].fillna(0.0).values.astype(np.float32)
            else:
                vals = last_match[away_cols].fillna(0.0).values.astype(np.float32)
            node_feats[idx] = vals

        return torch.tensor(node_feats, dtype=torch.float)

    def build_train_test_graphs(self):
        """
        Build graph data for training and testing.
        
        APPROACH: Transductive edge classification
        - Build ONE graph with all matches as edges
        - Train on edges from 2015-2024 seasons (9 seasons)
        - Test on edges from 2024-25 season
        - Edge features: match stats (NO goals/result to prevent leakage),
          StandardScaler-normalized (fit ONLY on train edges)
        - Node features: latest rolling + cumulative team stats (fully populated)
        - TEA-GNN extras: edge_time (recency) and league_id (cross-league context)
        """
        train_seasons = [1516, 1617, 1718, 1819, 1920, 2021, 2122, 2223, 2324]
        test_seasons = [2425]
        
        train_mask_df = self.df['Season'].isin(train_seasons)
        test_mask_df = self.df['Season'].isin(test_seasons)
        valid_mask = train_mask_df | test_mask_df
        
        df_sub = self.df[valid_mask].copy().reset_index(drop=True)
        is_train = df_sub['Season'].isin(train_seasons).values
        is_test = df_sub['Season'].isin(test_seasons).values

        # ── 1. Node features (train vs test snapshots) ──
        node_features = self._compute_latest_node_features(self.df[train_mask_df])
        test_node_features = self._compute_latest_node_features(self.df[valid_mask])

        # ── 2. Vectorized Edge attributes, topology & targets ──
        src_nodes = df_sub['HomeTeam'].map(self.team_to_idx).values.astype(np.int64)
        dst_nodes = df_sub['AwayTeam'].map(self.team_to_idx).values.astype(np.int64)
        edge_index = torch.tensor(np.array([src_nodes, dst_nodes]), dtype=torch.long)

        # Ensure all columns exist and fillna
        for c in self.HIST_EDGE_FEATURE_COLS:
            if c not in df_sub.columns:
                df_sub[c] = 0.0
        for c in self.TABULAR_FEATURES:
            if c not in df_sub.columns:
                df_sub[c] = 0.0

        edge_attrs_np = df_sub[self.HIST_EDGE_FEATURE_COLS].fillna(0.0).values.astype(np.float32)
        tabular_np = df_sub[self.TABULAR_FEATURES].fillna(0.0).values.astype(np.float32)
        labels_np = df_sub['FTR'].map(self.label_map).fillna(1).values.astype(np.int64)

        train_mask_t = torch.tensor(is_train, dtype=torch.bool)
        test_mask_t = torch.tensor(is_test, dtype=torch.bool)
        edge_y = torch.tensor(labels_np, dtype=torch.long)

        # ── 3. StandardScaler fit strictly on train split only ──
        edge_scaler = StandardScaler()
        edge_scaler.fit(edge_attrs_np[is_train])
        edge_attr_scaled = torch.tensor(edge_scaler.transform(edge_attrs_np), dtype=torch.float)

        tabular_scaler = StandardScaler()
        tabular_scaler.fit(tabular_np[is_train])
        tabular_tensor = torch.tensor(tabular_scaler.transform(tabular_np), dtype=torch.float)

        # ── 4. TEA-GNN edge_time (years-ago recency signal) ──
        dates = pd.to_datetime(df_sub['Date'])
        ref_date = dates.max()
        years_ago = ((ref_date - dates).dt.days / 365.0).values.astype(np.float32)
        edge_time = torch.tensor(years_ago, dtype=torch.float32)

        graph_data = {
            'x': node_features,                  # Node features (train)
            'x_test': test_node_features,         # Node features (updated for test)
            'edge_index': edge_index,             # All edges
            'edge_attr': edge_attr_scaled,        # Edge features (scaled on train only, no goals!)
            'edge_y': edge_y,                     # Labels
            'train_mask': train_mask_t,            # Which edges are training
            'test_mask': test_mask_t,              # Which edges are testing
            'tabular_features': tabular_tensor,    # Per-edge tabular features (for Hybrid)
            'num_nodes': self.num_teams,
            'num_node_features': len(self.NODE_FEATURE_SUFFIXES),
            'num_edge_features': len(self.HIST_EDGE_FEATURE_COLS),
            'num_tabular_features': len(self.TABULAR_FEATURES),
            # TEA-GNN extras:
            'edge_time': edge_time,                # [num_edges] years-ago per match
            'league_id': self.team_to_league_idx,  # [num_nodes] league 0..4 per team
            'num_leagues': len(self.LEAGUE_TO_ID), # 5
        }
        
        print(f"\n[OK] Graph built:")
        print(f"  Nodes:            {self.num_teams}")
        print(f"  Total edges:      {edge_idx}")
        print(f"  Train edges:      {len(train_edge_indices)}")
        print(f"  Test edges:       {len(test_edge_indices)}")
        print(f"  Node features:    {len(self.NODE_FEATURE_SUFFIXES)} (rolling + cumulative + season)")
        print(f"  Edge features:    {len(self.HIST_EDGE_FEATURE_COLS)} (scaled, no goals!)")
        print(f"  Tabular features: {len(self.TABULAR_FEATURES)} (for Hybrid)")
        print(f"  Edge time:        [num_edges] years-ago (for TEA-GNN)")
        print(f"  League id:        [{self.num_teams}] per team (for TEA-GNN, {len(self.LEAGUE_TO_ID)} leagues)")
        print(f"  Train label dist: {torch.bincount(edge_y[train_mask_t], minlength=3).tolist()}")
        print(f"  Test label dist:  {torch.bincount(edge_y[test_mask_t], minlength=3).tolist()}")
        
        return graph_data


if __name__ == '__main__':
    builder = FootballGraphBuilder()
    data = builder.build_train_test_graphs()
    
    print(f"\nNode x shape:      {data['x'].shape}")
    print(f"Edge index shape:  {data['edge_index'].shape}")
    print(f"Edge attr shape:   {data['edge_attr'].shape}")
    print(f"Edge labels shape: {data['edge_y'].shape}")
    print(f"Edge time shape:   {data['edge_time'].shape}")
    print(f"League id shape:   {data['league_id'].shape}")
    print(f"Edge attr mean/std: {data['edge_attr'].mean():.4f} / {data['edge_attr'].std():.4f}")
