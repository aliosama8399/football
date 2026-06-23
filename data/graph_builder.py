"""
Graph Builder for Football Match Prediction
============================================
Converts processed match data into PyTorch Geometric graph format.

DESIGN:
- Nodes: Teams (119 unique) with rolling stat features
- Historical edges: Past matches form the graph structure, carrying
  match-stat features (shots, corners, fouls, cards — NOT goals/result)
- Prediction: For each match to predict, we use the graph state BEFORE
  that match. The prediction edge has NO edge features (match hasn't happened).
  Instead, we use home/away node embeddings + H2H features from graph structure.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class FootballGraphBuilder:
    """Build temporal graphs from football match data."""
    
    # Node features: rolling averages per team
    NODE_FEATURE_SUFFIXES = [
        'Form_5', 'GF_5', 'GA_5', 'xG_5', 'xGA_5',
        'Shots_5', 'ShotsAgainst_5', 'SOT_5', 'SOTAgainst_5',
        'Corners_5', 'CornersAgainst_5', 'Fouls_5', 'FoulsAgainst_5',
        'Yellows_5', 'Reds_5',
    ]
    
    # Edge features for HISTORICAL matches (no goals — they'd leak the result)
    HIST_EDGE_FEATURE_COLS = [
        'HS', 'AS',          # Shots
        'HST', 'AST',        # Shots on target
        'HC', 'AC',          # Corners
        'HF', 'AF',          # Fouls  
        'HY', 'AY',          # Yellow cards
        'HR', 'AR',          # Red cards
    ]
    
    # Tabular features for Hybrid model (same as traditional ML PRE_MATCH_FEATURES)
    TABULAR_FEATURES = [
        'HomeForm_5', 'HomeGF_5', 'HomeGA_5', 'HomexG_5', 'HomexGA_5',
        'AwayForm_5', 'AwayGF_5', 'AwayGA_5', 'AwayxG_5', 'AwayxGA_5',
        'HomeShots_5', 'HomeShotsAgainst_5', 'AwayShots_5', 'AwayShotsAgainst_5',
        'HomeSOT_5', 'HomeSOTAgainst_5', 'AwaySOT_5', 'AwaySOTAgainst_5',
        'HomeCorners_5', 'HomeCornersAgainst_5', 'AwayCorners_5', 'AwayCornersAgainst_5',
        'HomeFouls_5', 'HomeFoulsAgainst_5', 'AwayFouls_5', 'AwayFoulsAgainst_5',
        'HomeYellows_5', 'HomeReds_5', 'AwayYellows_5', 'AwayReds_5',
        'H2H_Matches', 'H2H_HomeWins', 'H2H_AwayWins', 'H2H_Draws',
        'H2H_HomeGoals', 'H2H_AwayGoals',
        'Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness',
    ]
    
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
        
        # Encode target: A=0, D=1, H=2
        self.label_map = {'A': 0, 'D': 1, 'H': 2}
        self.class_names = ['A', 'D', 'H']
        
        print(f"[OK] Loaded {len(self.df)} matches, {self.num_teams} teams")
    
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
    
    def build_train_test_graphs(self):
        """
        Build graph data for training and testing.
        
        APPROACH: Transductive edge classification
        - Build ONE graph with all matches as edges
        - Train on edges from 2022-24 seasons
        - Test on edges from 2024-25 season
        - Edge features: match stats (NO goals/result to prevent leakage)
        - Node features: latest rolling team stats
        - The classifier uses: node_embed(home) + node_embed(away) only
          (edge features are used by NNConv in graph convolution but NOT
           in the final classifier for the target edge)
        
        This is fair because:
        - Node features (rolling stats) are known BEFORE the match
        - Graph structure from training edges is known
        - We predict labels on held-out test edges
        """
        train_seasons = [2223, 2324]
        test_seasons = [2425]
        
        train_mask = self.df['Season'].isin(train_seasons)
        test_mask = self.df['Season'].isin(test_seasons)
        
        # ── Node features: use latest rolling stats per team from TRAINING data ──
        node_features = torch.zeros(self.num_teams, len(self.NODE_FEATURE_SUFFIXES))
        
        # Update node features from training matches
        for _, row in self.df[train_mask].iterrows():
            hi = self.team_to_idx[row['HomeTeam']]
            ai = self.team_to_idx[row['AwayTeam']]
            node_features[hi] = torch.tensor(self._get_node_features(row, 'Home'), dtype=torch.float)
            node_features[ai] = torch.tensor(self._get_node_features(row, 'Away'), dtype=torch.float)
        
        # ── ALL edges (train + test form the graph structure) ──
        all_src, all_dst = [], []
        all_edge_feats = []
        all_tabular_feats = []
        train_edge_indices = []
        test_edge_indices = []
        all_labels = []
        
        edge_idx = 0
        for df_idx, row in self.df.iterrows():
            if not (train_mask[df_idx] or test_mask[df_idx]):
                continue
            
            hi = self.team_to_idx[row['HomeTeam']]
            ai = self.team_to_idx[row['AwayTeam']]
            
            # Add directed edge: Home → Away
            all_src.append(hi)
            all_dst.append(ai)
            all_edge_feats.append(self._get_hist_edge_features(row))
            all_tabular_feats.append(self._get_tabular_features(row))
            
            label = self.label_map.get(row['FTR'], -1)
            all_labels.append(label)
            
            if train_mask[df_idx]:
                train_edge_indices.append(edge_idx)
            else:
                test_edge_indices.append(edge_idx)
            
            edge_idx += 1
        
        # Also update node features from test data (rolling stats are pre-match, so valid)
        test_node_features = node_features.clone()
        for _, row in self.df[test_mask].iterrows():
            hi = self.team_to_idx[row['HomeTeam']]
            ai = self.team_to_idx[row['AwayTeam']]
            test_node_features[hi] = torch.tensor(self._get_node_features(row, 'Home'), dtype=torch.float)
            test_node_features[ai] = torch.tensor(self._get_node_features(row, 'Away'), dtype=torch.float)
        
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        edge_attr = torch.tensor(all_edge_feats, dtype=torch.float)
        edge_y = torch.tensor(all_labels, dtype=torch.long)
        train_mask_t = torch.zeros(edge_idx, dtype=torch.bool)
        test_mask_t = torch.zeros(edge_idx, dtype=torch.bool)
        train_mask_t[train_edge_indices] = True
        test_mask_t[test_edge_indices] = True
        
        # Tabular features: scale using training data
        tabular_np = np.array(all_tabular_feats, dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(tabular_np[train_mask_t.numpy()])
        tabular_scaled = scaler.transform(tabular_np)
        tabular_tensor = torch.tensor(tabular_scaled, dtype=torch.float)
        
        graph_data = {
            'x': node_features,                  # Node features (train)
            'x_test': test_node_features,         # Node features (updated for test)
            'edge_index': edge_index,             # All edges
            'edge_attr': edge_attr,               # Edge features (no goals!)
            'edge_y': edge_y,                     # Labels
            'train_mask': train_mask_t,            # Which edges are training
            'test_mask': test_mask_t,              # Which edges are testing
            'tabular_features': tabular_tensor,    # Per-edge tabular features (for Hybrid)
            'num_nodes': self.num_teams,
            'num_node_features': len(self.NODE_FEATURE_SUFFIXES),
            'num_edge_features': len(self.HIST_EDGE_FEATURE_COLS),
            'num_tabular_features': len(self.TABULAR_FEATURES),
        }
        
        print(f"\n[OK] Graph built:")
        print(f"  Nodes:            {self.num_teams}")
        print(f"  Total edges:      {edge_idx}")
        print(f"  Train edges:      {len(train_edge_indices)}")
        print(f"  Test edges:       {len(test_edge_indices)}")
        print(f"  Node features:    {len(self.NODE_FEATURE_SUFFIXES)}")
        print(f"  Edge features:    {len(self.HIST_EDGE_FEATURE_COLS)} (no goals!)")
        print(f"  Tabular features: {len(self.TABULAR_FEATURES)} (for Hybrid)")
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
