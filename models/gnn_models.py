"""
GNN Model Architectures for Football Match Prediction
=====================================================
7 different graph neural network architectures for edge classification
(predicting match outcome: Home Win / Draw / Away Win).

All models follow the same pattern:
1. GNN layers process node features using graph structure
2. For each edge (match), concatenate source + target node embeddings + edge features
3. MLP classifier predicts outcome probabilities

The 7th model (TEA-GNN) is implemented in models/tea_gnn.py and adds three
novel components beyond the baselines: edge-conditioned attention, learned
temporal decay via edge_time, and cross-league context pooling via league_id.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, NNConv,
    global_mean_pool
)

# TEA-GNN: novel architecture with edge-conditioned temporal attention
# + cross-league context. tea_gnn.py imports EdgeClassifier from this module
# below, so the import is placed AFTER EdgeClassifier is defined. See bottom.



class EdgeClassifier(nn.Module):
    """Shared edge classification head used by all GNN models."""
    
    def __init__(self, node_embed_dim, edge_feat_dim, num_classes=3, dropout=0.3):
        super().__init__()
        # Input: concat(home_embed, away_embed, edge_features)
        input_dim = node_embed_dim * 2 + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, node_embeds, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        src_embed = node_embeds[src]
        dst_embed = node_embeds[dst]
        # Concat: home_team_embed | away_team_embed | match_features
        edge_repr = torch.cat([src_embed, dst_embed, edge_attr], dim=-1)
        return self.mlp(edge_repr)


# ═══════════════════════════════════════════════════════════
# 1. GCN — Graph Convolutional Network
# ═══════════════════════════════════════════════════════════

class GCN_Model(nn.Module):
    """Basic GCN: averages neighbor features with normalization."""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# 2. GraphSAGE — Sample and Aggregate
# ═══════════════════════════════════════════════════════════

class SAGE_Model(nn.Module):
    """GraphSAGE: inductive, samples neighbors, aggregates with mean/max."""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# 3. GAT — Graph Attention Network
# ═══════════════════════════════════════════════════════════

class GAT_Model(nn.Module):
    """GAT: multi-head attention to weight neighbor contributions."""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=3,
                 heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# 4. GIN — Graph Isomorphism Network
# ═══════════════════════════════════════════════════════════

class GIN_Model(nn.Module):
    """GIN: sum aggregation with MLP — maximally expressive."""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# 5. EdgeConv (NNConv) — Edge-Conditioned Convolution
# ═══════════════════════════════════════════════════════════

class EdgeConv_Model(nn.Module):
    """NNConv: uses edge features directly in the convolution operator."""
    
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Edge network: maps edge features → weight matrix for convolution
        self.edge_nn1 = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features * hidden_dim),
        )
        self.conv1 = NNConv(num_node_features, hidden_dim, self.edge_nn1, aggr='mean')
        
        self.edge_nn2 = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * hidden_dim),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nn2, aggr='mean')
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)
    
    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# 6. Hybrid — GNN embeddings + Traditional ML features
# ═══════════════════════════════════════════════════════════

class Hybrid_Model(nn.Module):
    """
    Combines GNN node embeddings with raw tabular features.
    Uses GraphSAGE backbone + edge features + extra tabular features.
    """
    
    def __init__(self, num_node_features, num_edge_features, num_tabular_features=0,
                 hidden_dim=64, num_classes=3, dropout=0.3):
        super().__init__()
        
        # GNN backbone
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        
        # Combined classifier: GNN embeds + edge features + tabular
        combined_dim = hidden_dim * 2 + num_edge_features + num_tabular_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x, edge_index, edge_attr, tabular_features=None):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        
        if tabular_features is not None:
            edge_repr = torch.cat([edge_repr, tabular_features], dim=-1)
        
        return self.classifier(edge_repr)


# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════

# Import TEA-GNN AFTER EdgeClassifier + baselines are defined.
# tea_gnn.py imports EdgeClassifier from this module (must exist first).
from models.tea_gnn import TEA_GNN_Model

MODEL_REGISTRY = {
    'GCN': GCN_Model,
    'GraphSAGE': SAGE_Model,
    'GAT': GAT_Model,
    'GIN': GIN_Model,
    'EdgeConv': EdgeConv_Model,
    'Hybrid': Hybrid_Model,
    'TEA-GNN': TEA_GNN_Model,
}


def get_model(name, num_node_features, num_edge_features, **kwargs):
    """Factory function to create a GNN model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_node_features, num_edge_features, **kwargs)
