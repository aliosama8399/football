"""
TEA-GNN: Temporal Edge-Attention Network
=========================================
A new GNN architecture for football match outcome prediction (edge
classification: Home Win / Draw / Away Win) that combines three ideas none
of your existing 6 architectures (GCN, GraphSAGE, GAT, GIN, EdgeConv,
Hybrid) do together:

  1. EDGE-CONDITIONED ATTENTION
     Attention weight for edge (home -> away) is a function of BOTH node
     embeddings AND the 12 match-level edge features (shots, corners,
     fouls, cards). GAT ignores edge_attr entirely; EdgeConv uses edge_attr
     but has no attention. This layer merges both.

  2. LEARNED TEMPORAL DECAY
     Your thesis doc proposes a hand-set exponential recency weight
     (Enhancement 3). Here the decay RATE is a learned parameter (one per
     attention head, via softplus), applied as a log-bias before softmax
     so it rescales attention weights multiplicatively.

  3. CROSS-LEAGUE CONTEXT POOLING
     Explicit mechanism for the 5-league graph: pool team embeddings per
     league, let leagues attend to each other, broadcast the result back.
     Strengthens Novelty Claim #4 (cross-league topology) from an implicit
     side effect of message passing into an explicit, inspectable module.

INTEGRATION
-----------
Drop this file next to gnn_models.py, then:

    from gnn_models import EdgeClassifier
    from tea_gnn import TEA_GNN_Model
    MODEL_REGISTRY['TEA-GNN'] = TEA_GNN_Model

forward() is backward compatible: edge_time and league_id are OPTIONAL.
- Omit both  -> behaves like a plain edge-featured GAT.
- Add edge_time only -> adds learned temporal decay.
- Add league_id only -> adds cross-league context.
- Add both -> full TEA-GNN as described in the thesis writeup.

REQUIRED DATA CHANGES (graph_builder.py)
-----------------------------------------
edge_time: a float tensor [num_edges], one value per match, e.g.:
    edge_time = (reference_date - match_date).days / 365.0
  i.e. "years ago", so recent matches -> small value -> less decay.
  Store match_date when building edges and compute this at load time
  (see `compute_edge_time` helper at the bottom of this file).

league_id: a LongTensor [num_nodes] with values 0..num_leagues-1, one per
  team node (EPL=0, La Liga=1, Serie A=2, Bundesliga=3, Ligue 1=4, or
  whatever mapping you already use in graph_builder.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

try:
    # Reuse your existing classifier head so TEA-GNN plugs straight into
    # your training/tuning pipeline (train_traditional.py-style usage).
    from gnn_models import EdgeClassifier
except ImportError:
    # Fallback so this file can be tested standalone.
    class EdgeClassifier(nn.Module):
        def __init__(self, node_embed_dim, edge_feat_dim, num_classes=3, dropout=0.3):
            super().__init__()
            input_dim = node_embed_dim * 2 + edge_feat_dim
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, node_embeds, edge_index, edge_attr):
            src, dst = edge_index[0], edge_index[1]
            edge_repr = torch.cat([node_embeds[src], node_embeds[dst], edge_attr], dim=-1)
            return self.mlp(edge_repr)


# ═══════════════════════════════════════════════════════════
# 1. Edge-Conditioned Temporal Attention Layer
# ═══════════════════════════════════════════════════════════

class TemporalEdgeAttention(MessagePassing):
    """
    Multi-head attention convolution where the attention score for edge
    (home -> away) depends on:
        - home team embedding
        - away team embedding
        - match features (shots, corners, fouls, cards, ...)
        - a LEARNED temporal decay term based on match recency

    This is the core novelty: GATConv attention is node-only; NNConv is
    edge-conditioned but has no attention. This layer does both, plus time.
    """

    def __init__(self, in_channels, out_channels, edge_dim, heads=4,
                 dropout=0.3, negative_slope=0.2):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Edge (match) feature encoder -> per-head edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, heads * out_channels),
            nn.ReLU(),
            nn.Linear(heads * out_channels, heads * out_channels),
        )

        # Attention params: e_ij = LeakyReLU(a^T [Wh_i || Wh_j || edge_embed])
        self.att = nn.Parameter(torch.empty(1, heads, 3 * out_channels))

        # Learned decay rate per head. softplus keeps it positive.
        # decay applied as: alpha_logit += -softplus(rate) * edge_time
        self.decay_rate_raw = nn.Parameter(torch.zeros(heads))

        self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_src.weight)
        glorot(self.lin_dst.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, edge_time=None):
        H, C, N = self.heads, self.out_channels, x.size(0)

        h_src = self.lin_src(x).view(N, H, C)
        h_dst = self.lin_dst(x).view(N, H, C)
        edge_embed = self.edge_encoder(edge_attr).view(-1, H, C)

        out = self.propagate(
            edge_index, h_src=h_src, h_dst=h_dst,
            edge_embed=edge_embed, edge_time=edge_time, size=(N, N),
        )
        return out.reshape(N, H * C) + self.bias

    def message(self, h_src_j, h_dst_i, edge_embed, edge_time, index, ptr, size_i):
        # h_src_j = home (source) embedding gathered per edge
        # h_dst_i = away (target) embedding gathered per edge
        feat = torch.cat([h_dst_i, h_src_j, edge_embed], dim=-1)   # [E, H, 3C]
        alpha = (feat * self.att).sum(dim=-1)                       # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if edge_time is not None:
            decay_rate = F.softplus(self.decay_rate_raw)            # [H] > 0
            log_decay = -decay_rate.unsqueeze(0) * edge_time.unsqueeze(-1)
            alpha = alpha + log_decay                                # [E, H]

        alpha = softmax(alpha, index, ptr, size_i)  # normalize per target (away) node
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = (h_src_j + edge_embed) * alpha.unsqueeze(-1)          # [E, H, C]
        return msg


# ═══════════════════════════════════════════════════════════
# 2. Cross-League Context Pooling
# ═══════════════════════════════════════════════════════════

class CrossLeagueContext(nn.Module):
    """
    Pools team embeddings per league, lets the 5 leagues attend to each
    other, then broadcasts the (now cross-league-aware) context back onto
    every node via a learned gate.

    league_id: LongTensor [num_nodes], values 0..num_leagues-1.
    """

    def __init__(self, hidden_dim, num_leagues=5, dropout=0.3):
        super().__init__()
        self.num_leagues = num_leagues
        self.node_att = nn.Linear(hidden_dim, 1)
        self.league_att = nn.MultiheadAttention(
            hidden_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_embeds, league_id):
        H = node_embeds.size(-1)

        # 1. Attention-pool nodes within each league -> [num_leagues, H]
        contexts = []
        for lg in range(self.num_leagues):
            mask = league_id == lg
            if mask.sum() == 0:
                contexts.append(torch.zeros(H, device=node_embeds.device))
                continue
            lg_nodes = node_embeds[mask]
            scores = self.node_att(lg_nodes).squeeze(-1)
            weights = F.softmax(scores, dim=0)
            contexts.append((weights.unsqueeze(-1) * lg_nodes).sum(dim=0))
        league_ctx = torch.stack(contexts, dim=0).unsqueeze(0)      # [1, L, H]

        # 2. Leagues attend to each other (e.g. lets Serie A strength
        #    inform how a La Liga team's context is read)
        league_ctx_attn, _ = self.league_att(league_ctx, league_ctx, league_ctx)
        league_ctx_attn = league_ctx_attn.squeeze(0)                # [L, H]

        # 3. Broadcast back to every node, gated so the model can choose
        #    how much cross-league context to actually use
        broadcast = league_ctx_attn[league_id]                      # [N, H]
        gate = torch.sigmoid(self.gate(torch.cat([node_embeds, broadcast], dim=-1)))
        return node_embeds + gate * broadcast


# ═══════════════════════════════════════════════════════════
# 3. TEA-GNN — full model
# ═══════════════════════════════════════════════════════════

class TEA_GNN_Model(nn.Module):
    """
    Temporal Edge-Attention Network.

    Same call signature style as your other 6 models, with two optional
    extra forward() args (edge_time, league_id) so it fits your existing
    train_gnn.py / tune_gnn.py loops with minimal changes.
    """

    def __init__(self, num_node_features, num_edge_features, hidden_dim=64,
                 num_classes=3, heads=4, num_leagues=5, dropout=0.3,
                 use_cross_league=True):
        super().__init__()
        self.use_cross_league = use_cross_league

        self.conv1 = TemporalEdgeAttention(
            num_node_features, hidden_dim // heads, num_edge_features,
            heads=heads, dropout=dropout,
        )
        self.conv2 = TemporalEdgeAttention(
            hidden_dim, hidden_dim // heads, num_edge_features,
            heads=heads, dropout=dropout,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        if use_cross_league:
            self.cross_league = CrossLeagueContext(hidden_dim, num_leagues, dropout)

        self.classifier = EdgeClassifier(hidden_dim, num_edge_features, num_classes, dropout)

    def forward(self, x, edge_index, edge_attr, edge_time=None, league_id=None):
        h = self.conv1(x, edge_index, edge_attr, edge_time)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_attr, edge_time)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.use_cross_league and league_id is not None:
            h = self.cross_league(h, league_id)

        return self.classifier(h, edge_index, edge_attr)


# ═══════════════════════════════════════════════════════════
# Helper: compute edge_time from match dates
# ═══════════════════════════════════════════════════════════

def compute_edge_time(match_dates, reference_date=None):
    """
    match_dates: list/array of datetime.date or pandas.Timestamp, one per
                 edge, in the same order as your edge_index columns.
    reference_date: the date to measure recency from (defaults to the
                 latest match date, i.e. "most recent match = 0").

    Returns: float32 torch.Tensor [num_edges], years-ago per match.
    """
    import pandas as pd
    dates = pd.to_datetime(pd.Series(match_dates))
    ref = pd.to_datetime(reference_date) if reference_date is not None else dates.max()
    years_ago = (ref - dates).dt.days / 365.0
    return torch.tensor(years_ago.values, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# Quick self-test (run this file directly to sanity-check shapes)
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    N, E = 20, 60                 # 20 teams, 60 matches
    num_node_feat, num_edge_feat = 15, 12

    x = torch.randn(N, num_node_feat)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, num_edge_feat)
    edge_time = torch.rand(E) * 2          # 0-2 years ago
    league_id = torch.randint(0, 5, (N,))

    model = TEA_GNN_Model(num_node_feat, num_edge_feat, hidden_dim=64, num_leagues=5)
    out = model(x, edge_index, edge_attr, edge_time=edge_time, league_id=league_id)
    print("Output shape:", out.shape)      # expected: [E, 3]
    assert out.shape == (E, 3)
    print("TEA-GNN forward pass OK.")
