"""
GNN Match Explainer & LLM Interpreter
======================================
1. Loads the best tuned EdgeConv model and graph.
2. Given a Home and Away team, predicts the outcome.
3. Uses GNNExplainer to extract the key node/edge features.
4. Passes the mathematical explanation to an LLM/SLM for human translation.

Configuration:
  All LLM provider settings (model names, API keys, URLs) live in
  llm_config.yaml — edit that file only. Never hardcode keys here.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path
from torch_geometric.explain import Explainer, GNNExplainer

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model
from models.llm_providers import get_llm_provider

BASE_DIR = Path(__file__).parent.parent
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model & graph loading ─────────────────────────────────────────────────────

def load_model_and_graph(model_path: Path):
    print("Loading graph data...")
    builder = FootballGraphBuilder(
        data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv"))
    graph = builder.build_train_test_graphs()

    print(f"Loading model from {model_path.name}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    bp = checkpoint.get('best_params', {})

    model = get_model(
        'EdgeConv', graph['num_node_features'], graph['num_edge_features'],
        hidden_dim=bp.get('hidden_dim', 64), dropout=bp.get('dropout', 0.3)
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(DEVICE)
    model.eval()

    return model, graph, builder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_tensors(graph: dict):
    """Extract x, edge_index, edge_attr from graph with safe fallbacks."""
    x  = graph.get('x_test', graph.get('x'))
    ei = graph.get('edge_index')
    ea = graph.get('edge_attr')
    if x  is None: raise KeyError("Graph missing 'x_test' / 'x'.")
    if ei is None: raise KeyError("Graph missing 'edge_index'.")
    if ea is None: raise KeyError("Graph missing 'edge_attr'. Check FootballGraphBuilder.")
    return x.to(DEVICE), ei.to(DEVICE), ea.to(DEVICE)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_match(model, graph: dict, home_idx: int, away_idx: int):
    x, ei, ea = _get_tensors(graph)

    with torch.no_grad():
        out = model(x, ei, ea)

    src_mask   = ei[0] == home_idx
    dst_mask   = ei[1] == away_idx
    match_edges = (src_mask & dst_mask).nonzero(as_tuple=True)[0]

    if len(match_edges) == 0:
        raise ValueError(
            "Match edge not found in the graph. "
            "Did they play during the test-set period?"
        )

    edge_idx   = match_edges[-1].item()
    logits     = out[edge_idx]
    probs      = F.softmax(logits, dim=0).cpu().numpy()
    pred_class = logits.argmax().item()

    class_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    return edge_idx, class_map[pred_class], {'A': probs[0], 'D': probs[1], 'H': probs[2]}


# ── Explanation ───────────────────────────────────────────────────────────────

def explain_prediction(model, graph: dict, builder, edge_idx: int):
    x, ei, ea = _get_tensors(graph)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='edge',
            return_type='raw',
        ),
    )

    print("Running GNNExplainer (calculating mathematical attributions)...")
    explanation = explainer(x, ei, edge_attr=ea, index=edge_idx)

    # Node feature attribution
    node_mask  = explanation.node_mask.cpu()
    home_idx   = ei[0, edge_idx].item()
    away_idx   = ei[1, edge_idx].item()
    home_team  = builder.idx_to_team[home_idx]
    away_team  = builder.idx_to_team[away_idx]

    home_imp = node_mask[home_idx].numpy()
    away_imp = node_mask[away_idx].numpy()

    feature_names = builder.NODE_FEATURE_SUFFIXES
    top_k = min(3, len(feature_names))

    top_home_feats = {
        f"{home_team}_{feature_names[i]}": float(home_imp[i])
        for i in np.argsort(home_imp)[-top_k:][::-1]
    }
    top_away_feats = {
        f"{away_team}_{feature_names[i]}": float(away_imp[i])
        for i in np.argsort(away_imp)[-top_k:][::-1]
    }

    # Edge attribution (historical matches)
    edge_mask = explanation.edge_mask.cpu().numpy()
    top_edges = np.argsort(edge_mask)[-6:][::-1]  # grab 6, exclude self

    top_matches = []
    for e in top_edges:
        if int(e) == int(edge_idx):
            continue
        h = builder.idx_to_team[ei[0, e].item()]
        a = builder.idx_to_team[ei[1, e].item()]
        top_matches.append({
            "match": f"{h} vs {a}",
            "influence_score": round(float(edge_mask[e]), 6)
        })
        if len(top_matches) == 5:
            break

    return {
        'top_node_features':       {'home_team': top_home_feats, 'away_team': top_away_feats},
        'top_influencing_matches': top_matches,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Football XAI: GNN + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explain_match.py --list-teams
  python explain_match.py --home "Arsenal" --away "Chelsea"
  python explain_match.py --home "Arsenal" --away "Chelsea" --provider openai
  python explain_match.py --home "Arsenal" --away "Chelsea" --provider ollama --model-name llama3:8b
        """
    )
    parser.add_argument('--home',        type=str, help="Home team name")
    parser.add_argument('--away',        type=str, help="Away team name")
    parser.add_argument('--provider',    type=str, default="",
                        help="LLM provider: ollama | openai | gemini | anthropic "
                             "(default: reads llm_config.yaml → default_provider)")
    parser.add_argument('--model-name',  type=str, default="",
                        help="Override model name (default: reads llm_config.yaml)")
    parser.add_argument('--list-teams',  action='store_true',
                        help="Print all available team names and exit")
    args = parser.parse_args()

    model_path = BASE_DIR / "models" / "saved" / "gnn_tea-gnn_tuned.pt"
    if not model_path.exists():
        print("Tuned TEA-GNN model not found. Run tune_gnn.py first.")
        return

    model, graph, builder = load_model_and_graph(model_path)

    # ── List mode ──
    if args.list_teams:
        print("\nAvailable teams:")
        for name in sorted(builder.team_to_idx.keys()):
            print(f"  {name}")
        return

    # ── Validate ──
    if not args.home or not args.away:
        parser.error("--home and --away are required (or use --list-teams).")

    for label, team in [("home", args.home), ("away", args.away)]:
        if team not in builder.team_to_idx:
            print(f"Unknown {label} team: '{team}'. Run --list-teams to see options.")
            return

    home_idx = builder.team_to_idx[args.home]
    away_idx = builder.team_to_idx[args.away]

    # ── Predict ──
    try:
        edge_idx, pred, probs = predict_match(model, graph, home_idx, away_idx)
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    print(f"\nPREDICTION: {args.home} vs {args.away}")
    print(f"  Outcome : {pred}")
    print(f"  Probs   : Home={probs['H']:.1%} | Draw={probs['D']:.1%} | Away={probs['A']:.1%}")

    # ── Explain ──
    try:
        math_exp = explain_prediction(model, graph, builder, edge_idx)
    except Exception as e:
        print(f"Explainer error (GNNExplainer failed): {e}")
        return

    print("\nSTRUCTURAL EXPLANATION (GNNExplainer)")
    print(json.dumps(math_exp, indent=2))

    # ── LLM translation ──
    # provider and model_name both read from llm_config.yaml if not passed via CLI
    print(f"\nCalling LLM for natural language translation...")
    try:
        llm = get_llm_provider(
            provider_type=args.provider,
            model_name=args.model_name,  # empty string → config default
        )
    except (ValueError, ImportError) as e:
        print(f"Provider setup error: {e}")
        return

    match_context = {
        'home_team':     args.home,
        'away_team':     args.away,
        'prediction':    pred,
        'probabilities': probs,
    }

    text_exp = llm.generate_explanation(match_context, math_exp)

    print("\nTACTICAL ANALYSIS:")
    print("=" * 70)
    print(text_exp)
    print("=" * 70)


if __name__ == "__main__":
    main()