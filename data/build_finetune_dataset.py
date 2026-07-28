"""
Fine-Tuning Dataset Builder
============================
Generates a JSONL dataset for fine-tuning an SLM on football tactical analysis.

Pipeline per match:
  1. Read pre-match stats from processed_matches.csv
  2. Run TEA-GNN prediction → outcome + probabilities
     (TEA-GNN: novel edge-conditioned temporal attention architecture,
      requires edge_time + league_id from the graph builder)
  3. Run GNNExplainer → top node features + influential edges
  4. Call a "teacher" LLM (e.g. Gemini Flash) to write the analysis
  5. Save (user_prompt, assistant_response) as a JSONL row

Dataset now spans ALL 9 seasons (1516-2324 train + 2425 test) with expanded
stats payload (rolling + cumulative + season context + weather + stadium).

Usage:
  # Dry run — print first 3 samples without calling the LLM
  python data/build_finetune_dataset.py --dry-run --max-samples 3

  # Full run — generate all samples with Gemini teacher
  python data/build_finetune_dataset.py --provider gemini

  # Resume after a crash (skip already-generated rows)
  python data/build_finetune_dataset.py --provider gemini --resume

  # Skip GNNExplainer (faster, less rich prompts)
  python data/build_finetune_dataset.py --provider gemini --skip-explainer
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model
from models.llm_providers import get_llm_provider

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "processed_matches.csv"
OUTPUT_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature name mapping: column suffix → plain English for the prompt.
# Grouped into three sections: rolling 5, cumulative season-to-date, misc.
FEATURE_LABELS = {
    # ── Rolling 5-match window ──
    'Form_5':           'Points per match (last 5)',
    'GF_5':             'Goals scored per match (last 5)',
    'GA_5':             'Goals conceded per match (last 5)',
    'xG_5':             'Expected goals per match (last 5)',
    'xGA_5':            'Expected goals against per match (last 5)',
    'Shots_5':          'Shots per match (last 5)',
    'ShotsAgainst_5':   'Shots conceded per match (last 5)',
    'SOT_5':            'Shots on target per match (last 5)',
    'SOTAgainst_5':     'Shots on target conceded per match (last 5)',
    'Corners_5':        'Corners per match (last 5)',
    'CornersAgainst_5': 'Corners conceded per match (last 5)',
    'Fouls_5':          'Fouls committed per match (last 5)',
    'FoulsAgainst_5':   'Fouls suffered per match (last 5)',
    'Yellows_5':        'Yellow cards per match (last 5)',
    'Reds_5':           'Red cards per match (last 5)',

    # ── Cumulative season-to-date averages ──
    'cum_Form':           'Cumulative points per match (season)',
    'cum_GF':             'Cumulative goals scored per match (season)',
    'cum_GA':             'Cumulative goals conceded per match (season)',
    'cum_xG':             'Cumulative expected goals per match (season)',
    'cum_xGA':            'Cumulative expected goals against per match (season)',
    'cum_Shots':          'Cumulative shots per match (season)',
    'cum_ShotsAgainst':   'Cumulative shots conceded per match (season)',
    'cum_SOT':            'Cumulative shots on target per match (season)',
    'cum_SOTAgainst':     'Cumulative shots on target conceded per match (season)',
    'cum_Corners':        'Cumulative corners per match (season)',
    'cum_CornersAgainst': 'Cumulative corners conceded per match (season)',
    'cum_Fouls':          'Cumulative fouls committed per match (season)',
    'cum_FoulsAgainst':   'Cumulative fouls suffered per match (season)',
    'cum_Yellows':        'Cumulative yellow cards per match (season)',
    'cum_Reds':           'Cumulative red cards per match (season)',

    # ── Season context ──
    'season_progress': 'Season progress (%)',
    'form_vs_season':  'Form vs season average (ratio)',
}


def format_match_date(date_val):
    """Format date: remove 00:00:00 time if it's zero or missing."""
    if date_val is None or pd.isna(date_val):
        return ""
    if hasattr(date_val, 'strftime'):
        if date_val.hour == 0 and date_val.minute == 0 and date_val.second == 0:
            return date_val.strftime('%Y-%m-%d')
        return date_val.strftime('%Y-%m-%d %H:%M:%S')
    s = str(date_val).strip()
    if s.endswith(" 00:00:00"):
        s = s[:-9]
    return s


def clean_match_id(raw_id):
    """Normalize match_id string to remove trailing 00:00:00."""
    if not raw_id:
        return ""
    return raw_id.replace(" 00:00:00", "").strip()


# ═══════════════════════════════════════════════════════════
# 429 / BROKEN-RECORD HANDLING  (merged from fix_finetune_dataset.py)
# ═══════════════════════════════════════════════════════════

#
# If the teacher LLM (especially Ollama) returns a 429 / API-error
# instead of a real analysis, we still write the record so we don't
# lose the (expensive) user prompt.  `is_broken()` later flags those
# records; `--fix-broken-only` re-submits them one-by-one with live
# saves so progress is never lost.  After fixing, `rebuild_splits()`
# regenerates train/val JSON files from the clean + substantial
# records only.

def is_broken(rec):
    """True if the assistant message contains an Ollama 429 / API error."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if ("429 Client Error" in content
                    or "[OLLAMA] API error" in content
                    or "429 Too Many Requests" in content
                    or "API error" in content[:120]):  # short prefixes only
                return True
    return False


def save_records(records, path):
    """Atomic full-rewrite of the JSONL file (used by fix-broken mode)."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic on Windows when on same volume


def get_user_prompt(rec):
    """Extract the user message from a record."""
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            return m["content"]
    return ""


def set_assistant_content(rec, new_content):
    """Update the assistant message in a record in-place."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            m["content"] = new_content
            return


def rebuild_splits(records, output_dir=None):
    """
    Regenerate train/val JSON files from clean + substantial records only.
    Reuses the same file names as the old fix script (football_train.json /
    football_val.json) so downstream training scripts don't need changes.
    """
    output_dir = output_dir or OUTPUT_DIR
    clean = [r for r in records if not is_broken(r)]
    # Filter: assistant response must be substantial (>100 chars)
    valid = []
    for r in clean:
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and len(m.get("content", "")) > 100:
                valid.append(r)
                break

    if len(valid) < 10:
        print(f"  ⚠ Only {len(valid)} valid records, skipping split (need ≥10).")
        return 0, 0

    train, val = train_test_split(valid, test_size=0.15, random_state=42)
    train_path = output_dir / "football_train.json"
    val_path   = output_dir / "football_val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print(f"  💾 football_train.json: {len(train)} records")
    print(f"  💾 football_val.json:   {len(val)} records")
    return len(train), len(val)


def _looks_like_api_error(text):
    """Lightweight detector: returns True if `text` looks like an LLM API error rather than real analysis."""
    if not text:
        return True
    text = text.strip()
    if len(text) < 50:
        return True
    # Common Ollama / OpenAI / Gemini error prefixes (case-insensitive)
    lower = text[:200].lower()
    if ("429" in lower
            or "rate limit" in lower
            or "too many requests" in lower
            or "api error" in lower
            or "internal server error" in lower
            or "service unavailable" in lower
            or "ollama" in lower and "error" in lower):
        return True
    return False


# ═══════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════

def load_model_and_graph(use_tea_gnn=True):
    """
    Load the best tuned GNN model and the full graph.

    TEA-GNN (ranked #1 in tuned comparison: 60.96% accuracy, 0.577 F1)
    is the default since it's the strongest model. Pass use_tea_gnn=False
    to fall back to EdgeConv.
    """
    if use_tea_gnn:
        model_path = BASE_DIR / "models" / "saved" / "gnn_tea-gnn_tuned.pt"
        if not model_path.exists():
            model_path = BASE_DIR / "models" / "saved" / "gnn_tea-gnn.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No TEA-GNN checkpoint found. Run train_gnn.py / tune_gnn.py first."
                )
    else:
        model_path = BASE_DIR / "models" / "saved" / "gnn_edgeconv_tuned.pt"
        if not model_path.exists():
            model_path = BASE_DIR / "models" / "saved" / "gnn_edgeconv.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No EdgeConv checkpoint found. Run train_gnn.py / tune_gnn.py first."
                )

    builder = FootballGraphBuilder(data_path=str(DATA_PATH))
    graph   = builder.build_train_test_graphs()

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    bp = checkpoint.get('best_params', {})
    model_name = checkpoint.get('model_name', 'TEA-GNN' if use_tea_gnn else 'EdgeConv')
    is_tea_gnn = ('TEA-GNN' in model_name) or use_tea_gnn

    nf = graph['num_node_features']
    ef = graph['num_edge_features']
    kwargs = dict(hidden_dim=bp.get('hidden_dim', 64),
                  dropout=bp.get('dropout', 0.3))
    if is_tea_gnn:
        kwargs['heads'] = bp.get('heads', 4)
        kwargs['num_leagues'] = graph.get('num_leagues', 5)
        kwargs['use_cross_league'] = bp.get('use_cross_league', True)

    model = get_model(model_name, nf, ef, **kwargs)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(DEVICE).eval()

    graph['is_tea_gnn'] = is_tea_gnn
    print(f"✓ Loaded {model_name} (nf={graph['num_node_features']}, "
          f"ef={graph['num_edge_features']}) from {model_path.name}")
    print(f"  TEA-GNN: {is_tea_gnn} | Use cross-league: {bp.get('use_cross_league', True)}")

    return model, graph, builder


# ═══════════════════════════════════════════════════════════
# PER-MATCH PROCESSING
# ═══════════════════════════════════════════════════════════

def _build_stats_block(row, side='Home'):
    """Build a dict of {label: value} for one team's stats (rolling + cumulative + context)."""
    stats = {}
    for suffix, label in FEATURE_LABELS.items():
        col = f'{side}{suffix}'
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            stats[label] = round(val, 2)
    return stats


def _build_env_block(row):
    """Extract weather + stadium info as a readable dict."""
    env = {}
    # Weather
    for col, label in [('temperature', 'Temperature (C)'),
                       ('precipitation', 'Precipitation (mm)'),
                       ('rain', 'Rain (mm)'),
                       ('wind_speed', 'Wind speed (km/h)'),
                       ('humidity', 'Humidity (%)')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            env[label] = round(val, 2)
    # Stadium name only (no coordinates — not useful for tactical analysis)
    stadium = row.get('stadium_name', None)
    if stadium and not pd.isna(stadium) and isinstance(stadium, str):
        env['stadium'] = stadium
    return env


def _build_match_info_block(row):
    """Extract kickoff time + referee data (pre-match, no leakage)."""
    info = {}
    # Kickoff time (only include if it is a real, non-zero time)
    time_val = row.get('Time', None)
    if time_val is not None and not pd.isna(time_val):
        if isinstance(time_val, (int, float)):
            hour = int(float(time_val))
            minute = int((float(time_val) - hour) * 60)
            if hour > 0 or minute > 0:
                info['kickoff_time'] = f'{hour:02d}:{minute:02d}'
        else:
            ts = str(time_val).strip()
            if ts and ts not in ('0', '00:00', '00:00:00', '0.0', 'nan', 'None'):
                info['kickoff_time'] = ts
    # Referee profile
    ref = {}
    for col, label in [('Ref_AvgYellows', 'avg_yellows_per_match'),
                       ('Ref_AvgReds',    'avg_reds_per_match'),
                       ('Ref_Strictness', 'strictness_score')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            ref[label] = round(val, 3)
    if ref:
        info['referee_profile'] = ref
    return info



def build_user_prompt(row: pd.Series, builder, graph, model, edge_idx: int,
                      pred: str, probs: dict, gnn_explanation: dict) -> str:
    """
    Build the human-readable user prompt from raw match data + GNN output.
    This is what the fine-tuned SLM will receive at inference time.
    """
    home = row['HomeTeam']
    away = row['AwayTeam']

    home_stats = _build_stats_block(row, 'Home')
    away_stats = _build_stats_block(row, 'Away')
    env_info   = _build_env_block(row)
    match_info = _build_match_info_block(row)

    # ── Match date (always available; used in addition to optional kickoff_time) ──
    date_val = row.get('Date', None)
    if date_val is not None and not pd.isna(date_val):
        # Pandas Timestamp or datetime → ISO date string
        if hasattr(date_val, 'strftime'):
            match_date = date_val.strftime('%Y-%m-%d')
        else:
            match_date = str(date_val)[:10]
    else:
        match_date = None

    # ── Head-to-head ──
    h2h_parts = {}
    for col, label in [('H2H_Matches', 'Total meetings'), ('H2H_HomeWins', 'Home wins'),
                        ('H2H_AwayWins', 'Away wins'), ('H2H_Draws', 'Draws')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            h2h_parts[label] = int(val)
    # H2H goals
    hg = row.get('H2H_HomeGoals', None)
    ag = row.get('H2H_AwayGoals', None)
    if hg is not None and not pd.isna(hg):
        h2h_parts['Total home goals (H2H)'] = round(float(hg), 1)
    if ag is not None and not pd.isna(ag):
        h2h_parts['Total away goals (H2H)'] = round(float(ag), 1)

    # ── Key historical matches from GNN ──
    hist_lines = []
    for m in gnn_explanation.get('top_influencing_matches', [])[:3]:
        hist_lines.append(m['match'])

    payload = {
        "match": {
            "home_team": home,
            "away_team": away,
            "league": str(row.get('League', '')),
            "season": int(row.get('Season', 0)),
            "match_date": match_date,
        },
        "statistics": {
            "home_form": home_stats,
            "away_form": away_stats,
            "head_to_head": h2h_parts,
        },
        "environment": {
            "matchday_conditions": env_info,
        },
        "match_context": match_info,
        "neural_network_prediction": {
            "model": "TEA-GNN",
            "outcome": pred,
            "probabilities": {
                "home_win": round(probs['H'], 3),
                "draw": round(probs['D'], 3),
                "away_win": round(probs['A'], 3)
            },
            "influential_historical_matches": hist_lines,
        }
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def get_edge_idx_for_match(graph, builder, home_team, away_team):
    """Find the graph edge index for a specific home/away pair."""
    ei = graph['edge_index']
    hi = builder.team_to_idx.get(home_team)
    ai = builder.team_to_idx.get(away_team)
    if hi is None or ai is None:
        return None
    mask = (ei[0] == hi) & (ei[1] == ai)
    edges = mask.nonzero(as_tuple=True)[0]
    if len(edges) == 0:
        return None
    return edges[-1].item()


def run_gnn_prediction(model, graph, edge_idx):
    """Get prediction + probabilities for a single edge.
    Handles both standard models (GCN/SAGE/GAT/GIN/EdgeConv) and TEA-GNN."""
    x  = graph.get('x_test', graph['x']).to(DEVICE)
    ei = graph['edge_index'].to(DEVICE)
    ea = graph['edge_attr'].to(DEVICE)
    is_tea_gnn = graph.get('is_tea_gnn', False)

    with torch.no_grad():
        if is_tea_gnn:
            edge_time = graph.get('edge_time')
            league_id = graph.get('league_id')
            if edge_time is not None:
                edge_time = edge_time.to(DEVICE)
            if league_id is not None:
                league_id = league_id.to(DEVICE)
            out = model(x, ei, ea, edge_time=edge_time, league_id=league_id)
        else:
            out = model(x, ei, ea)

    logits = out[edge_idx]
    probs  = F.softmax(logits, dim=0).cpu().numpy()
    pred   = logits.argmax().item()

    class_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
    return class_map[pred], {'A': float(probs[0]), 'D': float(probs[1]), 'H': float(probs[2])}


def run_gnn_explainer(model, graph, builder, edge_idx):
    """Extract structural explanation for one edge (lightweight version)."""
    from torch_geometric.explain import Explainer, GNNExplainer

    x  = graph.get('x_test', graph['x']).to(DEVICE)
    ei = graph['edge_index'].to(DEVICE)
    ea = graph['edge_attr'].to(DEVICE)
    is_tea_gnn = graph.get('is_tea_gnn', False)

    # TEA-GNN passes extra args — the Explainer wraps the model, so
    # we need a wrapper that matches the Explainer's expected signature.
    if is_tea_gnn:
        edge_time = graph.get('edge_time')
        league_id = graph.get('league_id')
        if edge_time is not None:
            edge_time = edge_time.to(DEVICE)
        if league_id is not None:
            league_id = league_id.to(DEVICE)

        class TeagnnWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
            def forward(self, x, edge_index, edge_attr=None):
                return self.base(x, edge_index, edge_attr,
                                 edge_time=edge_time, league_id=league_id)

        explain_model = TeagnnWrapper(model)
    else:
        explain_model = model

    explainer = Explainer(
        model=explain_model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='edge', return_type='raw'),
    )

    explanation = explainer(x, ei, edge_attr=ea, index=edge_idx)

    # Node feature importance
    node_mask = explanation.node_mask.cpu()
    home_idx  = ei[0, edge_idx].item()
    away_idx  = ei[1, edge_idx].item()
    home_imp  = node_mask[home_idx].numpy()
    away_imp  = node_mask[away_idx].numpy()

    feat_names = builder.NODE_FEATURE_SUFFIXES
    home_team  = builder.idx_to_team[home_idx]
    away_team  = builder.idx_to_team[away_idx]

    top_home = {f"{home_team}_{feat_names[i]}": float(home_imp[i])
                for i in np.argsort(home_imp)[-3:][::-1]}
    top_away = {f"{away_team}_{feat_names[i]}": float(away_imp[i])
                for i in np.argsort(away_imp)[-3:][::-1]}

    # Edge importance
    edge_mask = explanation.edge_mask.cpu().numpy()
    top_edges = np.argsort(edge_mask)[-6:][::-1]
    top_matches = []
    for e in top_edges:
        if int(e) == int(edge_idx):
            continue
        h = builder.idx_to_team[ei[0, e].item()]
        a = builder.idx_to_team[ei[1, e].item()]
        top_matches.append({"match": f"{h} vs {a}", "influence_score": round(float(edge_mask[e]), 6)})
        if len(top_matches) == 3:
            break

    return {
        'top_node_features':       {'home_team': top_home, 'away_team': top_away},
        'top_influencing_matches': top_matches,
    }


# ═══════════════════════════════════════════════════════════
# FIX-BROKEN MODE  (merged from fix_finetune_dataset.py)
# ═══════════════════════════════════════════════════════════

def _fix_broken_mode(output_path, provider_name, model_name):
    """
    Scan existing JSONL, find broken (429/API-error) records,
    re-generate their assistant responses one-by-one with live saves.
    After fixing, rebuild train/val splits from clean + substantial records.
    Safe to Ctrl+C anytime — progress is never lost.
    """
    if not output_path.exists():
        print(f"❌ Cannot find {output_path}")
        return

    print("=" * 70)
    print("  FIX-BROKEN MODE")
    print(f"  File:      {output_path}")
    print(f"  Provider:  {provider_name}")
    print(f"{'=' * 70}")

    # Load LLM
    teacher = get_llm_provider(provider_type=provider_name, model_name=model_name)
    print(f"✓ Teacher LLM ready: {provider_name}")

    # Load all records
    records = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("⚠ File is empty — nothing to fix.")
        return

    # Find broken ones
    bad_indices = [i for i, r in enumerate(records) if is_broken(r)]
    total = len(records)
    good = total - len(bad_indices)

    if not bad_indices:
        print(f"✅ No broken records! Dataset is clean ({good} records).")
        print("\n🔄 Rebuilding train/val splits...")
        rebuild_splits(records, OUTPUT_DIR)
        print("🚀 DONE!")
        return

    print(f"📊 Total: {total} | ✅ Good: {good} | ❌ Broken: {len(bad_indices)}")
    print(f"⏳ Fixing one-by-one with live saves (safe to Ctrl+C anytime)...\n")

    fixed_count = 0
    for num, idx in enumerate(bad_indices, 1):
        rec = records[idx]
        match_id = rec.get("match_id", "Unknown")
        prompt = get_user_prompt(rec)
        if not prompt:
            print(f"  [{num}/{len(bad_indices)}] ⏭ No user prompt, skipping {match_id}")
            continue

        try:
            print(f"  [{num}/{len(bad_indices)}] 🔄 {match_id}...", end=" ", flush=True)
            response = teacher.generate_explanation(
                {'home_team': 'Fix', 'away_team': 'Mode', 'prediction': 'N/A', 'probabilities': {}},
                {}
            )

            if _looks_like_api_error(response):
                print(f"❌ Still failing (API error), will retry next run")
                time.sleep(5)
                continue

            set_assistant_content(rec, response.strip())
            fixed_count += 1

            # SAVE immediately so progress is never lost
            save_records(records, output_path)
            print(f"✅ saved")

            time.sleep(1.5)  # rate limit

        except KeyboardInterrupt:
            print(f"\n\n🛑 Interrupted! {fixed_count} records fixed so far.")
            print(f"   Run --fix-broken-only again to resume.\n")
            save_records(records, output_path)
            return
        except Exception as e:
            print(f"❌ {e}")
            time.sleep(5)

    # Final summary
    remaining = sum(1 for r in records if is_broken(r))
    print(f"\n{'=' * 70}")
    print(f"  ✅ Fixed {fixed_count} records this run")
    print(f"  📊 Remaining broken: {remaining}")
    print(f"{'=' * 70}")

    if remaining == 0:
        print("\n🎉 ALL records are now clean! Rebuilding train/val splits...")
        rebuild_splits(records, OUTPUT_DIR)
        print("🚀 DONE!")
    else:
        print(f"\n⚠ {remaining} records still broken. Run --fix-broken-only again to retry them.")



# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning dataset for football SLM")
    parser.add_argument('--provider',     type=str, default='gemini',
                        help="Teacher LLM provider (default: gemini)")
    parser.add_argument('--model-name',   type=str, default='',
                        help="Teacher LLM model name (default: from config)")
    parser.add_argument('--max-samples',  type=int, default=0,
                        help="Max samples to generate (0 = all)")
    parser.add_argument('--dry-run',      action='store_true',
                        help="Print prompts without calling teacher LLM")
    parser.add_argument('--resume',       action='store_true',
                        help="Skip matches already in the output file")
    parser.add_argument('--output',       type=str, default='football_tactical.jsonl',
                        help="Output filename inside data/finetune/")
    parser.add_argument('--skip-explainer', action='store_true',
                        help="Skip GNNExplainer (faster, less rich prompts)")
    parser.add_argument('--use-edgeconv', action='store_true',
                        help="Use EdgeConv instead of TEA-GNN (fallback)")
    parser.add_argument('--test-only',    action='store_true',
                        help="Process only test season (2425) instead of all 9 seasons")
    parser.add_argument('--fix-broken-only', action='store_true',
                        help="Scan existing JSONL, fix broken (429/API-error) records one-by-one with live saves")
    args = parser.parse_args()

    output_path = OUTPUT_DIR / args.output

    # ── Fix-broken-only mode ──
    if args.fix_broken_only:
        _fix_broken_mode(output_path, args.provider, args.model_name)
        return  # done

    print("=" * 70)
    print("  FINE-TUNING DATASET BUILDER")
    print(f"  Provider:  {args.provider}")
    print(f"  Output:    {output_path}")
    print(f"  Dry run:   {args.dry_run}")
    print(f"  Seasons:   {'Test only (2425)' if args.test_only else 'All 9 seasons'}")
    print(f"  Device:    {DEVICE}")
    print("=" * 70)

    # ── Load model & graph ──
    model, graph, builder = load_model_and_graph(use_tea_gnn=not args.use_edgeconv)
    df = builder.df.copy()

    # ── Select matches to process ──
    if args.test_only:
        process_seasons = [2425]
        print("\n✓ TEST-ONLY mode: processing season 2425 only")
    else:
        process_seasons = [1516, 1617, 1718, 1819, 1920, 2021, 2122, 2223, 2324, 2425]
        print(f"\n✓ ALL-SEASONS mode: processing seasons {process_seasons}")

    process_df = df[df['Season'].isin(process_seasons)].copy()
    print(f"✓ Matches to process: {len(process_df)}")

    # ── Resume support ──
    existing_keys = set()
    if args.resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    mid = rec.get('match_id', '')
                    if mid:
                        existing_keys.add(clean_match_id(mid))
                except json.JSONDecodeError:
                    pass
        print(f"✓ Resuming: {len(existing_keys)} already generated")

    # ── Teacher LLM ──
    teacher = None
    if not args.dry_run:
        teacher = get_llm_provider(provider_type=args.provider, model_name=args.model_name)
        print(f"✓ Teacher LLM ready: {args.provider}")

    # ── Process matches ──
    samples_written = 0
    errors = 0
    broken_count = 0
    max_s = args.max_samples if args.max_samples > 0 else len(process_df)

    with open(output_path, 'a', encoding='utf-8') as out_f:
        for idx, (_, row) in enumerate(tqdm(process_df.iterrows(), total=min(len(process_df), max_s),
                                            desc="Building dataset")):
            if samples_written >= max_s:
                break

            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            date_str  = format_match_date(row.get('Date', ''))
            match_id  = f"{date_str}_{home_team}_vs_{away_team}"

            if clean_match_id(match_id) in existing_keys:
                continue


            # Find edge in graph
            edge_idx = get_edge_idx_for_match(graph, builder, home_team, away_team)
            if edge_idx is None:
                errors += 1
                continue

            # GNN prediction
            try:
                pred, probs = run_gnn_prediction(model, graph, edge_idx)
            except Exception as e:
                errors += 1
                continue

            # GNN explanation (optional)
            gnn_exp = {'top_node_features': {}, 'top_influencing_matches': []}
            if not args.skip_explainer:
                try:
                    gnn_exp = run_gnn_explainer(model, graph, builder, edge_idx)
                except Exception:
                    pass  # Proceed without explanation

            # ── Data validity: warn if any key stat is NaN (xG should never be NaN/0) ──
            # cum_* = 0 is correct for first match of a team-season, not a bug.
            nan_warnings = []
            # Check rolling xG (these should NEVER be NaN or 0 once a team has 5 matches)
            for side in ('Home', 'Away'):
                for stat in ('xG_5', 'xGA_5'):
                    val = row.get(f'{side}{stat}')
                    if val is None or (not isinstance(val, str) and pd.isna(val)):
                        nan_warnings.append(f'{side}{stat}')
                    elif isinstance(val, (int, float)) and float(val) == 0.0:
                        nan_warnings.append(f'{side}{stat}=0')
            if nan_warnings:
                # Only warn; do not skip. The teacher LLM can still work with it.
                if not args.dry_run:
                    tqdm.write(f"  ⚠ [{match_id}] suspicious stats: {nan_warnings}")

            # Build the user prompt
            user_prompt = build_user_prompt(row, builder, graph, model, edge_idx,
                                            pred, probs, gnn_exp)

            if args.dry_run:
                print(f"\n{'─' * 70}")
                print(f"MATCH: {home_team} vs {away_team}")
                print(f"{'─' * 70}")
                print(f"USER PROMPT:\n{user_prompt}")
                print(f"\n[Would call {args.provider} here for the assistant response]")
                samples_written += 1
                continue

            # Call teacher LLM
            match_context = {
                'home_team': home_team, 'away_team': away_team,
                'prediction': pred, 'probabilities': probs,
            }
            try:
                assistant_response = teacher.generate_explanation(match_context, gnn_exp)
            except Exception as e:
                print(f"  ✗ LLM error for {match_id}: {e}")
                errors += 1
                continue

            # ── Inline 429 / API-error detection ──
            # If the teacher returned an error instead of real analysis,
            # still write the record (preserving the user prompt) but mark
            # it as broken so `--fix-broken-only` can pick it up later.
            if _looks_like_api_error(assistant_response):
                tqdm.write(f"  ⚠ [{match_id}] teacher returned API error — writing broken record")
                broken_record = {
                    "match_id": match_id,
                    "actual_result": row.get('FTR', ''),
                    "gnn_prediction": pred,
                    "messages": [
                        {"role": "user",      "content": user_prompt},
                        {"role": "assistant", "content": "[API error] " + assistant_response.strip()[:200]},
                    ],
                }
                out_f.write(json.dumps(broken_record, ensure_ascii=False) + "\n")
                out_f.flush()
                broken_count += 1
                time.sleep(5)  # back off after an API error
                continue

            if not assistant_response or len(assistant_response.strip()) < 50:
                errors += 1
                continue

            # Build the JSONL record
            record = {
                "match_id": match_id,
                "actual_result": row.get('FTR', ''),
                "gnn_prediction": pred,
                "messages": [
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": assistant_response.strip()},
                ],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            samples_written += 1

            # Rate limiting (avoid API throttling)
            time.sleep(0.5)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  DATASET GENERATION COMPLETE")
    print(f"  Samples written:  {samples_written}")
    print(f"  Broken (API err): {broken_count}")
    print(f"  Errors/skipped:   {errors}")
    print(f"  Output file:      {output_path}")
    print(f"{'=' * 70}")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")

    # ── Rebuild train/val splits from clean + substantial records ──
    if not args.dry_run and output_path.exists():
        all_records = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if all_records:
            clean_count = sum(1 for r in all_records if not is_broken(r))
            print(f"\n  🔄 Rebuilding train/val splits from {len(all_records)} records ({clean_count} clean)...")
            train_n, val_n = rebuild_splits(all_records, OUTPUT_DIR)
            if train_n + val_n > 0:
                print(f"  ✅ Splits ready for fine-tuning")
            else:
                print(f"  ⚠ No valid records to split — run --fix-broken-only first if there are API errors")


if __name__ == '__main__':
    main()