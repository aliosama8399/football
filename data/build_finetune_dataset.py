"""
Fine-Tuning Dataset Builder
============================
Generates a JSONL dataset for fine-tuning an SLM on football tactical analysis.

Pipeline per match:
  1. Read pre-match stats from processed_matches.csv
  2. Run TEA-GNN prediction → outcome + probabilities
  3. Run GNNExplainer → top node features + influential edges
  4. Call a "teacher" LLM to write the analysis
  5. Save (user_prompt, assistant_response) as a JSONL row
  6. Perform a strictly chronological, golden-set-excluded train/val split
"""

import sys, os
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
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

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model
from models.llm_providers import get_llm_provider

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "processed" / "processed_matches.csv"
OUTPUT_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def is_broken(rec):
    """True if the assistant message contains an Ollama 429 / API error."""
    for m in rec.get("messages", []):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if ("429 Client Error" in content
                    or "[OLLAMA] API error" in content
                    or "429 Too Many Requests" in content
                    or "API error" in content[:120]):
                return True
    return False


def save_records(records, path):
    """Atomic full-rewrite of the JSONL file (used by fix-broken mode)."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


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


def _load_golden_exclusion_keys() -> set[str]:
    """Load match_id / team-pair keys from eval/golden/rag_golden_set.json to prevent train/eval overlap."""
    golden_path = BASE_DIR / "eval" / "golden" / "rag_golden_set.json"
    excluded = set()
    if not golden_path.exists():
        return excluded
    try:
        with open(golden_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, list):
            for item in items:
                mid = item.get("match_id")
                if mid:
                    excluded.add(clean_match_id(str(mid)))
                ht = item.get("home_team")
                at = item.get("away_team")
                dt = item.get("match_date") or item.get("date")
                if ht and at and dt:
                    excluded.add(f"{str(dt)[:10]}_{ht}_vs_{at}")
    except Exception:
        pass
    return excluded


def _extract_record_date_sort_key(rec: dict) -> str:
    """Extract YYYY-MM-DD date prefix from match_id or embedded JSON user prompt for chronological sorting."""
    mid = clean_match_id(rec.get("match_id", ""))
    if len(mid) >= 10 and mid[4:5] == "-" and mid[7:8] == "-":
        return mid
    user_txt = get_user_prompt(rec)
    if user_txt:
        try:
            obj = json.loads(user_txt)
            md = obj.get("match", {}).get("match_date", "")
            if md:
                return f"{md}_{mid}"
        except Exception:
            pass
    return mid


def rebuild_splits(records, output_dir=None):
    """
    Regenerate train/val JSON files from clean + substantial records using a
    strictly CHRONOLOGICAL split (oldest 85% -> train, newest 15% -> val)
    while excluding any records overlapping with eval/golden/rag_golden_set.json.
    """
    output_dir = output_dir or OUTPUT_DIR
    golden_excluded = _load_golden_exclusion_keys()
    clean = [r for r in records if not is_broken(r)]

    valid = []
    excluded_count = 0
    for r in clean:
        mid = clean_match_id(r.get("match_id", ""))
        if mid and mid in golden_excluded:
            excluded_count += 1
            continue
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and len(m.get("content", "")) > 100:
                valid.append(r)
                break

    if excluded_count > 0:
        print(f"  🔒 Excluded {excluded_count} golden evaluation set records from fine-tuning splits.")

    if len(valid) < 10:
        print(f"  ⚠ Only {len(valid)} valid records, skipping split (need ≥10).")
        return 0, 0

    # Strictly chronological ordering by match date (prevents future temporal leakage into training)
    valid_sorted = sorted(valid, key=_extract_record_date_sort_key)
    split_idx = int(len(valid_sorted) * 0.85)
    train = valid_sorted[:split_idx]
    val = valid_sorted[split_idx:]

    train_path = output_dir / "football_train.json"
    val_path = output_dir / "football_val.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)
    print(f"  💾 Chronological football_train.json: {len(train)} records")
    print(f"  💾 Chronological football_val.json:   {len(val)} records")

    models_dir = BASE_DIR / "models"
    if models_dir.exists():
        try:
            with open(models_dir / "football_train.json", "w", encoding="utf-8") as f:
                json.dump(train, f, ensure_ascii=False, indent=2)
            with open(models_dir / "football_val.json", "w", encoding="utf-8") as f:
                json.dump(val, f, ensure_ascii=False, indent=2)
            print(f"  💾 Mirrored chronological splits to models/football_train.json & models/football_val.json")
        except Exception as e:
            print(f"  [Warn] Failed mirroring splits to models/: {e}")

    return len(train), len(val)


def _looks_like_api_error(text):
    """Lightweight detector: returns True if `text` looks like an LLM API error rather than real analysis."""
    if not text:
        return True
    text = text.strip()
    if len(text) < 50:
        return True
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


def load_model_and_graph(use_tea_gnn=True):
    """Load the best tuned GNN model and the full graph."""
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
        state_keys = checkpoint.get('model_state', {}).keys()
        has_cross = any('cross_league.' in k for k in state_keys)
        kwargs['heads'] = bp.get('heads', 4)
        kwargs['num_leagues'] = graph.get('num_leagues', 5)
        kwargs['use_cross_league'] = bp.get('use_cross_league', has_cross)

    model = get_model(model_name, nf, ef, **kwargs)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model = model.to(DEVICE).eval()

    graph['is_tea_gnn'] = is_tea_gnn
    print(f"[OK] Loaded {model_name} (nf={graph['num_node_features']}, "
          f"ef={graph['num_edge_features']}) from {model_path.name}")
    print(f"  TEA-GNN: {is_tea_gnn} | Use cross-league: {bp.get('use_cross_league', True)}")

    return model, graph, builder


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
    for col, label in [('temperature', 'Temperature (C)'),
                       ('precipitation', 'Precipitation (mm)'),
                       ('rain', 'Rain (mm)'),
                       ('wind_speed', 'Wind speed (km/h)'),
                       ('humidity', 'Humidity (%)')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            env[label] = round(val, 2)
    stadium = row.get('stadium_name', None)
    if stadium and not pd.isna(stadium) and isinstance(stadium, str):
        env['stadium'] = stadium
    return env


def _build_match_info_block(row):
    """Extract kickoff time + referee data (pre-match, no leakage)."""
    info = {}
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
    """Build the human-readable user prompt from raw match data + GNN output."""
    home = row['HomeTeam']
    away = row['AwayTeam']

    home_stats = _build_stats_block(row, 'Home')
    away_stats = _build_stats_block(row, 'Away')
    env_info   = _build_env_block(row)
    match_info = _build_match_info_block(row)

    date_val = row.get('Date', None)
    if date_val is not None and not pd.isna(date_val):
        if hasattr(date_val, 'strftime'):
            match_date = date_val.strftime('%Y-%m-%d')
        else:
            match_date = str(date_val)[:10]
    else:
        match_date = None

    h2h_parts = {}
    for col, label in [('H2H_Matches', 'Total meetings'), ('H2H_HomeWins', 'Home wins'),
                        ('H2H_AwayWins', 'Away wins'), ('H2H_Draws', 'Draws')]:
        val = row.get(col, None)
        if val is not None and not pd.isna(val):
            h2h_parts[label] = int(val)
    hg = row.get('H2H_HomeGoals', None)
    ag = row.get('H2H_AwayGoals', None)
    if hg is not None and not pd.isna(hg):
        h2h_parts['Total home goals (H2H)'] = round(float(hg), 1)
    if ag is not None and not pd.isna(ag):
        h2h_parts['Total away goals (H2H)'] = round(float(ag), 1)

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
    """Get prediction + probabilities for a single edge."""
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


def _fix_broken_mode(output_path, provider_name, model_name):
    """Scan existing JSONL, find broken (429/API-error) records, and regenerate them."""
    if not output_path.exists():
        print(f"❌ Cannot find {output_path}")
        return

    print("=" * 70)
    print("  FIX-BROKEN MODE")
    print(f"  File:      {output_path}")
    print(f"  Provider:  {provider_name}")
    print(f"{'=' * 70}")

    teacher = get_llm_provider(provider_type=provider_name, model_name=model_name)
    print(f"✓ Teacher LLM ready: {provider_name}")

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

            save_records(records, output_path)
            print(f"✅ saved")

            time.sleep(1.5)

        except KeyboardInterrupt:
            print(f"\n\n🛑 Interrupted! {fixed_count} records fixed so far.")
            print(f"   Run --fix-broken-only again to resume.\n")
            save_records(records, output_path)
            return
        except Exception as e:
            print(f"❌ {e}")
            time.sleep(5)

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
    parser.add_argument('--seasons',      nargs='+', type=int, default=None,
                        help="Specific season(s) to process (e.g. --seasons 2526 2627)")
    parser.add_argument('--incremental',  action='store_true', default=True,
                        help="Skip matches already generated in output file (default: True)")
    parser.add_argument('--resume',       action='store_true',
                        help="Skip matches already in the output file")
    parser.add_argument('--output',       type=str, default='football_tactical.jsonl',
                        help="Output filename inside data/finetune/")
    parser.add_argument('--skip-explainer', action='store_true',
                        help="Skip GNNExplainer (faster, less rich prompts)")
    parser.add_argument('--use-edgeconv', action='store_true',
                        help="Use EdgeConv instead of TEA-GNN (fallback)")
    parser.add_argument('--test-only',    action='store_true',
                        help="Process only test season instead of all seasons")
    parser.add_argument('--fix-broken-only', action='store_true',
                        help="Scan existing JSONL, fix broken (429/API-error) records one-by-one with live saves")
    args = parser.parse_args()

    output_path = OUTPUT_DIR / args.output

    if args.fix_broken_only:
        _fix_broken_mode(output_path, args.provider, args.model_name)
        return

    print("=" * 70)
    print("  FINE-TUNING DATASET BUILDER")
    print(f"  Provider:  {args.provider}")
    print(f"  Output:    {output_path}")
    print(f"  Dry run:   {args.dry_run}")
    print(f"  Device:    {DEVICE}")
    print("=" * 70)

    model, graph, builder = load_model_and_graph(use_tea_gnn=not args.use_edgeconv)
    df = builder.df.copy()

    from data._config import load_config
    cfg = load_config()
    cfg_seasons = [int(s) for s in cfg.get('seasons', [])]

    if args.seasons:
        process_seasons = args.seasons
        print(f"\n[OK] CUSTOM-SEASONS mode: processing seasons {process_seasons}")
    elif args.test_only:
        latest_s = cfg_seasons[-1] if cfg_seasons else 2627
        process_seasons = [latest_s]
        print(f"\n[OK] TEST-ONLY mode: processing latest season {latest_s} only")
    else:
        process_seasons = cfg_seasons
        print(f"\n[OK] ALL-SEASONS mode: processing seasons {process_seasons}")

    process_df = df[df['Season'].isin(process_seasons)].copy()
    print(f"[OK] Matches in selected seasons: {len(process_df)}")

    existing_keys = set()
    if (args.resume or args.incremental) and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    mid = rec.get('match_id', '')
                    if mid:
                        existing_keys.add(clean_match_id(mid))
                except json.JSONDecodeError:
                    pass
        print(f"[OK] Incremental: {len(existing_keys)} already annotated matches found in {output_path.name}")

    teacher = None
    if not args.dry_run:
        teacher = get_llm_provider(provider_type=args.provider, model_name=args.model_name)
        print(f"[OK] Teacher LLM ready: {args.provider}")

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

            edge_idx = get_edge_idx_for_match(graph, builder, home_team, away_team)
            if edge_idx is None:
                errors += 1
                continue

            try:
                pred, probs = run_gnn_prediction(model, graph, edge_idx)
            except Exception:
                errors += 1
                continue

            gnn_exp = {'top_node_features': {}, 'top_influencing_matches': []}
            if not args.skip_explainer:
                try:
                    gnn_exp = run_gnn_explainer(model, graph, builder, edge_idx)
                except Exception:
                    pass

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
                time.sleep(5)
                continue

            if not assistant_response or len(assistant_response.strip()) < 50:
                errors += 1
                continue

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
            time.sleep(0.5)

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
            print(f"\n  🔄 Rebuilding chronological train/val splits from {len(all_records)} records ({clean_count} clean)...")
            train_n, val_n = rebuild_splits(all_records, OUTPUT_DIR)
            if train_n + val_n > 0:
                print(f"  ✅ Splits ready for fine-tuning")
            else:
                print(f"  ⚠ No valid records to split — run --fix-broken-only first if there are API errors")


if __name__ == '__main__':
    main()
