"""
GNN Hyperparameter Tuning with Optuna
======================================
Tunes all 7 GNN architectures using Bayesian optimization.
Each model gets its own search space tailored to its architecture.

Models tuned: GCN, GraphSAGE, GAT, GIN, EdgeConv, Hybrid, TEA-GNN.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import optuna
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix,
                             roc_auc_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data.graph_builder import FootballGraphBuilder
from models.gnn_models import get_model
from models.plotting import (
    ensure_per_model_dir, plot_all_per_model,
    plot_per_class_f1_comparison, write_master_summary,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "models" / "results"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 40
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════

def ranked_probability_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    rps = 0.0
    for i in range(len(y_true)):
        cum_p, cum_t, rps_m = 0.0, 0.0, 0.0
        for j in range(n_classes - 1):
            cum_p += y_prob[i, j]
            cum_t += 1.0 if y_true[i] <= j else 0.0
            rps_m += (cum_p - cum_t) ** 2
        rps += rps_m / (n_classes - 1)
    return rps / len(y_true)


# ═══════════════════════════════════════════════════════════
# TRAINING FUNCTION (used by Optuna objectives)
# ═══════════════════════════════════════════════════════════

def train_and_evaluate(model, graph_data, lr, weight_decay, epochs=250, patience=30,
                       is_hybrid=False, is_tea_gnn=False):
    """Train model and return test accuracy. Used by Optuna objectives.
    
    Dispatches on (is_hybrid, is_tea_gnn):
      - Hybrid  : forward(x, ei, ea, tabular_features=tabular)
      - TEA-GNN : forward(x, ei, ea, edge_time=edge_time, league_id=league_id)
      - others  : forward(x, ei, ea)
    """
    
    x = graph_data['x'].to(DEVICE)
    ei = graph_data['edge_index'].to(DEVICE)
    ea = graph_data['edge_attr'].to(DEVICE)
    ey = graph_data['edge_y'].to(DEVICE)
    train_mask = graph_data['train_mask'].to(DEVICE)
    test_mask = graph_data['test_mask'].to(DEVICE)
    tabular = graph_data['tabular_features'].to(DEVICE) if is_hybrid else None
    edge_time = graph_data.get('edge_time')
    if edge_time is not None:
        edge_time = edge_time.to(DEVICE)
    league_id = graph_data.get('league_id')
    if league_id is not None:
        league_id = league_id.to(DEVICE)
    
    model = model.to(DEVICE)
    
    def _forward(m):
        if is_hybrid:
            return m(x, ei, ea, tabular_features=tabular)
        elif is_tea_gnn:
            return m(x, ei, ea, edge_time=edge_time, league_id=league_id)
        else:
            return m(x, ei, ea)
    
    # Class weights
    train_labels = ey[train_mask]
    cc = torch.bincount(train_labels, minlength=3).float()
    cw = ((1.0 / cc.clamp(min=1)) * cc.sum() / 3.0).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out = _forward(model)
        
        loss = criterion(out[train_mask], ey[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_out = _forward(model)
            val_loss = criterion(val_out[test_mask], ey[test_mask])
        
        scheduler.step(val_loss)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Final eval
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        final_out = _forward(model)
        test_logits = final_out[test_mask]
        y_prob = F.softmax(test_logits, dim=1).cpu().numpy()
        y_pred = test_logits.argmax(dim=1).cpu().numpy()
    
    y_true = ey[test_mask].cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average='macro')
    ll = log_loss(y_true, y_prob, labels=[0, 1, 2])
    rps = ranked_probability_score(y_true, y_prob)
    f1_w = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc_macro = float(roc_auc_score(y_true, y_prob, multi_class='ovr',
                                         average='macro', labels=[0, 1, 2]))
    except Exception:
        auc_macro = None
    
    return {
        'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w,
        'log_loss': ll, 'rps': rps, 'auc_macro': auc_macro, 'confusion_matrix': cm,
        'model_state': best_state or model.state_dict(),
        'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob,
        'train_losses': train_losses, 'epochs': epoch,
        'per_class_f1': {c: f1_score(y_true, y_pred, average=None, labels=[i])[0] 
                         for i, c in enumerate(['A','D','H'])},
    }


# ═══════════════════════════════════════════════════════════
# OPTUNA OBJECTIVES (one per model)
# ═══════════════════════════════════════════════════════════

def objective_gcn(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('GCN', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout)
    result = train_and_evaluate(model, graph_data, lr, wd)
    return result['accuracy']


def objective_sage(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('GraphSAGE', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout)
    result = train_and_evaluate(model, graph_data, lr, wd)
    return result['accuracy']


def objective_gat(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('GAT', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout, heads=heads)
    result = train_and_evaluate(model, graph_data, lr, wd)
    return result['accuracy']


def objective_gin(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('GIN', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout)
    result = train_and_evaluate(model, graph_data, lr, wd)
    return result['accuracy']


def objective_edgeconv(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('EdgeConv', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout)
    result = train_and_evaluate(model, graph_data, lr, wd)
    return result['accuracy']


def objective_hybrid(trial, graph_data):
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    model = get_model('Hybrid', graph_data['num_node_features'], graph_data['num_edge_features'],
                      hidden_dim=hidden, dropout=dropout,
                      num_tabular_features=graph_data['num_tabular_features'])
    result = train_and_evaluate(model, graph_data, lr, wd, is_hybrid=True)
    return result['accuracy']


def objective_tea_gnn(trial, graph_data):
    """Optuna objective for TEA-GNN (novel architecture)."""
    hidden = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    use_cl = trial.suggest_categorical('use_cross_league', [True, False])
    
    model = get_model('TEA-GNN', graph_data['num_node_features'],
                      graph_data['num_edge_features'],
                      hidden_dim=hidden, heads=heads, dropout=dropout,
                      num_leagues=graph_data.get('num_leagues', 5),
                      use_cross_league=use_cl)
    result = train_and_evaluate(model, graph_data, lr, wd, is_tea_gnn=True)
    return result['accuracy']


# ═══════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════

def plot_tuned_comparison(baseline_df, tuned_df):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    metrics = [('accuracy', 'Accuracy ↑'),
               ('f1_macro', 'Macro F1 ↑'),
               ('rps', 'RPS ↓'),
               ('f1_D', 'Draw F1 ↑ (hardest class)')]
    
    for ax, (m, title) in zip(axes, metrics):
        models = tuned_df['model'].tolist()
        y = np.arange(len(models))
        
        base_vals, tuned_vals = [], []
        for model in models:
            brow = baseline_df[baseline_df['model'] == model]
            base_vals.append(brow[m].values[0] if (not brow.empty and m in brow) else 0)
            trow = tuned_df[tuned_df['model'] == model]
            tuned_vals.append(trow[m].values[0] if (not trow.empty and m in trow) else 0)
        
        ax.barh(y + 0.2, base_vals, 0.35, label='Baseline', color='#3498db', alpha=0.7)
        ax.barh(y - 0.2, tuned_vals, 0.35, label='Tuned', color='#2ecc71', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(models)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.suptitle('GNN Baseline vs Tuned (7 models)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_tuning_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(all_results):
    n = len(all_results)
    cols = min(n, 3); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    cn = ['A','D','H']
    
    for idx, (name, res) in enumerate(all_results.items()):
        cm = res['confusion_matrix']
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=cn, yticklabels=cn, ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAcc={res["accuracy"]:.3f}')
        axes[idx].set_ylabel('True'); axes[idx].set_xlabel('Pred')
    for idx in range(n, len(axes)): axes[idx].axis('off')
    
    plt.suptitle('Tuned GNN Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_tuned_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_vs_traditional(gnn_df):
    trad_path = RESULTS_DIR / 'tuned_comparison.csv'
    if not trad_path.exists():
        trad_path = RESULTS_DIR / 'model_comparison.csv'
    if not trad_path.exists():
        return
    
    trad_df = pd.read_csv(trad_path)
    best_gnn = gnn_df.sort_values('accuracy', ascending=False).iloc[0]
    best_trad = trad_df.sort_values('accuracy', ascending=False).iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ['accuracy', 'f1_macro', 'rps']
    labels = ['Accuracy ↑', 'Macro F1 ↑', 'RPS ↓']
    x = np.arange(len(metrics)); w = 0.35
    
    b1 = ax.bar(x-w/2, [best_trad[m] for m in metrics], w,
                label=f'Trad: {best_trad["model"]}', color='#3498db', alpha=0.8)
    b2 = ax.bar(x+w/2, [best_gnn[m] for m in metrics], w,
                label=f'GNN: {best_gnn["model"]}', color='#2ecc71', alpha=0.8)
    
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('Best Traditional ML vs Best Tuned GNN', fontsize=14, fontweight='bold')
    ax.legend()
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.4f}', ha='center', fontsize=9)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{bar.get_height():.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'gnn_tuned_vs_traditional.png', dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  GNN HYPERPARAMETER TUNING (Optuna)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE} | Trials per model: {N_TRIALS}")
    print("=" * 70)
    
    builder = FootballGraphBuilder(
        data_path=str(BASE_DIR / "data" / "processed" / "processed_matches.csv"))
    graph_data = builder.build_train_test_graphs()
    
    # Load baseline for comparison
    baseline_path = RESULTS_DIR / 'gnn_comparison.csv'
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    
    tuning_config = {
        'GCN': objective_gcn,
        'GraphSAGE': objective_sage,
        'GAT': objective_gat,
        'GIN': objective_gin,
        'EdgeConv': objective_edgeconv,
        'Hybrid': objective_hybrid,
        'TEA-GNN': objective_tea_gnn,
    }
    
    all_results = {}
    rows = []
    all_params = {}
    per_model_dir = ensure_per_model_dir(RESULTS_DIR)
    
    for i, (name, obj_fn) in enumerate(tuning_config.items(), 1):
        print(f"\n{'=' * 70}")
        print(f"  [{i}/{len(tuning_config)}] Tuning {name} ({N_TRIALS} trials)")
        print(f"{'=' * 70}")
        
        start = time.time()
        
        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(lambda trial: obj_fn(trial, graph_data),
                       n_trials=N_TRIALS, show_progress_bar=False)
        
        bp = study.best_params
        best_cv = study.best_value
        tune_time = time.time() - start
        
        print(f"  Best trial accuracy: {best_cv:.4f} ({tune_time:.1f}s)")
        print(f"  Best params: {bp}")
        
        all_params[name] = bp
        
        # Rebuild best model and get full metrics
        is_hybrid = (name == 'Hybrid')
        is_tea_gnn = (name == 'TEA-GNN')
        kwargs = {k: v for k, v in bp.items() if k not in ['lr', 'weight_decay']}
        if is_hybrid:
            kwargs['num_tabular_features'] = graph_data['num_tabular_features']
        if is_tea_gnn:
            kwargs['num_leagues'] = graph_data.get('num_leagues', 5)
        # heads is already in kwargs for GAT / TEA-GNN (from trial.suggest_categorical)
        
        best_model = get_model(name, graph_data['num_node_features'],
                                graph_data['num_edge_features'], **kwargs)
        metrics = train_and_evaluate(best_model, graph_data,
                                      bp.get('lr', 0.003), bp.get('weight_decay', 5e-4),
                                      is_hybrid=is_hybrid, is_tea_gnn=is_tea_gnn)
        # Tag with tune time + epochs for plotting
        metrics['tune_time_s'] = tune_time
        
        all_results[name] = metrics
        
        print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:       {metrics['f1_macro']:.4f}")
        print(f"  RPS:            {metrics['rps']:.4f}")
        if metrics.get('auc_macro') is not None:
            print(f"  AUC macro:      {metrics['auc_macro']:.4f}")
        print(f"  Per-class: A={metrics['per_class_f1']['A']:.3f} "
              f"D={metrics['per_class_f1']['D']:.3f} H={metrics['per_class_f1']['H']:.3f}")
        
        rows.append({
            'model': name, 'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'], 'f1_weighted': metrics['f1_weighted'],
            'log_loss': metrics['log_loss'], 'rps': metrics['rps'],
            'auc_macro': metrics.get('auc_macro', None),
            'tune_time_s': round(tune_time, 1),
            'f1_A': metrics['per_class_f1']['A'],
            'f1_D': metrics['per_class_f1']['D'],
            'f1_H': metrics['per_class_f1']['H'],
        })
        
        # Save model
        model_path = MODELS_DIR / f'gnn_{name.lower()}_tuned.pt'
        torch.save({'model_state': metrics['model_state'],
                    'model_name': name, 'best_params': bp}, model_path)
        print(f"  ✓ Saved to {model_path.name}")
        
        # ── Per-model plotting (5 PNGs) ──
        print(f"  → Saving per-model tuned reports to {per_model_dir}...")
        try:
            tuned_name = f'{name} (tuned)'
            plot_all_per_model(
                tuned_name, metrics,
                metrics.get('y_true'), metrics.get('y_prob'),
                class_names=['A', 'D', 'H'], save_dir=per_model_dir
            )
        except Exception as e:
            print(f"    [warn] per-model plotting failed for {name}: {e}")
    
    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  TUNED GNN RESULTS")
    print(f"{'=' * 70}\n")
    
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
    df.index += 1
    show_cols = ['model', 'accuracy', 'f1_macro', 'rps', 'auc_macro',
                 'f1_D', 'tune_time_s']
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string())
    
    df.to_csv(RESULTS_DIR / 'gnn_tuned_comparison.csv', index=False)
    
    # Save params
    with open(RESULTS_DIR / 'gnn_best_params.json', 'w') as f:
        serializable = {n: {k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}
                        for n, p in all_params.items()}
        json.dump(serializable, f, indent=2)
    
    # ── Comparison ──
    if not baseline_df.empty:
        print(f"\n{'=' * 70}")
        print(f"  BASELINE vs TUNED")
        print(f"{'=' * 70}\n")
        for _, row in df.iterrows():
            base = baseline_df[baseline_df['model'] == row['model']]
            if not base.empty:
                b_acc = base['accuracy'].values[0]
                diff = row['accuracy'] - b_acc
                arrow = '↑' if diff > 0 else ('↓' if diff < 0 else '→')
                print(f"  {row['model']:12s}: {b_acc:.4f} → {row['accuracy']:.4f} ({arrow} {abs(diff):.4f})")
        
        plot_tuned_comparison(baseline_df, df)
    
    # Plots
    plot_confusion_matrices(all_results)
    plot_vs_traditional(df)
    
    # ── NEW: per-class F1 comparison + master summary (from shared plotting.py) ──
    # Note: keys in all_results are "ModelName (tuned)" — trim to match baseline keys
    cleaned_results = {}
    for k, v in all_results.items():
        clean_key = k.replace(' (tuned)', '')
        cleaned_results[clean_key] = v
    try:
        plot_per_class_f1_comparison(
            cleaned_results, class_names=['A', 'D', 'H'],
            save_path=RESULTS_DIR / 'gnn_tuned_per_class_f1.png',
            title_suffix=' (Tuned)'
        )
        print(f"✓ Saved gnn_tuned_per_class_f1.png")
    except Exception as e:
        print(f"  [warn] tuned per-class comparison failed: {e}")
    
    try:
        write_master_summary(
            cleaned_results,
            RESULTS_DIR / 'gnn_tuned_master_summary.txt',
            title='GNN Tuned Master Summary'
        )
        print(f"✓ Saved gnn_tuned_master_summary.txt")
    except Exception as e:
        print(f"  [warn] tuned master summary failed: {e}")
    
    best = df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST TUNED GNN: {best['model']}")
    print(f"     Accuracy:  {best['accuracy']:.4f}")
    print(f"     Macro F1:  {best['f1_macro']:.4f}")
    print(f"     RPS:       {best['rps']:.4f}")
    if 'auc_macro' in df.columns and pd.notna(best.get('auc_macro')):
        print(f"     AUC macro: {best['auc_macro']:.4f}")
    print(f"     Draw F1:   {best['f1_D']:.4f}")
    print(f"{'=' * 70}\n")
    
    print("✓ GNN tuning complete!")
    return df, all_results


if __name__ == '__main__':
    main()
