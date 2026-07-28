"""
Shared Visualization Module for GNN Training/Tuning
=====================================================
Plotters used by both train_gnn.py and tune_gnn.py to avoid duplication.

Two layers of visualizations:

1. PER-MODEL (saved to results/per_model/gnn_{name}_*.png):
   - plot_per_model_report       confusion matrix + loss curve + per-class F1 + metrics table
   - plot_roc_per_model          one-vs-rest ROC curves for 3 classes + macro AUC
   - plot_pr_per_model           one-vs-rest Precision-Recall curves
   - plot_calibration_per_model reliability diagram per class
   - plot_probdist_per_model     predicted-probability histogram per true class

2. ALL-MODEL COMPARISONS (saved to results/gnn_*.png / gnn_*.txt):
   - plot_per_class_f1_comparison  grouped bars (3 classes × N models)
   - write_master_summary          plain-text metrics dump per model
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve


# Class name canonicalization
CLASS_NAMES = ['A', 'D', 'H']


# ═══════════════════════════════════════════════════════════
# Directory helper
# ═══════════════════════════════════════════════════════════

def ensure_per_model_dir(results_dir):
    """Create results/per_model/ if missing. Returns the path."""
    per_model_dir = Path(results_dir) / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    return per_model_dir


# ═══════════════════════════════════════════════════════════
# Per-Model Reports
# ═══════════════════════════════════════════════════════════

def plot_per_model_report(name, result, class_names=CLASS_NAMES, save_dir=None):
    """
    2×2 grid saved to save_dir/gnn_{name_lower}.png:
      [0,0] Normalized confusion matrix (heatmap)
      [0,1] Training-loss curve
      [1,0] Per-class F1 bar chart
      [1,1] Metrics table rendered as text-in-axes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    name_lower = name.lower().replace('-', '').replace(' ', '_')

    # ── [0,0] Confusion matrix ──
    cm = result.get('confusion_matrix')
    if cm is not None:
        cm_n = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[0, 0], cbar=False, annot_kws={'size': 11})
        axes[0, 0].set_title('Confusion Matrix (Normalized)', fontweight='bold')
        axes[0, 0].set_ylabel('True')
        axes[0, 0].set_xlabel('Predicted')
    else:
        axes[0, 0].axis('off')

    # ── [0,1] Training-loss curve ──
    losses = result.get('train_losses', [])
    if losses:
        axes[0, 1].plot(losses, color='#3498db', linewidth=1.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Training Loss')
        axes[0, 1].set_title('Training-Loss Curve', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].axis('off')

    # ── [1,0] Per-class F1 bars ──
    per_class = result.get('per_class_f1', {})
    if per_class:
        classes = list(per_class.keys())
        f1s = [per_class[c] for c in classes]
        bars = axes[1, 0].bar(classes, f1s, color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.85)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Per-Class F1', fontweight='bold')
        for bar, v in zip(bars, f1s):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}',
                            ha='center', fontsize=10)
    else:
        axes[1, 0].axis('off')

    # ── [1,1] Metrics table ──
    axes[1, 1].axis('off')
    table_rows = [
        ['Accuracy',  f"{result.get('accuracy', 0):.4f}"],
        ['Macro F1',  f"{result.get('f1_macro', 0):.4f}"],
        ['Weighted F1', f"{result.get('f1_weighted', 0):.4f}"],
        ['Log Loss',  f"{result.get('log_loss', 0):.4f}"],
        ['RPS',       f"{result.get('rps', 0):.4f}"],
        ['AUC (macro)', f"{result.get('auc_macro', result.get('auc', 0)):.4f}"],
        ['Train time', f"{result.get('train_time', result.get('tune_time_s', 0)):.1f}s"],
        ['Epochs',    f"{result.get('epochs', '-')}".replace('-', '-')],
    ]
    tbl = axes[1, 1].table(cellText=table_rows, colLabels=['Metric', 'Value'],
                           loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)
    axes[1, 1].set_title('Metrics Summary', fontweight='bold', pad=15)

    plt.suptitle(f'GNN Model Report — {name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir is not None:
        out = Path(save_dir) / f'gnn_{name_lower}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out)
    plt.close()


def plot_roc_per_model(name, y_true, y_prob, class_names=CLASS_NAMES, save_dir=None):
    """One-vs-rest ROC curves (3 class curves + macro-average) + AUC legend."""
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # Binarize for OvR
    y_true_bin = np.eye(n_classes)[y_true]

    fpr, tpr, roc_auc = {}, {}, {}
    for i, c in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], color=colors[i], linewidth=2,
                label=f'Class {c} (AUC = {roc_auc[i]:.3f})')

    # Macro ROC
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color='black', linewidth=2.5, linestyle='--',
            label=f'Macro-avg (AUC = {macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {name}', fontweight='bold')
    ax.legend(loc='lower right', fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name_lower = name.lower().replace('-', '').replace(' ', '_')
    if save_dir is not None:
        out = Path(save_dir) / f'gnn_{name_lower}_roc.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out), macro_auc
    plt.close()
    return None, macro_auc


def plot_pr_per_model(name, y_true, y_prob, class_names=CLASS_NAMES, save_dir=None):
    """One-vs-rest Precision-Recall curves."""
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    y_true_bin = np.eye(n_classes)[y_true]
    for i, c in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=colors[i], linewidth=2,
                label=f'Class {c} (AUC = {pr_auc:.3f})')

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves — {name}', fontweight='bold')
    ax.legend(loc='lower left', fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name_lower = name.lower().replace('-', '').replace(' ', '_')
    if save_dir is not None:
        out = Path(save_dir) / f'gnn_{name_lower}_pr.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out)
    plt.close()


def plot_calibration_per_model(name, y_true, y_prob, class_names=CLASS_NAMES,
                               n_bins=10, save_dir=None):
    """Per-class reliability (calibration) diagram."""
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    y_true_bin = np.eye(n_classes)[y_true]
    for i, c in enumerate(class_names):
        frac_pos, mean_pred = calibration_curve(
            y_true_bin[:, i], y_prob[:, i], n_bins=n_bins, strategy='uniform'
        )
        ax.plot(mean_pred, frac_pos, 'o-', color=colors[i], linewidth=2,
                markersize=8, label=f'Class {c}')

    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', alpha=0.7,
            label='Perfectly calibrated')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration (Reliability) — {name}', fontweight='bold')
    ax.legend(loc='upper left', fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    name_lower = name.lower().replace('-', '').replace(' ', '_')
    if save_dir is not None:
        out = Path(save_dir) / f'gnn_{name_lower}_calibration.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out)
    plt.close()


def plot_probdist_per_model(name, y_true, y_prob, class_names=CLASS_NAMES, save_dir=None):
    """Predicted-probability histogram for each true class."""
    n_classes = y_prob.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=True)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for i, c in enumerate(class_names):
        ax = axes[i] if n_classes > 1 else axes
        mask = y_true == i
        # Show distribution of predicted prob for the TRUE class
        # + the other two classes overlaid
        for j, (cn, ccolor) in enumerate(zip(class_names, colors)):
            probs = y_prob[mask, j]
            alpha = 0.7 if j == i else 0.25
            ls = '-' if j == i else '--'
            ax.hist(probs, bins=20, range=(0, 1), color=ccolor, alpha=alpha,
                    linestyle=ls, label=f'Pred: {cn}', edgecolor='white', linewidth=0.5)
        ax.set_title(f'True {c}', fontweight='bold')
        ax.set_xlabel('Predicted Probability')
        if i == 0:
            ax.set_ylabel('Count')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Predicted-Prob Distributions — {name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    name_lower = name.lower().replace('-', '').replace(' ', '_')
    if save_dir is not None:
        out = Path(save_dir) / f'gnn_{name_lower}_probdist.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        return str(out)
    plt.close()


# ═══════════════════════════════════════════════════════════
# All-Model Comparisons
# ═══════════════════════════════════════════════════════════

def plot_per_class_f1_comparison(all_results, class_names=CLASS_NAMES,
                                 save_path=None, title_suffix=''):
    """
    Grouped-bar chart: 3 classes × N models side-by-side.
    Saved to save_path (e.g. results/gnn_per_class_f1.png).
    """
    names = list(all_results.keys())
    n = len(names)
    x = np.arange(len(class_names))
    width = 0.8 / max(n, 1)
    # Distinguishable colorblind-safe palette
    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 6))
    for i, (model_name, res) in enumerate(all_results.items()):
        per_class = res.get('per_class_f1', {})
        f1s = [per_class.get(c, 0) for c in class_names]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, f1s, width, label=model_name,
                      color=palette[i % len(palette)], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, f1s):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f'{v:.2f}', ha='center', fontsize=7, rotation=90)

    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel('F1 Score'); ax.set_ylim(0, 1)
    ax.set_title(f'Per-Class F1 Comparison{title_suffix}', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(save_path)
    plt.close()


def write_master_summary(all_results, save_path, title='GNN Master Summary',
                         extra_fields=None):
    """
    Plain-text dump of each model's metrics for copy-paste into the thesis.
    save_path: e.g. results/gnn_master_summary.txt
    """
    lines = []
    lines.append('=' * 70)
    lines.append(f'  {title}')
    lines.append('=' * 70)
    header = f"  {'Model':<12} {'Acc':>7} {'F1-mac':>7} {'F1-wt':>7} {'LogLoss':>7} {'RPS':>7} {'AUC':>7} {'F1_A':>6} {'F1_D':>6} {'F1_H':>6} {'Time':>7} {'Epochs':>6}"
    lines.append(header)
    lines.append('-' * len(header))

    for name, res in all_results.items():
        per = res.get('per_class_f1', {})
        time_str = f"{res.get('train_time', res.get('tune_time_s', 0)):.1f}s"
        epochs = str(res.get('epochs', '-'))
        row = (f"  {name:<12} "
               f"{res.get('accuracy', 0):>7.4f} "
               f"{res.get('f1_macro', 0):>7.4f} "
               f"{res.get('f1_weighted', 0):>7.4f} "
               f"{res.get('log_loss', 0):>7.4f} "
               f"{res.get('rps', 0):>7.4f} "
               f"{res.get('auc_macro', res.get('auc', 0)):>7.4f} "
               f"{per.get('A', 0):>6.3f} "
               f"{per.get('D', 0):>6.3f} "
               f"{per.get('H', 0):>6.3f} "
               f"{time_str:>7} "
               f"{epochs:>6}")
        lines.append(row)

        if extra_fields and name in extra_fields:
            lines.append(f"    [extra] {extra_fields[name]}")

    lines.append('=' * 70)
    # Best model line
    if all_results:
        best_name = max(all_results.keys(),
                        key=lambda k: all_results[k].get('accuracy', 0))
        best = all_results[best_name]
        lines.append(
            f"  BEST: {best_name}  "
            f"Acc={best.get('accuracy', 0):.4f}  "
            f"F1-mac={best.get('f1_macro', 0):.4f}  "
            f"RPS={best.get('rps', 0):.4f}  "
            f"AUC={best.get('auc_macro', best.get('auc', 0)):.4f}"
        )
    lines.append('=' * 70)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return str(save_path)


# ═══════════════════════════════════════════════════════════
# Convenience wrappers
# ═══════════════════════════════════════════════════════════

def plot_all_per_model(name, result, y_true, y_prob, class_names=CLASS_NAMES,
                       save_dir=None):
    """
    Run all 5 per-model plotters for one model.
    Returns a dict describing what was saved + computed AUCs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out = {'name': name, 'paths': {}}
    # Augment result with AUC if missing
    if 'auc_macro' not in result and y_true is not None and y_prob is not None:
        try:
            result['auc_macro'] = float(
                roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro',
                              labels=[0, 1, 2])
            )
        except Exception:
            result['auc_macro'] = None

    out['paths']['report'] = plot_per_model_report(name, result, class_names, save_dir)
    roc_path, macro_auc = plot_roc_per_model(name, y_true, y_prob, class_names, save_dir)
    out['auc_macro'] = macro_auc
    out['paths']['roc'] = roc_path
    out['paths']['pr'] = plot_pr_per_model(name, y_true, y_prob, class_names, save_dir)
    out['paths']['calibration'] = plot_calibration_per_model(
        name, y_true, y_prob, class_names, save_dir=save_dir
    )
    out['paths']['probdist'] = plot_probdist_per_model(
        name, y_true, y_prob, class_names, save_dir=save_dir
    )
    return out
