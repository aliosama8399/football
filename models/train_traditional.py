"""
Traditional ML Training Pipeline for Football Match Prediction
=============================================================
Trains and compares 10 model families for 3-class prediction (H/D/A)
with probability outputs.

Models:
  1. Decision Tree        (Tree-based)
  2. Random Forest        (Ensemble - Bagging)
  3. XGBoost              (Ensemble - Boosting)
  4. LightGBM             (Ensemble - Boosting)
  5. CatBoost             (Ensemble - Boosting)
  6. Logistic Regression  (Statistical)
  7. Naive Bayes          (Statistical)
  8. K-Nearest Neighbors  (Distance-based)
  9. MLP Classifier       (Neural)
  10. Voting Classifier   (Meta-ensemble)
"""

# ── Workaround for threadpoolctl "NoneType has no 'split' attribute" ──
# Must be applied BEFORE any imports that might trigger sklearn → threadpoolctl.
# threadpoolctl (used by scikit-learn inside cross_validate / kneighbors /
# KNN.predict / SVM.predict) calls get_config().split() on loaded BLAS DLLs.
# On some Windows conda installs, _OpenBLASModule.get_version uses a lambda
# that returns None then calls .split() on it (BUG in threadpoolctl 3.x).
# _BLISModule.get_version has same issue with .decode(None).
# Fix: monkeypatch ALL _Module subclasses' get_version to guard against None.
import threadpoolctl
# Patch base class
_orig_base_get_version = threadpoolctl._Module.get_version
def _safe_base_get_version(self):
    try:
        return _orig_base_get_version(self)
    except (AttributeError, TypeError):
        return '0.0.0'
threadpoolctl._Module.get_version = _safe_base_get_version

# Patch _OpenBLASModule (the main culprit: lambda returns None then .split())
if hasattr(threadpoolctl, '_OpenBLASModule'):
    _orig_openblas_get_version = threadpoolctl._OpenBLASModule.get_version
    def _safe_openblas_get_version(self):
        try:
            return _orig_openblas_get_version(self)
        except (AttributeError, TypeError):
            return '0.0.0'
    threadpoolctl._OpenBLASModule.get_version = _safe_openblas_get_version

# Patch _BLISModule (lambda returns None then .decode() crashes)
if hasattr(threadpoolctl, '_BLISModule'):
    _orig_blis_get_version = threadpoolctl._BLISModule.get_version
    def _safe_blis_get_version(self):
        try:
            return _orig_blis_get_version(self)
        except (AttributeError, TypeError):
            return '0.0.0'
    threadpoolctl._BLISModule.get_version = _safe_blis_get_version

# Patch _MKLModule and _OpenMPModule for consistency
for cls_name in ('_MKLModule', '_OpenMPModule'):
    if hasattr(threadpoolctl, cls_name):
        cls = getattr(threadpoolctl, cls_name)
        if hasattr(cls, 'get_version'):
            orig = cls.get_version
            def make_safe(orig_fn):
                def _safe(self):
                    try:
                        return orig_fn(self)
                    except (AttributeError, TypeError):
                        return '0.0.0'
                return _safe
            cls.get_version = make_safe(orig)

# Also patch _OpenBLASModule._get_extra_info / get_threading_layer / get_architecture
# to handle missing OpenBLAS functions (older versions don't have openblas_get_parallel)
if hasattr(threadpoolctl, '_OpenBLASModule'):
    cls = threadpoolctl._OpenBLASModule
    # Patch get_threading_layer
    if hasattr(cls, 'get_threading_layer'):
        orig_tl = cls.get_threading_layer
        def _safe_get_threading_layer(self):
            try:
                return orig_tl(self)
            except (AttributeError, TypeError, OSError):
                return "unknown"
        cls.get_threading_layer = _safe_get_threading_layer
    
    # Patch get_architecture
    if hasattr(cls, 'get_architecture'):
        orig_arch = cls.get_architecture
        def _safe_get_architecture(self):
            try:
                return orig_arch(self)
            except (AttributeError, TypeError, OSError):
                return "unknown"
        cls.get_architecture = _safe_get_architecture
    
    # Patch _get_extra_info
    if hasattr(cls, '_get_extra_info'):
        orig_extra = cls._get_extra_info
        def _safe_get_extra_info(self):
            try:
                return orig_extra(self)
            except (AttributeError, TypeError, OSError):
                self.threading_layer = "unknown"
                self.architecture = "unknown"
        cls._get_extra_info = _safe_get_extra_info

import pandas as pd
import numpy as np
import warnings
import json
import os
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, classification_report,
    confusion_matrix, make_scorer
)
from sklearn.pipeline import Pipeline

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠ XGBoost not installed, skipping")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠ LightGBM not installed, skipping")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠ CatBoost not installed, skipping")

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_matches.csv"
RESULTS_DIR = BASE_DIR / "models" / "results" / "traditional"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Features that are SAFE to use (known before kickoff)
PRE_MATCH_FEATURES = [
    # Rolling form features (last 5 matches)
    'HomeForm_5', 'HomeGF_5', 'HomeGA_5', 'HomexG_5', 'HomexGA_5',
    'AwayForm_5', 'AwayGF_5', 'AwayGA_5', 'AwayxG_5', 'AwayxGA_5',
    # Rolling match stats — team playing style (last 5 matches)
    'HomeShots_5', 'HomeShotsAgainst_5',        # Shots for/against
    'AwayShots_5', 'AwayShotsAgainst_5',
    'HomeSOT_5', 'HomeSOTAgainst_5',            # Shots on target for/against
    'AwaySOT_5', 'AwaySOTAgainst_5',
    'HomeCorners_5', 'HomeCornersAgainst_5',    # Corners for/against
    'AwayCorners_5', 'AwayCornersAgainst_5',
    'HomeFouls_5', 'HomeFoulsAgainst_5',        # Fouls committed/received
    'AwayFouls_5', 'AwayFoulsAgainst_5',
    'HomeYellows_5', 'HomeReds_5',              # Discipline
    'AwayYellows_5', 'AwayReds_5',
    # Head-to-head
    'H2H_Matches', 'H2H_HomeWins', 'H2H_AwayWins', 'H2H_Draws',
    'H2H_HomeGoals', 'H2H_AwayGoals',
    # Referee
    'Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness',
]

TARGET = 'FTR'
RANDOM_STATE = 42
N_FOLDS = 5


# ═══════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════

def load_and_prepare_data():
    """Load data, select features, encode target and league."""
    print("=" * 70)
    print("  FOOTBALL MATCH PREDICTION — TRADITIONAL ML PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Loaded {len(df)} matches from {DATA_PATH.name}")

    # League distribution
    print(f"\n  Leagues:")
    for league, count in df['League'].value_counts().items():
        print(f"    {league}: {count} matches")

    # Select features
    feature_cols = [c for c in PRE_MATCH_FEATURES if c in df.columns]
    print(f"\n✓ Using {len(feature_cols)} pre-match features")

    # One-hot encode League
    league_dummies = pd.get_dummies(df['League'], prefix='League', dtype=int)
    print(f"✓ Added {len(league_dummies.columns)} league dummy variables")

    # Build feature matrix
    X = pd.concat([df[feature_cols], league_dummies], axis=1)

    # Encode target: H=0, D=1, A=2
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    class_names = le.classes_
    print(f"\n✓ Target classes: {dict(zip(class_names, range(len(class_names))))}")
    print(f"  Distribution: {dict(zip(class_names, np.bincount(y)))}")

    # Handle any remaining NaN
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"⚠ Filling {nan_count} NaN values with column means")
        X = X.fillna(X.mean())

    # Time-based split: train on 2223+2324, test on 2425
    train_mask = df['Season'].isin([2223, 2324])
    test_mask = df['Season'] == 2425

    X_train_full = X[train_mask].values
    y_train_full = y[train_mask]
    X_test = X[test_mask].values
    y_test = y[test_mask]

    print(f"\n✓ Time-based split:")
    print(f"  Train (2022-23 + 2023-24): {len(X_train_full)} matches")
    print(f"  Test  (2024-25):           {len(X_test)} matches")
    print(f"\n  Feature names: {X.columns.tolist()}")

    return X, y, X_train_full, y_train_full, X_test, y_test, class_names, X.columns.tolist()


# ═══════════════════════════════════════════════════════════
# RANKED PROBABILITY SCORE (RPS)
# ═══════════════════════════════════════════════════════════

def ranked_probability_score(y_true, y_prob):
    """
    Calculate Ranked Probability Score for ordered outcomes.
    Lower is better. Ordering: A(0) < D(1) < H(2)
    """
    n_classes = y_prob.shape[1]
    rps_sum = 0.0
    for i in range(len(y_true)):
        cum_pred = 0.0
        cum_true = 0.0
        rps_match = 0.0
        for j in range(n_classes - 1):
            cum_pred += y_prob[i, j]
            cum_true += 1.0 if y_true[i] <= j else 0.0
            rps_match += (cum_pred - cum_true) ** 2
        rps_sum += rps_match / (n_classes - 1)
    return rps_sum / len(y_true)


# ═══════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════

def get_models():
    """Return dict of model name → model instance."""
    models = {}

    # 1. Decision Tree
    models['Decision Tree'] = DecisionTreeClassifier(
        max_depth=8, min_samples_split=10, min_samples_leaf=5,
        random_state=RANDOM_STATE
    )

    # 2. Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE  # n_jobs=1: threadpoolctl crash workaround
    )

    # 3. XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0
        )

    # 4. LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1
        )

    # 5. CatBoost
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=0
        )

    # 6. Logistic Regression (with scaling)
    models['Logistic Regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            multi_class='multinomial', solver='lbfgs',
            max_iter=1000, C=1.0, random_state=RANDOM_STATE
        ))
    ])

    # 7. Naive Bayes
    models['Naive Bayes'] = GaussianNB()

    # 8. KNN (with scaling)
    models['KNN'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=1))
    ])

    # 9. MLP (with scaling)
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE
        ))
    ])

    # 10. SVM (with scaling)
    models['SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf', C=1.0, probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    return models


# ═══════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    """Train model and compute all evaluation metrics."""
    start = time.time()

    # Train
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_prob)
    rps = ranked_probability_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'log_loss': logloss,
        'rps': rps,
        'train_time': train_time,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def run_cross_validation(model, X, y, n_folds=5):
    """Run stratified k-fold cross validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(f1_score, average='macro'),
        'neg_log_loss': 'neg_log_loss'
    }
    cv_results = cross_validate(
        model, X, y, cv=skf, scoring=scoring,
        return_train_score=False, n_jobs=1  # n_jobs=1 to avoid threadpoolctl crash on Windows
    )
    return {
        'cv_accuracy': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_f1_macro': cv_results['test_f1_macro'].mean(),
        'cv_f1_macro_std': cv_results['test_f1_macro'].std(),
        'cv_log_loss': -cv_results['test_neg_log_loss'].mean(),
        'cv_log_loss_std': cv_results['test_neg_log_loss'].std(),
    }


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrices(results, class_names):
    """Plot confusion matrix for each model."""
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = res['confusion_matrix']
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name}\nAcc={res["accuracy"]:.3f}', fontsize=10)
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved confusion matrices to {RESULTS_DIR / 'confusion_matrices.png'}")


def plot_model_comparison(results_df):
    """Bar chart comparing models on key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('accuracy', 'Accuracy ↑', 'Blues_d'),
        ('f1_macro', 'Macro F1 ↑', 'Greens_d'),
        ('log_loss', 'Log Loss ↓', 'Reds_d'),
        ('rps', 'Ranked Probability Score ↓', 'Oranges_d')
    ]

    for ax, (metric, title, palette) in zip(axes.flatten(), metrics):
        sorted_df = results_df.sort_values(metric, ascending=(metric in ['log_loss', 'rps']))
        colors = sns.color_palette(palette, n_colors=len(sorted_df))
        if metric in ['log_loss', 'rps']:
            colors = colors[::-1]
        bars = ax.barh(sorted_df['model'], sorted_df[metric], color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(metric)
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=8)

    plt.suptitle('Model Comparison — Traditional ML', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved model comparison chart to {RESULTS_DIR / 'model_comparison.png'}")


def plot_class_f1_heatmap(results, class_names):
    """Heatmap of per-class F1 scores for each model."""
    data = {}
    for name, res in results.items():
        report = res['classification_report']
        data[name] = [report[cls]['f1-score'] for cls in class_names]

    df_f1 = pd.DataFrame(data, index=class_names).T
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_f1, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'F1-Score'})
    ax.set_title('Per-Class F1-Score by Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model')
    ax.set_xlabel('Class')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'class_f1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class F1 heatmap to {RESULTS_DIR / 'class_f1_heatmap.png'}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # Load data
    X, y, X_train, y_train, X_test, y_test, class_names, feature_names = load_and_prepare_data()

    # Get models
    models = get_models()
    print(f"\n{'=' * 70}")
    print(f"  TRAINING {len(models)} MODELS")
    print(f"{'=' * 70}\n")

    all_results = {}
    summary_rows = []

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] {name}...")

        # Cross-validation on full data
        try:
            cv_res = run_cross_validation(model, X.values, y, N_FOLDS)
            print(f"  CV Accuracy: {cv_res['cv_accuracy']:.4f} ± {cv_res['cv_accuracy_std']:.4f}")
        except Exception as e:
            print(f"  ⚠ CV failed: {e}")
            cv_res = {'cv_accuracy': None, 'cv_accuracy_std': None,
                     'cv_f1_macro': None, 'cv_f1_macro_std': None,
                     'cv_log_loss': None, 'cv_log_loss_std': None}

        # Time-based evaluation
        try:
            eval_res = evaluate_model(model, X_train, y_train, X_test, y_test, class_names)
            print(f"  Test Accuracy: {eval_res['accuracy']:.4f}")
            print(f"  Macro F1:      {eval_res['f1_macro']:.4f}")
            print(f"  Log Loss:      {eval_res['log_loss']:.4f}")
            print(f"  RPS:           {eval_res['rps']:.4f}")
            print(f"  Train time:    {eval_res['train_time']:.2f}s")
            all_results[name] = eval_res
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            continue

        # Summary row
        row = {
            'model': name,
            'accuracy': eval_res['accuracy'],
            'f1_macro': eval_res['f1_macro'],
            'f1_weighted': eval_res['f1_weighted'],
            'log_loss': eval_res['log_loss'],
            'rps': eval_res['rps'],
            'train_time_s': round(eval_res['train_time'], 2),
        }
        row.update({k: v for k, v in cv_res.items() if v is not None})
        summary_rows.append(row)
        print()

    # ── Build Voting Ensemble from top 3 ──
    if len(all_results) >= 3:
        print(f"[BONUS] Building Voting Ensemble from top-3 models...")
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top3_names = [name for name, _ in sorted_models[:3]]
        print(f"  Top 3: {top3_names}")

        # Re-instantiate fresh models for voting
        fresh_models = get_models()
        estimators = [(n, fresh_models[n]) for n in top3_names if n in fresh_models]

        if len(estimators) >= 2:
            voting_clf = VotingClassifier(
                estimators=estimators, voting='soft', n_jobs=1  # n_jobs=1: threadpoolctl crash workaround
            )
            eval_res = evaluate_model(voting_clf, X_train, y_train, X_test, y_test, class_names)
            all_results['Voting (Top-3)'] = eval_res
            print(f"  Test Accuracy: {eval_res['accuracy']:.4f}")
            print(f"  Macro F1:      {eval_res['f1_macro']:.4f}")
            print(f"  Log Loss:      {eval_res['log_loss']:.4f}")
            print(f"  RPS:           {eval_res['rps']:.4f}")
            summary_rows.append({
                'model': 'Voting (Top-3)',
                'accuracy': eval_res['accuracy'],
                'f1_macro': eval_res['f1_macro'],
                'f1_weighted': eval_res['f1_weighted'],
                'log_loss': eval_res['log_loss'],
                'rps': eval_res['rps'],
                'train_time_s': round(eval_res['train_time'], 2),
            })

    # ── Results Summary ──
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}\n")

    results_df = pd.DataFrame(summary_rows)
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1  # 1-indexed rank

    # Print table
    display_cols = ['model', 'accuracy', 'f1_macro', 'log_loss', 'rps', 'train_time_s']
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=True))

    # Save to CSV
    csv_path = RESULTS_DIR / 'model_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    # ── Plots ──
    print(f"\n{'=' * 70}")
    print(f"  GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")

    plot_confusion_matrices(all_results, class_names)
    plot_model_comparison(results_df)
    plot_class_f1_heatmap(all_results, class_names)

    # ── Best Model ──
    best = results_df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST MODEL: {best['model']}")
    print(f"     Accuracy:  {best['accuracy']:.4f}")
    print(f"     Macro F1:  {best['f1_macro']:.4f}")
    print(f"     Log Loss:  {best['log_loss']:.4f}")
    print(f"     RPS:       {best['rps']:.4f}")
    print(f"{'=' * 70}")

    # ── Per-class report for best model ──
    best_name = best['model']
    if best_name in all_results:
        print(f"\n  Classification Report ({best_name}):")
        report = all_results[best_name]['classification_report']
        for cls in class_names:
            r = report[cls]
            print(f"    {cls}: P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  N={r['support']}")

    print(f"\n✓ All results saved to: {RESULTS_DIR}")
    print("✓ Pipeline complete!\n")

    return results_df, all_results


if __name__ == '__main__':
    results_df, all_results = main()
