"""
Traditional ML Training Pipeline for Football Match Prediction
=============================================================
Trains and compares 10 model families for 3-class prediction (H/D/A)
with probability outputs, expanded pre-match features (rolling + cumulative
+ season context + H2H + referee + weather), dynamic season splitting,
and MLflow experiment tracking.
"""

import threadpoolctl
_orig_base_get_version = threadpoolctl._Module.get_version
def _safe_base_get_version(self):
    try:
        return _orig_base_get_version(self)
    except (AttributeError, TypeError):
        return '0.0.0'
threadpoolctl._Module.get_version = _safe_base_get_version

if hasattr(threadpoolctl, '_OpenBLASModule'):
    _orig_openblas_get_version = threadpoolctl._OpenBLASModule.get_version
    def _safe_openblas_get_version(self):
        try:
            return _orig_openblas_get_version(self)
        except (AttributeError, TypeError):
            return '0.0.0'
    threadpoolctl._OpenBLASModule.get_version = _safe_openblas_get_version

if hasattr(threadpoolctl, '_BLISModule'):
    _orig_blis_get_version = threadpoolctl._BLISModule.get_version
    def _safe_blis_get_version(self):
        try:
            return _orig_blis_get_version(self)
        except (AttributeError, TypeError):
            return '0.0.0'
    threadpoolctl._BLISModule.get_version = _safe_blis_get_version

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

if hasattr(threadpoolctl, '_OpenBLASModule'):
    cls = threadpoolctl._OpenBLASModule
    if hasattr(cls, 'get_threading_layer'):
        orig_tl = cls.get_threading_layer
        def _safe_get_threading_layer(self):
            try:
                return orig_tl(self)
            except (AttributeError, TypeError, OSError):
                return "unknown"
        cls.get_threading_layer = _safe_get_threading_layer
    if hasattr(cls, 'get_architecture'):
        orig_arch = cls.get_architecture
        def _safe_get_architecture(self):
            try:
                return orig_arch(self)
            except (AttributeError, TypeError, OSError):
                return "unknown"
        cls.get_architecture = _safe_get_architecture
    if hasattr(cls, '_get_extra_info'):
        orig_extra = cls._get_extra_info
        def _safe_get_extra_info(self):
            try:
                return orig_extra(self)
            except (AttributeError, TypeError, OSError):
                self.threading_layer = "unknown"
                self.architecture = "unknown"
        cls._get_extra_info = _safe_get_extra_info

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, classification_report,
    confusion_matrix, make_scorer
)
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_matches.csv"
RESULTS_DIR = BASE_DIR / "models" / "results" / "traditional"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Expanded pre-match feature suite (rolling 5 + cumulative season + context + H2H + referee + weather)
PRE_MATCH_FEATURES = [
    # Rolling 5-match form features (30)
    'HomeForm_5', 'HomeGF_5', 'HomeGA_5', 'HomexG_5', 'HomexGA_5',
    'HomeShots_5', 'HomeShotsAgainst_5', 'HomeSOT_5', 'HomeSOTAgainst_5',
    'HomeCorners_5', 'HomeCornersAgainst_5', 'HomeFouls_5', 'HomeFoulsAgainst_5',
    'HomeYellows_5', 'HomeReds_5',
    'AwayForm_5', 'AwayGF_5', 'AwayGA_5', 'AwayxG_5', 'AwayxGA_5',
    'AwayShots_5', 'AwayShotsAgainst_5', 'AwaySOT_5', 'AwaySOTAgainst_5',
    'AwayCorners_5', 'AwayCornersAgainst_5', 'AwayFouls_5', 'AwayFoulsAgainst_5',
    'AwayYellows_5', 'AwayReds_5',
    # Cumulative season-to-date features (30)
    'Homecum_Form', 'Homecum_GF', 'Homecum_GA', 'Homecum_xG', 'Homecum_xGA',
    'Homecum_Shots', 'Homecum_ShotsAgainst', 'Homecum_SOT', 'Homecum_SOTAgainst',
    'Homecum_Corners', 'Homecum_CornersAgainst', 'Homecum_Fouls', 'Homecum_FoulsAgainst',
    'Homecum_Yellows', 'Homecum_Reds',
    'Awaycum_Form', 'Awaycum_GF', 'Awaycum_GA', 'Awaycum_xG', 'Awaycum_xGA',
    'Awaycum_Shots', 'Awaycum_ShotsAgainst', 'Awaycum_SOT', 'Awaycum_SOTAgainst',
    'Awaycum_Corners', 'Awaycum_CornersAgainst', 'Awaycum_Fouls', 'Awaycum_FoulsAgainst',
    'Awaycum_Yellows', 'Awaycum_Reds',
    # Season context (4)
    'Homeseason_progress', 'Homeform_vs_season',
    'Awayseason_progress', 'Awayform_vs_season',
    # Head-to-head (6)
    'H2H_Matches', 'H2H_HomeWins', 'H2H_AwayWins', 'H2H_Draws',
    'H2H_HomeGoals', 'H2H_AwayGoals',
    # Referee (3)
    'Ref_AvgYellows', 'Ref_AvgReds', 'Ref_Strictness',
    # Weather & Stadium (7)
    'temperature', 'precipitation', 'rain', 'wind_speed', 'humidity',
    'stadium_lat', 'stadium_lon',
]

TARGET = 'FTR'
RANDOM_STATE = 42
N_FOLDS = 5


def load_and_prepare_data():
    """Load data, select features, encode target and league, and split dynamically by season."""
    print("=" * 70)
    print("  FOOTBALL MATCH PREDICTION — TRADITIONAL ML PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[TARGET].isin(['H', 'D', 'A'])].copy().reset_index(drop=True)
    print(f"\n✓ Loaded {len(df)} matches from {DATA_PATH.name}")

    feature_cols = [c for c in PRE_MATCH_FEATURES if c in df.columns]
    print(f"\n✓ Using {len(feature_cols)} pre-match features")

    league_dummies = pd.get_dummies(df['League'], prefix='League', dtype=int)
    X = pd.concat([df[feature_cols], league_dummies], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    class_names = le.classes_

    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        X = X.fillna(X.mean()).fillna(0.0)

    # Dynamic season split: test on latest available season with >= 100 matches, train on prior seasons
    season_counts = df['Season'].value_counts()
    valid_seasons = sorted([int(s) for s, cnt in season_counts.items() if cnt >= 100])
    if not valid_seasons:
        valid_seasons = sorted([int(s) for s in df['Season'].dropna().unique()])

    test_season = valid_seasons[-1]
    train_seasons = [s for s in valid_seasons if s < test_season] or [test_season]

    train_mask = df['Season'].isin(train_seasons)
    test_mask = df['Season'] == test_season

    X_train_full = X[train_mask].values
    y_train_full = y[train_mask]
    X_test = X[test_mask].values
    y_test = y[test_mask]

    print(f"\n✓ Dynamic Time-based split:")
    print(f"  Train seasons ({train_seasons[0]}..{train_seasons[-1]}): {len(X_train_full)} matches")
    print(f"  Test season   ({test_season}):           {len(X_test)} matches")

    return X, y, X_train_full, y_train_full, X_test, y_test, class_names, X.columns.tolist()


def ranked_probability_score(y_true, y_prob):
    n_classes = y_prob.shape[1]
    rps_sum = 0.0
    for i in range(len(y_true)):
        cum_pred, cum_true, rps_match = 0.0, 0.0, 0.0
        for j in range(n_classes - 1):
            cum_pred += y_prob[i, j]
            cum_true += 1.0 if y_true[i] <= j else 0.0
            rps_match += (cum_pred - cum_true) ** 2
        rps_sum += rps_match / (n_classes - 1)
    return rps_sum / len(y_true)


def get_models():
    models = {}
    models['Decision Tree'] = DecisionTreeClassifier(
        max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, n_jobs=1, random_state=RANDOM_STATE
    )
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=RANDOM_STATE, verbosity=0
        )
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbose=-1
        )
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=0
        )
    models['Logistic Regression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE))
    ])
    models['Naive Bayes'] = GaussianNB()
    models['KNN'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=1))
    ])
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=RANDOM_STATE
        ))
    ])
    models['SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE))
    ])
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, class_names):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'log_loss': log_loss(y_test, y_prob),
        'rps': ranked_probability_score(y_test, y_prob),
        'train_time': train_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True),
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def run_cross_validation(model, X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'f1_macro': make_scorer(f1_score, average='macro'),
        'neg_log_loss': 'neg_log_loss'
    }
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=1)
    return {
        'cv_accuracy': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_f1_macro': cv_results['test_f1_macro'].mean(),
        'cv_f1_macro_std': cv_results['test_f1_macro'].std(),
        'cv_log_loss': -cv_results['test_neg_log_loss'].mean(),
        'cv_log_loss_std': cv_results['test_neg_log_loss'].std(),
    }


def plot_confusion_matrices(results, class_names):
    n_models = len(results)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for idx, (name, res) in enumerate(results.items()):
        cm = res['confusion_matrix']
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


def plot_model_comparison(results_df):
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


def plot_class_f1_heatmap(results, class_names):
    data = {}
    for name, res in results.items():
        report = res['classification_report']
        data[name] = [report[cls]['f1-score'] for cls in class_names]
    df_f1 = pd.DataFrame(data, index=class_names).T
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_f1, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'F1-Score'})
    ax.set_title('Per-Class F1-Score by Model', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'class_f1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    X, y, X_train, y_train, X_test, y_test, class_names, feature_names = load_and_prepare_data()

    try:
        from models.mlflow_config import setup_mlflow, safe_end_run
        safe_end_run()
        mlflow = setup_mlflow("football-traditional-training")
    except Exception:
        mlflow = None

    models = get_models()
    all_results = {}
    summary_rows = []

    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] {name}...")
        try:
            cv_res = run_cross_validation(model, X.values, y, N_FOLDS)
        except Exception:
            cv_res = {}

        try:
            eval_res = evaluate_model(model, X_train, y_train, X_test, y_test, class_names)
            all_results[name] = eval_res
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            continue

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

        if mlflow:
            try:
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model", name)
                    mlflow.log_param("num_features", len(feature_names))
                    mlflow.log_metrics({
                        "accuracy": float(eval_res['accuracy']),
                        "f1_macro": float(eval_res['f1_macro']),
                        "log_loss": float(eval_res['log_loss']),
                        "rps": float(eval_res['rps']),
                    })
            except Exception:
                pass

    if len(all_results) >= 3:
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        top3_names = [name for name, _ in sorted_models[:3]]
        fresh_models = get_models()
        estimators = [(n, fresh_models[n]) for n in top3_names if n in fresh_models]
        if len(estimators) >= 2:
            voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
            eval_res = evaluate_model(voting_clf, X_train, y_train, X_test, y_test, class_names)
            all_results['Voting (Top-3)'] = eval_res
            summary_rows.append({
                'model': 'Voting (Top-3)',
                'accuracy': eval_res['accuracy'],
                'f1_macro': eval_res['f1_macro'],
                'f1_weighted': eval_res['f1_weighted'],
                'log_loss': eval_res['log_loss'],
                'rps': eval_res['rps'],
                'train_time_s': round(eval_res['train_time'], 2),
            })

    results_df = pd.DataFrame(summary_rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1

    csv_path = RESULTS_DIR / 'model_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    plot_confusion_matrices(all_results, class_names)
    plot_model_comparison(results_df)
    plot_class_f1_heatmap(all_results, class_names)

    return results_df, all_results


if __name__ == '__main__':
    results_df, all_results = main()
