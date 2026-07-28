"""
Hyperparameter Tuning for Top Football Prediction Models
========================================================
Uses Optuna (Bayesian optimization) for efficient hyperparameter search.
Tunes all 10 models, saves best configs, and re-evaluates on held-out test set.
"""

import pandas as pd
import numpy as np
import optuna
import joblib
import json
import os
import time
import warnings
from pathlib import Path
from datetime import datetime

# ── Workaround for threadpoolctl "NoneType has no 'split' attribute" ──
# Same monkeypatch as train_traditional.py: guard against get_config()
# returning None when BLAS DLL is not recognized on Windows conda.
# Patch ALL _Module subclasses (_OpenBLASModule is the main culprit).
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

# Also patch _OpenBLASModule._get_extra_info / get_threading_layer / get_architecture
# to handle missing OpenBLAS functions (older versions don't have openblas_get_parallel)
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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "processed_matches.csv"
RESULTS_DIR = BASE_DIR / "models" / "results"
MODELS_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 60         # Optuna trials per model
N_FOLDS = 5           # CV folds
RANDOM_STATE = 42

PRE_MATCH_FEATURES = [
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


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_data():
    """Load and prepare data with time-based split."""
    print("=" * 70)
    print("  HYPERPARAMETER TUNING — FOOTBALL MATCH PREDICTION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"\n✓ Loaded {len(df)} matches")

    feature_cols = [c for c in PRE_MATCH_FEATURES if c in df.columns]
    league_dummies = pd.get_dummies(df['League'], prefix='League', dtype=int)
    X = pd.concat([df[feature_cols], league_dummies], axis=1)
    X = X.fillna(X.mean())

    le = LabelEncoder()
    y = le.fit_transform(df['FTR'])
    class_names = le.classes_

    # Time-based split
    train_mask = df['Season'].isin([2223, 2324])
    test_mask = df['Season'] == 2425

    X_train = X[train_mask].values
    y_train = y[train_mask]
    X_test = X[test_mask].values
    y_test = y[test_mask]

    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"✓ Classes: {dict(zip(class_names, np.bincount(y)))}")

    return X_train, y_train, X_test, y_test, class_names, X.columns.tolist()


# ═══════════════════════════════════════════════════════════
# RPS METRIC
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
# OPTUNA OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def make_cv_scorer(model, X, y, n_folds):
    """Return mean CV accuracy."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=1)  # n_jobs=1: avoid threadpoolctl crash on Windows
    return scores.mean()


def objective_logistic(trial, X, y):
    C = trial.suggest_float('C', 0.001, 100, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'saga'])
    penalty = 'l2'

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            C=C, solver=solver, penalty=penalty,
            multi_class='multinomial', max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 100, 800)
    max_depth = trial.suggest_int('max_depth', 4, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=1, random_state=RANDOM_STATE  # n_jobs=1: threadpoolctl crash workaround
    )
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    }

    model = XGBClassifier(
        **params, use_label_encoder=False, eval_metric='mlogloss',
        random_state=RANDOM_STATE, verbosity=0
    )
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_lgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_catboost(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 800),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
    }

    model = CatBoostClassifier(**params, random_state=RANDOM_STATE, verbose=0)
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_mlp(trial, X, y):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 32, 256))

    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=tuple(layers), alpha=alpha,
            learning_rate_init=learning_rate_init,
            activation='relu', solver='adam', max_iter=500,
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_STATE
        ))
    ])
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_svm(trial, X, y):
    C = trial.suggest_float('C', 0.01, 100, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])

    params = {'C': C, 'gamma': gamma, 'kernel': kernel, 'probability': True,
              'random_state': RANDOM_STATE}
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(**params))
    ])
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_decision_tree(trial, X, y):
    max_depth = trial.suggest_int('max_depth', 3, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, criterion=criterion,
        max_features=max_features, random_state=RANDOM_STATE
    )
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_naive_bayes(trial, X, y):
    var_smoothing = trial.suggest_float('var_smoothing', 1e-12, 1e-3, log=True)

    model = GaussianNB(var_smoothing=var_smoothing)
    return make_cv_scorer(model, X, y, N_FOLDS)


def objective_knn(trial, X, y):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    p = trial.suggest_categorical('p', [1, 2])  # 1=Manhattan, 2=Euclidean
    leaf_size = trial.suggest_int('leaf_size', 10, 100)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            p=p, leaf_size=leaf_size, n_jobs=1  # n_jobs=1: threadpoolctl crash workaround
        ))
    ])
    return make_cv_scorer(model, X, y, N_FOLDS)


# ═══════════════════════════════════════════════════════════
# MODEL REBUILDING FROM BEST PARAMS
# ═══════════════════════════════════════════════════════════

def build_best_model(name, params):
    """Reconstruct model from best Optuna params."""
    if name == 'Logistic Regression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=params['C'], solver=params['solver'], penalty='l2',
                multi_class='multinomial', max_iter=2000, random_state=RANDOM_STATE
            ))
        ])
    elif name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'], max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            n_jobs=1, random_state=RANDOM_STATE  # n_jobs=1: threadpoolctl crash workaround
        )
    elif name == 'XGBoost':
        p = {k: v for k, v in params.items()}
        return XGBClassifier(**p, use_label_encoder=False, eval_metric='mlogloss',
                            random_state=RANDOM_STATE, verbosity=0)
    elif name == 'LightGBM':
        return LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    elif name == 'CatBoost':
        return CatBoostClassifier(**params, random_state=RANDOM_STATE, verbose=0)
    elif name == 'MLP':
        layers = tuple(params[f'n_units_{i}'] for i in range(params['n_layers']))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                hidden_layer_sizes=layers, alpha=params['alpha'],
                learning_rate_init=params['learning_rate_init'],
                activation='relu', solver='adam', max_iter=500,
                early_stopping=True, validation_fraction=0.15,
                random_state=RANDOM_STATE
            ))
        ])
    elif name == 'SVM':
        svc_params = {'C': params['C'], 'gamma': params['gamma'],
                      'kernel': params['kernel'], 'probability': True,
                      'random_state': RANDOM_STATE}
        if params['kernel'] == 'poly':
            svc_params['degree'] = params['degree']
        return Pipeline([('scaler', StandardScaler()), ('clf', SVC(**svc_params))])
    elif name == 'Decision Tree':
        return DecisionTreeClassifier(
            max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'], criterion=params['criterion'],
            max_features=params['max_features'], random_state=RANDOM_STATE
        )
    elif name == 'Naive Bayes':
        return GaussianNB(var_smoothing=params['var_smoothing'])
    elif name == 'KNN':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(
                n_neighbors=params['n_neighbors'], weights=params['weights'],
                algorithm=params['algorithm'], p=params['p'],
                leaf_size=params['leaf_size'], n_jobs=1  # threadpoolctl workaround
            ))
        ])


# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════

def plot_tuning_comparison(baseline_df, tuned_df):
    """Side-by-side comparison chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('accuracy', 'Accuracy ↑'), ('f1_macro', 'Macro F1 ↑'), ('rps', 'RPS ↓')]

    for ax, (metric, title) in zip(axes, metrics):
        models = tuned_df['model'].tolist()
        base_vals = []
        tuned_vals = []
        for m in models:
            bv = baseline_df.loc[baseline_df['model'] == m, metric]
            base_vals.append(bv.values[0] if len(bv) > 0 else 0)
            tuned_vals.append(tuned_df.loc[tuned_df['model'] == m, metric].values[0])

        x = np.arange(len(models))
        w = 0.35
        ax.barh(x - w/2, base_vals, w, label='Baseline', color='#4a90d9', alpha=0.7)
        ax.barh(x + w/2, tuned_vals, w, label='Tuned', color='#2ecc71', alpha=0.9)
        ax.set_yticks(x)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

        for i, (bv, tv) in enumerate(zip(base_vals, tuned_vals)):
            ax.text(max(bv, tv) + 0.002, i, f'{tv:.4f}', va='center', fontsize=8, color='green')

    plt.suptitle('Baseline vs Tuned Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'tuning_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved tuning comparison to {RESULTS_DIR / 'tuning_comparison.png'}")


def plot_tuned_confusion_matrices(results, class_names):
    """Confusion matrices for tuned models."""
    n = len(results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, res) in enumerate(results.items()):
        cm = res['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{name} (Tuned)\nAcc={res["accuracy"]:.3f}', fontsize=10)
        axes[idx].set_ylabel('True')
        axes[idx].set_xlabel('Predicted')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Tuned Models — Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'tuned_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved tuned confusion matrices")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    X_train, y_train, X_test, y_test, class_names, feature_names = load_data()

    # Models to tune with their objective functions
    tuning_config = {
        'Logistic Regression': objective_logistic,
        'Random Forest': objective_rf,
        'XGBoost': objective_xgb,
        'LightGBM': objective_lgbm,
        'CatBoost': objective_catboost,
        'MLP': objective_mlp,
        'SVM': objective_svm,
        'Decision Tree': objective_decision_tree,
        'Naive Bayes': objective_naive_bayes,
        'KNN': objective_knn,
    }

    # Load baseline results for comparison
    baseline_path = RESULTS_DIR / 'model_comparison.csv'
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()

    print(f"\n{'=' * 70}")
    print(f"  TUNING {len(tuning_config)} MODELS ({N_TRIALS} trials each)")
    print(f"{'=' * 70}\n")

    all_results = {}
    summary_rows = []
    all_best_params = {}

    for i, (name, objective_fn) in enumerate(tuning_config.items(), 1):
        print(f"\n[{i}/{len(tuning_config)}] Tuning {name}...")
        start = time.time()

        # Run Optuna
        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(lambda trial: objective_fn(trial, X_train, y_train),
                       n_trials=N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        best_cv_acc = study.best_value
        tune_time = time.time() - start

        print(f"  Best CV Accuracy: {best_cv_acc:.4f} ({tune_time:.1f}s)")
        print(f"  Best params: {best_params}")

        all_best_params[name] = best_params

        # Rebuild and evaluate on test set
        model = build_best_model(name, best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_m = f1_score(y_test, y_pred, average='macro')
        f1_w = f1_score(y_test, y_pred, average='weighted')
        ll = log_loss(y_test, y_prob)
        rps = ranked_probability_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Macro F1:      {f1_m:.4f}")
        print(f"  Log Loss:      {ll:.4f}")
        print(f"  RPS:           {rps:.4f}")

        all_results[name] = {
            'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w,
            'log_loss': ll, 'rps': rps, 'confusion_matrix': cm,
            'classification_report': report
        }

        summary_rows.append({
            'model': name, 'accuracy': acc, 'f1_macro': f1_m,
            'f1_weighted': f1_w, 'log_loss': ll, 'rps': rps,
            'best_cv_accuracy': best_cv_acc, 'tune_time_s': round(tune_time, 1),
        })

        # Save model
        model_path = MODELS_DIR / f'{name.lower().replace(" ", "_")}_tuned.pkl'
        joblib.dump(model, model_path)
        print(f"  ✓ Saved model to {model_path.name}")

    # ── Results ──
    print(f"\n{'=' * 70}")
    print(f"  TUNED RESULTS SUMMARY")
    print(f"{'=' * 70}\n")

    results_df = pd.DataFrame(summary_rows)
    results_df = results_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1

    display_cols = ['model', 'accuracy', 'f1_macro', 'log_loss', 'rps', 'best_cv_accuracy', 'tune_time_s']
    print(results_df[display_cols].to_string(index=True))

    # Save
    results_df.to_csv(RESULTS_DIR / 'tuned_comparison.csv', index=False)
    print(f"\n✓ Saved tuned results to {RESULTS_DIR / 'tuned_comparison.csv'}")

    # Save best params
    with open(RESULTS_DIR / 'best_params.json', 'w') as f:
        serializable = {}
        for name, params in all_best_params.items():
            serializable[name] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()}
        json.dump(serializable, f, indent=2)
    print(f"✓ Saved best params to {RESULTS_DIR / 'best_params.json'}")

    # ── Comparison with baseline ──
    if not baseline_df.empty:
        print(f"\n{'=' * 70}")
        print(f"  BASELINE vs TUNED COMPARISON")
        print(f"{'=' * 70}\n")

        for _, row in results_df.iterrows():
            name = row['model']
            base = baseline_df.loc[baseline_df['model'] == name]
            if not base.empty:
                base_acc = base['accuracy'].values[0]
                diff = row['accuracy'] - base_acc
                arrow = '↑' if diff > 0 else ('↓' if diff < 0 else '→')
                print(f"  {name:25s}: {base_acc:.4f} → {row['accuracy']:.4f} ({arrow} {abs(diff):.4f})")

        plot_tuning_comparison(baseline_df, results_df)

    # ── Plots ──
    plot_tuned_confusion_matrices(all_results, class_names)

    # ── Best model ──
    best = results_df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  🏆 BEST TUNED MODEL: {best['model']}")
    print(f"     Accuracy:  {best['accuracy']:.4f}")
    print(f"     Macro F1:  {best['f1_macro']:.4f}")
    print(f"     Log Loss:  {best['log_loss']:.4f}")
    print(f"     RPS:       {best['rps']:.4f}")
    print(f"{'=' * 70}\n")

    return results_df, all_results


if __name__ == '__main__':
    results_df, all_results = main()
