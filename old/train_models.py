import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import joblib
from preprocess_data import preprocess_data
import seaborn as sns
import matplotlib.pyplot as plt

def select_features(X_df, y, n_features=20):
    """Select top features based on correlation and feature importance"""
    # Calculate correlation with target
    correlations = []
    for i in range(y.shape[1]):  # For both home and away goals
        corr = np.abs(pd.DataFrame(X_df).corrwith(pd.Series(y[:, i])))
        correlations.append(corr)
    
    # Average correlation for both targets
    mean_corr = pd.concat(correlations, axis=1).mean(axis=1)
    
    # Select top features
    selected_features = mean_corr.nlargest(n_features).index.tolist()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pd.DataFrame(X_df[:, selected_features]).corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png')
    plt.close()
    
    return selected_features

def custom_score(y_true, y_pred):
    """Custom scoring function for exact score prediction"""
    # Convert predictions to rounded integers
    y_pred_rounded = np.round(y_pred).astype(int)
    # Calculate exact match accuracy
    exact_matches = np.all(y_true == y_pred_rounded, axis=1)
    return np.mean(exact_matches)

def train_models():
    # Get preprocessed data
    X, y_scores, y_expected_goals, y_results, feature_list, label_encoders, feature_scaler = preprocess_data()

    # Convert X to DataFrame for feature selection
    X_df = pd.DataFrame(X, columns=feature_list)

    # Split data
    X_train, X_test, y_scores_train, y_scores_test = train_test_split(
        X, y_scores, test_size=0.2, random_state=42
    )

    # Define cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    custom_scorer = make_scorer(custom_score)

    # 1. Train on all features
    print("\nTraining model with all features...")
    xgb_full = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )

    # Cross-validation for full model
    cv_scores_full = cross_val_score(
        xgb_full, X_train, y_scores_train, 
        cv=kfold, scoring=custom_scorer
    )

    # Train and evaluate full model
    xgb_full.fit(X_train, y_scores_train)
    full_predictions = xgb_full.predict(X_test)
    full_mse = mean_squared_error(y_scores_test, full_predictions, multioutput='raw_values')
    
    print("\nModel with all features:")
    print(f"Cross-validation scores (exact match): {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std() * 2:.4f})")
    print(f"Home goals MSE: {full_mse[0]:.4f}")
    print(f"Away goals MSE: {full_mse[1]:.4f}")

    # 2. Feature selection and training with selected features
    print("\nPerforming feature selection...")
    n_features = 20  # Number of features to select
    selected_features = select_features(X_train, y_scores_train, n_features)
    
    # Train model with selected features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    xgb_selected = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        random_state=42
    )

    # Cross-validation for selected features model
    cv_scores_selected = cross_val_score(
        xgb_selected, X_train_selected, y_scores_train, 
        cv=kfold, scoring=custom_scorer
    )

    # Train and evaluate selected features model
    xgb_selected.fit(X_train_selected, y_scores_train)
    selected_predictions = xgb_selected.predict(X_test_selected)
    selected_mse = mean_squared_error(y_scores_test, selected_predictions, multioutput='raw_values')
    
    print("\nModel with selected features:")
    print(f"Cross-validation scores (exact match): {cv_scores_selected.mean():.4f} (+/- {cv_scores_selected.std() * 2:.4f})")
    print(f"Home goals MSE: {selected_mse[0]:.4f}")
    print(f"Away goals MSE: {selected_mse[1]:.4f}")

    # Save feature importance plot
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(xgb_full, max_num_features=20)
    plt.title('Feature Importance (Full Model)')
    plt.tight_layout()
    plt.savefig('feature_importance_full.png')
    plt.close()

    # Save models and metadata
    models = {
        'full_model': xgb_full,
        'selected_model': xgb_selected,
        'feature_scaler': feature_scaler,
        'label_encoders': label_encoders,
        'all_features': feature_list,
        'selected_features': selected_features,
        'cv_scores_full': cv_scores_full,
        'cv_scores_selected': cv_scores_selected
    }
    
    joblib.dump(models, 'match_prediction_models.joblib')
    
    return models

if __name__ == "__main__":
    train_models()




