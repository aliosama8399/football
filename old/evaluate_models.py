import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, confusion_matrix
from preprocess_data import preprocess_data

def evaluate_predictions(y_true, y_pred, threshold=0.5):
    """Evaluate predictions with various metrics"""
    # Round predictions to nearest integer
    y_pred_rounded = np.round(y_pred)
    
    # Exact match accuracy
    exact_matches = np.all(y_true == y_pred_rounded, axis=1)
    exact_accuracy = np.mean(exact_matches)
    
    # Goal difference accuracy
    true_diff = y_true[:, 0] - y_true[:, 1]
    pred_diff = y_pred_rounded[:, 0] - y_pred_rounded[:, 1]
    diff_accuracy = np.mean(true_diff == pred_diff)
    
    # Result accuracy (win/draw/loss)
    true_results = np.sign(true_diff)
    pred_results = np.sign(pred_diff)
    result_accuracy = np.mean(true_results == pred_results)
    
    # Mean absolute error for goals
    mae_home = np.mean(np.abs(y_true[:, 0] - y_pred_rounded[:, 0]))
    mae_away = np.mean(np.abs(y_true[:, 1] - y_pred_rounded[:, 1]))
    
    return {
        'exact_accuracy': exact_accuracy,
        'goal_diff_accuracy': diff_accuracy,
        'result_accuracy': result_accuracy,
        'mae_home': mae_home,
        'mae_away': mae_away
    }

def plot_score_distribution(y_true, y_pred, title):
    """Plot actual vs predicted score distribution"""
    plt.figure(figsize=(12, 5))
    
    # Plot home goals
    plt.subplot(1, 2, 1)
    sns.histplot(data={
        'Actual': y_true[:, 0],
        'Predicted': np.round(y_pred[:, 0])
    }, multiple="dodge")
    plt.title(f'{title} - Home Goals Distribution')
    
    # Plot away goals
    plt.subplot(1, 2, 2)
    sns.histplot(data={
        'Actual': y_true[:, 1],
        'Predicted': np.round(y_pred[:, 1])
    }, multiple="dodge")
    plt.title(f'{title} - Away Goals Distribution')
    
    plt.tight_layout()
    return plt

def main():
    # Load models and data
    models = joblib.load('match_prediction_models.joblib')
    X, y_scores, _, _, features, _, _ = preprocess_data()
    
    # Split data (use same seed as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y_scores, test_size=0.2, random_state=42)
    
    # Get predictions from both models
    full_pred = models['full_model'].predict(X_test)
    X_test_selected = X_test[:, models['selected_features']]
    selected_pred = models['selected_model'].predict(X_test_selected)
    
    # Evaluate both models
    print("\nFull Model Evaluation:")
    full_metrics = evaluate_predictions(y_test, full_pred)
    for metric, value in full_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nSelected Features Model Evaluation:")
    selected_metrics = evaluate_predictions(y_test, selected_pred)
    for metric, value in selected_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    plot_score_distribution(y_test, full_pred, "Full Model")
    plt.savefig('full_model_distribution.png')
    plt.close()
    
    plot_score_distribution(y_test, selected_pred, "Selected Features Model")
    plt.savefig('selected_model_distribution.png')
    plt.close()
    
    # Compare cross-validation results
    cv_comparison = pd.DataFrame({
        'Full Model': models['cv_scores_full'],
        'Selected Features': models['cv_scores_selected']
    })
    
    plt.figure(figsize=(8, 6))
    cv_comparison.boxplot()
    plt.title('Cross-validation Scores Comparison')
    plt.ylabel('Exact Match Accuracy')
    plt.savefig('cv_comparison.png')
    plt.close()
    
    print("\nCross-validation Results:")
    print(f"Full Model: {models['cv_scores_full'].mean():.4f} (+/- {models['cv_scores_full'].std() * 2:.4f})")
    print(f"Selected Features: {models['cv_scores_selected'].mean():.4f} (+/- {models['cv_scores_selected'].std() * 2:.4f})")

if __name__ == "__main__":
    main()



