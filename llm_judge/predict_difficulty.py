"""
Predict IRT difficulty from task features.

This script extracts heuristic features from SWE-bench tasks and trains
a model to predict the IRT difficulty parameter (b).

Usage:
    python llm_judge/predict_difficulty.py --output_dir chris_output/difficulty_prediction
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract heuristic features from task data."""
    features = pd.DataFrame(index=df.index)

    # Text length features
    features['problem_len'] = df['problem_statement'].str.len()
    features['problem_lines'] = df['problem_statement'].str.count('\n')
    features['problem_words'] = df['problem_statement'].str.split().str.len()

    # Patch features
    features['patch_len'] = df['patch'].str.len()
    features['patch_lines'] = df['patch'].str.count('\n')
    features['patch_files'] = df['patch'].str.count(r'diff --git')

    # Test patch features
    if 'test_patch' in df.columns:
        features['test_patch_len'] = df['test_patch'].str.len()
        features['test_patch_lines'] = df['test_patch'].str.count('\n')

    # Code indicators in problem statement
    features['has_code_block'] = df['problem_statement'].str.contains('```').astype(int)
    features['has_traceback'] = df['problem_statement'].str.lower().str.contains(
        'traceback|error|exception|typeerror|valueerror|keyerror'
    ).astype(int)
    features['has_example'] = df['problem_statement'].str.lower().str.contains(
        'example|e\\.g\\.|for instance'
    ).astype(int)

    # Repository (categorical)
    features['repo'] = df['repo']

    # SWE-bench difficulty label (if available)
    if 'swebench_difficulty' in df.columns:
        difficulty_map = {
            '<15 min fix': 0,
            '15 min - 1 hour': 1,
            '1-4 hours': 2,
            '>4 hours': 3,
        }
        features['swebench_difficulty_num'] = df['swebench_difficulty'].map(difficulty_map)

    return features


def build_model_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Build a sklearn pipeline with preprocessing and model."""

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])

    return model


def evaluate_models(X: pd.DataFrame, y: pd.Series,
                   numeric_features: List[str],
                   categorical_features: List[str]) -> Dict:
    """Evaluate multiple models using cross-validation."""

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ]
    )

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    }

    results = {}
    for name, regressor in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        cv_rmse = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')

        results[name] = {
            'r2_mean': cv_scores.mean(),
            'r2_std': cv_scores.std(),
            'rmse_mean': -cv_rmse.mean(),
            'rmse_std': cv_rmse.std(),
        }

    return results


def get_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.Series:
    """Extract feature importance from trained model."""
    regressor = model.named_steps['regressor']

    if hasattr(regressor, 'feature_importances_'):
        # For tree-based models
        importances = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        # For linear models
        importances = np.abs(regressor.coef_)
    else:
        return None

    # Get transformed feature names
    preprocessor = model.named_steps['preprocessor']
    transformed_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == 'num':
            transformed_names.extend(columns)
        elif name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
            transformed_names.extend(transformer.get_feature_names_out(columns))

    if len(transformed_names) == len(importances):
        return pd.Series(importances, index=transformed_names).sort_values(ascending=False)
    else:
        return pd.Series(importances).sort_values(ascending=False)


def plot_results(y_true: np.ndarray, y_pred: np.ndarray,
                feature_importance: pd.Series,
                output_dir: Path) -> None:
    """Create visualization of prediction results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.5, s=30)

    # Add diagonal line
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax1.plot(lims, lims, 'r--', alpha=0.8, label='Perfect prediction')

    ax1.set_xlabel('Actual IRT Difficulty (b)')
    ax1.set_ylabel('Predicted IRT Difficulty (b)')
    ax1.set_title(f'Actual vs Predicted (r={np.corrcoef(y_true, y_pred)[0,1]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Feature Importance
    ax2 = axes[1]
    if feature_importance is not None:
        top_features = feature_importance.head(15)
        colors = ['green' if 'repo' not in str(idx) else 'blue' for idx in top_features.index]
        bars = ax2.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels([str(x)[:30] for x in top_features.index], fontsize=8)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 15 Features')
        ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict IRT difficulty from task features')
    parser.add_argument('--irt_path', type=str,
                       default='clean_data/swebench_verified_20250930_full/1d/items.csv',
                       help='Path to IRT items.csv')
    parser.add_argument('--output_dir', type=str,
                       default='chris_output/difficulty_prediction',
                       help='Output directory for results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load IRT difficulty
    print("Loading IRT difficulty parameters...")
    items = pd.read_csv(args.irt_path, index_col=0)

    # Load SWE-bench dataset
    if not HAS_DATASETS:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    swebench_df = pd.DataFrame({
        'instance_id': ds['instance_id'],
        'repo': ds['repo'],
        'problem_statement': ds['problem_statement'],
        'swebench_difficulty': ds['difficulty'],
        'patch': ds['patch'],
        'test_patch': ds['test_patch'],
    })
    swebench_df = swebench_df.set_index('instance_id')

    # Merge with IRT difficulty
    merged = swebench_df.join(items[['b', 'a']], how='inner')
    print(f"Merged dataset: {len(merged)} tasks")

    # Extract features
    print("Extracting features...")
    features = extract_features(merged)

    # Define feature groups
    numeric_features = [
        'problem_len', 'problem_lines', 'problem_words',
        'patch_len', 'patch_lines', 'patch_files',
        'test_patch_len', 'test_patch_lines',
        'has_code_block', 'has_traceback', 'has_example',
        'swebench_difficulty_num',
    ]
    categorical_features = ['repo']

    # Prepare data
    X = features[numeric_features + categorical_features].copy()
    y = merged['b']

    # Handle missing values
    X = X.fillna(0)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION (5-fold CV)")
    print("=" * 60)

    results = evaluate_models(X, y, numeric_features, categorical_features)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('r2_mean', ascending=False)
    print(results_df.to_string())

    # Train best model on all data for feature importance
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    model = build_model_pipeline(numeric_features, categorical_features)
    model.fit(X, y)

    # Get predictions
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Feature importance
    feature_importance = get_feature_importance(model, numeric_features + categorical_features)

    print("\nTop 10 Feature Importances:")
    if feature_importance is not None:
        print(feature_importance.head(10).to_string())

    # Plot results
    print(f"\nSaving results to {output_dir}...")
    plot_results(y.values, y_pred, feature_importance, output_dir)

    # Save predictions
    predictions_df = pd.DataFrame({
        'actual_b': y,
        'predicted_b': y_pred,
        'residual': y - y_pred,
    }, index=merged.index)
    predictions_df.to_csv(output_dir / 'predictions.csv')

    # Save model evaluation results
    results_df.to_csv(output_dir / 'model_comparison.csv')

    # Summary statistics
    corr = np.corrcoef(y, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Correlation (actual vs predicted): {corr:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Difficulty range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
