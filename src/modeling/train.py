"""
Trains a LightGBM classifier to predict 48-hour post-earnings price direction
from sentiment features. Includes SHAP explainability and evaluation report.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.processing.feature_engineering import get_feature_columns, merge_features_with_labels

MODEL_DIR = Path("data/processed/model")
REPORTS_DIR = Path("data/processed/reports")


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = merge_features_with_labels()
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["direction"]
    logger.info(f"Dataset: {len(df)} samples, {len(feature_cols)} features, class balance: {y.mean():.2%} positive")
    return X, y, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[lgb.LGBMClassifier, np.ndarray]:
    """Train LightGBM with cross-validation."""
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X, y)
    return model, cv_scores


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Generate full evaluation metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "auc_roc": round(roc_auc_score(y, y_prob), 4),
        "classification_report": classification_report(y, y_pred, output_dict=True),
    }
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
    return metrics


def generate_shap_report(model, X: pd.DataFrame, feature_cols: list[str]):
    """Generate SHAP feature importance plot and values."""
    logger.info("Computing SHAP values...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, take class 1 SHAP values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, feature_names=feature_cols, show=False, plot_type="bar")
    plt.title("SHAP Feature Importance — Sentiment → Price Direction", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X, feature_names=feature_cols, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save mean SHAP values
    mean_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    mean_shap.to_csv(REPORTS_DIR / "shap_values.csv", index=False)

    logger.info(f"Top 5 features by SHAP:\n{mean_shap.head()}")
    return mean_shap


def generate_confusion_matrix_plot(y_true, y_pred):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — 48h Price Direction")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_training():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = load_data()
    model, cv_scores = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    generate_confusion_matrix_plot(y, model.predict(X))
    shap_df = generate_shap_report(model, X, feature_cols)

    # Save model
    model_path = MODEL_DIR / "lgbm_sentiment.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    # Save full report
    report = {
        **metrics,
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "top_feature": shap_df.iloc[0]["feature"],
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
    }
    report_path = REPORTS_DIR / "performance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved → {report_path}")
    logger.info(f"\n{'='*50}\nFINAL RESULTS\n{'='*50}\nAUC-ROC : {metrics['auc_roc']}\nAccuracy: {metrics['accuracy']}\nCV AUC  : {report['cv_auc_mean']} ± {report['cv_auc_std']}\nTop feat: {report['top_feature']}\n{'='*50}")


if __name__ == "__main__":
    run_training()
