"""
Correlation analysis between sentiment features and post-earnings price change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.processing.feature_engineering import get_feature_columns, merge_features_with_labels

REPORTS_DIR = Path("data/processed/reports")


def run_correlation_analysis():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = merge_features_with_labels()
    feature_cols = get_feature_columns(df)

    target = "pct_change_48h"
    corr = df[feature_cols + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)

    # Bar chart of correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with 48h Price Change")
    ax.set_title("Sentiment Feature Correlations with Post-Earnings Return")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Heatmap of top features
    top_features = corr.abs().nlargest(10).index.tolist()
    corr_matrix = df[top_features + [target]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, square=True, linewidths=0.5, ax=ax
    )
    ax.set_title("Correlation Matrix — Top Sentiment Features vs Price Change")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save correlation table
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "correlation_with_48h_return"]
    corr_df.to_csv(REPORTS_DIR / "feature_correlations.csv", index=False)
    logger.info(f"Top correlated features:\n{corr_df.head(10)}")

    return corr_df


if __name__ == "__main__":
    run_correlation_analysis()
