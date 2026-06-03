"""
Merges sentiment features with price change labels.
Produces the final feature matrix used for modeling.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def merge_features_with_labels() -> pd.DataFrame:
    """
    Merge call_features.csv with all_price_changes.csv on (ticker, event_date).
    Returns the labeled feature matrix.
    """
    features_path = PROCESSED_DIR / "call_features.csv"
    prices_path = Path("data/raw/prices/all_price_changes.csv")

    if not features_path.exists():
        raise FileNotFoundError(f"Run sentiment_pipeline.py first: {features_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Run price_fetcher.py first: {prices_path}")

    features = pd.read_csv(features_path)
    prices = pd.read_csv(prices_path)

    df = features.merge(prices, on=["ticker", "event_date"], how="inner")
    logger.info(f"Merged dataset: {len(df)} rows, {df.shape[1]} columns")

    df = _clean_features(df)
    out = PROCESSED_DIR / "feature_matrix.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved feature matrix → {out}")
    return df


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN sentiment values with 0 and ensure numeric types."""
    sentiment_cols = [c for c in df.columns if any(
        k in c for k in ["sentiment", "ratio", "shift", "volatility", "count"]
    )]
    df[sentiment_cols] = df[sentiment_cols].fillna(0)
    df["direction"] = df["direction"].astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns to use for modeling."""
    exclude = {
        "ticker", "event_date", "source_file", "file",
        "direction", "pct_change_48h", "close_event",
        "close_48h", "post_date_48h",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


if __name__ == "__main__":
    df = merge_features_with_labels()
    feature_cols = get_feature_columns(df)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    logger.info(f"Class distribution:\n{df['direction'].value_counts()}")
