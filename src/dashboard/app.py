"""
Streamlit dashboard for the Earnings Sentiment Analyzer.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

PROCESSED_DIR = Path("data/processed")
MODEL_PATH = PROCESSED_DIR / "model/lgbm_sentiment.pkl"
FEATURES_PATH = PROCESSED_DIR / "call_features.csv"
MATRIX_PATH = PROCESSED_DIR / "feature_matrix.csv"
PRICES_DIR = Path("data/raw/prices")
REPORTS_DIR = PROCESSED_DIR / "reports"

st.set_page_config(
    page_title="Earnings Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_matrix():
    if not MATRIX_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATRIX_PATH)


@st.cache_data
def load_call_features():
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_report():
    p = REPORTS_DIR / "performance_report.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_shap_values():
    p = REPORTS_DIR / "shap_values.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_price_data(ticker: str, event_date: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}_{event_date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def sentiment_color(score: float) -> str:
    if score > 0.1:
        return "#2ecc71"
    elif score < -0.1:
        return "#e74c3c"
    return "#f39c12"


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("📈 Earnings Sentiment Analyzer")
st.caption("NLP-driven analysis of earnings call transcripts correlated with post-earnings price movement.")

model = load_model()
df_matrix = load_feature_matrix()
df_features = load_call_features()
report = load_report()
shap_df = load_shap_values()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Search")

    available_tickers = sorted(df_features["ticker"].unique().tolist()) if not df_features.empty else []
    ticker = st.selectbox("Ticker", available_tickers if available_tickers else ["No data yet"])

    available_dates = []
    if ticker and not df_features.empty:
        available_dates = sorted(
            df_features[df_features["ticker"] == ticker]["event_date"].unique().tolist(),
            reverse=True,
        )

    event_date = st.selectbox("Earnings Date", available_dates if available_dates else ["No data yet"])

    st.divider()
    st.header("📊 Model Performance")
    if report:
        st.metric("AUC-ROC", report.get("auc_roc", "—"))
        st.metric("Accuracy", report.get("accuracy", "—"))
        st.metric("CV AUC", f"{report.get('cv_auc_mean', '—')} ± {report.get('cv_auc_std', '')}")
        st.metric("Top Feature", report.get("top_feature", "—"))
        st.metric("Samples", report.get("n_samples", "—"))
    else:
        st.info("Run `python src/modeling/train.py` to populate model metrics.")

# ── Main content ──────────────────────────────────────────────────────────────

if df_features.empty:
    st.warning("No processed data found. Run the full pipeline first:")
    st.code(
        "python src/ingestion/edgar_fetcher.py --tickers AAPL MSFT --years 2023 2024\n"
        "python src/processing/sentiment_pipeline.py\n"
        "python src/modeling/train.py",
        language="bash",
    )
    st.stop()

row = df_features[(df_features["ticker"] == ticker) & (df_features["event_date"] == event_date)]
if row.empty:
    st.error(f"No sentiment data found for {ticker} on {event_date}")
    st.stop()

row = row.iloc[0]

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall Sentiment", f"{row.get('overall_sentiment', 0):.3f}")
col2.metric("Sentiment Volatility", f"{row.get('sentiment_volatility', 0):.3f}")
col3.metric("Positive Ratio", f"{row.get('positive_ratio', 0):.1%}")
col4.metric("Negative Ratio", f"{row.get('negative_ratio', 0):.1%}")
col5.metric("Tone Shift (P→Q&A)", f"{row.get('tone_shift', 0):+.3f}")

st.divider()

# ── Sentiment by speaker and section ─────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎤 Sentiment by Speaker")
    speaker_data = {
        role: row.get(f"{role.lower()}_sentiment", 0)
        for role in ["CEO", "CFO", "ANALYST"]
        if not pd.isna(row.get(f"{role.lower()}_sentiment", np.nan))
    }
    if speaker_data:
        fig = go.Figure(go.Bar(
            x=list(speaker_data.keys()),
            y=list(speaker_data.values()),
            marker_color=[sentiment_color(v) for v in speaker_data.values()],
            text=[f"{v:.3f}" for v in speaker_data.values()],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Speaker data not available for this transcript.")

with col_right:
    st.subheader("📑 Prepared Remarks vs Q&A")
    section_data = {
        "Prepared": row.get("prepared_sentiment", 0),
        "Q&A": row.get("qa_sentiment", 0),
    }
    fig2 = go.Figure(go.Bar(
        x=list(section_data.keys()),
        y=list(section_data.values()),
        marker_color=[sentiment_color(v) for v in section_data.values()],
        text=[f"{v:.3f}" for v in section_data.values()],
        textposition="outside",
    ))
    fig2.update_layout(
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1]),
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Price chart with sentiment overlay ───────────────────────────────────────

st.subheader("💹 5-Day Price Movement + Sentiment Overlay")
price_df = load_price_data(ticker, event_date)

if not price_df.empty:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=price_df.index, y=price_df["Close"],
        mode="lines+markers", name="Close Price",
        line=dict(color="#6E96F7", width=2),
        marker=dict(size=6),
    ))
    # Shade the event date
    event_ts = pd.Timestamp(event_date)
    fig3.add_vline(
        x=event_ts, line_dash="dash", line_color="orange",
        annotation_text="Earnings Call", annotation_position="top right",
    )
    # Add sentiment as secondary annotation
    overall = row.get("overall_sentiment", 0)
    fig3.add_annotation(
        x=event_ts, y=price_df["Close"].min(),
        text=f"Sentiment: {overall:+.3f}",
        showarrow=False,
        font=dict(color=sentiment_color(overall), size=12),
        yshift=-20,
    )
    fig3.update_layout(
        xaxis_title="Date", yaxis_title="Price (USD)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Price data not available for this ticker/date combination.")

st.divider()

# ── SHAP feature importance ───────────────────────────────────────────────────

st.subheader("🔬 SHAP Feature Importance")

if not shap_df.empty:
    top_n = shap_df.nlargest(10, "mean_abs_shap")
    fig4 = go.Figure(go.Bar(
        x=top_n["mean_abs_shap"],
        y=top_n["feature"],
        orientation="h",
        marker_color="#6E96F7",
    ))
    fig4.update_layout(
        xaxis_title="Mean |SHAP Value|",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    st.plotly_chart(fig4, use_container_width=True)

    shap_img = REPORTS_DIR / "shap_beeswarm.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="SHAP Beeswarm — full feature impact distribution")
else:
    st.info("SHAP values not available. Run `python src/modeling/train.py`.")

st.divider()

# ── Model prediction ──────────────────────────────────────────────────────────

st.subheader("🤖 Model Prediction")

if model is not None and not df_matrix.empty and report:
    feature_cols = report.get("feature_columns", [])
    match = df_matrix[
        (df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == event_date)
    ]
    if not match.empty and feature_cols:
        X_input = match[feature_cols].fillna(0).iloc[[0]]
        prob = model.predict_proba(X_input)[0]
        pred = int(model.predict(X_input)[0])
        confidence = prob[pred]

        col_pred, col_conf, col_actual = st.columns(3)
        col_pred.metric(
            "Predicted Direction",
            "📈 UP" if pred == 1 else "📉 DOWN",
        )
        col_conf.metric("Confidence", f"{confidence:.1%}")

        actual_row = df_matrix[
            (df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == event_date)
        ]
        if not actual_row.empty and "pct_change_48h" in actual_row.columns:
            actual_chg = actual_row["pct_change_48h"].iloc[0]
            col_actual.metric("Actual 48h Change", f"{actual_chg:+.2f}%")
    else:
        st.info("No matching feature row found for this ticker/date in the model dataset.")
else:
    st.info("Model not loaded. Run `python src/modeling/train.py` to train.")
