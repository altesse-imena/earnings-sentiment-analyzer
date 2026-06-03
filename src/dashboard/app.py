"""
Earnings Sentiment Analyzer — Dashboard
Dark Stripe-inspired design system.
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODEL_PATH    = PROCESSED_DIR / "model/lgbm_sentiment.pkl"
FEATURES_PATH = PROCESSED_DIR / "call_features.csv"
MATRIX_PATH   = PROCESSED_DIR / "feature_matrix.csv"
PRICES_DIR    = Path("data/raw/prices")
REPORTS_DIR   = PROCESSED_DIR / "reports"

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Earnings Sentiment Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────

BG         = "#0A0A0F"
SURFACE    = "#111118"
SURFACE_2  = "#16161F"
BORDER     = "#1E1E2E"
ACCENT     = "#635BFF"
ACCENT_DIM = "#4B44CC"
POS        = "#00C48C"
NEG        = "#FF4D6A"
TEXT_PRI   = "#F0F0F5"
TEXT_MUT   = "#8A8A9A"
TEXT_DIM   = "#525266"
WHITE      = "#FFFFFF"

# ── CSS ────────────────────────────────────────────────────────────────────────

st.html(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {{
    background-color: {BG} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: {TEXT_PRI} !important;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, header[data-testid="stHeader"] {{
    display: none !important;
}}

.block-container {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
    padding: 2rem 1.5rem !important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: {TEXT_PRI} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}}

/* Selectboxes */
[data-baseweb="select"] > div {{
    background-color: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    color: {TEXT_PRI} !important;
}}
[data-baseweb="select"] svg {{ fill: {TEXT_MUT} !important; }}
[data-baseweb="popover"] {{
    background-color: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[role="option"] {{
    background-color: {SURFACE} !important;
    color: {TEXT_PRI} !important;
    font-size: 0.85rem !important;
}}
[role="option"]:hover {{
    background-color: {SURFACE_2} !important;
}}

/* Buttons */
.stButton > button {{
    background-color: {ACCENT} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    width: 100% !important;
    transition: background-color 0.15s ease !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background-color: {ACCENT_DIM} !important;
}}

/* Plotly charts */
.stPlotlyChart {{ background: transparent !important; }}
.js-plotly-plot .plotly {{ background: transparent !important; }}

hr {{
    border: none !important;
    border-top: 1px solid {BORDER} !important;
    margin: 1.5rem 0 !important;
}}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* Code blocks */
.stCodeBlock code {{
    background-color: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    color: {TEXT_PRI} !important;
    font-size: 0.8125rem !important;
}}

/* Remove default metric styling */
[data-testid="metric-container"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}
</style>
""")

# ── Plotly base configs ────────────────────────────────────────────────────────
# Kept separate to avoid duplicate-key errors when overriding xaxis/yaxis per chart.

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_MUT, size=12),
    margin=dict(l=8, r=8, t=8, b=8),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUT, size=11),
    ),
)

def xaxis(**kwargs):
    base = dict(
        gridcolor=BORDER, gridwidth=1, showgrid=False,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(color=TEXT_MUT, size=11),
        title_font=dict(color=TEXT_MUT, size=11),
        zeroline=False,
    )
    base.update(kwargs)
    return base

def yaxis(**kwargs):
    base = dict(
        gridcolor=BORDER, gridwidth=1, showgrid=True,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=TEXT_MUT, size=11),
        title_font=dict(color=TEXT_MUT, size=11),
    )
    base.update(kwargs)
    return base


# ── SVG icons ──────────────────────────────────────────────────────────────────

def icon(name: str, color: str = TEXT_MUT, size: int = 18) -> str:
    s = size
    icons = {
        "trending-up":   f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "trending-down": f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
        "activity":      f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
        "user":          f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
        "cpu":           f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "bar-chart":     f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
        "message":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        "dollar":        f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
        "search":        f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
        "sliders":       f'<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    }
    return icons.get(name, "")


# ── Reusable components ────────────────────────────────────────────────────────

def section_label(title: str, icon_name: str = "activity"):
    st.html(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
        {icon(icon_name, ACCENT, 15)}
        <span style="font-size:0.72rem;font-weight:600;letter-spacing:0.09em;
                     text-transform:uppercase;color:{TEXT_MUT};">{title}</span>
    </div>""")


def stat_card(icon_name: str, value: str, label: str,
              value_color: str = TEXT_PRI, badge_text: str = "",
              badge_color: str = TEXT_MUT) -> str:
    badge = ""
    if badge_text:
        badge = f"""<span style="font-size:0.68rem;font-weight:600;color:{badge_color};
                    background:{hex_rgba(badge_color, 0.09)};
                    border:1px solid {hex_rgba(badge_color, 0.19)};
                    padding:1px 7px;border-radius:99px;">{badge_text}</span>"""
    return f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:1.1rem 1.1rem 0.9rem;height:100%;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    margin-bottom:0.7rem;">
            <div style="width:32px;height:32px;background:{SURFACE_2};border-radius:7px;
                        border:1px solid {BORDER};display:flex;align-items:center;
                        justify-content:center;flex-shrink:0;">
                {icon(icon_name, ACCENT, 15)}
            </div>
            {badge}
        </div>
        <div style="font-size:1.55rem;font-weight:700;color:{value_color};
                    letter-spacing:-0.02em;line-height:1.1;margin-bottom:0.2rem;">
            {value}
        </div>
        <div style="font-size:0.78rem;color:{TEXT_MUT};">{label}</div>
    </div>"""


def divider():
    st.html(
        f'<div style="border-top:1px solid {BORDER};margin:1.75rem 0;"></div>',
    )


def sidebar_row(label: str, value: str):
    st.html(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.45rem 0;border-bottom:1px solid {BORDER};">
        <span style="font-size:0.78rem;color:{TEXT_MUT};">{label}</span>
        <span style="font-size:0.78rem;font-weight:600;color:{TEXT_PRI};">{value}</span>
    </div>""")


def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color string to rgba() CSS notation."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning default for None, NaN, or non-numeric values."""
    try:
        v = float(val)
        return v if not math.isnan(v) else default
    except (TypeError, ValueError):
        return default


def sentiment_color(v: float) -> str:
    if v > 0.05: return POS
    if v < -0.05: return NEG
    return TEXT_DIM


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data
def load_feature_matrix():
    return pd.read_csv(MATRIX_PATH) if MATRIX_PATH.exists() else pd.DataFrame()

@st.cache_data
def load_call_features():
    return pd.read_csv(FEATURES_PATH) if FEATURES_PATH.exists() else pd.DataFrame()

@st.cache_data
def load_report():
    p = REPORTS_DIR / "performance_report.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data
def load_shap_values():
    p = REPORTS_DIR / "shap_values.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def load_price_data(ticker: str, event_date: str) -> pd.DataFrame:
    p = PRICES_DIR / f"{ticker}_{event_date}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    idx = pd.to_datetime(df.index)
    df.index = idx.tz_convert(None) if idx.tz is not None else idx
    return df


# ── Load data ──────────────────────────────────────────────────────────────────

model       = load_model()
df_matrix   = load_feature_matrix()
df_features = load_call_features()
report      = load_report()
shap_df     = load_shap_values()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.html(f"""
    <div style="margin-bottom:2rem;">
        <div style="font-size:1rem;font-weight:700;color:{TEXT_PRI};letter-spacing:-0.01em;">
            Earnings Sentiment
        </div>
        <div style="font-size:0.73rem;color:{TEXT_DIM};margin-top:3px;">
            NLP + price analytics
        </div>
    </div>""")

    st.html(f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:0.5rem;">
        {icon("search", TEXT_DIM, 12)}
        <span style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;
                     text-transform:uppercase;color:{TEXT_DIM};">Selection</span>
    </div>""")

    available_tickers = (
        sorted(df_features["ticker"].unique().tolist()) if not df_features.empty else []
    )
    ticker = st.selectbox(
        "Ticker",
        available_tickers if available_tickers else ["No data"],
        label_visibility="collapsed",
    )

    available_dates = []
    if ticker and not df_features.empty:
        available_dates = sorted(
            df_features[df_features["ticker"] == ticker]["event_date"].unique().tolist(),
            reverse=True,
        )
    event_date = st.selectbox(
        "Earnings Date",
        available_dates if available_dates else ["No data"],
        label_visibility="collapsed",
    )

    st.html("<div style='height:0.75rem'></div>")
    st.button("Analyze")

    st.html(
        f'<div style="border-top:1px solid {BORDER};margin:1.5rem 0;"></div>',
    )

    st.html(f"""
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:0.75rem;">
        {icon("sliders", TEXT_DIM, 12)}
        <span style="font-size:0.68rem;font-weight:600;letter-spacing:0.1em;
                     text-transform:uppercase;color:{TEXT_DIM};">Model Performance</span>
    </div>""")

    if report:
        sidebar_row("AUC-ROC",    str(report.get("auc_roc", "—")))
        sidebar_row("Accuracy",   str(report.get("accuracy", "—")))
        sidebar_row("CV AUC",     f"{report.get('cv_auc_mean','—')} ± {report.get('cv_auc_std','')}")
        sidebar_row("Top Feature",str(report.get("top_feature", "—"))[:22])
        sidebar_row("Samples",    str(report.get("n_samples", "—")))
    else:
        st.html(
            f'<p style="font-size:0.78rem;color:{TEXT_DIM};">Run train.py to populate.</p>',
        )

# ── Header ─────────────────────────────────────────────────────────────────────

st.html(f"""
<div style="padding:0.25rem 0 1.25rem;">
    <h1 style="margin:0;font-size:1.7rem;font-weight:700;color:{TEXT_PRI};
               letter-spacing:-0.03em;line-height:1.2;">
        Earnings Call Intelligence
    </h1>
    <p style="margin:0.35rem 0 0;font-size:0.875rem;color:{TEXT_MUT};font-weight:400;">
        FinBERT sentiment correlated with 48-hour post-earnings price movement
        &nbsp;&middot;&nbsp; {ticker} &nbsp;&middot;&nbsp; {event_date}
    </p>
</div>
<div style="border-top:1px solid {BORDER};margin-bottom:1.75rem;"></div>
""")

# ── No data guard ──────────────────────────────────────────────────────────────

if df_features.empty:
    st.html(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:2rem;text-align:center;color:{TEXT_MUT};font-size:0.875rem;">
        No processed data found. Run the pipeline first.
    </div>""")
    st.code(
        "python src/ingestion/edgar_fetcher.py --tickers AAPL MSFT --years 2023 2024\n"
        "python src/processing/sentiment_pipeline.py\n"
        "python src/modeling/train.py",
        language="bash",
    )
    st.stop()

row_df = df_features[
    (df_features["ticker"] == ticker) & (df_features["event_date"] == event_date)
]
if row_df.empty:
    st.html(
        f'<div style="color:{NEG};font-size:0.875rem;">No data for {ticker} on {event_date}.</div>',
    )
    st.stop()
row = row_df.iloc[0]

# ── Compute model prediction up front ─────────────────────────────────────────

pred_label = None
model_conf = None
actual_chg = None

if model is not None and not df_matrix.empty and report:
    fc = report.get("feature_columns", [])
    match = df_matrix[
        (df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == event_date)
    ]
    if not match.empty and fc:
        X_input    = match[fc].fillna(0).iloc[[0]]
        prob       = model.predict_proba(X_input)[0]
        pred_label = int(model.predict(X_input)[0])
        model_conf = float(prob[pred_label])
    if not match.empty and "pct_change_48h" in df_matrix.columns:
        actual_chg = float(match["pct_change_48h"].iloc[0])

# ── Stat cards ─────────────────────────────────────────────────────────────────

section_label("Overview", "activity")

overall  = safe_float(row.get("overall_sentiment"))
ceo_sent = safe_float(row.get("ceo_sentiment"))
tone_sh  = safe_float(row.get("tone_shift"))

c1, c2, c3, c4 = st.columns(4, gap="small")

with c1:
    st.html(stat_card(
        "activity", f"{overall:+.3f}", "Overall Sentiment",
        value_color=sentiment_color(overall),
    ))
with c2:
    st.html(stat_card(
        "user", f"{ceo_sent:+.3f}", "CEO Tone",
        value_color=sentiment_color(ceo_sent),
    ))
with c3:
    st.html(stat_card(
        "message", f"{tone_sh:+.3f}", "Q&A Sentiment Shift",
        value_color=sentiment_color(tone_sh),
    ))
with c4:
    if model_conf is not None:
        conf_color   = POS if pred_label == 1 else NEG
        badge_label  = "UP" if pred_label == 1 else "DOWN"
        st.html(stat_card(
            "cpu", f"{model_conf:.1%}", "Model Confidence",
            value_color=conf_color,
            badge_text=badge_label, badge_color=conf_color,
        ))
    else:
        st.html(stat_card("cpu", "—", "Model Confidence"))

divider()

# ── Sentiment breakdown charts ─────────────────────────────────────────────────

col_l, col_r = st.columns(2, gap="medium")

with col_l:
    section_label("Sentiment by Speaker", "user")
    speaker_vals = {
        role: safe_float(row.get(f"{role.lower()}_sentiment"))
        for role in ["CEO", "CFO", "ANALYST"]
    }
    fig_speaker = go.Figure(go.Bar(
        x=list(speaker_vals.keys()),
        y=list(speaker_vals.values()),
        marker=dict(
            color=[sentiment_color(v) for v in speaker_vals.values()],
            line=dict(width=0),
        ),
        text=[f"{v:+.3f}" for v in speaker_vals.values()],
        textposition="outside",
        textfont=dict(color=TEXT_MUT, size=11, family="Inter"),
        width=0.45,
    ))
    fig_speaker.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig_speaker.update_layout(
        **PLOT_BASE,
        xaxis=xaxis(),
        yaxis=yaxis(range=[-0.65, 0.65], title="Sentiment Score"),
        height=270,
    )
    st.plotly_chart(fig_speaker, use_container_width=True, config={"displayModeBar": False})

with col_r:
    section_label("Prepared Remarks vs Q&A", "message")
    prep = safe_float(row.get("prepared_sentiment"))
    qa   = safe_float(row.get("qa_sentiment"))
    fig_sections = go.Figure(go.Bar(
        x=["Prepared Remarks", "Q&A"],
        y=[prep, qa],
        marker=dict(
            color=[sentiment_color(prep), sentiment_color(qa)],
            line=dict(width=0),
        ),
        text=[f"{prep:+.3f}", f"{qa:+.3f}"],
        textposition="outside",
        textfont=dict(color=TEXT_MUT, size=11, family="Inter"),
        width=0.35,
    ))
    fig_sections.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig_sections.update_layout(
        **PLOT_BASE,
        xaxis=xaxis(),
        yaxis=yaxis(range=[-0.65, 0.65], title="Sentiment Score"),
        height=270,
    )
    st.plotly_chart(fig_sections, use_container_width=True, config={"displayModeBar": False})

divider()

# ── Price movement overlay ─────────────────────────────────────────────────────

section_label("5-Day Price Movement", "dollar")
price_df = load_price_data(ticker, event_date)

if not price_df.empty:
    fig_price = go.Figure()

    x_dates  = [d.strftime("%Y-%m-%d") for d in price_df.index]
    x_min    = price_df.index.min().strftime("%Y-%m-%d")
    x_max    = price_df.index.max().strftime("%Y-%m-%d")
    vline_x  = pd.Timestamp(event_date).strftime("%Y-%m-%d")

    fig_price.add_trace(go.Scatter(
        x=x_dates, y=price_df["Close"].tolist(),
        mode="lines+markers", name="Close Price",
        line=dict(color=TEXT_PRI, width=2),
        marker=dict(size=5, color=TEXT_PRI),
        yaxis="y1",
    ))
    fig_price.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[overall, overall],
        mode="lines",
        name=f"Sentiment ({overall:+.3f})",
        line=dict(color=ACCENT, width=1.5, dash="dot"),
        yaxis="y2",
        opacity=0.8,
    ))
    fig_price.add_shape(
        type="line",
        x0=vline_x, x1=vline_x, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=ACCENT, width=1, dash="dash"),
    )
    fig_price.add_annotation(
        x=vline_x, y=1, xref="x", yref="paper",
        text="Earnings Call",
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(color=TEXT_DIM, size=10, family="Inter"),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    )
    price_layout = dict(**PLOT_BASE)
    price_layout.update(
        xaxis=xaxis(showgrid=False),
        yaxis=yaxis(title="Price (USD)", showgrid=True),
        yaxis2=dict(
            overlaying="y", side="right",
            title="Sentiment", range=[-1.2, 1.2],
            showgrid=False, zeroline=False,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_DIM, size=10),
        ),
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(color=TEXT_MUT, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=320,
        margin=dict(l=8, r=8, t=32, b=8),
    )
    fig_price.update_layout(**price_layout)
    st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})
else:
    st.html(
        f'<p style="color:{TEXT_DIM};font-size:0.875rem;">Price data not available.</p>',
    )

divider()

# ── SHAP + Prediction ──────────────────────────────────────────────────────────

col_shap, col_pred = st.columns([3, 2], gap="medium")

with col_shap:
    section_label("Feature Importance", "bar-chart")

    if not shap_df.empty:
        top_n = shap_df.nlargest(8, "mean_abs_shap").sort_values("mean_abs_shap")
        n = len(top_n)
        shap_colors = [
            f"rgba(99,91,255,{0.25 + 0.75 * (i / max(n - 1, 1))})"
            for i in range(n)
        ]
        fig_shap = go.Figure(go.Bar(
            x=top_n["mean_abs_shap"],
            y=top_n["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker=dict(color=shap_colors, line=dict(width=0)),
            text=[f"{v:.4f}" for v in top_n["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color=TEXT_DIM, size=10, family="Inter"),
        ))
        fig_shap.update_layout(
            **PLOT_BASE,
            xaxis=xaxis(showgrid=True, title="Mean |SHAP Value|"),
            yaxis=yaxis(showgrid=False, tickfont=dict(color=TEXT_MUT, size=11)),
            height=330,
        )
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
    else:
        st.html(
            f'<p style="color:{TEXT_DIM};font-size:0.875rem;">Run train.py to generate SHAP values.</p>',
        )

with col_pred:
    section_label("Model Prediction", "cpu")

    if pred_label is not None and model_conf is not None:
        pred_color = POS if pred_label == 1 else NEG
        pred_text  = "UP" if pred_label == 1 else "DOWN"
        pred_icon  = "trending-up" if pred_label == 1 else "trending-down"
        conf_pct   = int(model_conf * 100)

        driver_text = ""
        if not shap_df.empty:
            top_feat     = shap_df.sort_values("mean_abs_shap", ascending=False).iloc[0]["feature"].replace("_", " ")
            dir_word     = "positive" if pred_label == 1 else "negative"
            driver_text  = f"Primary driver: <strong style='color:{TEXT_PRI};'>{top_feat}</strong> signals {dir_word} momentum heading into the call."

        actual_html = ""
        if actual_chg is not None:
            chg_color  = POS if actual_chg > 0 else NEG
            actual_html = f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.6rem 0.75rem;background:{SURFACE_2};border-radius:7px;
                        border:1px solid {BORDER};margin-bottom:1.1rem;">
                <span style="font-size:0.78rem;color:{TEXT_MUT};">Actual 48h change</span>
                <span style="font-size:0.875rem;font-weight:600;color:{chg_color};">
                    {actual_chg:+.2f}%
                </span>
            </div>"""

        st.html(f"""
        <div style="padding:0.25rem 0;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.4rem;">
                <div style="width:50px;height:50px;background:{hex_rgba(pred_color, 0.09)};border-radius:11px;
                            border:1px solid {hex_rgba(pred_color, 0.21)};display:flex;align-items:center;
                            justify-content:center;flex-shrink:0;">
                    {icon(pred_icon, pred_color, 22)}
                </div>
                <div>
                    <div style="font-size:1.9rem;font-weight:700;color:{pred_color};
                                letter-spacing:-0.03em;line-height:1;">{pred_text}</div>
                    <div style="font-size:0.75rem;color:{TEXT_DIM};margin-top:2px;">
                        48-hour direction prediction
                    </div>
                </div>
            </div>

            <div style="margin-bottom:1.3rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:0.78rem;color:{TEXT_MUT};">Confidence</span>
                    <span style="font-size:0.78rem;font-weight:600;color:{TEXT_PRI};">{conf_pct}%</span>
                </div>
                <div style="height:4px;background:{BORDER};border-radius:99px;overflow:hidden;">
                    <div style="height:100%;width:{conf_pct}%;background:{ACCENT};
                                border-radius:99px;"></div>
                </div>
            </div>

            {actual_html}

            <div style="font-size:0.79rem;color:{TEXT_DIM};line-height:1.6;
                        padding-top:0.75rem;border-top:1px solid {BORDER};">
                {driver_text}
            </div>
        </div>
        """)
    else:
        st.html(
            f'<p style="color:{TEXT_DIM};font-size:0.875rem;">Run train.py to generate predictions.</p>',
        )
