"""
Earnings Sentiment Analyzer — Dashboard
Dark Stripe-inspired design system.
"""

import json
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

# ── Page config (must be first Streamlit call) ─────────────────────────────────

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

# ── Global CSS injection ───────────────────────────────────────────────────────

def inject_css():
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

    <style>
    /* ── Reset & base ── */
    *, *::before, *::after {{ box-sizing: border-box; }}

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp {{
        background-color: {BG} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: {TEXT_PRI} !important;
    }}

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"],
    .stDeployButton, header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* ── Remove default padding ── */
    .block-container {{
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1400px !important;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding: 2rem 1.5rem !important;
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{
        color: {TEXT_PRI} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_PRI} !important;
        border-radius: 6px !important;
    }}

    /* ── Selectbox dropdown ── */
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_PRI} !important;
    }}
    .stSelectbox [data-baseweb="select"] svg {{
        fill: {TEXT_MUT} !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        background-color: {ACCENT} !important;
        color: {WHITE} !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1.25rem !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease !important;
        width: 100% !important;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_DIM} !important;
    }}

    /* ── Plotly chart containers ── */
    .stPlotlyChart {{
        background: transparent !important;
        border: none !important;
    }}
    .js-plotly-plot .plotly {{
        background: transparent !important;
    }}

    /* ── Horizontal rule ── */
    hr {{
        border: none !important;
        border-top: 1px solid {BORDER} !important;
        margin: 1.5rem 0 !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

    /* ── Code blocks ── */
    .stCodeBlock, code {{
        background-color: {SURFACE_2} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_PRI} !important;
        font-size: 0.8125rem !important;
    }}

    /* ── Info / warning / error boxes ── */
    .stAlert {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        color: {TEXT_PRI} !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {{
        background-color: {ACCENT} !important;
    }}
    .stProgress > div > div > div {{
        background-color: {BORDER} !important;
        border-radius: 99px !important;
    }}

    /* ── Divider ── */
    [data-testid="stVerticalBlock"] > hr {{
        border-top: 1px solid {BORDER} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


inject_css()

# ── Plotly dark template ───────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_MUT, size=12),
    xaxis=dict(
        gridcolor=BORDER, gridwidth=1, showgrid=False,
        linecolor=BORDER, tickcolor=BORDER,
        tickfont=dict(color=TEXT_MUT, size=11),
        title_font=dict(color=TEXT_MUT, size=11),
    ),
    yaxis=dict(
        gridcolor=BORDER, gridwidth=1, showgrid=True,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=TEXT_MUT, size=11),
        title_font=dict(color=TEXT_MUT, size=11),
    ),
    margin=dict(l=8, r=8, t=8, b=8),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUT, size=11),
    ),
)

# ── SVG icon library ───────────────────────────────────────────────────────────

def icon(name: str, color: str = TEXT_MUT, size: int = 18) -> str:
    s = size
    icons = {
        "trending-up": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>""",
        "trending-down": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>""",
        "activity": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>""",
        "user": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>""",
        "cpu": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>""",
        "bar-chart": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>""",
        "message-square": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>""",
        "dollar-sign": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>""",
        "zap": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>""",
        "search": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>""",
        "sliders": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>""",
        "arrow-up-right": f"""<svg width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>""",
    }
    return icons.get(name, "")


# ── HTML component helpers ─────────────────────────────────────────────────────

def section_header(title: str, icon_name: str = "activity", subtitle: str = ""):
    sub_html = f'<p style="margin:0;font-size:0.8rem;color:{TEXT_DIM};letter-spacing:0.02em;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.25rem;">
        <div style="flex-shrink:0;">{icon(icon_name, ACCENT, 18)}</div>
        <div>
            <h3 style="margin:0;font-size:0.875rem;font-weight:600;
                       letter-spacing:0.06em;text-transform:uppercase;
                       color:{TEXT_MUT};">{title}</h3>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(icon_name: str, value: str, label: str, value_color: str = TEXT_PRI, badge: str = ""):
    badge_html = f'<span style="font-size:0.7rem;font-weight:500;color:{TEXT_MUT};background:{BORDER};padding:2px 8px;border-radius:99px;margin-left:8px;">{badge}</span>' if badge else ""
    return f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:1.25rem 1.25rem 1rem;display:flex;flex-direction:column;gap:0.6rem;
                height:100%;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="width:34px;height:34px;background:{SURFACE_2};border-radius:8px;
                        border:1px solid {BORDER};display:flex;align-items:center;
                        justify-content:center;">
                {icon(icon_name, ACCENT, 16)}
            </div>
            {badge_html}
        </div>
        <div>
            <div style="font-size:1.6rem;font-weight:700;color:{value_color};
                        letter-spacing:-0.02em;line-height:1.1;">{value}</div>
            <div style="font-size:0.8rem;color:{TEXT_MUT};margin-top:0.2rem;
                        font-weight:400;">{label}</div>
        </div>
    </div>"""


def divider():
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:2rem 0;">', unsafe_allow_html=True)


def chart_card_open(title: str, icon_name: str = "bar-chart", subtitle: str = ""):
    sub_html = f'<span style="font-size:0.8rem;color:{TEXT_DIM};margin-left:8px;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:1.25rem 1.25rem 0.5rem;margin-bottom:0;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem;">
            {icon(icon_name, ACCENT, 16)}
            <span style="font-size:0.8rem;font-weight:600;letter-spacing:0.05em;
                         text-transform:uppercase;color:{TEXT_MUT};">{title}</span>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def chart_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────

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


# ── Data ───────────────────────────────────────────────────────────────────────

model       = load_model()
df_matrix   = load_feature_matrix()
df_features = load_call_features()
report      = load_report()
shap_df     = load_shap_values()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:2rem;">
        <div style="font-size:1rem;font-weight:700;color:{TEXT_PRI};letter-spacing:-0.01em;">
            Earnings Sentiment
        </div>
        <div style="font-size:0.75rem;color:{TEXT_DIM};margin-top:2px;">
            NLP + price analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
              text-transform:uppercase;color:{TEXT_DIM};margin-bottom:0.5rem;">
        {icon("search", TEXT_DIM, 12)}&nbsp; Selection
    </p>
    """, unsafe_allow_html=True)

    available_tickers = sorted(df_features["ticker"].unique().tolist()) if not df_features.empty else []
    ticker = st.selectbox("Ticker", available_tickers if available_tickers else ["No data yet"], label_visibility="collapsed")

    available_dates = []
    if ticker and not df_features.empty:
        available_dates = sorted(
            df_features[df_features["ticker"] == ticker]["event_date"].unique().tolist(),
            reverse=True,
        )
    event_date = st.selectbox("Earnings Date", available_dates if available_dates else ["No data yet"], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Analyze")

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:1.5rem 0;">', unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
              text-transform:uppercase;color:{TEXT_DIM};margin-bottom:0.75rem;">
        {icon("sliders", TEXT_DIM, 12)}&nbsp; Model Performance
    </p>
    """, unsafe_allow_html=True)

    if report:
        def sidebar_metric(label, value):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.5rem 0;border-bottom:1px solid {BORDER};">
                <span style="font-size:0.78rem;color:{TEXT_MUT};">{label}</span>
                <span style="font-size:0.78rem;font-weight:600;color:{TEXT_PRI};">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        sidebar_metric("AUC-ROC", report.get("auc_roc", "—"))
        sidebar_metric("Accuracy", report.get("accuracy", "—"))
        sidebar_metric("CV AUC", f"{report.get('cv_auc_mean','—')} ± {report.get('cv_auc_std','')}")
        sidebar_metric("Top Feature", str(report.get("top_feature", "—"))[:20])
        sidebar_metric("Samples", report.get("n_samples", "—"))
    else:
        st.markdown(f'<p style="font-size:0.78rem;color:{TEXT_DIM};">Run train.py to populate metrics.</p>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="padding:0.5rem 0 1.5rem;">
    <h1 style="margin:0;font-size:1.75rem;font-weight:700;color:{TEXT_PRI};
               letter-spacing:-0.03em;line-height:1.2;">
        Earnings Call Intelligence
    </h1>
    <p style="margin:0.4rem 0 0;font-size:0.9rem;color:{TEXT_MUT};font-weight:400;">
        FinBERT sentiment analysis correlated with 48-hour post-earnings price movement
    </p>
</div>
<hr style="border:none;border-top:1px solid {BORDER};margin:0 0 2rem;">
""", unsafe_allow_html=True)

# ── No data state ──────────────────────────────────────────────────────────────

if df_features.empty:
    st.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
                padding:2rem;text-align:center;">
        <div style="color:{TEXT_MUT};font-size:0.875rem;margin-bottom:1rem;">
            No processed data found. Run the full pipeline first.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.code(
        "python src/ingestion/edgar_fetcher.py --tickers AAPL MSFT --years 2023 2024\n"
        "python src/processing/sentiment_pipeline.py\n"
        "python src/modeling/train.py",
        language="bash",
    )
    st.stop()

row = df_features[(df_features["ticker"] == ticker) & (df_features["event_date"] == event_date)]
if row.empty:
    st.markdown(f'<div style="color:{NEG};font-size:0.875rem;">No data for {ticker} on {event_date}.</div>', unsafe_allow_html=True)
    st.stop()
row = row.iloc[0]

# ── Stat cards ─────────────────────────────────────────────────────────────────

overall   = row.get("overall_sentiment", 0.0)
ceo_sent  = row.get("ceo_sentiment", 0.0)
tone_sh   = row.get("tone_shift", 0.0)

def sentiment_label(v):
    if v > 0.1: return POS
    if v < -0.1: return NEG
    return TEXT_MUT

model_conf = None
pred_label = None
if model is not None and not df_matrix.empty and report:
    fc = report.get("feature_columns", [])
    match = df_matrix[(df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == event_date)]
    if not match.empty and fc:
        X_input = match[fc].fillna(0).iloc[[0]]
        prob = model.predict_proba(X_input)[0]
        pred = int(model.predict(X_input)[0])
        model_conf = prob[pred]
        pred_label = pred

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(stat_card(
        "activity",
        f"{overall:+.3f}",
        "Overall Sentiment Score",
        value_color=sentiment_label(overall),
    ), unsafe_allow_html=True)

with col2:
    st.markdown(stat_card(
        "user",
        f"{ceo_sent:+.3f}",
        "CEO Tone",
        value_color=sentiment_label(ceo_sent),
    ), unsafe_allow_html=True)

with col3:
    ts_val = f"{tone_sh:+.3f}"
    st.markdown(stat_card(
        "message-square",
        ts_val,
        "Q&A Sentiment Shift",
        value_color=sentiment_label(tone_sh),
    ), unsafe_allow_html=True)

with col4:
    if model_conf is not None:
        conf_color = POS if pred_label == 1 else NEG
        st.markdown(stat_card(
            "cpu",
            f"{model_conf:.1%}",
            "Model Confidence",
            value_color=conf_color,
            badge="UP" if pred_label == 1 else "DOWN",
        ), unsafe_allow_html=True)
    else:
        st.markdown(stat_card("cpu", "—", "Model Confidence"), unsafe_allow_html=True)

divider()

# ── Sentiment breakdown charts ─────────────────────────────────────────────────

col_l, col_r = st.columns(2, gap="medium")

with col_l:
    chart_card_open("Sentiment by Speaker", "user")

    speaker_data = {
        role: float(row.get(f"{role.lower()}_sentiment", 0) or 0)
        for role in ["CEO", "CFO", "ANALYST"]
    }
    bar_colors = [POS if v > 0.05 else (NEG if v < -0.05 else TEXT_DIM) for v in speaker_data.values()]

    fig = go.Figure(go.Bar(
        x=list(speaker_data.keys()),
        y=list(speaker_data.values()),
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in speaker_data.values()],
        textposition="outside",
        textfont=dict(color=TEXT_MUT, size=11, family="Inter"),
        width=0.45,
    ))
    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig.update_layout(
        **PLOT_LAYOUT,
        yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[-0.6, 0.6], title="Sentiment Score"),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    chart_card_close()

with col_r:
    chart_card_open("Prepared Remarks vs Q&A", "message-square")

    prep = float(row.get("prepared_sentiment", 0) or 0)
    qa   = float(row.get("qa_sentiment", 0) or 0)
    section_vals = [prep, qa]
    section_labels = ["Prepared Remarks", "Q&A"]
    s_colors = [POS if v > 0.05 else (NEG if v < -0.05 else TEXT_DIM) for v in section_vals]

    fig2 = go.Figure(go.Bar(
        x=section_labels,
        y=section_vals,
        marker=dict(color=s_colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in section_vals],
        textposition="outside",
        textfont=dict(color=TEXT_MUT, size=11, family="Inter"),
        width=0.35,
    ))
    fig2.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig2.update_layout(
        **PLOT_LAYOUT,
        yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[-0.6, 0.6], title="Sentiment Score"),
        height=260,
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    chart_card_close()

divider()

# ── Price movement + sentiment overlay ─────────────────────────────────────────

chart_card_open("5-Day Price Movement", "dollar-sign", f"{ticker} · {event_date}")
price_df = load_price_data(ticker, event_date)

if not price_df.empty:
    event_ts = pd.Timestamp(event_date)

    fig3 = go.Figure()

    # Price line
    fig3.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df["Close"],
        mode="lines+markers",
        name="Close Price",
        line=dict(color=TEXT_PRI, width=2),
        marker=dict(size=5, color=TEXT_PRI),
        yaxis="y1",
    ))

    # Sentiment score as secondary line (constant for the window, just reference)
    fig3.add_trace(go.Scatter(
        x=[price_df.index.min(), price_df.index.max()],
        y=[overall, overall],
        mode="lines",
        name=f"Sentiment ({overall:+.3f})",
        line=dict(color=ACCENT, width=1.5, dash="dot"),
        yaxis="y2",
        opacity=0.75,
    ))

    # Earnings call marker
    fig3.add_vline(
        x=event_ts,
        line=dict(color=ACCENT, width=1, dash="dash"),
        annotation_text="Earnings Call",
        annotation_position="top",
        annotation=dict(
            font=dict(color=TEXT_MUT, size=11, family="Inter"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
    )

    fig3.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(**PLOT_LAYOUT["xaxis"], showgrid=False),
        yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Price (USD)", showgrid=True),
        yaxis2=dict(
            overlaying="y", side="right",
            title="Sentiment Score",
            range=[-1.2, 1.2],
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
            tickfont=dict(color=TEXT_DIM, size=10),
            title_font=dict(color=TEXT_DIM, size=10),
        ),
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(color=TEXT_MUT, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=320,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown(f'<p style="color:{TEXT_DIM};font-size:0.875rem;padding:1rem 0;">Price data not available for this selection.</p>', unsafe_allow_html=True)

chart_card_close()

divider()

# ── SHAP feature importance ────────────────────────────────────────────────────

col_shap, col_pred = st.columns([3, 2], gap="medium")

with col_shap:
    chart_card_open("Feature Importance", "bar-chart", "SHAP mean absolute values")

    if not shap_df.empty:
        top_n = shap_df.nlargest(8, "mean_abs_shap").sort_values("mean_abs_shap")

        # Gradient from dim to accent based on rank
        n = len(top_n)
        bar_colors_shap = [
            f"rgba(99, 91, 255, {0.3 + 0.7 * (i / max(n - 1, 1))})"
            for i in range(n)
        ]

        fig4 = go.Figure(go.Bar(
            x=top_n["mean_abs_shap"],
            y=top_n["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker=dict(color=bar_colors_shap, line=dict(width=0)),
            text=[f"{v:.4f}" for v in top_n["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color=TEXT_DIM, size=10, family="Inter"),
        ))
        fig4.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(**PLOT_LAYOUT["xaxis"], showgrid=True, title="Mean |SHAP Value|"),
            yaxis=dict(**PLOT_LAYOUT["yaxis"], showgrid=False, tickfont=dict(color=TEXT_MUT, size=11)),
            height=340,
        )
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown(f'<p style="color:{TEXT_DIM};font-size:0.875rem;padding:1rem 0;">Run train.py to generate SHAP values.</p>', unsafe_allow_html=True)

    chart_card_close()

# ── Model prediction panel ─────────────────────────────────────────────────────

with col_pred:
    chart_card_open("Model Prediction", "cpu")

    if model is not None and pred_label is not None and model_conf is not None:
        pred_color  = POS if pred_label == 1 else NEG
        pred_text   = "UP" if pred_label == 1 else "DOWN"
        pred_icon   = "trending-up" if pred_label == 1 else "trending-down"
        conf_pct    = int(model_conf * 100)

        # Actual change
        actual_chg = None
        if not df_matrix.empty and "pct_change_48h" in df_matrix.columns:
            m = df_matrix[(df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == event_date)]
            if not m.empty:
                actual_chg = float(m["pct_change_48h"].iloc[0])

        # Top SHAP driver interpretation
        driver_text = ""
        if not shap_df.empty:
            top_feat = shap_df.iloc[0]["feature"].replace("_", " ")
            direction_word = "positive" if pred_label == 1 else "negative"
            driver_text = f"Primary driver: <strong>{top_feat}</strong> indicates {direction_word} sentiment momentum heading into the call."

        st.markdown(f"""
        <div style="padding:0.5rem 0;">
            <!-- Direction badge -->
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
                <div style="width:52px;height:52px;background:{pred_color}18;border-radius:12px;
                            border:1px solid {pred_color}40;display:flex;align-items:center;
                            justify-content:center;">
                    {icon(pred_icon, pred_color, 24)}
                </div>
                <div>
                    <div style="font-size:2rem;font-weight:700;color:{pred_color};
                                letter-spacing:-0.03em;line-height:1;">{pred_text}</div>
                    <div style="font-size:0.78rem;color:{TEXT_DIM};margin-top:2px;">
                        48-hour prediction
                    </div>
                </div>
            </div>

            <!-- Confidence bar -->
            <div style="margin-bottom:1.5rem;">
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:6px;">
                    <span style="font-size:0.78rem;color:{TEXT_MUT};">Confidence</span>
                    <span style="font-size:0.78rem;font-weight:600;
                                 color:{TEXT_PRI};">{conf_pct}%</span>
                </div>
                <div style="height:4px;background:{BORDER};border-radius:99px;overflow:hidden;">
                    <div style="height:100%;width:{conf_pct}%;
                                background:{ACCENT};border-radius:99px;
                                transition:width 0.4s ease;"></div>
                </div>
            </div>

            {"<!-- Actual outcome -->" + f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.75rem;background:{SURFACE_2};border-radius:8px;
                        border:1px solid {BORDER};margin-bottom:1.25rem;">
                <span style="font-size:0.78rem;color:{TEXT_MUT};">Actual 48h change</span>
                <span style="font-size:0.875rem;font-weight:600;
                             color:{POS if actual_chg and actual_chg > 0 else NEG};">
                    {f"{actual_chg:+.2f}%" if actual_chg is not None else "—"}
                </span>
            </div>""" if actual_chg is not None else ""}

            <!-- Interpretation -->
            <div style="font-size:0.8rem;color:{TEXT_DIM};line-height:1.6;
                        padding-top:0.75rem;border-top:1px solid {BORDER};">
                {driver_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color:{TEXT_DIM};font-size:0.875rem;">Run train.py to generate predictions.</p>', unsafe_allow_html=True)

    chart_card_close()
