# -*- coding: utf-8 -*-
from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import qrcode
import requests
import streamlit as st

from predictor import generate_projection_chart_data, train_predict_for_ticker

APP_URL = "https://rotrivex-systems-rbtustwa2cfqcegsrs4zem.streamlit.app/"

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.markdown(
    """
<style>
:root {
    --bg: #0b1220;
    --bg-2: #0f172a;
    --panel: rgba(17, 24, 39, 0.92);
    --panel-2: rgba(15, 23, 42, 0.92);
    --border: #223046;
    --muted: #9fb0c7;
    --text: #f8fafc;
    --blue: #60a5fa;
    --green: #34d399;
    --red: #f87171;
    --amber: #fbbf24;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.16), transparent 30%),
        radial-gradient(circle at top left, rgba(16, 185, 129, 0.10), transparent 22%),
        linear-gradient(180deg, var(--bg) 0%, #101828 100%);
    color: var(--text);
}

.main .block-container {
    max-width: 1160px;
    padding-top: 0.8rem;
    padding-bottom: 2.2rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

.hero {
    padding: 1rem 1.05rem;
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(15,23,42,0.92));
    box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    margin-bottom: 0.8rem;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 0.85rem;
    align-items: center;
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 0.2rem 0;
}

.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 0.55rem;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.pill {
    display: inline-block;
    padding: 0.35rem 0.62rem;
    border-radius: 999px;
    border: 1px solid #2a3a53;
    background: rgba(19, 31, 51, 0.9);
    color: #d8e4f3;
    font-size: 0.78rem;
    font-weight: 700;
}

.hero-side {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.55rem;
}

.mini-stat {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.88);
    padding: 0.72rem;
}

.mini-stat-label {
    color: var(--muted);
    font-size: 0.75rem;
    margin-bottom: 0.2rem;
}

.mini-stat-value {
    font-size: 1rem;
    font-weight: 800;
    color: var(--text);
}

.section-card {
    padding: 0.9rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--panel);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    margin-bottom: 0.8rem;
}

.metric-strip {
    margin: 0.2rem 0 0.55rem 0;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(14, 22, 36, 0.96));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.55rem 0.6rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 700;
    font-size: 0.76rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.05rem !important;
}

.signal-buy, .signal-sell, .signal-watch {
    padding: 0.72rem 0.92rem;
    border-radius: 14px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0.75rem;
    border: 1px solid transparent;
    letter-spacing: 0.05em;
}
.signal-buy { background: rgba(6, 78, 59, 0.34); color: #6ee7b7; border-color: rgba(16,185,129,0.28); }
.signal-sell { background: rgba(127, 29, 29, 0.34); color: #fca5a5; border-color: rgba(239,68,68,0.28); }
.signal-watch { background: rgba(146, 64, 14, 0.32); color: #fcd34d; border-color: rgba(251,191,36,0.26); }

.kv-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 0.65rem;
}

.kv-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel-2);
    padding: 0.7rem;
}

.kv-label {
    color: var(--muted);
    font-size: 0.75rem;
    margin-bottom: 0.15rem;
}

.kv-value {
    color: var(--text);
    font-size: 1rem;
    font-weight: 800;
}

.flag {
    padding: 0.36rem 0.6rem;
    border-radius: 999px;
    display: inline-block;
    margin: 0.12rem 0.14rem 0.12rem 0;
    background: #1b2638;
    border: 1px solid #314158;
    color: #e5edf8;
    font-size: 0.76rem;
    font-weight: 700;
}

.note-list {
    margin: 0;
    padding-left: 1rem;
    color: #dbe4f0;
    line-height: 1.55;
}

.headline-card {
    padding: 0.76rem 0.82rem;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel-2);
    margin-bottom: 0.5rem;
}

.headline-meta {
    color: #9ec6ff;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.18rem;
}

.headline-card a {
    color: var(--text) !important;
    text-decoration: none !important;
    font-weight: 700;
}

.data-badge {
    display: inline-block;
    margin-top: 0.1rem;
    color: #cfe0f5;
    font-size: 0.78rem;
    background: rgba(19,31,51,0.84);
    border: 1px solid #28405f;
    border-radius: 999px;
    padding: 0.28rem 0.55rem;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
}

.stTabs [data-baseweb="tab"] {
    background: #162032;
    border-radius: 12px 12px 0 0;
    color: #d9e3f0;
    padding: 0.45rem 0.8rem;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #24324a !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1729 0%, #111827 100%);
    border-right: 1px solid #1f2a3d;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0.8rem;
}

.stButton > button,
.stFormSubmitButton > button,
.stLinkButton > a {
    border-radius: 12px !important;
    font-weight: 800 !important;
}

.stFormSubmitButton > button,
.stLinkButton > a {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
}

@media (max-width: 900px) {
    .hero-grid { grid-template-columns: 1fr; }
    .hero-side { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 768px) {
    .main .block-container { padding-top: 0.5rem; padding-left: 0.55rem; padding-right: 0.55rem; }
    .hero { padding: 0.85rem; }
    .hero-title { font-size: 1.4rem; }
    .hero-side, .kv-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


def safe_attr(obj, name, default):
    return getattr(obj, name, default)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_headlines(ticker: str, limit: int = 8) -> list[dict]:
    query = urllib.parse.quote(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    items = []
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        channel = root.find("channel")
        if channel is None:
            return items
        for item in channel.findall("item")[:limit]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            source_el = item.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else ""
            if title:
                items.append({"title": title, "link": link, "source": source, "published": pub_date})
    except Exception:
        return []
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def build_qr_code(url: str) -> bytes:
    qr = qrcode.make(url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(ttl=1800, show_spinner=False)
def run_dashboard(ticker: str, period: str, threshold: float, forecast_days: int, n_sims: int):
    result = train_predict_for_ticker(ticker, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)
    headlines = fetch_live_headlines(ticker, limit=8)
    return result, summary, headlines



def signal_class(signal: str) -> str:
    return {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-watch")



def build_candlestick_chart(hist: pd.DataFrame, result) -> go.Figure:
    chart_data = hist.tail(75).copy()
    chart_data["SMA20"] = chart_data["Close"].rolling(20).mean()
    current_price = safe_attr(result, "latest_close", float(chart_data["Close"].iloc[-1]))
    support = safe_attr(result, "support_level", float(chart_data["Low"].tail(30).min()))
    resistance = safe_attr(result, "resistance_level", float(chart_data["High"].tail(30).max()))

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
            increasing_line_color="#34d399",
            decreasing_line_color="#f87171",
            increasing_fillcolor="#34d399",
            decreasing_fillcolor="#f87171",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["SMA20"],
            mode="lines",
            name="20D Avg",
            line=dict(color="#60a5fa", width=2),
        )
    )
    for y, label, color in [
        (current_price, "Current", "#cbd5e1"),
        (support, "Support", "#fbbf24"),
        (resistance, "Resistance", "#a78bfa"),
    ]:
        fig.add_hline(
            y=y,
            line_dash="dot",
            line_color=color,
            line_width=1.0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=color,
        )
    fig.update_layout(
        height=380,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(color="#e5edf8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig



def build_projection_chart(summary: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["High Band (90%)"],
            mode="lines",
            line=dict(color="rgba(96,165,250,0)", width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["Low Band (10%)"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(96,165,250,0.18)",
            line=dict(color="rgba(96,165,250,0)", width=0),
            name="Projected Range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary.index,
            y=summary["Median"],
            mode="lines",
            name="Median Path",
            line=dict(color="#60a5fa", width=3),
        )
    )
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="#cbd5e1",
        line_width=1.0,
        annotation_text="Current",
        annotation_position="right",
        annotation_font_color="#cbd5e1",
    )
    fig.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=8, r=8, t=40, b=8),
        font=dict(color="#e5edf8"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


with st.sidebar:
    st.markdown("### Dashboard controls")
    st.caption("Adjust settings, then run the model.")
    with st.form("controls"):
        search_ticker = st.text_input("Ticker", value="AAPL", placeholder="AAPL").strip().upper()
        period = st.selectbox("History period", options=["6mo", "1y", "2y", "5y"], index=1)
        threshold = st.slider("Signal threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
        forecast_days = st.slider("Projection days", min_value=5, max_value=60, value=20, step=5)
        n_sims = st.slider("Projection paths", min_value=50, max_value=500, value=200, step=50)
        run = st.form_submit_button("Run dashboard", use_container_width=True)

if "last_run" not in st.session_state:
    st.session_state.last_run = ("AAPL", "1y", 0.55, 20, 200)

if run:
    st.session_state.last_run = (search_ticker or "AAPL", period, threshold, forecast_days, n_sims)

ticker, period, threshold, forecast_days, n_sims = st.session_state.last_run

try:
    with st.spinner(f"Loading {ticker}..."):
        result, summary, live_headlines = run_dashboard(ticker, period, threshold, forecast_days, n_sims)
except Exception as exc:
    st.error(f"Could not load market data for {ticker}. Details: {exc}")
    st.stop()

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="hero-title">{ticker} dashboard</div>
                <div class="hero-sub">Probability, trade levels, projection range, and headline context in one place.</div>
                <div class="hero-badges">
                    <span class="pill">Signal model</span>
                    <span class="pill">{period} history</span>
                    <span class="pill">{forecast_days}-day outlook</span>
                    <span class="pill">{n_sims} paths</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="mini-stat">
                    <div class="mini-stat-label">Signal</div>
                    <div class="mini-stat-value">{safe_attr(result, 'model_signal', 'WATCH')}</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-label">Mood</div>
                    <div class="mini-stat-value">{safe_attr(result, 'mood', 'Neutral')}</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-label">Data source</div>
                    <div class="mini-stat-value">{safe_attr(result, 'data_source', 'Unknown')}</div>
                </div>
                <div class="mini-stat">
                    <div class="mini-stat-label">News tone</div>
                    <div class="mini-stat-value">{safe_attr(result, 'sentiment_label', 'Neutral')}</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="{signal_class(safe_attr(result, "model_signal", "WATCH"))}">{safe_attr(result, "model_signal", "WATCH")}</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Price", f"${safe_attr(result, 'latest_close', 0.0):,.2f}")
mc2.metric("Up probability", f"{safe_attr(result, 'next_day_up_probability', 0.5):.2%}")
mc3.metric("Accuracy", f"{safe_attr(result, 'holdout_accuracy', 0.0):.2%}")
mc4.metric("Earnings", safe_attr(result, "earnings_flag", "No date found"))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="metric-strip">', unsafe_allow_html=True)
md1, md2, md3, md4 = st.columns(4)
md1.metric("Momentum", f"{safe_attr(result, 'momentum_20d', 0.0):.2%}")
md2.metric("Volume ratio", f"{safe_attr(result, 'volume_ratio', 1.0):.2f}x")
md3.metric("RSI", f"{safe_attr(result, 'rsi_14', 50.0):.1f}")
md4.metric("MACD", f"{safe_attr(result, 'macd', 0.0):.2f}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="data-badge">Data source: {safe_attr(result, "data_source", "Unknown")}</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Details", "Headlines", "Share"])

with tab1:
    c1, c2 = st.columns([1.55, 1])
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Price action")
        st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Trade setup")
        st.markdown(
            f"""
            <div class="kv-grid">
                <div class="kv-card"><div class="kv-label">Support</div><div class="kv-value">${safe_attr(result, 'support_level', 0.0):,.2f}</div></div>
                <div class="kv-card"><div class="kv-label">Resistance</div><div class="kv-value">${safe_attr(result, 'resistance_level', 0.0):,.2f}</div></div>
                <div class="kv-card"><div class="kv-label">Stop loss</div><div class="kv-value">${safe_attr(result, 'stop_loss', 0.0):,.2f}</div></div>
                <div class="kv-card"><div class="kv-label">Target 1</div><div class="kv-value">${safe_attr(result, 'target_1', 0.0):,.2f}</div></div>
                <div class="kv-card"><div class="kv-label">Target 2</div><div class="kv-value">${safe_attr(result, 'target_2', 0.0):,.2f}</div></div>
                <div class="kv-card"><div class="kv-label">20D average</div><div class="kv-value">${safe_attr(result, 'sma20', 0.0):,.2f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Flags")
        st.markdown(
            "".join([f'<span class="flag">{flag}</span>' for flag in safe_attr(result, "watchlist_flags", [])]),
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Projection range")
    st.plotly_chart(build_projection_chart(summary, safe_attr(result, "latest_close", 0.0)), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model summary")
        rsi = safe_attr(result, "rsi_14", 50.0)
        notes = [
            f"Signal is {safe_attr(result, 'model_signal', 'WATCH')}.",
            f"RSI is {rsi:.1f}.",
            f"Price is {'above' if safe_attr(result, 'latest_close', 0.0) > safe_attr(result, 'sma20', 0.0) else 'below'} the 20-day average.",
            f"MACD is {'above' if safe_attr(result, 'macd', 0.0) > safe_attr(result, 'macd_signal', 0.0) else 'below'} its signal line.",
            f"Volume ratio is {safe_attr(result, 'volume_ratio', 1.0):.2f}x.",
            f"Data source used: {safe_attr(result, 'data_source', 'Unknown')}.",
        ]
        st.markdown("<ul class='note-list'>" + "".join([f"<li>{n}</li>" for n in notes]) + "</ul>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature importance")
        feat_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
        if not feat_df.empty:
            feat_df["Importance"] = feat_df["Importance"].map(lambda x: f"{x:.3f}")
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.write("No feature importance data available.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Levels")
        levels_df = pd.DataFrame(
            {
                "Item": ["Support", "Resistance", "Stop loss", "Target 1", "Target 2", "52W position"],
                "Value": [
                    f"${safe_attr(result, 'support_level', 0.0):,.2f}",
                    f"${safe_attr(result, 'resistance_level', 0.0):,.2f}",
                    f"${safe_attr(result, 'stop_loss', 0.0):,.2f}",
                    f"${safe_attr(result, 'target_1', 0.0):,.2f}",
                    f"${safe_attr(result, 'target_2', 0.0):,.2f}",
                    f"{safe_attr(result, 'range_52w_position', 0.0):.0%}",
                ],
            }
        )
        st.dataframe(levels_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Latest headlines")
    if live_headlines:
        for item in live_headlines:
            source_line = " - ".join([x for x in [item.get("source", ""), item.get("published", "")] if x])
            st.markdown(
                f'<div class="headline-card"><div class="headline-meta">{source_line}</div><div><a href="{item.get("link", "")}" target="_blank">{item.get("title", "")}</a></div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.write("No live headlines were returned right now.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    ticker_url = f"{APP_URL}?ticker={ticker}"
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Share this ticker")
        st.write("Use this link to open the same ticker dashboard.")
        st.code(ticker_url, language=None)
        st.link_button("Open direct ticker link", ticker_url, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("QR code")
        st.image(build_qr_code(ticker_url), caption="Scan to open this ticker", width=170)
        st.markdown('</div>', unsafe_allow_html=True)
