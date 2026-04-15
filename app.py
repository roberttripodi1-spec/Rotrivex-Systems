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

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0b1220 0%, #101828 100%); color: #f8fafc; }
.main .block-container { max-width: 1100px; padding-top: 0.8rem; padding-bottom: 2rem; }
.hero, .section-card { padding: 0.85rem; border: 1px solid #223046; border-radius: 14px; background: #111827; margin-bottom: 0.75rem; }
.signal-buy, .signal-sell, .signal-watch { padding: .58rem .74rem; border-radius: 12px; font-weight: 800; text-align: center; margin-bottom: .7rem; }
.signal-buy { background: #0b3b2e; color: #6ee7b7; }
.signal-sell { background: #4c1717; color: #fca5a5; }
.signal-watch { background: #5a3b10; color: #fcd34d; }
.headline-card { padding: .68rem .72rem; border: 1px solid #223046; border-radius: 12px; background: #0f172a; margin-bottom: .45rem; }
.flag { padding: .32rem .52rem; border-radius: 999px; display: inline-block; margin: .12rem .14rem .12rem 0; background: #1b2638; border: 1px solid #314158; color: #e5edf8; font-size: .76rem; }
[data-testid="stMetric"] { background: #111827; border: 1px solid #223046; border-radius: 12px; padding: .4rem .46rem; }
.stTabs [data-baseweb="tab"] { background: #162032; border-radius: 10px 10px 0 0; color: #d9e3f0; }
.stTabs [aria-selected="true"] { background: #24324a !important; }
</style>
""", unsafe_allow_html=True)


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
    fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"], low=chart_data["Low"], close=chart_data["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA20"], mode="lines", name="20D Avg"))
    fig.add_hline(y=current_price, line_dash="dot", annotation_text="Current", annotation_position="right")
    fig.add_hline(y=support, line_dash="dot", annotation_text="Support", annotation_position="right")
    fig.add_hline(y=resistance, line_dash="dot", annotation_text="Resistance", annotation_position="right")
    fig.update_layout(height=360, xaxis_rangeslider_visible=False, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", margin=dict(l=8, r=8, t=38, b=8))
    return fig


def build_projection_chart(summary: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary.index, y=summary["High Band (90%)"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Low Band (10%)"], mode="lines", fill="tonexty", name="Projected Range"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Median"], mode="lines", name="Median Path"))
    fig.add_hline(y=current_price, line_dash="dot", annotation_text="Current", annotation_position="right")
    fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", margin=dict(l=8, r=8, t=38, b=8))
    return fig


st.markdown('<div class="hero"><h1>Stock Predictor</h1><div>Fast Streamlit build with Yahoo fallback.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    with st.form("controls"):
        search_ticker = st.text_input("Search ticker", value="AAPL").strip().upper()
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

st.markdown(f'<div class="{signal_class(safe_attr(result, "model_signal", "WATCH"))}">{safe_attr(result, "model_signal", "WATCH")}</div>', unsafe_allow_html=True)
st.caption(f"Data source: {safe_attr(result, 'data_source', 'Unknown')}")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Price", f"${safe_attr(result, 'latest_close', 0.0):,.2f}")
mc2.metric("Mood", safe_attr(result, "mood", "Neutral"))
mc3.metric("Up probability", f"{safe_attr(result, 'next_day_up_probability', 0.5):.2%}")
mc4.metric("Accuracy", f"{safe_attr(result, 'holdout_accuracy', 0.0):.2%}")

md1, md2, md3, md4 = st.columns(4)
md1.metric("Momentum", f"{safe_attr(result, 'momentum_20d', 0.0):.2%}")
md2.metric("Volume", f"{safe_attr(result, 'volume_ratio', 1.0):.2f}x")
md3.metric("News tone", safe_attr(result, "sentiment_label", "Neutral"))
md4.metric("Earnings", safe_attr(result, "earnings_flag", "No date found"))

tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Details", "Headlines", "Share"])

with tab1:
    st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
    st.plotly_chart(build_projection_chart(summary, safe_attr(result, "latest_close", 0.0)), use_container_width=True)

with tab2:
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Key levels")
        levels_df = pd.DataFrame({
            "Item": ["Support", "Resistance", "Stop loss", "Target 1", "Target 2"],
            "Value": [safe_attr(result, "support_level", 0.0), safe_attr(result, "resistance_level", 0.0), safe_attr(result, "stop_loss", 0.0), safe_attr(result, "target_1", 0.0), safe_attr(result, "target_2", 0.0)],
        })
        levels_df["Value"] = levels_df["Value"].map(lambda x: f"${x:,.2f}")
        st.dataframe(levels_df, use_container_width=True, hide_index=True)
        st.markdown("Flags")
        st.markdown("".join([f'<span class="flag">{flag}</span>' for flag in safe_attr(result, "watchlist_flags", [])]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model summary")
        rsi = safe_attr(result, "rsi_14", 50.0)
        notes = [
            f"Signal is {safe_attr(result, 'model_signal', 'WATCH')}",
            f"RSI is {rsi:.1f}",
            f"Price vs 20D average: {'above' if safe_attr(result, 'latest_close', 0.0) > safe_attr(result, 'sma20', 0.0) else 'below'}",
            f"MACD vs signal: {'above' if safe_attr(result, 'macd', 0.0) > safe_attr(result, 'macd_signal', 0.0) else 'below'}",
            f"Data source used: {safe_attr(result, 'data_source', 'Unknown')}",
        ]
        for n in notes:
            st.write("-", n)
        feat_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
        if not feat_df.empty:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    if live_headlines:
        for item in live_headlines:
            source_line = " - ".join([x for x in [item.get("source", ""), item.get("published", "")] if x])
            st.markdown(f'<div class="headline-card"><div>{source_line}</div><div style="margin-top:.2rem;"><a href="{item.get("link", "")}" target="_blank">{item.get("title", "")}</a></div></div>', unsafe_allow_html=True)
    else:
        st.write("No live headlines were returned right now.")

with tab4:
    ticker_url = f"{APP_URL}?ticker={ticker}"
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Share this ticker")
    st.code(ticker_url, language=None)
    st.link_button("Open direct ticker link", ticker_url, use_container_width=True)
    st.image(build_qr_code(ticker_url), caption="Scan to open this ticker", width=160)
    st.markdown('</div>', unsafe_allow_html=True)
