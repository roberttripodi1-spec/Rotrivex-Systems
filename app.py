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

APP_URL = "https://stock-predictor-app-chqgww4vn5xvfzytgesxvv.streamlit.app/"

st.set_page_config(page_title="Stock Predictor", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_headlines(ticker: str, limit: int = 8) -> list[dict]:
    query = urllib.parse.quote(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    channel = root.find("channel")
    if channel is None:
        return []

    items: list[dict] = []
    for item in channel.findall("item")[:limit]:
        title = item.findtext("title", default="").strip()
        if not title:
            continue
        items.append(
            {
                "title": title,
                "link": item.findtext("link", default="").strip(),
                "published": item.findtext("pubDate", default="").strip(),
                "source": (item.find("source").text.strip() if item.find("source") is not None and item.find("source").text else ""),
            }
        )
    return items


@st.cache_data(ttl=3600, show_spinner=False)
def build_qr_code(url: str) -> bytes:
    qr = qrcode.make(url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(ttl=1800, show_spinner=False)
def load_dashboard_data(
    ticker: str,
    period: str,
    threshold: float,
    forecast_days: int,
    n_sims: int,
):
    result = train_predict_for_ticker(ticker, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)
    headlines = fetch_live_headlines(ticker, limit=8)
    return result, summary, headlines


def safe_attr(obj, name: str, default):
    return getattr(obj, name, default)


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
        (support, "Support", "#f59e0b"),
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
        title=f"{safe_attr(result, 'ticker', 'Ticker')} Price",
        height=360,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=8, r=8, t=38, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#e5edf8"),
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
            line=dict(color="rgba(96,165,250,0.0)", width=0),
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
            line=dict(color="rgba(96,165,250,0.0)", width=0),
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
        title="Projection Range",
        height=260,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=8, r=8, t=38, b=8),
        font=dict(color="#e5edf8"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


def build_simple_gauge_html(title: str, value: float, min_value: float, max_value: float, value_fmt: str) -> str:
    pct = 0.0 if max_value <= min_value else (value - min_value) / (max_value - min_value)
    pct = max(0.0, min(1.0, pct))
    deg = 180 * pct

    if value_fmt == "pct0":
        display_value = f"{value:.0f}"
    elif value_fmt == "float2":
        display_value = f"{value:.2f}"
    else:
        display_value = str(value)

    return f"""
    <div class="indicator-card">
        <div class="indicator-title">{title}</div>
        <div class="indicator-wrap">
            <div class="indicator-arch-base"></div>
            <div class="indicator-arch-fill">
                <div style="
                    position:absolute;
                    inset:0;
                    background:conic-gradient(from 180deg, #60a5fa 0deg, #60a5fa {deg}deg, transparent {deg}deg, transparent 180deg);
                    -webkit-mask: radial-gradient(circle at 50% 100%, transparent 44px, black 45px);
                    mask: radial-gradient(circle at 50% 100%, transparent 44px, black 45px);
                "></div>
            </div>
            <div class="indicator-value">{display_value}</div>
        </div>
    </div>
    """


st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #101828 100%);
        color: #f8fafc;
    }
    .main .block-container {
        padding-top: 0.6rem;
        padding-bottom: 2rem;
        max-width: 1050px;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }
    .hero, .section-card, .movers-card {
        padding: 0.8rem;
        border: 1px solid #223046;
        border-radius: 14px;
        background: #111827;
        box-shadow: 0 4px 16px rgba(0,0,0,0.16);
        margin-bottom: 0.7rem;
    }
    .muted {
        color: #a5b4c7;
        font-size: 0.85rem;
    }
    .signal-buy, .signal-sell, .signal-watch {
        padding: 0.65rem 0.8rem;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        border: 1px solid transparent;
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }
    .signal-buy { background: #0b3b2e; color: #6ee7b7; border-color: #14532d; }
    .signal-sell { background: #4c1717; color: #fca5a5; border-color: #7f1d1d; }
    .signal-watch { background: #5a3b10; color: #fcd34d; border-color: #92400e; }
    .headline-card {
        padding: 0.68rem 0.72rem;
        border: 1px solid #223046;
        border-radius: 12px;
        background: #0f172a;
        margin-bottom: 0.45rem;
    }
    .headline-source {
        color: #93c5fd;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .flag {
        padding: 0.32rem 0.52rem;
        border-radius: 999px;
        display: inline-block;
        margin: 0.12rem 0.14rem 0.12rem 0;
        background: #1b2638;
        border: 1px solid #314158;
        color: #e5edf8;
        font-size: 0.76rem;
    }
    .indicator-card {
        padding: 0.48rem 0.52rem;
        border: 1px solid #223046;
        border-radius: 12px;
        background: #0f172a;
        margin-bottom: 0.45rem;
    }
    .indicator-title {
        color: #93c5fd;
        font-size: 0.76rem;
        font-weight: 700;
        text-align: center;
    }
    .indicator-wrap {
        position: relative;
        width: 100%;
        max-width: 210px;
        height: 108px;
        margin: 0 auto;
    }
    .indicator-arch-base,
    .indicator-arch-fill {
        position: absolute;
        left: 50%;
        top: 8px;
        transform: translateX(-50%);
        width: 148px;
        height: 74px;
        border-top-left-radius: 148px;
        border-top-right-radius: 148px;
        border-bottom: 0;
        box-sizing: border-box;
        overflow: hidden;
    }
    .indicator-arch-base { border: 10px solid #223046; }
    .indicator-arch-fill { border: 10px solid transparent; }
    .indicator-value {
        position: absolute;
        left: 50%;
        bottom: 10px;
        transform: translateX(-50%);
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1;
        text-align: center;
        width: 100%;
    }
    .small-note { color: #a5b4c7; font-size: 0.84rem; }
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #223046;
        border-radius: 12px;
        padding: 0.4rem 0.46rem;
    }
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1f2a3d;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }
        .indicator-wrap { max-width: 180px; height: 96px; }
        .indicator-value { font-size: 0.92rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

query_ticker = str(st.query_params.get("ticker", "AAPL") or "AAPL").strip().upper()
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = query_ticker

with st.sidebar:
    st.title("Stock Predictor")
    st.markdown("<div class='muted'>Enter a ticker and run the dashboard.</div>", unsafe_allow_html=True)

    with st.form("dashboard_form", clear_on_submit=False):
        ticker = st.text_input("Ticker", value=st.session_state.active_ticker).strip().upper()
        period = st.selectbox("History period", options=["1y", "2y", "5y", "10y"], index=2)
        threshold = st.slider("Signal threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
        forecast_days = st.slider("Projection days", min_value=5, max_value=60, value=20, step=5)
        n_sims = st.slider("Projection paths", min_value=50, max_value=500, value=200, step=50)
        submitted = st.form_submit_button("Run dashboard", use_container_width=True)

    st.caption("The dashboard only recomputes when you submit the form.")

if submitted or "result_bundle" not in st.session_state or st.session_state.get("active_ticker") != ticker:
    st.session_state.active_ticker = ticker or "AAPL"
    st.query_params["ticker"] = st.session_state.active_ticker
    try:
        with st.spinner(f"Running model for {st.session_state.active_ticker}..."):
            result, summary, headlines = load_dashboard_data(
                st.session_state.active_ticker,
                period,
                threshold,
                forecast_days,
                n_sims,
            )
        st.session_state.result_bundle = {
            "result": result,
            "summary": summary,
            "headlines": headlines,
            "period": period,
            "threshold": threshold,
            "forecast_days": forecast_days,
            "n_sims": n_sims,
        }
    except Exception as exc:
        st.session_state.result_bundle = None
        st.error(f"Unable to run the dashboard for {st.session_state.active_ticker}: {exc}")

bundle = st.session_state.get("result_bundle")

st.markdown('<div class="hero">', unsafe_allow_html=True)
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Stock Predictor")
    st.caption("Clean rebuild for reliable Streamlit deployment.")
with header_right:
    st.link_button("Open app", APP_URL, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if not bundle:
    st.info("Run the dashboard from the sidebar to load a ticker.")
    st.stop()

result = bundle["result"]
summary = bundle["summary"]
live_headlines = bundle["headlines"]

st.markdown(
    f'<div class="{signal_class(safe_attr(result, "model_signal", "WATCH"))}">{safe_attr(result, "model_signal", "WATCH")}</div>',
    unsafe_allow_html=True,
)

metric_row_1 = st.columns(4)
metric_row_1[0].metric("Ticker", safe_attr(result, "ticker", st.session_state.active_ticker))
metric_row_1[1].metric("Price", f"${safe_attr(result, 'latest_close', 0.0):,.2f}")
metric_row_1[2].metric("Up probability", f"{safe_attr(result, 'next_day_up_probability', 0.0):.2%}")
metric_row_1[3].metric("Accuracy", f"{safe_attr(result, 'holdout_accuracy', 0.0):.2%}")

metric_row_2 = st.columns(4)
metric_row_2[0].metric("Mood", safe_attr(result, "mood", "Neutral"))
metric_row_2[1].metric("Momentum", f"{safe_attr(result, 'momentum_20d', 0.0):.2%}")
metric_row_2[2].metric("Volume", f"{safe_attr(result, 'volume_ratio', 1.0):.2f}x")
metric_row_2[3].metric("News tone", safe_attr(result, "sentiment_label", "Neutral"))


tab_dashboard, tab_details, tab_headlines, tab_share = st.tabs(["Dashboard", "Details", "Headlines", "Share"])

with tab_dashboard:
    st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
    st.plotly_chart(build_projection_chart(summary, safe_attr(result, "latest_close", 0.0)), use_container_width=True)

with tab_details:
    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Indicators")
        rsi_value = max(0, min(100, safe_attr(result, "rsi_14", 50.0)))
        sentiment_value = max(-1, min(1, safe_attr(result, "sentiment_score", 0.0)))
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(build_simple_gauge_html("RSI", rsi_value, 0, 100, "pct0"), unsafe_allow_html=True)
        with g2:
            st.markdown(build_simple_gauge_html("News Sentiment", sentiment_value, -1, 1, "float2"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Key levels")
        levels_df = pd.DataFrame(
            {
                "Item": ["Support", "Resistance", "Stop loss", "Target 1", "Target 2"],
                "Value": [
                    safe_attr(result, "support_level", 0.0),
                    safe_attr(result, "resistance_level", 0.0),
                    safe_attr(result, "stop_loss", 0.0),
                    safe_attr(result, "target_1", 0.0),
                    safe_attr(result, "target_2", 0.0),
                ],
            }
        )
        levels_df["Value"] = levels_df["Value"].map(lambda x: f"${x:,.2f}")
        st.dataframe(levels_df, use_container_width=True, hide_index=True)
        st.caption(f"Earnings: {safe_attr(result, 'earnings_flag', 'No date found')}")
        flags = safe_attr(result, "watchlist_flags", ["No major alert flags"])
        st.markdown("".join([f'<span class="flag">{flag}</span>' for flag in flags]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model summary")
        rsi = safe_attr(result, "rsi_14", 50.0)
        notes = [
            f"Signal is {safe_attr(result, 'model_signal', 'WATCH')}.",
            f"RSI is {'high' if rsi > 70 else 'low' if rsi < 40 else 'middle-range'} at {rsi:.1f}.",
            f"Price is {'above' if safe_attr(result, 'latest_close', 0.0) > safe_attr(result, 'sma20', 0.0) else 'below'} the 20-day average.",
            f"MACD is {'above' if safe_attr(result, 'macd', 0.0) > safe_attr(result, 'macd_signal', 0.0) else 'below'} its signal line.",
            f"20-day momentum is {safe_attr(result, 'momentum_20d', 0.0):.2%}.",
        ]
        for note in notes:
            st.write(f"• {note}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Feature importance")
        feature_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
        if feature_df.empty:
            st.write("No feature importance data available.")
        else:
            feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_headlines:
    if live_headlines:
        st.caption(f"Latest headlines for {safe_attr(result, 'ticker', st.session_state.active_ticker)}")
        for item in live_headlines:
            source = item.get("source", "")
            published = item.get("published", "")
            title = item.get("title", "")
            link = item.get("link", "")
            source_line = f"{source} • {published}" if source and published else source or published
            st.markdown(
                f"""
                <div class="headline-card">
                    <div class="headline-source">{source_line}</div>
                    <div style="margin-top:.22rem;">
                        <a href="{link}" target="_blank" style="color:#f8fafc; text-decoration:none; font-weight:600;">{title}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.write("No live headlines were returned right now.")

with tab_share:
    ticker_url = f"{APP_URL}?ticker={safe_attr(result, 'ticker', st.session_state.active_ticker)}"
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Share this ticker")
    st.code(ticker_url, language=None)
    st.link_button("Open direct ticker link", ticker_url, use_container_width=True)
    st.image(build_qr_code(ticker_url), caption="Scan to open this ticker on your phone", width=160)
    st.markdown('</div>', unsafe_allow_html=True)
