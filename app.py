from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import qrcode
import requests
import streamlit as st
import yfinance as yf

from predictor import generate_projection_chart_data, train_predict_for_ticker


APP_URL = "https://stock-predictor-app-chqgww4vn5xvfzytgesxvv.streamlit.app/"
PAGE_OPTIONS = ["Dashboard", "Details", "Headlines", "Share"]

st.set_page_config(page_title="Stock Predictor", layout="wide", initial_sidebar_state="expanded")

query_params = st.query_params
query_ticker = str(query_params.get("ticker", "AAPL")).strip().upper() if query_params.get("ticker", None) else "AAPL"

if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = query_ticker
if query_ticker and query_ticker != st.session_state.active_ticker:
    st.session_state.active_ticker = query_ticker
if "selected_view" not in st.session_state:
    st.session_state.selected_view = "Dashboard"
if "last_run_key" not in st.session_state:
    st.session_state.last_run_key = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "last_headlines" not in st.session_state:
    st.session_state.last_headlines = []
if "form_period" not in st.session_state:
    st.session_state.form_period = "5y"
if "form_threshold" not in st.session_state:
    st.session_state.form_threshold = 0.55
if "form_forecast_days" not in st.session_state:
    st.session_state.form_forecast_days = 20
if "form_n_sims" not in st.session_state:
    st.session_state.form_n_sims = 200


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


@st.cache_data(ttl=600, show_spinner=False)
def fetch_sp500_symbols() -> list[str]:
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/s-and-p-500-companies.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            for col in ["Symbol", "symbol"]:
                if col in df.columns:
                    symbols = df[col].astype(str).str.replace(".", "-", regex=False).dropna().unique().tolist()
                    if symbols:
                        return symbols
        except Exception:
            continue
    return []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_sp500_top_movers(limit: int = 8) -> pd.DataFrame:
    symbols = fetch_sp500_symbols()
    if not symbols:
        return pd.DataFrame(columns=["Ticker", "Last", "Change %", "Direction"])

    rows = []

    def chunks(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i + size]

    for group in chunks(symbols, 100):
        try:
            data = yf.download(
                tickers=" ".join(group),
                period="5d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            continue

        if data is None or len(data) == 0:
            continue

        for ticker in group:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    ticker_df = data[ticker].copy()
                else:
                    ticker_df = data.copy()

                if "Close" not in ticker_df.columns:
                    continue

                close = pd.to_numeric(ticker_df["Close"], errors="coerce").dropna()
                if len(close) < 2:
                    continue

                prev_close = float(close.iloc[-2])
                last_close = float(close.iloc[-1])
                if prev_close <= 0:
                    continue

                pct = ((last_close / prev_close) - 1.0) * 100.0
                rows.append({"Ticker": ticker, "Last": last_close, "Change %": pct})
            except Exception:
                continue

    movers = pd.DataFrame(rows)
    if movers.empty:
        return pd.DataFrame(columns=["Ticker", "Last", "Change %", "Direction"])

    movers = movers.drop_duplicates(subset=["Ticker"]).copy()
    movers["Abs Move"] = movers["Change %"].abs()
    movers = movers.sort_values("Abs Move", ascending=False).head(limit).copy()
    movers["Direction"] = movers["Change %"].apply(lambda x: "Up" if x >= 0 else "Down")
    movers = movers.drop(columns=["Abs Move"])
    return movers.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def run_dashboard_data(
    ticker: str,
    period: str,
    threshold: float,
    forecast_days: int,
    n_sims: int,
) -> tuple[object, pd.DataFrame, list[dict]]:
    result = train_predict_for_ticker(ticker, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)
    headlines = fetch_live_headlines(ticker, limit=8)
    return result, summary, headlines


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #101828 100%);
        color: #f8fafc;
    }
    .main .block-container {
        padding-top: 0.55rem;
        padding-bottom: 1.8rem;
        max-width: 1020px;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
            padding-top: 0.35rem;
            padding-bottom: 1.25rem;
        }
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.04rem !important; }
        h3 { font-size: 0.95rem !important; }
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem !important;
    }

    .hero, .section-card, .movers-card {
        padding: 0.72rem 0.76rem;
        border: 1px solid #223046;
        border-radius: 14px;
        background: #111827;
        box-shadow: 0 4px 16px rgba(0,0,0,0.16);
        margin-bottom: 0.6rem;
    }

    .muted, .small-note {
        color: #a5b4c7;
        font-size: .84rem;
        margin-top: .18rem;
    }

    .flag {
        padding: .32rem .52rem;
        border-radius: 999px;
        display: inline-block;
        margin: .12rem .14rem .12rem 0;
        background: #1b2638;
        border: 1px solid #314158;
        color: #e5edf8;
        font-size: .76rem;
    }

    .headline-card {
        padding: .68rem .72rem;
        border: 1px solid #223046;
        border-radius: 12px;
        background: #0f172a;
        margin-bottom: .45rem;
    }
    .headline-source {
        color: #93c5fd;
        font-size: .76rem;
        font-weight: 600;
    }

    .signal-buy, .signal-sell, .signal-watch {
        padding: .58rem .74rem;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        border: 1px solid transparent;
        margin-bottom: .5rem;
        letter-spacing: .02em;
        font-size: .92rem;
    }
    .signal-buy { background: #0b3b2e; color: #6ee7b7; border-color: #14532d; }
    .signal-sell { background: #4c1717; color: #fca5a5; border-color: #7f1d1d; }
    .signal-watch { background: #5a3b10; color: #fcd34d; border-color: #92400e; }

    .detail-list {
        margin: 0;
        padding-left: 1rem;
        color: #dbe4f0;
        line-height: 1.45;
        font-size: .9rem;
    }
    .detail-label {
        color: #93c5fd;
        font-weight: 700;
        margin-bottom: .26rem;
        display: block;
        font-size: .82rem;
    }

    .indicator-card {
        padding: .48rem .52rem;
        border: 1px solid #223046;
        border-radius: 12px;
        background: #0f172a;
        margin-bottom: .45rem;
    }
    .indicator-title {
        color: #93c5fd;
        font-size: .76rem;
        font-weight: 700;
        margin-bottom: 0;
        text-align: center;
    }
    .indicator-wrap {
        position: relative;
        width: 100%;
        max-width: 210px;
        height: 114px;
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
        bottom: 8px;
        transform: translateX(-50%);
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1;
        text-align: center;
        width: 100%;
    }

    .details-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.6rem;
    }
    @media (min-width: 900px) {
        .details-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 768px) {
        .hero, .section-card, .movers-card { padding: 0.66rem 0.68rem; }
        .indicator-card { padding: 0.4rem; }
        .indicator-wrap { max-width: 160px; height: 95px; }
        .indicator-value { font-size: .9rem; }
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #223046;
        border-radius: 12px;
        padding: .4rem .46rem;
    }
    [data-testid="stMetricLabel"] {
        color: #b8c4d6 !important;
        font-weight: 600;
        font-size: .76rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: .96rem !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #223046;
        border-radius: 12px;
        overflow: hidden;
        background: #111827;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .25rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background: #162032;
        border-radius: 12px 12px 0 0;
        color: #d9e3f0;
        padding: .38rem .65rem;
        font-size: .82rem;
        border: 1px solid #223046;
    }
    .stTabs [aria-selected="true"] {
        background: #24324a !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1f2a3d;
    }
    section[data-testid="stSidebar"]::before {
        content: "Search tickers";
        display: block;
        color: #f8fafc;
        font-size: .96rem;
        font-weight: 800;
        padding: .72rem .86rem 0.12rem .86rem;
    }

    .stButton > button, .stLinkButton > a {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        width: 100%;
        text-align: center;
        min-height: 2.38rem;
    }
    .stButton > button:hover, .stLinkButton > a:hover { background: #1d4ed8 !important; }

    .mover-button-wrap button {
        min-height: 2rem !important;
        font-size: .78rem !important;
        border-radius: 12px !important;
    }

    .stSelectbox label, .stTextInput label, .stSlider label {
        color: #dbe4f0 !important;
        font-weight: 600;
        font-size: .84rem !important;
    }
</style>
""", unsafe_allow_html=True)


def signal_class(signal: str) -> str:
    return {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-watch")


def build_candlestick_chart(hist: pd.DataFrame, result) -> go.Figure:
    chart_data = hist.tail(75).copy()
    chart_data["SMA20"] = chart_data["Close"].rolling(20).mean()
    current_price = safe_attr(result, "latest_close", float(chart_data["Close"].iloc[-1]))
    support = safe_attr(result, "support_level", float(chart_data["Low"].tail(30).min()))
    resistance = safe_attr(result, "resistance_level", float(chart_data["High"].tail(30).max()))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
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
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["SMA20"], mode="lines", name="20D Avg",
        line=dict(color="#60a5fa", width=2)
    ))
    for y, label, color in [
        (current_price, "Current", "#cbd5e1"),
        (support, "Support", "#f59e0b"),
        (resistance, "Resistance", "#a78bfa"),
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1.0, annotation_text=label, annotation_position="right", annotation_font_color=color)
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
    fig.add_trace(go.Scatter(
        x=summary.index, y=summary["High Band (90%)"], mode="lines",
        line=dict(color="rgba(96,165,250,0.0)", width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=summary.index, y=summary["Low Band (10%)"], mode="lines",
        fill="tonexty", fillcolor="rgba(96,165,250,0.18)",
        line=dict(color="rgba(96,165,250,0.0)", width=0), name="Projected Range"
    ))
    fig.add_trace(go.Scatter(
        x=summary.index, y=summary["Median"], mode="lines", name="Median Path",
        line=dict(color="#60a5fa", width=3)
    ))
    fig.add_hline(
        y=current_price, line_dash="dot", line_color="#cbd5e1", line_width=1.0,
        annotation_text="Current", annotation_position="right", annotation_font_color="#cbd5e1"
    )
    fig.update_layout(
        title="Projection Range",
        height=250,
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
    pct = 0.0
    if max_value > min_value:
        pct = (value - min_value) / (max_value - min_value)
    pct = max(0.0, min(1.0, pct))
    deg = 180 * pct

    if value_fmt == "pct0":
        display_value = f"{value:.0f}"
    elif value_fmt == "float2":
        display_value = f"{value:.2f}"
    else:
        display_value = f"{value}"

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


def set_active_ticker(ticker: str) -> None:
    st.session_state.active_ticker = ticker.upper()
    query_params["ticker"] = st.session_state.active_ticker


header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Stock Predictor")
    st.caption("Smoothed Streamlit build with cached model runs and lighter navigation.")
    st.markdown('</div>', unsafe_allow_html=True)
with header_right:
    st.link_button("📲 Share App", APP_URL, use_container_width=True)

movers = fetch_sp500_top_movers(limit=8)
st.markdown('<div class="movers-card">', unsafe_allow_html=True)
st.subheader("Live S&P 500 movers")
if not movers.empty:
    mover_cols = st.columns(4)
    for idx, (_, row) in enumerate(movers.iterrows()):
        ticker = str(row["Ticker"])
        pct = float(row["Change %"])
        label = f"{ticker} {'▲' if pct >= 0 else '▼'} {pct:+.2f}%"
        with mover_cols[idx % 4]:
            if st.button(label, key=f"mover_{ticker}", use_container_width=True):
                set_active_ticker(ticker)
else:
    st.caption("Top movers were not available right now.")
st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='muted'>Adjust settings, then run once. Slider changes will not retrigger heavy work until you submit.</div>", unsafe_allow_html=True)
    with st.form("dashboard_controls"):
        search_ticker = st.text_input("Search ticker", value=st.session_state.active_ticker).strip().upper()
        period = st.selectbox("History period", options=["1y", "2y", "5y", "10y"], index=["1y", "2y", "5y", "10y"].index(st.session_state.form_period))
        threshold = st.slider("Signal threshold", min_value=0.50, max_value=0.75, value=float(st.session_state.form_threshold), step=0.01)
        forecast_days = st.slider("Projection days", min_value=5, max_value=60, value=int(st.session_state.form_forecast_days), step=5)
        n_sims = st.slider("Projection paths", min_value=50, max_value=500, value=int(st.session_state.form_n_sims), step=50)
        run = st.form_submit_button("Run dashboard", use_container_width=True)

focus = (search_ticker or st.session_state.active_ticker or "AAPL").upper()
st.session_state.form_period = period
st.session_state.form_threshold = threshold
st.session_state.form_forecast_days = forecast_days
st.session_state.form_n_sims = n_sims

run_key = (focus, period, float(threshold), int(forecast_days), int(n_sims))
should_run = run or st.session_state.last_result is None or focus != st.session_state.active_ticker

result = st.session_state.last_result
summary = st.session_state.last_summary
live_headlines = st.session_state.last_headlines

if should_run:
    set_active_ticker(focus)
    with st.spinner(f"Running {focus} dashboard..."):
        result, summary, live_headlines = run_dashboard_data(
            st.session_state.active_ticker,
            period,
            float(threshold),
            int(forecast_days),
            int(n_sims),
        )
    st.session_state.last_run_key = run_key
    st.session_state.last_result = result
    st.session_state.last_summary = summary
    st.session_state.last_headlines = live_headlines
elif st.session_state.last_run_key != run_key and st.session_state.last_result is not None:
    st.caption("Settings changed. Press Run dashboard to refresh the model.")

if result is not None:
    st.markdown(f'<div class="{signal_class(safe_attr(result, "model_signal", "WATCH"))}">{safe_attr(result, "model_signal", "WATCH")}</div>', unsafe_allow_html=True)

    metric_cols = st.columns(2)
    metric_cols[0].metric("Price", f"${safe_attr(result, 'latest_close', 0.0):,.2f}")
    metric_cols[1].metric("Mood", safe_attr(result, "mood", "Neutral"))

    metric_cols = st.columns(2)
    metric_cols[0].metric("Up probability", f"{safe_attr(result, 'next_day_up_probability', 0.5):.2%}")
    metric_cols[1].metric("Accuracy", f"{safe_attr(result, 'holdout_accuracy', 0.0):.2%}")

    metric_cols = st.columns(2)
    metric_cols[0].metric("Momentum", f"{safe_attr(result, 'momentum_20d', 0.0):.2%}")
    metric_cols[1].metric("Volume", f"{safe_attr(result, 'volume_ratio', 1.0):.2f}x")

    metric_cols = st.columns(2)
    metric_cols[0].metric("News tone", safe_attr(result, "sentiment_label", "Neutral"))
    metric_cols[1].metric("Earnings", safe_attr(result, "earnings_flag", "No date found"))

    tab_dashboard, tab_details, tab_headlines, tab_share = st.tabs(PAGE_OPTIONS)

    with tab_dashboard:
        st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
        st.plotly_chart(build_projection_chart(summary, safe_attr(result, "latest_close", 0.0)), use_container_width=True)

    with tab_details:
        left, right = st.columns(2)
        with left:
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
            levels_df = pd.DataFrame({
                "Item": ["Support", "Resistance", "Stop loss", "Target 1", "Target 2"],
                "Value": [
                    safe_attr(result, "support_level", 0.0),
                    safe_attr(result, "resistance_level", 0.0),
                    safe_attr(result, "stop_loss", 0.0),
                    safe_attr(result, "target_1", 0.0),
                    safe_attr(result, "target_2", 0.0),
                ],
            })
            levels_df["Value"] = levels_df["Value"].map(lambda x: f"${x:,.2f}")
            st.dataframe(levels_df, use_container_width=True, hide_index=True)
            st.markdown('<span class="detail-label">Flags</span>', unsafe_allow_html=True)
            flags = safe_attr(result, "watchlist_flags", ["No major alert flags"])
            flags_html = "".join([f'<span class="flag">{flag}</span>' for flag in flags])
            st.markdown(flags_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Model summary")
            rsi = safe_attr(result, "rsi_14", 50.0)
            notes = [
                f"Signal is {safe_attr(result, 'model_signal', 'WATCH')}.",
                f"RSI is {'high' if rsi > 70 else 'low' if rsi < 40 else 'middle-range'} at {rsi:.1f}.",
                f"Price is {'above' if safe_attr(result, 'latest_close', 0.0) > safe_attr(result, 'sma20', 0.0) else 'below'} the 20-day average.",
                f"MACD is {'above' if safe_attr(result, 'macd', 0.0) > safe_attr(result, 'macd_signal', 0.0) else 'below'} its signal line.",
                f"Earnings status is {safe_attr(result, 'earnings_flag', 'No date found')}.",
            ]
            st.markdown("<ul class='detail-list'>" + "".join([f"<li>{n}</li>" for n in notes]) + "</ul>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Feature importance")
            feature_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
            if not feature_df.empty:
                feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
            else:
                st.write("No feature importance data available.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_headlines:
        if live_headlines:
            st.caption(f"Latest headlines for {st.session_state.active_ticker}. Refresh or rerun later for newer items.")
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
            st.write("No live headlines were returned right now. Try rerunning in a few minutes.")

    with tab_share:
        ticker_url = f"{APP_URL}?ticker={st.session_state.active_ticker}"
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Share this ticker")
        st.write(f"Share a direct link to the {st.session_state.active_ticker} dashboard.")
        st.code(ticker_url, language=None)
        st.link_button("Open direct ticker link", ticker_url, use_container_width=True)
        st.image(build_qr_code(ticker_url), caption="Scan to open this ticker on your phone", width=160)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Search one ticker on the left or tap a mover above.")
