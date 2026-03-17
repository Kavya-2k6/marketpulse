# app.py  ──  MarketPulse: Stock Price vs. News Sentiment Dashboard
# Run with:  streamlit run app.py
#
# INTERVIEW EXPLAINER: "The Streamlit app ties everything together.
# The sidebar lets users pick a ticker and date range. On submit, it
# runs the full pipeline — fetch → score → merge → visualize — and
# displays a dual-axis chart where stock price and sentiment are layered."

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Import our custom modules ──────────────────────────────────────────────
from data_fetcher import fetch_stock_data
from sentiment_engine import process_news_csv, generate_sample_news_csv, score_headline
from merger import merge_stock_and_sentiment

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — clean dark-finance aesthetic
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1d27;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #13151f; }

    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Sentiment color badge helper */
    .badge-positive { color: #48bb78; font-weight: 700; }
    .badge-negative { color: #fc8181; font-weight: 700; }
    .badge-neutral  { color: #90cdf4; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("📈 MarketPulse")
st.caption("Stock Price · News Sentiment · Correlation Dashboard")
st.divider()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR  — User Inputs
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Try: TSLA, GOOGL, MSFT, AMZN",
    ).upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date(2024, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date(2024, 6, 30))

    st.divider()

    # News data source selector
    news_source = st.radio(
        "News Data Source",
        options=["Use Sample Data (Demo)", "Upload My CSV"],
        index=0,
    )

    uploaded_file = None
    if news_source == "Upload My CSV":
        st.caption("CSV must have columns: `Date`, `Headline`")
        uploaded_file = st.file_uploader("Upload News CSV", type=["csv"])

    st.divider()
    run_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════
# HEADLINE SENTIMENT TESTER  (always visible, sidebar)
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.subheader("🧪 Test a Headline")
    test_headline = st.text_area("Paste any news headline:", height=80)
    if test_headline.strip():
        score = score_headline(test_headline)
        label = "🟢 Positive" if score > 0.05 else ("🔴 Negative" if score < -0.05 else "🔵 Neutral")
        st.metric("VADER Score", f"{score:.3f}", label)


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE  — runs when user clicks the button
# ══════════════════════════════════════════════════════════════════════════
if run_button:

    # ── 1. Validate dates ──────────────────────────────────────────
    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    with st.spinner("Fetching stock data from Yahoo Finance…"):
        stock_df = fetch_stock_data(
            ticker,
            start=str(start_date),
            end=str(end_date),
        )

    # ── 2. Load / generate news sentiment ─────────────────────────
    if news_source == "Use Sample Data (Demo)":
        with st.spinner("Generating sample news data…"):
            generate_sample_news_csv("_temp_news.csv")
            sentiment_df = process_news_csv("_temp_news.csv")

    else:  # User uploaded a file
        if uploaded_file is None:
            st.warning("⚠️ Please upload a CSV file, or switch to Sample Data.")
            st.stop()
        with st.spinner("Scoring your headlines with VADER…"):
            news_df = pd.read_csv(uploaded_file)
            sentiment_df = process_news_csv(uploaded_file)

    # ── 3. Merge ───────────────────────────────────────────────────
    with st.spinner("Merging datasets…"):
        merged = merge_stock_and_sentiment(stock_df, sentiment_df)

    if merged.empty:
        st.error("❌ No overlapping dates found after merge. "
                 "Check that your news dates fall within the stock date range.")
        st.stop()

    # ══════════════════════════════════════════════════════════════
    # METRIC CARDS  — quick summary stats
    # ══════════════════════════════════════════════════════════════
    st.subheader(f"📊 Summary — {ticker}")

    avg_sentiment = merged["Sentiment"].mean()
    price_change  = merged["Close"].iloc[-1] - merged["Close"].iloc[0]
    pct_change    = (price_change / merged["Close"].iloc[0]) * 100
    corr          = merged["Close"].corr(merged["Sentiment"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Daily Sentiment", f"{avg_sentiment:+.3f}",
              "Positive" if avg_sentiment > 0 else "Negative")
    m2.metric("Price Change", f"${price_change:+.2f}", f"{pct_change:+.1f}%")
    m3.metric("Data Points", f"{len(merged)} days")
    m4.metric("Price–Sentiment Correlation", f"{corr:.3f}",
              help="Close to +1 = strong positive link, -1 = inverse, 0 = no link")

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # DUAL-AXIS CHART  — Stock Price (line) + Sentiment (bars)
    # ══════════════════════════════════════════════════════════════
    st.subheader("📈 Stock Price vs. News Sentiment Over Time")

    # ── Color the bars: green if positive, red if negative ────────
    bar_colors = merged["Sentiment"].apply(
        lambda s: "#48bb78" if s >= 0 else "#fc8181"
    ).tolist()

    fig = go.Figure()

    # ── Trace 1: Sentiment BARS (left Y-axis) ─────────────────────
    fig.add_trace(go.Bar(
        x=merged["Date"],
        y=merged["Sentiment"],
        name="Daily Sentiment",
        marker_color=bar_colors,
        opacity=0.65,
        yaxis="y1",                 # Bind to left axis
    ))

    # ── Trace 2: Sentiment 7-day MA (left Y-axis, thin line) ──────
    fig.add_trace(go.Scatter(
        x=merged["Date"],
        y=merged["Sentiment_MA7"],
        name="Sentiment (7-day MA)",
        mode="lines",
        line=dict(color="#90cdf4", width=1.5, dash="dot"),
        yaxis="y1",
    ))

    # ── Trace 3: Stock PRICE LINE (right Y-axis) ──────────────────
    fig.add_trace(go.Scatter(
        x=merged["Date"],
        y=merged["Close"],
        name=f"{ticker} Close Price",
        mode="lines",
        line=dict(color="#f6e05e", width=2.5),
        yaxis="y2",                 # Bind to RIGHT axis
    ))

    # ── Layout: dual Y-axes ───────────────────────────────────────
    fig.update_layout(
        # Paper & plot background
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font=dict(color="#e2e8f0"),

        # Left Y-axis → Sentiment
        yaxis=dict(
            title="Sentiment Score (-1 to +1)",
            title_font=dict(color="#90cdf4"),
            tickfont=dict(color="#90cdf4"),
            range=[-1.1, 1.1],
            gridcolor="#2d3748",
            zeroline=True,
            zerolinecolor="#4a5568",
        ),

        # Right Y-axis → Stock Price (overlaid)
        yaxis2=dict(
            title=f"{ticker} Price (USD)",
            title_font=dict(color="#f6e05e"),
            tickfont=dict(color="#f6e05e"),
            overlaying="y",         # KEY: overlays on same chart
            side="right",
            gridcolor="rgba(0,0,0,0)",   # Hide right-axis grid lines (avoid clutter)
        ),

        xaxis=dict(
            gridcolor="#2d3748",
            showgrid=True,
        ),

        legend=dict(
            bgcolor="#1a1d27",
            bordercolor="#2d3748",
            borderwidth=1,
        ),

        hovermode="x unified",      # Hover shows both values at same date
        height=520,
        margin=dict(l=60, r=60, t=30, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # RAW DATA TABLE  (collapsible)
    # ══════════════════════════════════════════════════════════════
    with st.expander("🗂️ View Raw Merged Data"):
        display_df = merged.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        display_df["Close"] = display_df["Close"].round(2)
        display_df["Sentiment"] = display_df["Sentiment"].round(4)
        display_df["Sentiment_MA7"] = display_df["Sentiment_MA7"].round(4)
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            data=display_df.to_csv(index=False),
            file_name=f"marketpulse_{ticker}.csv",
            mime="text/csv",
        )

    # ══════════════════════════════════════════════════════════════
    # CORRELATION EXPLANATION  (interview-ready insight box)
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔬 Correlation Insight")

    if abs(corr) < 0.2:
        insight = f"**Weak correlation ({corr:.2f})** — News sentiment and {ticker}'s price don't move together much in this period. Markets are driven by many factors beyond headlines."
    elif corr >= 0.2:
        insight = f"**Positive correlation ({corr:.2f})** — In this period, positive news days tend to align with higher {ticker} prices. Stronger as score approaches +1."
    else:
        insight = f"**Negative correlation ({corr:.2f})** — Interesting! Positive news days seem to coincide with lower prices, or vice versa. This can happen with 'buy the rumor, sell the news' behavior."

    st.info(insight)


# ══════════════════════════════════════════════════════════════════════════
# DEFAULT STATE  — shown before clicking Run
# ══════════════════════════════════════════════════════════════════════════
else:
    st.info("👈 Configure your settings in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    **How this works:**
    1. 📥 Fetches stock closing prices via `yfinance`
    2. 🧠 Scores news headlines using VADER NLP (-1 to +1)
    3. 🔗 Merges both datasets on `Date` using `pandas.merge()`
    4. 📊 Plots a dual-axis chart (price line + sentiment bars)
    """)