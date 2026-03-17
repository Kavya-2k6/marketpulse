# merger.py
# PURPOSE: Join the stock price data + sentiment data on the 'Date' column
#
# INTERVIEW EXPLAINER: "I used a pandas inner merge on 'Date'. Inner join means
# we only keep dates where BOTH datasets have data — so weekends (no stock trading)
# and news-less days are automatically dropped. No manual filtering needed."
#
# ── PANDAS MERGE CHEATSHEET (for your interview prep) ──────────────────────
#
#   pd.merge(left_df, right_df, on="Date", how="inner")
#   │                                       │
#   │                                       └─ "inner" = only matching dates
#   │                                          "left"  = all stock dates, NaN for missing sentiment
#   │                                          "outer" = all dates from both
#   └─ We pick "inner" because charting NaN values creates ugly gaps
#
# ───────────────────────────────────────────────────────────────────────────

import pandas as pd


def merge_stock_and_sentiment(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges stock prices and daily sentiment on the 'Date' column.

    Args:
        stock_df     : DataFrame with ['Date', 'Close']
        sentiment_df : DataFrame with ['Date', 'Sentiment']

    Returns:
        Merged DataFrame with ['Date', 'Close', 'Sentiment'], sorted by Date
    """

    # ── Step 1: Make sure both Date columns are the same TYPE ──────
    # Bug magnet: one might be datetime.date, other might be string
    # Fix: convert both to pandas Timestamp (datetime64)
    stock_df = stock_df.copy()
    sentiment_df = sentiment_df.copy()

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])

    # ── Step 2: The merge ──────────────────────────────────────────
    merged = pd.merge(
        stock_df,       # Left table  → stock prices
        sentiment_df,   # Right table → sentiment scores
        on="Date",      # Match rows where Date is equal
        how="inner",    # Keep ONLY rows present in BOTH tables
    )

    # ── Step 3: Sort chronologically ──────────────────────────────
    merged = merged.sort_values("Date").reset_index(drop=True)

    # ── Step 4: Add a 7-day rolling average of sentiment ──────────
    # Why? Daily sentiment is noisy. Rolling average shows the trend.
    # INTERVIEW: "rolling(7).mean() creates a moving average window of 7 days"
    merged["Sentiment_MA7"] = (
        merged["Sentiment"]
        .rolling(window=7, min_periods=1)  # min_periods=1 avoids NaN at start
        .mean()
        .round(3)
    )

    print(f"✅ Merged dataset: {len(merged)} rows | "
          f"Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")
    return merged


def load_and_merge_from_csvs(
    stock_csv: str = "stock_prices.csv",
    sentiment_csv: str = "daily_sentiment.csv",
) -> pd.DataFrame:
    """
    Convenience wrapper: reads both CSVs and returns the merged DataFrame.
    Use this in app.py to keep the Streamlit code clean.
    """
    stock_df = pd.read_csv(stock_csv)
    sentiment_df = pd.read_csv(sentiment_csv)
    return merge_stock_and_sentiment(stock_df, sentiment_df)


if __name__ == "__main__":
    # ── Quick test ──────────────────────────────────────────────────
    merged = load_and_merge_from_csvs()
    merged.to_csv("merged_data.csv", index=False)
    print(merged.head(10))
    print("\nColumn types:\n", merged.dtypes)
    # ────────────────────────────────────────────────────────────────
