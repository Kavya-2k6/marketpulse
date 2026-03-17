# sentiment_engine.py
# PURPOSE: Read a CSV of news headlines → add a VADER sentiment score per row
#          → group by Date → get ONE daily sentiment score
#
# INTERVIEW EXPLAINER: "VADER gives a 'compound' score from -1 (very negative)
# to +1 (very positive). Since there are multiple headlines per day, I averaged
# them so each date gets one score — making it easy to merge with stock data."

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Initialize VADER once (expensive to recreate per row) ──────────────────
analyzer = SentimentIntensityAnalyzer()


def score_headline(text: str) -> float:
    """
    Returns the compound sentiment score for a single headline.
    Range: -1.0 (most negative) to +1.0 (most positive)
    """
    scores = analyzer.polarity_scores(text)
    return scores["compound"]   # We only need compound for our chart


def process_news_csv(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV that has at minimum two columns: 'Date' and 'Headline'.
    Returns a DataFrame with one row per date: ['Date', 'Sentiment']

    Args:
        filepath: Path to your news CSV file

    Returns:
        DataFrame with columns: ['Date', 'Sentiment']
    """
    # ── Step 1: Load the CSV ───────────────────────────────────────
    df = pd.read_csv(filepath)

    # Guard: make sure required columns exist
    required = {"Date", "Headline"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {set(df.columns)}")

    # ── Step 2: Drop rows with missing headlines ───────────────────
    df = df.dropna(subset=["Headline"])

    # ── Step 3: Score every headline (apply runs score_headline on each row) ──
    # This is the core NLP step — one number per headline
    df["Sentiment"] = df["Headline"].apply(score_headline)

    # ── Step 4: Convert Date column to proper datetime ─────────────
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # ── Step 5: Group by Date → average all headline scores that day ──
    # Why average? Because 5 positive + 1 negative = net positive day
    daily_sentiment = (
        df.groupby("Date")["Sentiment"]
        .mean()
        .reset_index()
    )
    daily_sentiment.columns = ["Date", "Sentiment"]

    print(f"✅ Scored sentiment for {len(daily_sentiment)} unique dates")
    return daily_sentiment


# ── Helper: generate SAMPLE data if you don't have a real CSV yet ──────────
def generate_sample_news_csv(output_path: str = "sample_news.csv"):
    """
    Creates a small fake news CSV so you can test the pipeline immediately.
    Replace with real data from NewsAPI later.
    """
    import random
    from datetime import date, timedelta

    headlines_pool = [
        # Positive
        "Apple reports record quarterly revenue, beating analyst expectations",
        "Strong jobs data boosts market confidence",
        "Tech stocks rally as Fed signals rate pause",
        "Apple unveils breakthrough AI chip, stock surges",
        "iPhone sales hit all-time high in emerging markets",
        # Negative
        "Recession fears grow as manufacturing data disappoints",
        "Apple faces antitrust probe in European Union",
        "Supply chain disruptions threaten holiday iPhone production",
        "Markets tumble on inflation concerns",
        "Apple misses revenue targets amid weak China demand",
        # Neutral
        "Apple announces annual developer conference dates",
        "Federal Reserve releases meeting minutes",
        "Apple updates privacy policy for app developers",
        "Tech sector holds steady ahead of earnings season",
    ]

    rows = []
    start = date(2024, 1, 2)
    for i in range(120):  # ~4 months of weekdays
        current_date = start + timedelta(days=i)
        if current_date.weekday() >= 5:   # Skip weekends
            continue
        # 1–3 headlines per day
        n = random.randint(1, 3)
        for _ in range(n):
            rows.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Headline": random.choice(headlines_pool),
            })

    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(output_path, index=False)
    print(f"✅ Sample news CSV written to: {output_path} ({len(sample_df)} rows)")


if __name__ == "__main__":
    # ── Quick test with sample data ──────────────────────────────────
    generate_sample_news_csv("sample_news.csv")
    result = process_news_csv("sample_news.csv")
    result.to_csv("daily_sentiment.csv", index=False)
    print(result.head(10))
    # ────────────────────────────────────────────────────────────────
