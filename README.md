# MarketPulse 📈
**Stock Price vs. News Sentiment Dashboard**

A Python dashboard that fetches live stock prices and analyzes 
news sentiment to find correlations between market mood and price movement.

## Tech Stack
- `yfinance` — live stock data
- `VADER NLP` — sentiment scoring
- `Pandas` — data merging & rolling averages
- `Streamlit` + `Plotly` — interactive dashboard

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Dual-axis chart: stock price line + sentiment bars
- 7-day rolling sentiment average
- Downloadable merged CSV
- Live headline sentiment tester in sidebar
