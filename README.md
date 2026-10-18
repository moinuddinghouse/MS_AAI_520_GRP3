# Multi-Agent Investment Research System

A Streamlit demo of a multi-agent financial research workflow. It retrieves market data from Yahoo Finance, calculates technical indicators, routes articles to specialist agents, evaluates evidence coverage, and saves a small local research history.

## Features

- Stock ticker input and market-price chart
- Technical indicators: 50-day SMA, 200-day SMA, RSI (14), annualized volatility, and one-year return
- Fundamental metrics when Yahoo Finance provides them
- Specialist roles for news, earnings, and market analysis
- Article routing with visibly labelled illustrative samples when live news is unavailable
- Evidence-aware evaluation that does not count sample articles as live news
- Optional Gemini-generated narrative analysis
- JSON report download and local research-memory table

## Important limitations

- Yahoo Finance may return incomplete data, rate-limit requests, or return no news.
- Illustrative sample articles are hypothetical and must not be described as real news.
- Local JSON memory is suitable for a demo, not reliable persistent storage on Streamlit Community Cloud. It may be lost after restart or redeployment.
- The app is for educational purposes only and does not provide personalized investment advice.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
streamlit run app.py
```

The app should open in your browser at the local URL printed by Streamlit.

## Optional Gemini setup

Without an API key, the app uses deterministic fallback summaries. To enable Gemini locally, set an environment variable before running the app:

```bash
export GOOGLE_API_KEY="your-key-here"   # macOS/Linux
# set GOOGLE_API_KEY=your-key-here      # Windows Command Prompt
```

Do not commit API keys to GitHub.

For Streamlit Community Cloud, add this in the app's **Settings → Secrets**:

```toml
GOOGLE_API_KEY = "your-key-here"
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io/ and connect your GitHub account.
4. Create an app from the repository and choose `app.py` as the entry point.
5. Add `GOOGLE_API_KEY` in the app's Secrets settings if using Gemini.
6. Deploy and test with a ticker such as `AAPL`.

## Architecture

1. **Research UI / orchestrator:** accepts a ticker and runs the research workflow.
2. **Market Specialist:** summarizes price-derived indicators.
3. **Earnings Specialist:** reports available fundamental metrics without inventing missing values.
4. **Router:** dispatches articles to the appropriate specialist.
5. **Evaluator:** scores market-data availability, fundamental-data availability, live-news availability, and whether article routing was demonstrated.
6. **Memory:** appends a compact record of each run to `research_memory.json`.

## Academic integrity and attribution

Adapt this project to your team's implementation, understand each component, and disclose AI assistance in accordance with your course requirements. Verify data outputs before using them in academic submissions.
