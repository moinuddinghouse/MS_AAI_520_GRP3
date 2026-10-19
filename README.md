# Multi-Agent Financial Analysis System

A Streamlit web application for educational financial research. It combines historical market data, company fundamentals, recent news metadata, a research plan, specialist routing, a news-processing workflow, report synthesis, completeness evaluation, and lightweight JSON memory.

> **Disclaimer:** This is an educational research prototype, not personalized investment advice. Market history and LLM-generated explanations do not guarantee future results or factual accuracy.

## Features

- Stock ticker and historical-period selection
- Yahoo Finance price history, returns, volatility, and moving averages
- Company fundamentals when available from Yahoo Finance
- Recent news metadata from Google News RSS
- Research planning with Gemini when configured, with a deterministic fallback
- News classification, topic extraction, and specialist routing
- Gemini-based news summary and report synthesis when configured
- Deterministic report fallback if Gemini is unavailable
- Completeness checks, evaluator feedback, and a second report-generation pass
- JSON memory records for later runs
- Downloadable Markdown report

## Repository files

```text
app.py
requirements.txt
README.md
```

## Run locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Enable Gemini

Create a Google AI API key and configure it as an environment variable for local development:

```bash
export GOOGLE_API_KEY="your-key-here"
```

Optionally set a model name:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

Do not commit API keys or `.env` files.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app and select your repository, branch, and `app.py` as the main file.
4. In the app's **Settings → Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your-key-here"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```

5. Deploy and test with a ticker such as `AAPL`.

The app can run without Gemini, but planning, summarization, and report writing will use deterministic fallbacks. Provider availability and rate limits may vary.

## Notes and limitations

- Google News RSS supplies article metadata; article headlines and snippets are not a substitute for reading and verifying original articles.
- Yahoo Finance fields can be missing, delayed, or revised.
- The routing fallback is category/keyword-based. Review routing output before treating it as reliable specialist judgment.
- The completeness score checks for expected report sections; it is not an independent factual-accuracy score.
- The JSON memory file may not survive a Streamlit Community Cloud restart or redeployment because the hosting filesystem can be ephemeral. Use a persistent external store if durable cross-session memory is required.
- This Streamlit application is a companion implementation based on the project's workflow. It does not load or execute the notebook file itself.
