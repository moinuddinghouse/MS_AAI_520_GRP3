# Multi-Agent Financial Analysis System — Streamlit Deployment

This folder contains a Streamlit companion app for the team's multi-agent financial research workflow.

## Files
- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `README.md` — Setup and deployment instructions.

## Features
- Research Manager workflow with market, fundamentals, and news specialist steps.
- Yahoo Finance market data and company fundamentals.
- Yahoo Finance news retrieval with Google News RSS fallback.
- Interactive price chart with 20-day and 50-day moving averages.
- Transparent evaluator checklist and completeness score.
- Session-level memory for previous runs during the current session.
- Optional Gemini report review using `GOOGLE_API_KEY`.
- Downloadable Markdown report.

## Run locally (macOS/Linux)
Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository for these files.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Sign in at https://share.streamlit.io/ with GitHub.
4. Select **Create app**, choose the repository and branch, and set **Main file path** to `app.py`.
5. Click **Deploy**.

## Optional Gemini review
The app works without Gemini. To enable the optional report review:
1. Obtain a Google Gemini API key.
2. In Streamlit Community Cloud, open your app's **Settings → Secrets**.
3. Add this TOML secret:

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

Never commit API keys to GitHub or hard-code them in `app.py`.

## Important notes
- This is a Streamlit deployment companion to the Jupyter notebook, not a hosted notebook runtime.
- Provider access and news availability can vary. Missing headlines are disclosed rather than fabricated.
- The evaluator score is a completeness checklist, not a financial prediction accuracy metric.
- Memory is session-level and may not persist across sessions or server restarts.
- This is an educational research tool, not investment advice.


## Updated report layout
The dashboard displays the executive summary, market analysis, fundamentals, news evidence, evaluation checklist, and session memory together on one page. The checklist score now requires adequate market history, at least three fundamental fields, and usable linked news to award those checks. It is a completeness score, not investment accuracy.
