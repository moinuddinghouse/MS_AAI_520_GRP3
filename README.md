# Multi-Agent Investment Research System

A Streamlit educational demo for market data, technical indicators, company fundamentals, news evidence, and evaluator checks.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Select a historical period

Use **Historical period** in the sidebar to choose 1 year, 2 years, 5 years, 10 years, or maximum available price history. The selected period applies to the market chart and period-return calculation. Indicators such as the 50-day and 200-day moving averages still use their named trading-day windows.

## Optional Gemini review

Set `GOOGLE_API_KEY` as an environment variable or in Streamlit Community Cloud under **App settings → Secrets**:

```toml
GOOGLE_API_KEY = "your-api-key"
```

The app uses deterministic fallback text when Gemini is not configured. Do not commit secrets to GitHub.

## Deploy to Streamlit Community Cloud

Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository, then create an app in Streamlit Community Cloud and select `app.py` as the entry point.

## Limitations

Market data/news availability depends on external providers and may be delayed or incomplete. Historical performance is not predictive. The evidence-coverage score measures process checks, not prediction accuracy or investment quality. This is an educational application, not personalized investment advice.
