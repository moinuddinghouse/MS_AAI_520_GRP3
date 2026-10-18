# Multi-Agent Investment Research System

A Streamlit companion app for an educational multi-agent financial research workflow. It presents market indicators, available fundamentals, retrieved news, an evidence-coverage evaluator, and research history in one scrollable report.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional Gemini review

Add `GOOGLE_API_KEY` to Streamlit secrets or the environment. Without a key, the app uses deterministic fallback summaries. Never commit API keys.

## Streamlit Community Cloud

Push `app.py`, `requirements.txt`, and this README to the repository root (or select the correct app path in deployment settings). Set `GOOGLE_API_KEY` in **App settings → Secrets** if desired.

## Evaluation score

The score measures evidence/process coverage only. It is not prediction accuracy or a buy/sell signal. Market-cap validation compares reported market cap with latest price × shares outstanding when both values are available.

## Limitations

Yahoo Finance and Google News RSS availability/rate limits can vary. Local JSON memory may not persist across cloud restarts. The app does not currently include peer-group valuation percentiles or a validated predictive backtest. Educational use only; not personalized financial advice.
