# Multi-Agent Investment Research System

Educational Streamlit demonstration with an orchestrator/router, News specialist,
Earnings specialist, Market specialist, evaluator, and local research memory.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set `GOOGLE_API_KEY` as an environment variable or in Streamlit Community Cloud
Secrets to enable Gemini narrative analysis. Without a valid key/model access, the
app uses deterministic fallback summaries.

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to the GitHub repository.
2. In Streamlit Community Cloud, set the app's main file to `app.py`.
3. Add `GOOGLE_API_KEY = "your-key"` in **App settings → Secrets**.
4. Restart/redeploy the app and inspect the specialist tabs and logs.

## Notes

- Yahoo Finance may return no news, depending on symbol, region, and provider response.
- If sample articles are enabled, they are explicitly illustrative, not real news.
- The evaluation score measures workflow-check completion, not predictive performance.
- Local JSON memory may not persist across Streamlit Cloud restarts/redeployments because
  the app filesystem is not a durable database.
- Market data can be delayed or incomplete. This is for education, not investment advice.
