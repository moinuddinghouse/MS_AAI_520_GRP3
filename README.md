# Multi-Agent Investment Research System

Educational Streamlit demonstration for multi-agent financial research. It retrieves market history, selected Yahoo Finance fundamentals, and news headlines; displays moving averages; groups headlines with deterministic keyword rules; and reports evidence-coverage checks. Gemini is optional and falls back to built-in summaries when unavailable or quota-exhausted.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to your GitHub repository.
2. In Streamlit Community Cloud, select the repository and `app.py` as the entry point.
3. Optionally add `GOOGLE_API_KEY` in app Secrets. The app remains usable without Gemini.

## Notes

- The chart displays closing price, 50-day SMA, and 200-day SMA using the selected historical period.
- Market-cap validation distinguishes missing data from a discrepancy. It compares reported market cap with price × shares outstanding when both are available; source timestamps and share-count definitions may differ.
- News themes use deterministic keyword matching, not model-based sentiment or causal inference.
- Evidence coverage is a process checklist, not prediction accuracy or investment quality.
- Market data can be delayed or incomplete. This is an educational demonstration, not personalized investment advice.
- Streamlit Community Cloud's local filesystem may not persist research memory across restarts.
