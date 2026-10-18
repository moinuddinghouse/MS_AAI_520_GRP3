# Multi-Agent Financial Analysis Dashboard

This folder adds a Streamlit web dashboard to the Multi-Agent Financial Analysis System. It is designed to complement the course notebook, not replace it.

## Run locally

From the `financial_dashboard` folder:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Deploy with Streamlit Community Cloud

1. Upload the `financial_dashboard` folder to your GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app** and select your repository and branch.
4. Set the app file path to `financial_dashboard/app.py`.
5. Deploy. Streamlit installs dependencies from `financial_dashboard/requirements.txt`.

## Notes

- The dashboard uses Yahoo Finance data through `yfinance`; live news and company data may be missing or delayed.
- No API key is required for the deterministic dashboard version.
- Memory is stored in a local JSON file. Hosted container storage may be ephemeral, so do not treat the hosted JSON file as durable long-term storage.
- This is an educational research prototype, not financial advice.
