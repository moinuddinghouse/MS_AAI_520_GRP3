"""Streamlit companion app for the Multi-Agent Financial Analysis System."""
from datetime import datetime, timezone
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Multi-Agent Financial Analysis", page_icon="📈", layout="wide")
st.title("📈 Multi-Agent Financial Analysis System")
st.caption("Research Manager → Market Specialist → Earnings Specialist → News Specialist → Evaluator–Optimizer")


def number(value, prefix="", suffix="", digits=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{prefix}{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def get_news(stock, ticker, limit=10):
    rows, diagnostics = [], []
    try:
        for item in (stock.news or []):
            content = item.get("content", item)
            if not isinstance(content, dict):
                content = item
            provider = content.get("provider", {})
            publisher = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
            canonical = content.get("canonicalUrl", {})
            link = canonical.get("url", "") if isinstance(canonical, dict) else item.get("link", "")
            title = content.get("title") or item.get("title")
            if title:
                rows.append({"Title": title, "Publisher": publisher, "Published": content.get("pubDate", ""), "Link": link, "Source": "Yahoo Finance"})
    except Exception as exc:
        diagnostics.append(f"Yahoo Finance news unavailable: {exc}")
    if not rows:
        try:
            q = urllib.parse.quote(f"{ticker} stock company finance")
            url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as response:
                root = ET.fromstring(response.read())
            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item"):
                    title = item.findtext("title", "")
                    if title:
                        source = item.find("source")
                        rows.append({"Title": title, "Publisher": source.text if source is not None else "Google News", "Published": item.findtext("pubDate", ""), "Link": item.findtext("link", ""), "Source": "Google News RSS"})
        except Exception as exc:
            diagnostics.append(f"Google News RSS fallback unavailable: {exc}")
    unique, seen = [], set()
    for row in rows:
        key = row["Title"].strip().lower()
        if key and key not in seen:
            unique.append(row)
            seen.add(key)
    return pd.DataFrame(unique[:limit]), diagnostics


def market_specialist(history):
    if history.empty or "Close" not in history:
        return {"error": "No historical market data was returned."}
    close = history["Close"].dropna()
    if close.empty:
        return {"error": "No valid closing prices were returned."}
    returns = close.pct_change().dropna()
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    latest = float(close.iloc[-1])
    return {
        "start_price": float(close.iloc[0]), "latest_price": latest,
        "period_return_pct": float((latest / float(close.iloc[0]) - 1) * 100),
        "annualized_volatility_pct": float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else None,
        "ma20": float(ma20) if pd.notna(ma20) else None,
        "ma50": float(ma50) if pd.notna(ma50) else None,
        "max_price": float(close.max()), "min_price": float(close.min()),
        "trend_signal": "Bullish" if ma20 is not None and latest > ma20 else ("Bearish" if ma20 is not None else "Insufficient data"),
        "observations": len(close),
    }


def make_report(ticker, market, info, news, prior_lesson):
    company = info.get("longName") or ticker
    lines = [f"# Investment Research Summary: {company} ({ticker})", "", "## Market specialist"]
    if "error" in market:
        lines.append(market["error"])
    else:
        lines += [
            f"- Latest adjusted close: {number(market.get('latest_price'), '$')}",
            f"- Period return: {number(market.get('period_return_pct'), suffix='%')}",
            f"- Annualized historical volatility: {number(market.get('annualized_volatility_pct'), suffix='%')}",
            f"- 20-day moving average: {number(market.get('ma20'), '$')}",
            f"- 50-day moving average: {number(market.get('ma50'), '$')}",
            f"- Trend heuristic: {market.get('trend_signal', 'N/A')}",
        ]
    lines += ["", "## Earnings / fundamentals specialist"]
    lines += [
        f"- Sector / industry: {info.get('sector') or 'N/A'} / {info.get('industry') or 'N/A'}",
        f"- Market capitalization: {number(info.get('marketCap'), '$', digits=0)}",
        f"- Trailing P/E: {number(info.get('trailingPE'))}",
        f"- Forward P/E: {number(info.get('forwardPE'))}",
        f"- Profit margin: {number((info.get('profitMargins') or 0) * 100, suffix='%') if info.get('profitMargins') is not None else 'N/A'}",
        f"- Return on equity: {number((info.get('returnOnEquity') or 0) * 100, suffix='%') if info.get('returnOnEquity') is not None else 'N/A'}",
        f"- Revenue growth: {number((info.get('revenueGrowth') or 0) * 100, suffix='%') if info.get('revenueGrowth') is not None else 'N/A'}",
    ]
    lines += ["", "## News specialist"]
    if news.empty:
        lines.append("No usable articles were returned by configured providers during this run. Current news sentiment cannot be assessed from missing data.")
    else:
        lines.extend([f"- {r['Title']} — {r['Publisher']}" for _, r in news.head(5).iterrows()])
    lines += ["", "## Risk and limitations", "- Historical returns and technical indicators do not guarantee future performance.", "- Financial metrics and headlines may be delayed, incomplete, or revised by providers.", "- This is an educational research tool, not personalized financial advice.", f"- Prior session memory: {prior_lesson or 'No prior lesson for this ticker in the current session.'}"]
    return "\n".join(lines)


with st.sidebar:
    st.header("Research configuration")
    ticker = st.text_input("Stock ticker", "AAPL").strip().upper()
    period = st.selectbox("Historical lookback", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    run = st.button("Run multi-agent analysis", type="primary", use_container_width=True)
    st.caption("Yahoo Finance data; Google News RSS fallback for headlines.")
    st.caption("Optional Gemini review can be enabled with GOOGLE_API_KEY in Streamlit Secrets.")

if "memory" not in st.session_state:
    st.session_state.memory = []
if "result" not in st.session_state:
    st.session_state.result = None

if run:
    if not ticker or any(c.isspace() for c in ticker):
        st.error("Enter a ticker such as AAPL or MSFT.")
        st.stop()
    try:
        with st.spinner(f"Research Manager coordinating specialists for {ticker}…"):
            stock = yf.Ticker(ticker)
            history = stock.history(period=period, auto_adjust=True)
            try:
                info = stock.info or {}
            except Exception:
                info = {}
            market = market_specialist(history)
            news, diagnostics = get_news(stock, ticker)
            prior = next((x["lesson"] for x in reversed(st.session_state.memory) if x["ticker"] == ticker), None)
            report = make_report(ticker, market, info, news, prior)
            # Score actual evidence coverage, not merely whether a variable exists.
            has_news_title = (
                not news.empty and "Title" in news.columns
                and news["Title"].fillna("").astype(str).str.strip().ne("").any()
            )
            has_news_link = (
                not news.empty and "Link" in news.columns
                and news["Link"].fillna("").astype(str).str.startswith("http").any()
            )
            fundamental_keys = ["marketCap", "trailingPE", "profitMargins", "returnOnEquity", "revenueGrowth"]
            fundamental_coverage = sum(info.get(k) is not None for k in fundamental_keys)
            news_failure_disclosed = "no usable articles" in report.lower() or "no usable headlines" in report.lower()
            checks = {
                "Market data coverage adequate (20+ observations)": (
                    "error" not in market and market.get("latest_price") is not None
                    and market.get("observations", 0) >= 20
                ),
                "Fundamental coverage adequate (3+ fields)": fundamental_coverage >= 3,
                "Usable news evidence retrieved": has_news_title and has_news_link,
                "Risk disclosure included": (
                    "Risk and limitations" in report and "do not guarantee" in report.lower()
                ),
                "Missing evidence handled transparently": (
                    (has_news_title and has_news_link) or news_failure_disclosed
                ) and ("N/A" in report or "not available" in report.lower() or "Risk and limitations" in report),
            }
            score = round(sum(bool(v) for v in checks.values()) / len(checks) * 100)
            st.session_state.result = {
                "ticker": ticker, "period": period, "time": datetime.now(timezone.utc).isoformat(),
                "history": history, "market": market, "info": info, "news": news,
                "diagnostics": diagnostics, "report": report, "checks": checks, "score": score,
            }
            st.session_state.memory.append({"ticker": ticker, "timestamp": datetime.now(timezone.utc).isoformat(), "lesson": f"{ticker}: checklist {score}/100; news items {len(news)}; trend {market.get('trend_signal', 'N/A')}. Disclose missing evidence."})
    except Exception as exc:
        st.error(f"Research run failed: {exc}")
        st.exception(exc)

r = st.session_state.result
if r is None:
    st.info("Choose a ticker in the sidebar and click **Run multi-agent analysis**.")
    st.markdown("**Workflow**: Research Manager → Market / Earnings / News specialists → Evaluator–Optimizer → Report + session memory")
    st.stop()

m, info, news = r["market"], r["info"], r["news"]
st.subheader(f"{info.get('longName') or r['ticker']} ({r['ticker']})")
st.caption(f"Lookback: {r['period']} · Run time (UTC): {r['time']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest adjusted close", number(m.get("latest_price"), "$"))
c2.metric("Period return", number(m.get("period_return_pct"), suffix="%"))
c3.metric("Annualized volatility", number(m.get("annualized_volatility_pct"), suffix="%"))
c4.metric("Checklist score", f"{r['score']}/100", help="Completeness checklist, not a measure of investment accuracy.")

# All sections are presented together as one consolidated report page (no separate tabs).
st.markdown("## Executive research report")
st.markdown("### 1. Executive summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest adjusted close", number(m.get("latest_price"), "$"))
c2.metric("Period return", number(m.get("period_return_pct"), suffix="%"))
c3.metric("Annualized volatility", number(m.get("annualized_volatility_pct"), suffix="%"))
c4.metric("Evidence checklist score", f"{r['score']}/100", help="Evidence completeness score, not investment accuracy.")
st.markdown(r["report"])
st.download_button(
    "Download complete research report (.md)", r["report"],
    file_name=f"{r['ticker']}_research_report.md", mime="text/markdown",
    key="download_complete_report",
)

st.divider()
st.markdown("### 2. Market analysis")
history = r["history"]
if not history.empty:
    chart = history.copy()
    chart["MA20"] = chart["Close"].rolling(20).mean()
    chart["MA50"] = chart["Close"].rolling(50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart.index, y=chart["Close"], name="Adjusted close"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart["MA20"], name="20-day MA"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart["MA50"], name="50-day MA"))
    fig.update_layout(title=f"{r['ticker']} price and moving averages", xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(history.tail(20), use_container_width=True)
else:
    st.warning("No market history returned.")
with st.expander("Market specialist details", expanded=False):
    st.json(m)

st.divider()
st.markdown("### 3. Earnings and fundamental analysis")
cols = st.columns(3)
metrics = [("Market cap", "marketCap", "money"), ("Trailing P/E", "trailingPE", "num"), ("Forward P/E", "forwardPE", "num"), ("Profit margin", "profitMargins", "pct"), ("Operating margin", "operatingMargins", "pct"), ("Return on equity", "returnOnEquity", "pct"), ("Revenue growth", "revenueGrowth", "pct"), ("Debt-to-equity", "debtToEquity", "num"), ("Beta", "beta", "num")]
for i, (label, key, kind) in enumerate(metrics):
    v = info.get(key)
    if kind == "money": value = number(v, "$", digits=0)
    elif kind == "pct": value = number(v * 100, suffix="%") if v is not None else "N/A"
    else: value = number(v)
    cols[i % 3].metric(label, value)
st.write("**Company:**", info.get("longName") or r["ticker"])
st.write("**Sector:**", info.get("sector") or "N/A", " | **Industry:**", info.get("industry") or "N/A")
st.caption("Provider fields may be missing, delayed, or revised.")

st.divider()
st.markdown("### 4. News evidence and specialist routing")
st.markdown("**Routing:** Research Manager → News Specialist → Report synthesis → Evaluator")
if news.empty:
    st.warning("No usable headlines were returned. Do not infer current sentiment from missing data.")
    for msg in r["diagnostics"]:
        st.caption(msg)
else:
    for _, row in news.iterrows():
        title = str(row["Title"])
        link = str(row.get("Link", ""))
        st.markdown(f"**[{title}]({link})**" if link.startswith("http") else f"**{title}**")
        st.caption(f"{row['Publisher']} · {row['Published']} · {row['Source']}")
        st.divider()

st.divider()
st.markdown("### 5. Evaluator–optimizer checklist")
st.metric("Evidence completeness", f"{r['score']}/100", help="This is not a measure of investment prediction accuracy.")
passed = sum(bool(v) for v in r["checks"].values())
st.caption(f"{passed} of {len(r['checks'])} checks passed. Failed checks indicate missing evidence or insufficient data coverage.")
for name, passed_check in r["checks"].items():
    st.write(("✅ " if passed_check else "⚠️ ") + name)

st.divider()
st.markdown("### 6. Session memory and execution record")
st.caption("Memory is held in the current Streamlit session; it may not persist across sessions or server restarts.")
st.dataframe(pd.DataFrame(st.session_state.memory), use_container_width=True)

if st.button("Optional Gemini report review"):
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.warning("Add GOOGLE_API_KEY under Streamlit app Settings → Secrets to enable Gemini review.")
    else:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            with st.spinner("Reviewing report with Gemini…"):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Review this educational financial research report for missing caveats and unsupported claims. Do not make buy/sell recommendations. Be concise.\n\n" + r["report"],
                )
            st.markdown("### Gemini review")
            st.write(response.text)
        except Exception as exc:
            st.warning(f"Gemini review unavailable: {exc}")

st.divider()
st.caption("Educational research only; not personalized financial advice. Validate data independently before making financial decisions.")
