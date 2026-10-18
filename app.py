"""
Multi-Agent Investment Research System — Streamlit demo.

This app uses Yahoo Finance for market/fundamental data and optionally Google
Gemini for narrative analysis. If Gemini is unavailable, it falls back to
deterministic summaries. Sample articles are clearly marked as illustrative
and are never represented as live news.
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatGoogleGenerativeAI = None
    HumanMessage = None


APP_DIR = Path(__file__).resolve().parent
MEMORY_PATH = APP_DIR / "research_memory.json"


# ---------- Data retrieval ----------
def fetch_market_data(ticker: str) -> dict[str, Any]:
    """Fetch recent price history and available company fundamentals."""
    stock = yf.Ticker(ticker)
    history = stock.history(period="1y", auto_adjust=True)
    if history is None or history.empty:
        raise ValueError(
            f"No price history was returned for {ticker}. Check the symbol "
            "and try again."
        )

    info: dict[str, Any] = {}
    try:
        info = stock.info or {}
    except Exception:
        # Yahoo Finance may rate-limit or omit fundamentals; prices can still work.
        info = {}

    close = history["Close"].dropna()
    returns = close.pct_change().dropna()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    current_price = float(close.iloc[-1])
    prior_price = float(close.iloc[-2]) if len(close) > 1 else current_price
    metrics = {
        "current_price": current_price,
        "daily_change_pct": (
            (current_price / prior_price - 1) * 100 if prior_price else 0.0
        ),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "profit_margin": info.get("profitMargins"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "sma50": float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None,
        "rsi14": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "annualized_volatility_pct": (
            float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else None
        ),
        "period_return_pct": (
            float((close.iloc[-1] / close.iloc[0] - 1) * 100)
            if len(close) > 1 else 0.0
        ),
    }
    return {"history": history, "metrics": metrics, "info": info}


# ---------- Specialist agents ----------
class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def _gemini(self, prompt: str) -> str | None:
        """Return a Gemini response when configured; otherwise return None."""
        api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
        if not api_key or ChatGoogleGenerativeAI is None:
            return None
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                google_api_key=api_key,
                temperature=0.2,
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            return str(response.content)
        except Exception as exc:
            st.warning(f"{self.name}: Gemini was unavailable; using fallback summary. ({exc})")
            return None


class NewsSpecialistAgent(BaseAgent):
    def analyze_article(self, article: dict[str, str]) -> dict[str, str]:
        prompt = (
            "You are a financial news analyst. Analyze the article below. "
            "Return a concise summary, sentiment (positive/neutral/negative), "
            "and possible market relevance. Do not invent facts.\n\n"
            f"Title: {article['title']}\nText: {article['text']}"
        )
        answer = self._gemini(prompt)
        if answer:
            return {"specialist": self.name, "analysis": answer}
        return {
            "specialist": self.name,
            "analysis": (
                f"Illustrative article: {article['title']}. "
                "Gemini is not configured; no independent sentiment conclusion "
                "is asserted by the fallback."
            ),
        }


class EarningsSpecialistAgent(BaseAgent):
    def analyze_article(self, article: dict[str, str]) -> dict[str, str]:
        prompt = (
            "You are a fundamental/earnings analyst. Explain what the supplied "
            "content says about revenue, profit, guidance, or financial results. "
            "If no financial figures are provided, explicitly say so. Do not "
            "invent facts.\n\n"
            f"Title: {article['title']}\nText: {article['text']}"
        )
        answer = self._gemini(prompt)
        if answer:
            return {"specialist": self.name, "analysis": answer}
        return {
            "specialist": self.name,
            "analysis": (
                f"{article['title']}: the illustrative text contains no verified "
                "financial statement data. Check official filings before drawing "
                "fundamental conclusions."
            ),
        }


class MarketSpecialistAgent(BaseAgent):
    def analyze(self, ticker: str, metrics: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "You are a market/technical analyst. Summarize the supplied calculated "
            "metrics. Distinguish data from interpretation; do not give personalized "
            "financial advice.\n"
            f"Ticker: {ticker}\nMetrics: {json.dumps(metrics, default=str)}"
        )
        answer = self._gemini(prompt)
        trend = "unavailable"
        if metrics.get("sma50") is not None and metrics.get("sma200") is not None:
            trend = "above" if metrics["sma50"] > metrics["sma200"] else "below"
        return {
            "specialist": self.name,
            "trend_signal": trend,
            "analysis": answer or (
                f"The 50-day SMA is {trend} the 200-day SMA where available. "
                f"One-year price return: {metrics.get('period_return_pct', 0):.2f}%. "
                "This is a descriptive indicator, not a forecast."
            ),
        }


class EarningsSpecialist:
    """Fundamental specialist for available Yahoo Finance company metrics."""
    def __init__(self, name: str):
        self.name = name

    def analyze(self, info: dict[str, Any]) -> dict[str, Any]:
        data = {
            "revenue": info.get("totalRevenue"),
            "net_income": info.get("netIncomeToCommon"),
            "profit_margin": info.get("profitMargins"),
            "trailing_PE": info.get("trailingPE"),
            "debt_to_equity": info.get("debtToEquity"),
        }
        return {
            "specialist": self.name,
            "metrics": data,
            "analysis": (
                "Fundamental metrics are shown when Yahoo Finance provides them. "
                "Unavailable values are left blank rather than estimated."
            ),
        }


class RouterAgent(BaseAgent):
    def __init__(self, name: str, news_agent: NewsSpecialistAgent,
                 earnings_agent: EarningsSpecialistAgent,
                 market_agent: MarketSpecialistAgent):
        super().__init__(name)
        self.news_agent = news_agent
        self.earnings_agent = earnings_agent
        self.market_agent = market_agent

    def route_article(self, article: dict[str, str]) -> dict[str, str]:
        category = article.get("category", "news").lower()
        if category == "earnings":
            result = self.earnings_agent.analyze_article(article)
        else:
            result = self.news_agent.analyze_article(article)
        result["category"] = category
        result["article_title"] = article["title"]
        return result

    def route_market(self, ticker: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return self.market_agent.analyze(ticker, metrics)


# ---------- Workflow and memory ----------
def load_memory() -> list[dict[str, Any]]:
    try:
        if MEMORY_PATH.exists():
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def save_memory(record: dict[str, Any]) -> None:
    history = load_memory()
    history.append(record)
    # Keep a small local demo history.
    MEMORY_PATH.write_text(json.dumps(history[-50:], indent=2, default=str), encoding="utf-8")


def run_research(ticker: str, use_sample_articles: bool) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    data = fetch_market_data(ticker)

    news_agent = NewsSpecialistAgent("News Specialist")
    earnings_article_agent = NewsSpecialistAgent("Earnings Specialist")
    market_agent = MarketSpecialistAgent("Market Specialist")
    earnings_agent = EarningsSpecialist("Earnings Specialist")
    router = RouterAgent("Router", news_agent, earnings_article_agent, market_agent)

    # Yahoo Finance news is attempted; examples are explicitly illustrative.
    raw_news: list[dict[str, Any]] = []
    try:
        raw_news = yf.Ticker(ticker).news or []
    except Exception:
        raw_news = []

    articles: list[dict[str, str]] = []
    for item in raw_news:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        summary = content.get("summary") or item.get("summary") or ""
        if title:
            articles.append({
                "title": str(title),
                "text": str(summary),
                "category": "news",
                "source": "Yahoo Finance",
            })

    news_is_live = bool(articles)
    if not articles and use_sample_articles:
        articles = [
            {
                "title": "Illustrative earnings update",
                "text": "Hypothetical example only: a company reports revenue growth but provides no verified figures.",
                "category": "earnings",
                "source": "Illustrative sample — not live news",
            },
            {
                "title": "Illustrative market commentary",
                "text": "Hypothetical example only: market participants discuss recent price volatility.",
                "category": "news",
                "source": "Illustrative sample — not live news",
            },
        ]

    routed_articles = [router.route_article(article) for article in articles]
    market_result = router.route_market(ticker, data["metrics"])
    earnings_result = earnings_agent.analyze(data["info"])

    # Evidence-aware evaluation: live-news evidence is required for full news coverage.
    checks = {
        "market_data_available": not data["history"].empty,
        "fundamental_metrics_available": any(
            value is not None for value in earnings_result["metrics"].values()
        ),
        "live_news_available": news_is_live,
        "specialist_routing_demonstrated": len(routed_articles) > 0,
    }
    score = sum(checks.values()) / len(checks) * 100
    feedback = []
    if not checks["fundamental_metrics_available"]:
        feedback.append("Fundamental metrics were unavailable from the data provider.")
    if not news_is_live:
        feedback.append("Live news was unavailable; sample articles are illustrative only.")
    if not checks["specialist_routing_demonstrated"]:
        feedback.append("No articles were available to demonstrate article-level routing.")
    final_report = {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": data["metrics"],
        "market_analysis": market_result,
        "fundamental_analysis": earnings_result,
        "routed_articles": routed_articles,
        "live_news_available": news_is_live,
        "evaluation": {"score": score, "checks": checks, "feedback": feedback},
        "limitations": [
            "Market data is provided by Yahoo Finance and may be delayed or incomplete.",
            "Illustrative articles are not real financial news.",
            "This application is for educational research, not investment advice.",
        ],
    }
    save_memory({
        "ticker": ticker,
        "generated_at": final_report["generated_at"],
        "score": score,
        "live_news_available": news_is_live,
    })
    return {"report": final_report, "history": data["history"], "info": data["info"]}


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Multi-Agent Investment Research", page_icon="📈", layout="wide")
st.title("📈 Multi-Agent Investment Research System")
st.caption("Research orchestrator · News specialist · Earnings specialist · Market specialist · Evaluator")
st.info(
    "Educational demonstration only. Market data may be delayed or incomplete. "
    "This app does not provide personalized investment advice."
)

with st.sidebar:
    st.header("Research settings")
    ticker_input = st.text_input("Stock symbol", value="AAPL", max_chars=12).strip().upper()
    use_sample_articles = st.checkbox(
        "Use clearly labelled sample articles if live news is unavailable",
        value=True,
    )
    st.caption("Live news is only reported when the Yahoo Finance provider returns articles.")
    run_button = st.button("Run research", type="primary", width="stretch")
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("- Yahoo Finance (prices/fundamentals/news)\n- Google Gemini (optional narrative analysis)")

if run_button:
    if not ticker_input or not ticker_input.replace(".", "").replace("-", "").isalnum():
        st.error("Enter a valid ticker symbol, such as AAPL or MSFT.")
    else:
        with st.spinner(f"Researching {ticker_input}..."):
            try:
                st.session_state["research_result"] = run_research(
                    ticker_input, use_sample_articles
                )
            except Exception as exc:
                st.error(f"Research failed: {exc}")
                st.stop()

result = st.session_state.get("research_result")
if result:
    report = result["report"]
    metrics = report["metrics"]
    st.subheader(f"Research dashboard: {report['ticker']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price", f"{metrics['current_price']:,.2f}",
              f"{metrics['daily_change_pct']:+.2f}%")
    c2.metric("1-year return", f"{metrics['period_return_pct']:+.2f}%")
    c3.metric("P/E ratio", f"{metrics['pe_ratio']:.2f}" if metrics["pe_ratio"] is not None else "N/A")
    c4.metric("Evaluation score", f"{report['evaluation']['score']:.0f}/100")

    tab_overview, tab_market, tab_earnings, tab_news, tab_logs = st.tabs(
        ["Executive summary", "Market analysis", "Earnings analysis", "News & routing", "Evaluation & memory"]
    )
    with tab_overview:
        st.markdown("### Executive summary")
        st.write(
            f"The latest available price for **{report['ticker']}** is "
            f"{metrics['current_price']:,.2f}. The one-year price return is "
            f"{metrics['period_return_pct']:+.2f}%. This summary describes historical "
            "data and should not be interpreted as a forecast."
        )
        st.markdown("### Research limitations")
        for item in report["limitations"]:
            st.write(f"- {item}")
        st.download_button(
            "Download research report (JSON)",
            data=json.dumps(report, indent=2, default=str),
            file_name=f"{report['ticker']}_research_report.json",
            mime="application/json",
        )

    with tab_market:
        st.markdown("### Price history")
        st.line_chart(result["history"]["Close"], width="stretch")
        st.markdown("### Technical indicators")
        technical = {
            "50-day SMA": metrics["sma50"],
            "200-day SMA": metrics["sma200"],
            "RSI (14)": metrics["rsi14"],
            "Annualized volatility (%)": metrics["annualized_volatility_pct"],
            "52-week high": metrics["fifty_two_week_high"],
            "52-week low": metrics["fifty_two_week_low"],
        }
        st.dataframe(pd.DataFrame(
            [{"Indicator": k, "Value": v} for k, v in technical.items()]
        ), width="stretch", hide_index=True)
        st.markdown("### Market specialist")
        st.write(report["market_analysis"]["analysis"])

    with tab_earnings:
        st.markdown("### Available company fundamentals")
        fundamental_rows = [
            {"Metric": k.replace("_", " ").title(), "Value": v}
            for k, v in report["fundamental_analysis"]["metrics"].items()
        ]
        st.dataframe(pd.DataFrame(fundamental_rows), width="stretch", hide_index=True)
        st.write(report["fundamental_analysis"]["analysis"])

    with tab_news:
        st.write(
            "Live news status: **Available**" if report["live_news_available"]
            else "**Unavailable — sample articles, if enabled, are illustrative only.**"
        )
        if report["routed_articles"]:
            for article in report["routed_articles"]:
                with st.expander(
                    f"{article['specialist']} · {article['article_title']}"
                ):
                    st.caption(f"Category: {article['category']}")
                    st.write(article["analysis"])
        else:
            st.warning("No articles were available for article-level routing.")

    with tab_logs:
        st.markdown("### Evaluation checks")
        st.json(report["evaluation"]["checks"])
        st.markdown("### Evaluator feedback")
        if report["evaluation"]["feedback"]:
            for feedback in report["evaluation"]["feedback"]:
                st.write(f"- {feedback}")
        else:
            st.success("All configured checks passed.")
        st.markdown("### Local research memory")
        memory = load_memory()
        if memory:
            st.dataframe(pd.DataFrame(memory[::-1]), width="stretch", hide_index=True)
        else:
            st.caption("No saved research runs yet.")
else:
    st.markdown("Enter a stock ticker in the sidebar and click **Run research** to begin.")
