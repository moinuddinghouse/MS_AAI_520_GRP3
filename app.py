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
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
def fetch_market_data(ticker: str, period: str = "1y") -> dict[str, Any]:
    """Fetch price history for the selected lookback period and company fundamentals."""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, auto_adjust=True)
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


def fetch_google_news_rss(ticker: str, limit: int = 8) -> list[dict[str, str]]:
    """Try Google News RSS when Yahoo Finance returns no usable headlines."""
    query = urllib.parse.quote(f"{ticker} stock company")
    url = (
        "https://news.google.com/rss/search?q="
        f"{query}&hl=en-US&gl=US&ceid=US:en"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; InvestmentResearchDemo/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        items: list[dict[str, str]] = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title", default="").strip()
            description = item.findtext("description", default="").strip()
            link = item.findtext("link", default="").strip()
            if title:
                items.append({
                    "title": title,
                    "text": description or title,
                    "category": "news",
                    "source": "Google News RSS",
                    "url": link,
                })
        return items
    except Exception:
        # Network access can be blocked on hosted platforms; keep fallback explicit.
        return []


# ---------- Specialist agents ----------
class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def _gemini(self, prompt: str) -> str | None:
        """Return a Gemini response when configured; otherwise return None."""
        if st.session_state.get("gemini_disabled_for_session", False):
            return None
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
            # Quota errors are session-latched so every article does not repeat
            # the same long 429 traceback or trigger more Gemini calls.
            message = str(exc)
            quota_error = "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower()
            st.session_state["gemini_disabled_for_session"] = True
            if not st.session_state.get("gemini_warning_shown", False):
                if quota_error:
                    st.warning("Gemini quota is exhausted for this session. Continuing with the built-in fallback; market, fundamentals, and news retrieval remain available.")
                else:
                    st.warning("Gemini is unavailable. Continuing with the built-in fallback summary.")
                st.session_state["gemini_warning_shown"] = True
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


def evaluate_research(data: dict[str, Any], earnings_result: dict[str, Any],
                      articles: list[dict[str, Any]], news_is_live: bool,
                      market_result: dict[str, Any], report_text: str) -> dict[str, Any]:
    """Evaluate evidence coverage and process quality, not investment accuracy."""
    metrics = data["metrics"]
    info = data["info"]
    fundamentals = earnings_result["metrics"]
    available_fundamentals = sum(v is not None for v in fundamentals.values())
    usable_news = [a for a in articles if a.get("title") and a.get("source") != "Illustrative sample — not live news"]
    shares = info.get("sharesOutstanding")
    price = metrics.get("current_price")
    market_cap = metrics.get("market_cap")
    implied_cap = float(price) * float(shares) if price is not None and shares else None
    cap_consistent = None
    if market_cap and implied_cap:
        cap_consistent = abs(float(market_cap) - implied_cap) / max(abs(float(market_cap)), 1) <= 0.15
    trend_method_explained = "50-day SMA" in market_result.get("analysis", "") or "200-day SMA" in market_result.get("analysis", "") or market_result.get("trend_signal") != "unavailable"
    checks = {
        "Sufficient price history": not data["history"].empty and len(data["history"]) >= 20,
        "Technical method is explicit": trend_method_explained and metrics.get("sma50") is not None,
        "Fundamental evidence coverage": available_fundamentals >= 3,
        "Usable live news evidence": news_is_live and len(usable_news) > 0,
        "Market-cap cross-check": cap_consistent is True if cap_consistent is not None else False,
        "Limitations and uncertainty disclosed": "Risk and limitations" in report_text and "do not guarantee" in report_text.lower(),
        "Missing evidence handled transparently": (news_is_live or "no usable news articles" in report_text.lower() or "news unavailable" in report_text.lower()) and ("N/A" in report_text or "not available" in report_text.lower()),
    }
    passed = sum(bool(v) for v in checks.values())
    feedback = []
    if not checks["Technical method is explicit"]:
        feedback.append("State the trend rule explicitly (for example, 50-day SMA compared with 200-day SMA).")
    if not checks["Fundamental evidence coverage"]:
        feedback.append("Limited fundamentals are available; avoid strong conclusions and verify company filings.")
    if not checks["Usable live news evidence"]:
        feedback.append("Live news evidence is missing or sample-only; do not present illustrative items as current news.")
    if not checks["Market-cap cross-check"]:
        if implied_cap is None or market_cap is None:
            feedback.append("Market-cap cross-check is unavailable because market capitalisation, current price, or shares outstanding is missing. This is a data-availability issue, not proof that the reported value is wrong.")
        else:
            difference_pct = abs(float(market_cap) - implied_cap) / max(abs(float(market_cap)), 1) * 100
            feedback.append(f"Market-cap cross-check differs by {difference_pct:.1f}% (tolerance: 15%). Verify timestamps, share classes, and share-count definitions before concluding the value is wrong.")
    feedback.append("This score measures evidence and process coverage, not prediction accuracy or investment quality.")
    return {"score": round(100 * passed / len(checks)), "checks": checks, "passed": passed, "total": len(checks),
            "available_fundamental_fields": available_fundamentals, "implied_market_cap": implied_cap, "market_cap_consistent": cap_consistent, "feedback": feedback}


def run_research(ticker: str, use_sample_articles: bool, period: str = "1y") -> dict[str, Any]:
    ticker = ticker.strip().upper()
    data = fetch_market_data(ticker, period=period)

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
    if not articles:
        # Secondary live source: Google News RSS. It may also be unavailable
        # when the deployment environment blocks external requests.
        articles = fetch_google_news_rss(ticker)
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

    # Build a compact report text first, then evaluate evidence coverage.
    report_text = (
        "Risk and limitations: data may be delayed or incomplete; historical indicators do not guarantee future results. "
        "Unavailable values are marked N/A. "
        + ("No usable news articles were retrieved." if not news_is_live else "Live news was retrieved.")
    )
    evaluation = evaluate_research(data, earnings_result, articles, news_is_live, market_result, report_text)
    final_report = {
        "ticker": ticker,
        "selected_period": period,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": data["metrics"],
        "market_analysis": market_result,
        "fundamental_analysis": earnings_result,
        "routed_articles": routed_articles,
        "raw_articles": articles,
        "live_news_available": news_is_live,
        "evaluation": evaluation,
        "report_text": report_text,
        "limitations": [
            "Market data is provided by Yahoo Finance and may be delayed or incomplete.",
            "Illustrative articles are not real financial news.",
            "This application is for educational research, not investment advice.",
        ],
    }
    save_memory({
        "ticker": ticker,
        "generated_at": final_report["generated_at"],
        "score": evaluation["score"],
        "live_news_available": news_is_live,
    })
    return {"report": final_report, "history": data["history"], "info": data["info"]}


def synthesize_news(articles: list[dict[str, Any]]) -> dict[str, Any]:
    """Group live headlines into transparent keyword themes without an LLM."""
    live = [a for a in articles if a.get("title") and not str(a.get("source", "")).startswith("Illustrative sample")]
    rules = {
        "Earnings and financial performance": ("earnings", "revenue", "profit", "quarter", "sales", "margin", "guidance", "results"),
        "Products, AI and strategy": ("product", "iphone", "ai", "artificial intelligence", "launch", "strategy", "innovation"),
        "Analyst views and valuation": ("analyst", "price target", "valuation", "buy", "sell", "stock to buy", "defensive play"),
        "Institutional ownership and trading": ("stake", "shares", "purchased", "investor", "fund", "ownership", "market cap", "market value"),
        "Other company or market news": (),
    }
    groups = {k: [] for k in rules}
    for article in live:
        title = str(article.get("title", ""))
        low = title.lower()
        chosen = "Other company or market news"
        for theme, keywords in rules.items():
            if theme != "Other company or market news" and any(word in low for word in keywords):
                chosen = theme
                break
        groups[chosen].append(title)
    return {"article_count": len(live), "groups": {k: v for k, v in groups.items() if v},
            "method": "Deterministic keyword grouping of headlines; not an LLM sentiment or causal-impact analysis."}


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
    period_options = {"1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y", "Maximum available": "max"}
    selected_period_label = st.selectbox("Historical period", list(period_options.keys()), index=0, help="Choose how much historical price data to analyze and display.")
    selected_period = period_options[selected_period_label]
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
                    ticker_input, use_sample_articles, selected_period
                )
            except Exception as exc:
                st.error(f"Research failed: {exc}")
                st.stop()

result = st.session_state.get("research_result")
if result:
    report = result["report"]
    metrics = report["metrics"]
    evaluation = report["evaluation"]

    st.subheader(f"Research report: {report['ticker']}")
    st.caption(f"Period: {report.get('selected_period', '1y')} · Generated {report['generated_at']} · Sources: Yahoo Finance; Google News RSS fallback · Gemini is optional")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest price", f"{metrics['current_price']:,.2f}", f"{metrics['daily_change_pct']:+.2f}%")
    c2.metric(f"{report.get('selected_period', '1y')} return", f"{metrics['period_return_pct']:+.2f}%")
    c3.metric("Annualized volatility", f"{metrics['annualized_volatility_pct']:.2f}%" if metrics['annualized_volatility_pct'] is not None else "N/A")
    c4.metric("Evidence coverage", f"{evaluation['score']}/100", f"{evaluation['passed']}/{evaluation['total']} checks")

    st.markdown("## 1. Executive summary")
    trend = report["market_analysis"].get("trend_signal", "unavailable")
    st.write(f"For **{report['ticker']}**, the latest available adjusted close is **{metrics['current_price']:,.2f}** and the selected-period ({report.get('selected_period', '1y')}) price return is **{metrics['period_return_pct']:+.2f}%**. The technical specialist reports the 50-day SMA as **{trend}** the 200-day SMA where both are available. This is a historical indicator, not a forecast or recommendation.")
    st.caption("Interpretation rule: the trend signal compares the 50-day simple moving average (SMA) with the 200-day SMA. ‘Above’ is a positive long/short moving-average configuration; it does not establish future performance.")

    st.markdown("## 2. Market analysis")
    chart_data = result["history"]["Close"].dropna()
    st.line_chart(chart_data, x_label="Date", y_label="Adjusted close")
    tech_rows = [
        {"Indicator": "50-day SMA", "Value": metrics.get("sma50")},
        {"Indicator": "200-day SMA", "Value": metrics.get("sma200")},
        {"Indicator": "RSI (14)", "Value": metrics.get("rsi14")},
        {"Indicator": "52-week high", "Value": metrics.get("fifty_two_week_high")},
        {"Indicator": "52-week low", "Value": metrics.get("fifty_two_week_low")},
    ]
    st.dataframe(pd.DataFrame(tech_rows), width="stretch", hide_index=True)
    st.write(report["market_analysis"]["analysis"])

    st.markdown("## 3. Fundamentals and valuation context")
    fundamentals = report["fundamental_analysis"]["metrics"]
    fundamental_rows = [{"Metric": k.replace("_", " ").title(), "Value": (f"{v:.2%}" if isinstance(v, (float, np.floating)) and k == "profit_margin" else (f"{v:,.2f}" if isinstance(v, (int, float, np.integer, np.floating)) else "N/A"))} for k, v in fundamentals.items()]
    st.dataframe(pd.DataFrame(fundamental_rows), width="stretch", hide_index=True)
    st.write("These figures are company-level snapshots from the data provider. The app does not currently calculate peer-group or historical valuation percentiles, so ratios should not be labelled cheap or expensive without additional comparison data.")
    cap = metrics.get("market_cap")
    implied = evaluation.get("implied_market_cap")
    if cap is not None:
        st.write(f"Reported market capitalisation: **{cap:,.0f}**")
        if implied is not None:
            st.write(f"Cross-check (latest price × shares outstanding): **{implied:,.0f}**")
            if evaluation.get("market_cap_consistent"):
                st.success("Market-cap cross-check is within 15%.")
            else:
                st.warning("Market-cap cross-check is unavailable or differs by more than 15%; verify source timestamps and share-count definitions.")
        else:
            st.info("Market-cap cross-check unavailable because shares outstanding were not provided.")

    st.markdown("## 4. News evidence and synthesis")
    if report["live_news_available"]:
        st.success("Live headlines retrieved. Headlines are evidence to inspect, not proof of causal market impact.")
        titles = [a.get("article_title", "") for a in report["routed_articles"] if a.get("article_title")]
        if titles:
            synthesis = synthesize_news(result["report"].get("raw_articles", []))
            st.write(f"Retrieved **{len(titles)}** article(s). The list below is a source-level summary; the app does not claim a quantified sentiment trend unless an analysis model is configured.")
            st.markdown("### Rule-based headline themes")
            if synthesis["groups"]:
                for theme, theme_titles in synthesis["groups"].items():
                    with st.expander(f"{theme} · {len(theme_titles)} headline(s)"):
                        for headline in theme_titles:
                            st.write(f"- {headline}")
            st.caption(synthesis["method"] + " Theme labels are approximate and can overlap conceptually.")
            for article in report["routed_articles"]:
                with st.expander(f"{article['specialist']} · {article['article_title']}"):
                    st.caption(f"Category: {article['category']}")
                    st.write(article["analysis"])
    else:
        st.warning("No live headlines were retrieved. Any sample articles are illustrative only and must not be treated as current news.")
        for article in report["routed_articles"]:
            with st.expander(f"{article['specialist']} · {article['article_title']}"):
                st.caption("Illustrative sample — not live news")
                st.write(article["analysis"])

    st.markdown("## 5. Evaluator–optimizer review")
    st.write("**Evidence coverage score — not model accuracy:** the score reflects whether key evidence and quality checks passed, not whether a stock prediction is correct.")
    st.progress(evaluation["score"] / 100)
    st.write(f"**{evaluation['passed']} of {evaluation['total']} checks passed**")
    st.dataframe(pd.DataFrame([{"Check": k, "Passed": bool(v)} for k, v in evaluation["checks"].items()]), width="stretch", hide_index=True)
    for item in evaluation["feedback"]:
        st.write(f"- {item}")

    st.markdown("## 6. Research memory")
    memory = load_memory()
    if memory:
        st.dataframe(pd.DataFrame(memory[::-1]), width="stretch", hide_index=True)
    else:
        st.caption("No saved research runs yet.")
    st.caption("Memory is stored in a local JSON file when the host filesystem permits it; on Streamlit Community Cloud it may not persist across restarts.")

    st.markdown("## 7. Limitations")
    st.markdown("- Market data and fundamentals may be delayed, incomplete, or rate-limited.\n- News retrieval and coverage vary by ticker and source.\n- No peer valuation comparison or validated predictive backtest is included.\n- Historical returns and technical indicators do not guarantee future results.\n- This educational application is not personalized investment advice.")
    st.download_button("Download full research report (JSON)", data=json.dumps(report, indent=2, default=str), file_name=f"{report['ticker']}_research_report.json", mime="application/json")
else:
    st.markdown("Enter a stock ticker in the sidebar and click **Run research** to generate the full report.")
