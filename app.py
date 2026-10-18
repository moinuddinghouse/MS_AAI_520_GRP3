from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Multi-Agent Financial Research",
    page_icon="📈",
    layout="wide",
)

MEMORY_PATH = Path(__file__).with_name("investment_agent_memory.json")


# ----------------------------
# Agent and workflow components
# ----------------------------
class MarketSpecialistAgent:
    def analyze(self, history: pd.DataFrame) -> dict:
        if history is None or history.empty or "Close" not in history:
            return {"error": "No market price history was available."}

        close = history["Close"].dropna()
        if close.empty:
            return {"error": "No valid closing prices were available."}

        returns = close.pct_change().dropna()
        latest = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        volatility = float(returns.std() * np.sqrt(252) * 100) if not returns.empty else None

        return {
            "start_price": float(close.iloc[0]),
            "latest_price": latest,
            "period_return_pct": float((latest / float(close.iloc[0]) - 1) * 100),
            "annualized_volatility_pct": volatility,
            "ma20": ma20,
            "ma50": ma50,
            "max_price": float(close.max()),
            "min_price": float(close.min()),
            "trend_signal": "Bullish" if ma20 is not None and latest > ma20 else (
                "Bearish" if ma20 is not None else "Insufficient data"
            ),
        }


class EarningsSpecialistAgent:
    """Retrieve company fundamentals from Yahoo Finance, with FMP fallback."""

    FUNDAMENTAL_KEYS = [
        "longName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "profitMargins",
        "returnOnEquity",
        "debtToEquity",
        "revenueGrowth",
    ]

    @staticmethod
    def _fmp_api_key() -> str | None:
        """Read the FMP key from Streamlit Secrets or an environment variable."""
        key = os.getenv("FMP_API_KEY", "").strip()
        try:
            key = (st.secrets.get("FMP_API_KEY", "") or key).strip()
        except Exception:
            # Streamlit Secrets may not be configured locally or in the cloud.
            pass
        return key or None

    @staticmethod
    def _fmp_get(endpoint: str, ticker: str, api_key: str) -> tuple[dict | None, str | None]:
        """Call one FMP stable endpoint without exposing the API key in errors."""
        url = f"https://financialmodelingprep.com/stable/{endpoint}"
        try:
            response = requests.get(
                url,
                params={"symbol": ticker, "apikey": api_key},
                timeout=12,
            )
            if response.status_code != 200:
                return None, f"FMP {endpoint} returned HTTP {response.status_code}."
            payload = response.json()
            if isinstance(payload, list):
                payload = payload[0] if payload and isinstance(payload[0], dict) else {}
            if not isinstance(payload, dict) or not payload:
                return None, f"FMP {endpoint} returned no usable data."
            if payload.get("Error Message") or payload.get("Error"):
                return None, f"FMP {endpoint} returned an API error. Check key, plan, and endpoint access."
            return payload, None
        except requests.RequestException as exc:
            return None, f"FMP {endpoint} request failed: {type(exc).__name__}."
        except (ValueError, TypeError):
            return None, f"FMP {endpoint} returned an unreadable response."

    def _from_fmp(self, ticker: str) -> dict:
        result = {key: None for key in self.FUNDAMENTAL_KEYS}
        result["_data_source"] = "Financial Modeling Prep"
        result["_available_info_keys"] = []
        api_key = self._fmp_api_key()
        if not api_key:
            result["_retrieval_error"] = (
                "Yahoo Finance did not return the requested fundamentals. "
                "To enable the fallback, add FMP_API_KEY to Streamlit Secrets. "
                "Do not place the key in app.py or commit it to GitHub."
            )
            return result

        errors = []
        profile, error = self._fmp_get("profile", ticker, api_key)
        if error:
            errors.append(error)
        ratios, error = self._fmp_get("ratios-ttm", ticker, api_key)
        if error:
            errors.append(error)
        growth, error = self._fmp_get("income-statement-growth", ticker, api_key)
        if error:
            errors.append(error)

        if profile:
            result["longName"] = profile.get("companyName") or profile.get("name")
            result["sector"] = profile.get("sector")
            result["industry"] = profile.get("industry")
            result["marketCap"] = self._as_number(profile.get("marketCap"))
            result["_available_info_keys"].extend(f"profile.{key}" for key in profile.keys())

        if ratios:
            result["trailingPE"] = self._as_number(
                ratios.get("priceToEarningsRatioTTM")
                or ratios.get("priceEarningsRatioTTM")
            )
            result["profitMargins"] = self._as_number(
                ratios.get("netProfitMarginTTM")
                or ratios.get("netProfitMargin")
            )
            result["returnOnEquity"] = self._as_number(
                ratios.get("returnOnEquityTTM")
                or ratios.get("returnOnEquity")
            )
            result["debtToEquity"] = self._as_number(
                ratios.get("debtEquityRatioTTM")
                or ratios.get("debtToEquityRatioTTM")
            )
            result["_available_info_keys"].extend(f"ratios.{key}" for key in ratios.keys())

        if growth:
            result["revenueGrowth"] = self._as_number(
                growth.get("growthRevenue")
                or growth.get("revenueGrowth")
            )
            result["_available_info_keys"].extend(f"growth.{key}" for key in growth.keys())

        # FMP's endpoints above do not consistently provide forward P/E. Leave it
        # missing rather than estimating or fabricating a value.
        found = any(result.get(key) is not None for key in self.FUNDAMENTAL_KEYS if key != "longName")
        if found:
            result["_retrieval_error"] = None
            if errors:
                result["_partial_data_note"] = "Some FMP fields/endpoints were unavailable: " + " ".join(errors)
        else:
            result["_retrieval_error"] = (
                "Yahoo Finance returned no usable fundamentals and FMP did not return requested metrics. "
                + " ".join(errors or ["Check your FMP API key, subscription access, and ticker symbol."])
            )
        return result

    @staticmethod
    def _as_number(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def analyze(self, ticker: str) -> dict:
        result = {key: None for key in self.FUNDAMENTAL_KEYS}
        result.update({
            "_retrieval_error": None,
            "_available_info_keys": [],
            "_data_source": "Yahoo Finance",
        })
        fundamental_keys = [key for key in self.FUNDAMENTAL_KEYS if key != "longName"]
        yahoo_error = None

        try:
            stock = yf.Ticker(ticker)
            info = {}
            try:
                info = stock.get_info() or {}
            except Exception as exc:
                yahoo_error = f"Yahoo Finance get_info failed: {type(exc).__name__}: {exc}"

            if not isinstance(info, dict) or not any(info.get(key) is not None for key in fundamental_keys):
                try:
                    fallback_info = stock.info or {}
                    if isinstance(fallback_info, dict) and fallback_info:
                        info = fallback_info
                except Exception as exc:
                    fallback_error = f"Yahoo Finance .info failed: {type(exc).__name__}: {exc}"
                    yahoo_error = f"{yahoo_error}; {fallback_error}" if yahoo_error else fallback_error

            if isinstance(info, dict):
                result.update({key: info.get(key) for key in self.FUNDAMENTAL_KEYS})
                result["_available_info_keys"] = sorted(str(key) for key in info.keys())

            if any(result.get(key) is not None for key in fundamental_keys):
                result["_retrieval_error"] = None
                return result

            available = result["_available_info_keys"][:30]
            yahoo_error = yahoo_error or (
                "Yahoo Finance returned no requested fundamental metrics. "
                f"Available response keys (first 30): {available}"
            )
        except Exception as exc:
            yahoo_error = f"Yahoo Finance retrieval failed: {type(exc).__name__}: {exc}"

        # Use FMP only when Yahoo fails to provide actual requested fundamentals.
        fallback = self._from_fmp(ticker)
        fallback["_yahoo_error"] = yahoo_error
        if any(fallback.get(key) is not None for key in fundamental_keys):
            return fallback

        fallback["_retrieval_error"] = f"{yahoo_error} Fallback status: {fallback.get('_retrieval_error')}"
        return fallback


class NewsSpecialistAgent:
    def ingest(self, ticker: str) -> pd.DataFrame:
        try:
            raw_items = yf.Ticker(ticker).news or []
        except Exception:
            raw_items = []

        rows = []
        for item in raw_items[:20]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            provider = content.get("provider", {}) or {}
            canonical = content.get("canonicalUrl", {}) or {}
            rows.append({
                "title": content.get("title") or item.get("title") or "",
                "publisher": provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", ""),
                "summary": content.get("summary") or item.get("summary") or "",
                "link": canonical.get("url", "") if isinstance(canonical, dict) else item.get("link", ""),
                "published": content.get("pubDate") or item.get("providerPublishTime", ""),
            })
        return pd.DataFrame(rows, columns=["title", "publisher", "summary", "link", "published"])

    def preprocess(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df is None or news_df.empty:
            return pd.DataFrame(columns=["title", "publisher", "summary", "link", "published", "text", "clean_text"])
        df = news_df.copy()
        df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()
        df["clean_text"] = (
            df["text"].str.lower()
            .str.replace(r"http\S+", " ", regex=True)
            .str.replace(r"[^a-z0-9\s$%.-]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return df

    def classify(self, text: str) -> str:
        text = str(text).lower()
        groups = {
            "earnings": ["earnings", "revenue", "profit", "eps", "quarter", "guidance"],
            "market": ["stock", "shares", "price", "market", "rally", "selloff", "upgrade", "downgrade"],
            "news": ["acquisition", "launch", "partnership", "lawsuit", "regulator", "product", "ai"],
        }
        scores = {name: sum(term in text for term in terms) for name, terms in groups.items()}
        return max(scores, key=scores.get) if max(scores.values()) else "news"

    def extract(self, text: str) -> list[str]:
        text = str(text).lower()
        groups = {
            "earnings": ["earnings", "revenue", "profit", "eps", "guidance"],
            "market": ["stock", "shares", "price", "market", "rally", "selloff"],
            "growth": ["growth", "expansion", "launch", "demand"],
            "risk": ["risk", "lawsuit", "regulator", "competition", "decline"],
        }
        topics = [topic for topic, terms in groups.items() if any(term in text for term in terms)]
        return topics or ["general"]

    def summarize(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No recent news was available from the selected source."
        return "\n".join(
            f"- {row['title']} ({row['publisher'] or 'publisher unavailable'})"
            for _, row in df.head(5).iterrows()
        )


class RouterAgent:
    """Dispatch market, earnings, and news tasks to specialist agents."""
    def dispatch(self, task, payload, market_agent, earnings_agent, news_agent):
        if task == "market":
            return market_agent.analyze(payload)
        if task == "earnings":
            return earnings_agent.analyze(payload)
        if task == "news":
            articles = news_agent.preprocess(payload)
            if not articles.empty:
                articles["category"] = articles["text"].apply(news_agent.classify)
                articles["topics"] = articles["text"].apply(news_agent.extract)
                articles["routed_to"] = articles["category"].map({
                    "market": "Market Specialist",
                    "earnings": "Earnings Specialist",
                    "news": "News Specialist",
                }).fillna("News Specialist")
            return {"articles": articles, "summary": news_agent.summarize(articles)}
        raise ValueError(f"Unknown research task: {task}")


class EvaluatorAgent:
    def evaluate(self, result: dict) -> dict:
        text = result.get("integrated_analysis", "").lower()
        checks = {
            "has_market_data": bool(result.get("market")) and "error" not in result.get("market", {}),
            "has_financial_data": any(
                result.get("financials", {}).get(key) is not None
                for key in [
                    "sector", "industry", "marketCap", "trailingPE", "forwardPE",
                    "profitMargins", "returnOnEquity", "debtToEquity", "revenueGrowth",
                ]
            ),
            "has_news_evidence": bool(result.get("news_summary")) and "no recent news" not in result.get("news_summary", "").lower(),
            "has_risk_discussion": "risk" in text,
            "has_uncertainty": any(word in text for word in ["uncertain", "limitation", "may", "could"]),
        }
        score = sum(checks.values()) / len(checks) * 100
        feedback = []
        if not checks["has_market_data"]:
            feedback.append("Market data is missing or incomplete.")
        if not checks["has_financial_data"]:
            feedback.append("Company financial metrics are unavailable.")
        if not checks["has_news_evidence"]:
            feedback.append("Recent news evidence is unavailable.")
        if not checks["has_risk_discussion"]:
            feedback.append("Add a risk discussion.")
        if not checks["has_uncertainty"]:
            feedback.append("State uncertainty and limitations.")
        return {"score": score, "checks": checks, "feedback": feedback}


class OptimizerAgent:
    def optimize(self, result: dict, evaluation: dict) -> dict:
        text = result.get("integrated_analysis", "")
        additions = []
        feedback = evaluation.get("feedback", [])
        if "Add a risk discussion." in feedback:
            additions.append("Risk: Market prices can be volatile and company fundamentals can change.")
        if "State uncertainty and limitations." in feedback:
            additions.append("Limitation: Public data may be incomplete or delayed; this is not personalized investment advice.")
        if "Recent news evidence is unavailable." in feedback:
            additions.append("News limitation: No recent news was retrieved in this run; conclusions should not assume news evidence that was not available.")
        if additions:
            text += "\n\n" + "\n".join(additions)
        result["optimized_analysis"] = text
        return result


@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker: str, period: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, auto_adjust=True)


def load_memory() -> list[dict]:
    if MEMORY_PATH.exists():
        try:
            data = json.loads(MEMORY_PATH.read_text())
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_memory(memory: list[dict]) -> None:
    try:
        MEMORY_PATH.write_text(json.dumps(memory[-10:], indent=2))
    except OSError:
        # Some hosted environments have ephemeral or restricted storage.
        pass


def run_research(ticker: str, period: str) -> dict:
    market_agent = MarketSpecialistAgent()
    earnings_agent = EarningsSpecialistAgent()
    news_agent = NewsSpecialistAgent()
    router = RouterAgent()
    evaluator = EvaluatorAgent()
    optimizer = OptimizerAgent()

    history = get_history(ticker, period)
    market = router.dispatch("market", history, market_agent, earnings_agent, news_agent)
    financials = router.dispatch("earnings", ticker, market_agent, earnings_agent, news_agent)
    raw_news = news_agent.ingest(ticker)
    news_result = router.dispatch("news", raw_news, market_agent, earnings_agent, news_agent)
    articles, news_summary = news_result["articles"], news_result["summary"]

    previous = [item for item in load_memory() if item.get("ticker") == ticker]
    memory_context = f"Previous run note: {previous[-1]}" if previous else "No previous run for this ticker was found in memory."
    company = financials.get("longName") or ticker
    sector = financials.get("sector") or "Unavailable"
    volatility = market.get("annualized_volatility_pct")
    ret = market.get("period_return_pct")
    if "error" in market:
        market_text = f"Market data unavailable: {market['error']}"
    else:
        market_text = (
            f"- Latest available price: {market['latest_price']:.2f}\n"
            f"- Return over selected period: {ret:.2f}%\n"
            f"- Trend signal based on 20-day moving average: {market['trend_signal']}\n"
            f"- Annualized historical volatility: {volatility:.2f}%" if volatility is not None else
            f"- Latest available price: {market['latest_price']:.2f}\n- Return over selected period: {ret:.2f}%\n- Trend signal: {market['trend_signal']}\n- Annualized volatility: unavailable"
        )
    integrated = f"""Investment Research Summary — {company} ({ticker})

MARKET EVIDENCE
{market_text}

COMPANY EVIDENCE
- Sector: {sector}
- Industry: {financials.get('industry') or 'Unavailable'}
- Market capitalization: {financials.get('marketCap') or 'Unavailable'}
- Trailing P/E: {financials.get('trailingPE') or 'Unavailable'}
- Profit margin: {financials.get('profitMargins') or 'Unavailable'}

NEWS EVIDENCE
{news_summary}

INTERPRETATION
Market evidence should be considered together with company fundamentals and recent news. A positive trend does not by itself prove that a stock is undervalued, and a negative trend does not by itself prove that the business is deteriorating.

RISK AND LIMITATIONS
Key risks include market volatility, changing company fundamentals, incomplete information, and the possibility that news sentiment does not reflect long-term performance. Data may be missing, delayed, or inconsistent across sources. This is a research aid, not personalized investment advice.

MEMORY
{memory_context}"""

    result = {
        "ticker": ticker,
        "period": period,
        "history": history,
        "market": market,
        "financials": financials,
        "news": articles,
        "news_summary": news_summary,
        "integrated_analysis": integrated,
    }
    first_eval = evaluator.evaluate(result)
    result["evaluation"] = first_eval
    result = optimizer.optimize(result, first_eval)
    final_eval = evaluator.evaluate({**result, "integrated_analysis": result["optimized_analysis"]})
    result["final_evaluation"] = final_eval

    memory = load_memory()
    memory.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "trend_signal": market.get("trend_signal"),
        "period_return_pct": market.get("period_return_pct"),
        "evaluation_score": final_eval.get("score"),
        "lesson": "Compare future runs with previous market trend and evaluation score.",
    })
    save_memory(memory)
    return result


# ----------------------------
# Streamlit dashboard interface
# ----------------------------
st.title("📈 Multi-Agent Financial Analysis System")
st.caption("Educational investment research prototype using specialist agents, task routing, evaluation, and refinement.")

with st.sidebar:
    st.header("Research settings")
    ticker = st.text_input("Stock ticker", value="AAPL", max_chars=12).strip().upper()
    period = st.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    run_button = st.button("Run investment research", type="primary", use_container_width=True)
    st.divider()
    st.caption("Market/news data is retrieved at run time and may be missing or delayed.")
    st.caption("For the fundamentals fallback, configure FMP_API_KEY in Streamlit Secrets.")

if run_button:
    if not ticker or not ticker.replace(".", "").isalnum():
        st.error("Enter a valid ticker symbol, such as AAPL, MSFT, or TSLA.")
    else:
        with st.spinner(f"Researching {ticker}... running market, earnings, and news agents."):
            try:
                st.session_state["research_result"] = run_research(ticker, period)
            except Exception as exc:
                st.error(f"The research run failed: {exc}")

result = st.session_state.get("research_result")
if result:
    market = result["market"]
    financials = result["financials"]
    history = result["history"]

    st.subheader(f"Research results: {result['ticker']}")
    if "error" in market:
        st.warning(market["error"])
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest price", f"{market['latest_price']:,.2f}")
        c2.metric("Period return", f"{market['period_return_pct']:.2f}%")
        vol = market.get("annualized_volatility_pct")
        c3.metric("Annualized volatility", f"{vol:.2f}%" if vol is not None else "N/A")
        c4.metric("Trend signal", market.get("trend_signal", "N/A"))

        st.subheader("Market performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history.index, y=history["Close"], name="Close price", mode="lines"))
        if len(history) >= 20:
            fig.add_trace(go.Scatter(x=history.index, y=history["Close"].rolling(20).mean(), name="20-day moving average", mode="lines"))
        if len(history) >= 50:
            fig.add_trace(go.Scatter(x=history.index, y=history["Close"].rolling(50).mean(), name="50-day moving average", mode="lines"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Price", legend_title="Series", margin=dict(l=10, r=10, t=20, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Company fundamentals source: {financials.get('_data_source', 'Unknown')}")
    if financials.get("_retrieval_error"):
        st.warning(
            "Company fundamentals are unavailable from the configured data sources. "
            "This is a data-retrieval issue, not a formatting issue. "
            + financials["_retrieval_error"]
        )
    if financials.get("_partial_data_note"):
        st.info(financials["_partial_data_note"])
    if financials.get("_retrieval_error") or financials.get("_yahoo_error"):
        with st.expander("Company-data diagnostic details"):
            if financials.get("_yahoo_error"):
                st.write("Yahoo Finance result:")
                st.code(financials["_yahoo_error"])
            available_keys = financials.get("_available_info_keys", [])
            if available_keys:
                st.write("Fields returned by the selected fallback provider:")
                st.code(", ".join(available_keys[:100]))
            elif not financials.get("_yahoo_error"):
                st.write("No usable company-information fields were returned.")

    with st.expander("Company financial metrics", expanded=True):
        display_metrics = {
            "Company": financials.get("longName") or result["ticker"],
            "Sector": financials.get("sector") or "Unavailable",
            "Industry": financials.get("industry") or "Unavailable",
            "Market capitalization": (
                f"${financials['marketCap']:,.0f}"
                if isinstance(financials.get("marketCap"), (int, float)) else "Unavailable"
            ),
            "Trailing P/E": (
                round(financials["trailingPE"], 2)
                if isinstance(financials.get("trailingPE"), (int, float)) else "Unavailable"
            ),
            "Forward P/E": (
                round(financials["forwardPE"], 2)
                if isinstance(financials.get("forwardPE"), (int, float)) else "Unavailable"
            ),
            "Profit margin": (
                f"{financials['profitMargins']:.2%}"
                if isinstance(financials.get("profitMargins"), (int, float)) else "Unavailable"
            ),
            "Return on equity": (
                f"{financials['returnOnEquity']:.2%}"
                if isinstance(financials.get("returnOnEquity"), (int, float)) else "Unavailable"
            ),
            "Revenue growth": (
                f"{financials['revenueGrowth']:.2%}"
                if isinstance(financials.get("revenueGrowth"), (int, float)) else "Unavailable"
            ),
        }
        st.dataframe(pd.DataFrame(display_metrics.items(), columns=["Metric", "Value"]), hide_index=True, use_container_width=True)

    st.subheader("News pipeline and routing")
    articles = result["news"]
    if articles.empty:
        st.info("No recent news was returned by the selected source in this run. The dashboard does not create substitute news evidence.")
    else:
        cols = [c for c in ["title", "publisher", "category", "topics", "routed_to", "link"] if c in articles.columns]
        st.dataframe(articles[cols], hide_index=True, use_container_width=True)
    with st.expander("News summary"):
        st.text(result["news_summary"])

    st.subheader("Agent workflow")
    flow_cols = st.columns(5)
    for col, title, detail in zip(
        flow_cols,
        ["Market agent", "Earnings agent", "News agent", "Evaluator", "Optimizer"],
        ["Prices and trends", "Company metrics", "Classify and extract", "Check quality", "Refine report"],
    ):
        col.markdown(f"**{title}**")
        col.caption(detail)

    eval_result = result["final_evaluation"]
    st.subheader("Final evaluation")
    st.metric("Rule-based quality score", f"{eval_result['score']:.0f}/100")
    st.caption("This is a simple checklist score, not a measure of investment performance or a guarantee of correctness.")
    st.dataframe(pd.DataFrame([{"Check": k.replace("_", " ").title(), "Passed": bool(v)} for k, v in eval_result["checks"].items()]), hide_index=True, use_container_width=True)

    st.subheader("Optimized research summary")
    st.text_area("Final analysis", result["optimized_analysis"], height=360)
    st.download_button(
        "Download research summary",
        data=result["optimized_analysis"],
        file_name=f"{result['ticker']}_investment_research.txt",
        mime="text/plain",
    )

    with st.expander("Persistent memory (recent runs)"):
        mem = load_memory()
        if mem:
            st.dataframe(pd.DataFrame(mem).tail(10), hide_index=True, use_container_width=True)
        else:
            st.info("No memory records are available yet.")
else:
    st.info("Choose a ticker in the sidebar and click **Run investment research** to start.")

st.divider()
st.caption("Disclaimer: Educational research only, not personalized financial advice. External data may be missing, delayed, or incomplete.")