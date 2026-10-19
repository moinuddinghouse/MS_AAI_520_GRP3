import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from google import genai
except ImportError:
    genai = None

APP_TITLE = "Multi-Agent Financial Analysis System"
MEMORY_FILE = Path("investment_agent_memory.json")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")


def get_api_key():
    """Read Gemini key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GOOGLE_API_KEY", "")


def call_gemini(prompt):
    """Return (text, error); gracefully fall back if Gemini is not configured."""
    key = get_api_key()
    if not key or genai is None:
        return None, "Gemini is unavailable; using deterministic fallback."
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        result = getattr(response, "text", None)
        if result and result.strip():
            return result.strip(), None
        return None, "Gemini returned an empty response."
    except Exception as exc:
        return None, f"Gemini request failed: {type(exc).__name__}: {str(exc)[:180]}"


def load_memory():
    try:
        if MEMORY_FILE.exists():
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)][-10:]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_memory(record):
    """Persist a short run summary when the hosting filesystem permits it."""
    memory = load_memory()
    memory.append(record)
    try:
        MEMORY_FILE.write_text(json.dumps(memory[-10:], indent=2, default=str), encoding="utf-8")
        return True
    except OSError:
        return False


def get_market_data(ticker, period):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period, auto_adjust=True)
    if history.empty:
        raise ValueError(f"No market history returned for {ticker}.")
    close = history["Close"].dropna()
    returns = close.pct_change().dropna()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    result = {
        "start_price": float(close.iloc[0]),
        "latest_price": float(close.iloc[-1]),
        "period_return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "annualized_volatility_pct": float(returns.std() * np.sqrt(252) * 100) if len(returns) > 1 else None,
        "ma20": float(ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None,
        "ma50": float(ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None,
        "trend_signal": (
            "Bullish" if pd.notna(ma20.iloc[-1]) and close.iloc[-1] > ma20.iloc[-1]
            else "Bearish" if pd.notna(ma20.iloc[-1]) else "Insufficient data"
        ),
    }
    chart = pd.DataFrame({"Close": close, "MA20": ma20, "MA50": ma50})
    return history, result, stock, chart


def get_fundamentals(stock):
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    fields = {
        "Company": info.get("longName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "Market capitalization": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Profit margin": info.get("profitMargins"),
        "Return on equity": info.get("returnOnEquity"),
        "Debt to equity": info.get("debtToEquity"),
        "Revenue growth": info.get("revenueGrowth"),
    }
    return fields


def get_news(ticker, company_name, limit=15):
    """Retrieve genuine article metadata from Google News RSS."""
    query = quote_plus(f'("{company_name}" OR "{ticker}") when:7d')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0 FinancialResearchStudentProject/1.0"}, timeout=20
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        source_node = item.find("source")
        publisher = (source_node.text or "").strip() if source_node is not None else ""
        if title:
            rows.append({
                "title": title,
                "publisher": publisher or "Publisher unavailable",
                "published": published or "Date unavailable",
                "link": link,
                "summary": re.sub("<[^>]+>", " ", description).strip(),
            })
    return pd.DataFrame(rows, columns=["title", "publisher", "published", "link", "summary"])


def classify_article(text):
    """Deterministic routing fallback used when Gemini is unavailable."""
    text = str(text).lower()
    categories = {
        "earnings": ["earnings", "revenue", "profit", "eps", "quarter", "guidance", "margin", "results"],
        "market": ["stock", "shares", "price", "market", "rally", "selloff", "trading", "investor", "upgrade"],
        "regulatory_legal": ["lawsuit", "court", "patent", "antitrust", "regulator", "fine", "investigation", "verdict"],
        "product_strategy": ["launch", "partnership", "product", "iphone", "ipad", "macbook", "artificial intelligence", "chip"],
    }
    scores = {category: sum(term in text for term in terms) for category, terms in categories.items()}
    if not scores or max(scores.values()) == 0:
        return "general"
    return max(categories, key=lambda category: scores[category])


def run_news_prompt_chain(news_df, ticker):
    """Ingest/preprocess/classify/extract/summarize; Gemini stages are optional."""
    if news_df.empty:
        return news_df.copy(), "No news retrieved", "No recent news articles were retrieved from Google News RSS."
    df = news_df.copy()
    df["clean_text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    df["category"] = df["clean_text"].apply(classify_article)
    df["routed_to"] = df["category"].map({
        "earnings": "Earnings Specialist",
        "market": "Market Specialist",
        "regulatory_legal": "News Specialist",
        "product_strategy": "News Specialist",
        "general": "News Specialist",
    }).fillna("News Specialist")
    df["topics"] = df["clean_text"].apply(lambda x: ", ".join(
        term for term in ["earnings", "revenue", "launch", "lawsuit", "patent", "AI", "market"]
        if term.lower() in x
    ) or "general")

    sample = df.head(10)[["title", "publisher", "published", "category", "topics"]].to_dict("records")
    prompt = f"""
You are the news specialist in a financial research workflow for {ticker}.
Use only the supplied article metadata. Headlines may be incomplete; do not treat allegations as proven.
Summarize recurring themes, distinguish reported facts from interpretation, mention uncertainty,
and avoid giving investment advice. Cite article titles/publishers where possible.
Articles:
{json.dumps(sample, ensure_ascii=False, default=str)}
"""
    summary, error = call_gemini(prompt)
    if summary:
        return df, "Gemini prompt chain + deterministic routing", summary
    bullets = []
    for _, row in df.head(8).iterrows():
        bullets.append(f"- {row['title']} ({row['publisher']}; {row['published']})")
    fallback = "Gemini summarization was unavailable, so this is a headline-based summary only:\n" + "\n".join(bullets)
    if error:
        fallback += f"\n\nNote: {error}"
    return df, "Deterministic fallback; headlines only", fallback


def create_plan(ticker, user_request, memory):
    memory_hint = json.dumps(memory[-3:], ensure_ascii=False, default=str)[:2500]
    prompt = f"""
Create a concise JSON research plan for ticker {ticker} and request: {user_request}
Return valid JSON with keys rationale, prior_learning_applied, tasks.
tasks must be an ordered list with task values chosen from news, fundamentals, market,
and a short focus for each. For a comprehensive request include all three.
Prior memory (may be empty; do not treat old market values as current evidence):
{memory_hint}
"""
    text, error = call_gemini(prompt)
    if text:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
            plan = json.loads(cleaned)
            if isinstance(plan.get("tasks"), list) and plan["tasks"]:
                plan["planning_method"] = "Gemini LLM"
                return plan
        except (json.JSONDecodeError, AttributeError):
            pass
    return {
        "rationale": "Use a balanced research plan covering market performance, company fundamentals, and recent news.",
        "prior_learning_applied": "Prior records are context only and should not override current evidence.",
        "tasks": [
            {"task": "news", "focus": "Recent material events, source attribution, and risks"},
            {"task": "fundamentals", "focus": "Valuation, profitability, growth, and leverage where available"},
            {"task": "market", "focus": "Historical return, volatility, and moving averages"},
        ],
        "planning_method": "Deterministic fallback",
        "planning_diagnostic": error or "Gemini did not return a valid JSON plan.",
    }


def build_report(ticker, market, fundamentals, news_summary, articles, evaluation_feedback):
    article_lines = []
    if not articles.empty:
        for _, row in articles.head(8).iterrows():
            article_lines.append(
                f"- {row['title']} | {row['publisher']} | {row['published']} | {row['link']}"
            )
    else:
        article_lines = ["No article metadata was available for this run."]
    prompt = f"""
Write a structured educational financial research report for {ticker}.
Use only supplied evidence. Explicitly mark unavailable metrics. Do not invent events, numbers,
or sources. Separate reported facts from interpretation, include specific risks and limitations,
and state that this is not personalized investment advice.
Market evidence: {json.dumps(market, default=str)}
Fundamentals: {json.dumps(fundamentals, default=str)}
News summary: {news_summary}
Source register:
{chr(10).join(article_lines)}
Evaluator feedback to address: {json.dumps(evaluation_feedback, default=str)}
"""
    text, error = call_gemini(prompt)
    if text:
        return text, "Gemini synthesis", error
    company = fundamentals.get("Company") or ticker
    report = f"""# Investment Research Summary — {company} ({ticker})

## Market evidence
- Latest available price: {market.get('latest_price')}
- Return over selected period: {market.get('period_return_pct', float('nan')):.2f}%
- Trend signal (20-day moving average): {market.get('trend_signal')}
- Annualized historical volatility: {market.get('annualized_volatility_pct')}

## Company evidence
""" + "\n".join(f"- **{k}:** {v if v is not None else 'Unavailable from provider'}" for k, v in fundamentals.items())
    report += "\n\n## News evidence\n" + news_summary
    report += "\n\n## Source register\n" + "\n".join(article_lines)
    report += "\n\n## Interpretation and risks\nThese indicators describe historical observations, not future performance. Provider data may be incomplete or delayed; headlines alone do not establish the full facts of a story. Verify material claims with original sources."
    report += "\n\n## Responsible use\nThis is an educational research summary, not personalized investment advice."
    return report, "Deterministic report fallback", error


def evaluate_report(report, market, fundamentals, articles):
    checks = {
        "Market evidence included": market.get("latest_price") is not None,
        "Fundamentals represented": any(v is not None for v in fundamentals.values()),
        "News evidence or explicit absence": bool(len(articles)) or "No recent news" in report,
        "Risk/limitations language": any(word in report.lower() for word in ["risk", "limitation", "uncertain"]),
        "Responsible-use disclaimer": "investment advice" in report.lower(),
    }
    score = round(100 * sum(checks.values()) / len(checks), 1)
    return {"score": score, "checks": checks}


st.title("📈 Multi-Agent Financial Analysis System")
st.caption("Educational research prototype — not personalized investment advice.")

with st.sidebar:
    st.header("Research settings")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    period = st.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    user_request = st.text_area(
        "Research request",
        value="Prepare a balanced research report covering market performance, company fundamentals, and recent material news.",
        height=120,
    )
    run_button = st.button("Run research", type="primary", use_container_width=True)
    st.caption(f"Gemini model: {MODEL_NAME}")
    st.caption("Add GOOGLE_API_KEY in Streamlit Secrets to enable Gemini calls.")

if run_button:
    if not ticker or not re.fullmatch(r"[A-Z0-9.^=-]{1,15}", ticker):
        st.error("Enter a valid ticker symbol.")
        st.stop()

    with st.spinner(f"Researching {ticker}..."):
        try:
            prior_memory = load_memory()
            plan = create_plan(ticker, user_request, prior_memory)
            st.session_state["plan"] = plan

            history, market, stock, chart = get_market_data(ticker, period)
            fundamentals = get_fundamentals(stock)

            try:
                company_name = fundamentals.get("Company") or ticker
                articles = get_news(ticker, company_name)
                news_error = None
            except Exception as exc:
                articles = pd.DataFrame(columns=["title", "publisher", "published", "link", "summary"])
                news_error = f"{type(exc).__name__}: {str(exc)[:180]}"

            articles, chain_status, news_summary = run_news_prompt_chain(articles, ticker)
            feedback = []
            report, report_method, report_error = build_report(
                ticker, market, fundamentals, news_summary, articles, feedback
            )
            evaluation = evaluate_report(report, market, fundamentals, articles)

            # Evaluator–optimizer pass: request a revised report using explicit checklist feedback.
            critique = [name for name, passed in evaluation["checks"].items() if not passed]
            optimized_report, optimizer_method, optimizer_error = build_report(
                ticker, market, fundamentals, news_summary, articles, critique
            )
            final_evaluation = evaluate_report(optimized_report, market, fundamentals, articles)

            st.session_state["results"] = {
                "ticker": ticker, "period": period, "history": history, "chart": chart,
                "market": market, "fundamentals": fundamentals, "articles": articles,
                "news_summary": news_summary, "chain_status": chain_status,
                "plan": plan, "report": optimized_report, "report_method": optimizer_method,
                "evaluation": final_evaluation, "initial_evaluation": evaluation,
                "critique": critique, "news_error": news_error,
                "report_error": report_error, "optimizer_error": optimizer_error,
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            saved = save_memory({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "period_return_pct": market.get("period_return_pct"),
                "trend_signal": market.get("trend_signal"),
                "evaluation_score": final_evaluation["score"],
                "lesson": "; ".join(critique) or "Verify sources and distinguish evidence from interpretation.",
                "llm_enabled": bool(get_api_key() and genai is not None),
            })
            st.session_state["memory_saved"] = saved
        except Exception as exc:
            st.error(f"Research failed: {type(exc).__name__}: {str(exc)[:300]}")
            st.stop()

if "results" not in st.session_state:
    st.info("Choose a ticker and click **Run research** to generate a report.")
    st.markdown(
        "**Workflow:** Research Manager → Market / Earnings / News specialists → "
        "News processing and routing → Synthesis → Evaluation and refinement → Memory."
    )
    st.stop()

r = st.session_state["results"]
st.subheader(f"{r['ticker']} — Research results")
st.caption(f"Period: {r['period']} · Run timestamp (UTC): {r['run_timestamp']}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest price", f"{r['market']['latest_price']:.2f}")
m2.metric("Period return", f"{r['market']['period_return_pct']:.2f}%")
m3.metric("20-day trend", r["market"]["trend_signal"])
vol = r["market"].get("annualized_volatility_pct")
m4.metric("Annualized volatility", f"{vol:.2f}%" if vol is not None else "N/A")

# Display the full report directly on the main page rather than hiding it in a tab.
st.markdown("---")
st.header("Full Financial Research Report")
st.markdown(r["report"])
st.download_button(
    "Download full report as Markdown",
    data=r["report"],
    file_name=f"{r['ticker']}_financial_research_report.md",
    mime="text/markdown",
    key="download_full_report",
)

st.markdown("### Evaluation summary")
eval_col1, eval_col2 = st.columns([1, 2])
with eval_col1:
    st.metric("Completeness score", f"{r['evaluation']['score']:.1f}%")
    st.caption("Completeness only—not factual accuracy or investment performance.")
with eval_col2:
    st.dataframe(pd.DataFrame([
        {"Check": k, "Passed": bool(v)} for k, v in r["evaluation"]["checks"].items()
    ]), use_container_width=True)
if r["critique"]:
    st.markdown("**Evaluator feedback**")
    for item in r["critique"]:
        st.write("- " + item)
else:
    st.success("All configured completeness checks passed.")
st.caption("Report generation: " + str(r["report_method"]))
for key in ["report_error", "optimizer_error"]:
    if r.get(key):
        st.caption(r[key])

st.markdown("---")
tab_overview, tab_plan, tab_news, tab_memory = st.tabs(
    ["Market & fundamentals", "Plan & routing", "News prompt chain", "Memory"]
)

with tab_overview:
    st.markdown("### Historical price and moving averages")
    st.line_chart(r["chart"])
    st.caption("Historical observations only; moving averages are not forecasts.")
    st.markdown("### Company fundamentals")
    st.dataframe(pd.DataFrame([r["fundamentals"]]), use_container_width=True)
    st.markdown("### Market metrics")
    st.dataframe(pd.DataFrame([r["market"]]), use_container_width=True)

with tab_plan:
    st.markdown("### Research plan")
    st.write("Planning method:", r["plan"].get("planning_method"))
    st.write(r["plan"].get("rationale", ""))
    st.write("Prior learning:", r["plan"].get("prior_learning_applied", ""))
    st.dataframe(pd.DataFrame(r["plan"].get("tasks", [])), use_container_width=True)
    st.markdown("### Specialist routing")
    if not r["articles"].empty:
        st.dataframe(r["articles"][["title", "category", "routed_to"]], use_container_width=True)
        st.bar_chart(r["articles"]["routed_to"].value_counts())
    else:
        st.info("No articles were retrieved; routing was not run.")

with tab_news:
    st.write("Prompt-chain status:", r["chain_status"])
    st.markdown("### News summary")
    st.write(r["news_summary"])
    if r["news_error"]:
        st.warning("News retrieval issue: " + r["news_error"])
    if not r["articles"].empty:
        st.markdown("### Retrieved articles and source metadata")
        st.dataframe(r["articles"][["title", "publisher", "published", "link", "category", "topics", "routed_to"]], use_container_width=True)
    else:
        st.info("No news articles were available for this run.")

with tab_memory:
    st.markdown("### Saved research memory")
    st.write("Current memory file:", str(MEMORY_FILE))
    st.write("Memory records currently readable:", len(load_memory()))
    st.write("Latest record saved:", "Yes" if st.session_state.get("memory_saved") else "No / hosting filesystem may be ephemeral.")
    st.caption("Streamlit Community Cloud storage may be ephemeral. Use external persistent storage if durable memory across redeployments is required.")
    memory = load_memory()
    if memory:
        st.dataframe(pd.DataFrame(memory[::-1]), use_container_width=True)
    else:
        st.info("No memory records have been saved yet.")
