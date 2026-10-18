# Multi-Agent Financial Analysis System

A team project for the MS Applied AI program demonstrating an agentic
investment-research workflow. The system coordinates specialist agents
to analyze market data, company financial information, and financial
news, then evaluates and refines an integrated research summary.

## Team Members

-   Moinuddin K
-   Ravisha Krishnamurthy
-   Priya

## Project Objectives

The project demonstrates the agent functions and workflow patterns
required by the assignment.

### Agent Functions

-   **Planning:** `ResearchManagerAgent` creates a research plan for a
    stock ticker.
-   **Tool use:** Retrieves market history, company information, and
    news through Yahoo Finance using `yfinance`.
-   **Self-reflection:** `EvaluatorAgent` checks the integrated analysis
    for evidence, risk discussion, and uncertainty.
-   **Learning across runs:** A persistent JSON memory file stores notes
    from previous runs for later analyses.

### Workflow Patterns

1.  **Prompt chaining:** News is ingested, preprocessed, classified,
    enriched with topics, and summarized.
2.  **Routing:** `RouterAgent` dispatches market, earnings, and news
    tasks to the appropriate specialist agents.
3.  **Evaluator--Optimizer:** The Evaluator provides feedback, the
    Optimizer refines the analysis, and the result is evaluated again.

## Main Components

-   `ResearchManagerAgent` --- creates the research plan.
-   `MarketSpecialistAgent` --- analyzes historical prices, returns,
    volatility, and moving averages.
-   `EarningsSpecialistAgent` --- retrieves available company-level
    financial metrics.
-   `NewsSpecialistAgent` --- ingests and processes financial news.
-   `RouterAgent` --- dispatches tasks to the relevant specialist.
-   `EvaluatorAgent` --- assesses analysis quality and identifies
    missing elements.
-   `OptimizerAgent` --- improves the analysis using evaluator feedback.
-   Persistent memory --- stores brief notes across runs.

## Technology Stack

-   Python and Jupyter Notebook
-   `yfinance`
-   `pandas` and `numpy`
-   `matplotlib`
-   Gemini API integration, when configured
-   JSON for persistent memory

Install the packages imported by the notebook if they are not already
installed.

## Getting Started

1.  Clone or download this repository.
2.  Open the project notebook in Jupyter Notebook, JupyterLab, or VS
    Code.
3.  Install the Python packages imported by the notebook.
4.  If using Gemini, configure your API key locally using the
    environment-variable or `.env` approach described in the notebook.
5.  Run notebook cells from top to bottom.
6.  Adjust the stock ticker and analysis period in the notebook
    configuration as needed.

**Security:** Never commit API keys or `.env` files to GitHub.

External data and news availability can vary between runs. When data is
unavailable, the system should report the limitation rather than
inventing evidence. The notebook includes a deterministic fallback for
cases where optional LLM functionality or external data is unavailable.

## Outputs

Depending on data availability, the notebook produces:

-   A research plan for the selected ticker
-   Market analysis and visualizations
-   Available company-level financial metrics
-   Processed and categorized news, topics, and a summary
-   An integrated investment research summary
-   Initial and final evaluation results
-   A persistent memory note for future runs

This is an educational research prototype, not financial advice. Its
results depend on the availability, quality, and timeliness of external
data and should not be used as the sole basis for investment decisions.

## GitHub Repository

https://github.com/moinuddinghouse/MS_AAI_520_GRP3

## AI-Assistance Disclosure

AI-assisted tools, including ChatGPT and Gemini, were used during
development for tasks such as discussing agent architecture, explaining
implementation approaches, troubleshooting code, and improving
documentation. The team should ensure this statement accurately reflects
its actual use. Team members remain responsible for understanding,
testing, and validating the submitted implementation.

## Academic Integrity and Team Contribution

Ensure that the submitted notebook, documentation, and peer evaluations
accurately reflect team contributions and comply with course
requirements.
