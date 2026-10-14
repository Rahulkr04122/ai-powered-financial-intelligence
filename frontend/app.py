from pathlib import Path
import sys

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.financial_agent import build_agent

DB_PATH = PROJECT_ROOT / "data" / "financial.duckdb"

st.set_page_config(
    page_title="Financial Intelligence AI",
    page_icon="📈",
    layout="wide",
)

# Build the agent only once
@st.cache_resource
def load_agent():
    return build_agent()


def get_available_tickers():
    if not DB_PATH.exists():
        return []

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker"
            ).fetchall()
        ]


def get_stock_history(ticker):
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(
            """
            SELECT date, close, volume
            FROM stock_prices
            WHERE ticker = ?
            ORDER BY date
            """,
            [ticker],
        ).df()


st.title("📈 AI-Powered Financial Intelligence")
st.caption(
    "Explore stock data, investigate unusual price movements, "
    "and ask questions about financial documents."
)

# Sidebar: financial chart
st.sidebar.header("Market Explorer")

tickers = get_available_tickers()

if tickers:
    selected_ticker = st.sidebar.selectbox(
        "Select a stock",
        options=tickers,
    )

    history = get_stock_history(selected_ticker)

    if not history.empty:
        latest_price = history["close"].iloc[-1]
        st.sidebar.metric(
            label=f"{selected_ticker} latest stored close",
            value=f"{latest_price:.2f}",
        )

        st.subheader(f"{selected_ticker} — Closing Price History")

        figure = px.line(
            history,
            x="date",
            y="close",
            title=f"{selected_ticker} closing prices",
            labels={
                "date": "Date",
                "close": "Closing Price",
            },
        )
        st.plotly_chart(figure, use_container_width=True)

        with st.expander("View recent stock records"):
            st.dataframe(
                history.tail(10).sort_values("date", ascending=False),
                use_container_width=True,
            )
else:
    st.sidebar.warning(
        "No stock data found. Run the ingestion and database-loading scripts first."
    )

st.divider()

# Chat interface
st.subheader("💬 Ask Your Financial Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask about stock prices, anomalies, or your financial documents..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            try:
                agent = load_agent()

                result = agent.invoke(
                    {
                        "question": question,
                        "route": "",
                        "result": "",
                    }
                )

                answer = result["result"]
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Something went wrong: {error}")