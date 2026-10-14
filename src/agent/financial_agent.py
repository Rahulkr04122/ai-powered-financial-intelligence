
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from src.agent.financial_tools import (
    get_stock_summary,
    get_recent_anomalies,
)
from src.rag.answer_question import answer_question


class AgentState(TypedDict):
    question: str
    route: str
    result: str


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


def route_question(state: AgentState):
    """Choose a tool based on the user's question."""

    question = state["question"].lower()

    anomaly_words = [
        "anomaly", "anomalies", "unusual",
        "abnormal", "sudden movement",
    ]

    stock_words = [
        "stock price", "stock performance",
        "closing price", "trading sessions",
        "historical price", "price statistics",
        "average price", "maximum price", "minimum price",
    ]

    if any(word in question for word in anomaly_words):
        route = "anomalies"
    elif any(word in question for word in stock_words):
        route = "stock"
    else:
        route = "documents"

    print(f"\nSelected route: {route}")
    return {"route": route}


def stock_node(state: AgentState):
    """Get stock statistics from DuckDB."""

    question = state["question"]
    words = question.upper().replace("?", "").split()

    # Use AAPL by default unless a known ticker is supplied.
    known_tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"}
    ticker = next(
        (word.strip(".,:;()") for word in words
         if word.strip(".,:;()") in known_tickers),
        "AAPL",
    )

    result = get_stock_summary(ticker)
    return {"result": result}


def anomaly_node(state: AgentState):
    """Get existing anomaly detection results."""

    question = state["question"]
    words = question.upper().replace("?", "").split()

    known_tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"}
    ticker = next(
        (word.strip(".,:;()") for word in words
         if word.strip(".,:;()") in known_tickers),
        "AAPL",
    )

    result = get_recent_anomalies(ticker)
    return {"result": result}


def document_node(state: AgentState):
    """Answer using retrieved financial document evidence."""

    result = answer_question(state["question"])
    return {"result": result}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", route_question)
    graph.add_node("stock", stock_node)
    graph.add_node("anomalies", anomaly_node)
    graph.add_node("documents", document_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "stock": "stock",
            "anomalies": "anomalies",
            "documents": "documents",
        },
    )

    graph.add_edge("stock", END)
    graph.add_edge("anomalies", END)
    graph.add_edge("documents", END)

    return graph.compile()


def main():
    agent = build_agent()

    print("Financial Intelligence Agent")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            result = agent.invoke({
                "question": question,
                "route": "",
                "result": "",
            })

            print("\nAgent:", result["result"])

        except Exception as error:
            print(f"\nAgent error: {error}")


if __name__ == "__main__":
    main()
