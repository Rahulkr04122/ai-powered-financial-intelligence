
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.rag.retrieve_documents import retrieve_documents


llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
You are a financial document analysis assistant.

Answer the question using only the supplied document context.

Rules:
- Do not invent financial facts.
- If the context does not contain the answer, say so.
- Distinguish facts from interpretation.
- This is informational analysis, not investment advice.

Document context:
{context}

Question:
{question}

Answer:
""")


def answer_question(question: str) -> str:
    documents = retrieve_documents(question, k=4)

    if not documents:
        return "I could not find relevant evidence in the indexed documents."

    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        context_parts.append(
            f"Source: {source}, Page: {page}\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
    })

    return response.content


def main():
    question = input("Ask a financial question: ")

    try:
        answer = answer_question(question)
        print("\nAI ANSWER:\n")
        print(answer)
    except Exception as error:
        print(f"Error generating answer: {error}")


if __name__ == "__main__":
    main()
