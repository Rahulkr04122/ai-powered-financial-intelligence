
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_vector_store():
    """Open the existing local financial document index."""

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. "
            "Run src/rag/ingest_documents.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    return Chroma(
        collection_name="financial_documents",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )


def retrieve_documents(question: str, k: int = 4):
    """Find the most relevant document chunks for a question."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        question,
        k=k,
    )

    return results


def main():
    question = input("Ask a question about your financial report: ")

    results = retrieve_documents(question)

    if not results:
        print("No relevant document chunks were found.")
        return

    for index, document in enumerate(results, start=1):
        print(f"\n{'=' * 20} RESULT {index} {'=' * 20}")

        print("Source:", document.metadata.get("source", "Unknown"))
        print("Page:", document.metadata.get("page", "Unknown"))
        print("\nContent:\n", document.page_content)


if __name__ == "__main__":
    main()
