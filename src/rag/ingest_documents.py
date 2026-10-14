
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs():
    """Load all PDF files from the documents directory."""

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}. "
            "Add at least one financial report PDF."
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} document pages.")
    return documents


def split_documents(documents):
    """Split pages into smaller searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks.")
    return chunks


def create_vector_database(chunks):
    """Embed text chunks and store them locally in ChromaDB."""

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    # Rebuild the index so rerunning this script
    # doesn't append duplicate chunks.
    if VECTOR_DB_DIR.exists():
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)

    VECTOR_DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name="financial_documents",
    )

    print(f"Vector database created at: {VECTOR_DB_DIR}")
    return vector_store


def main():
    documents = load_pdfs()
    chunks = split_documents(documents)
    create_vector_database(chunks)

    print("Financial document ingestion completed!")


if __name__ == "__main__":
    main()
