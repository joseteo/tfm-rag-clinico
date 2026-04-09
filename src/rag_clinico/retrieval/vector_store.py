"""Gestión del vector store (ChromaDB)."""
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

CHROMA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "chroma_db"


def build_vector_store(
    chunks: list[Document],
    embedding_model: Embeddings,
    collection_name: str = "esc_guidelines",
    persist_directory: Path | str | None = None,
) -> Chroma:
    """Construye un vector store a partir de fragmentos.

    Args:
        chunks: Fragmentos de documentos ya segmentados.
        embedding_model: Modelo de embedding a utilizar.
        collection_name: Nombre de la colección en ChromaDB.
        persist_directory: Directorio de persistencia.

    Returns:
        Instancia de Chroma con los fragmentos indexados.
    """
    persist_dir = str(persist_directory or CHROMA_DIR)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def load_vector_store(
    embedding_model: Embeddings,
    collection_name: str = "esc_guidelines",
    persist_directory: Path | str | None = None,
) -> Chroma:
    """Carga un vector store existente."""
    persist_dir = str(persist_directory or CHROMA_DIR)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )
