"""Chunking semántico basado en similitud entre oraciones (C3)."""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def semantic_chunker(
    documents: list[Document],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    breakpoint_threshold: float = 0.3,
    max_chunk_size: int = 1500,
) -> list[Document]:
    """Divide documentos agrupando oraciones semánticamente similares.

    Calcula embeddings de oraciones consecutivas y corta donde la similitud
    coseno desciende por debajo del umbral, indicando cambio temático.

    Args:
        documents: Lista de documentos LangChain.
        embedding_model_name: Modelo para calcular similitud entre oraciones.
        breakpoint_threshold: Umbral de caída de similitud para crear un corte.
        max_chunk_size: Tamaño máximo de caracteres (safety cap).

    Returns:
        Lista de fragmentos como documentos LangChain.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold * 100,
        )
        chunks = splitter.split_documents(documents)
    except ImportError:
        # Fallback: si langchain_experimental no está instalado,
        # usar recursive como aproximación con separadores de oración
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", " "],
            chunk_size=max_chunk_size,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

    # Safety cap: fragmentos demasiado grandes se re-dividen
    capped = []
    cap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=100
    )
    for chunk in chunks:
        if len(chunk.page_content) > max_chunk_size:
            capped.extend(cap_splitter.split_documents([chunk]))
        else:
            capped.append(chunk)

    return capped
