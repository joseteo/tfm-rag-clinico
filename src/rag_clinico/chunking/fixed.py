"""Chunking de tamaño fijo (C1)."""
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


def fixed_size_chunker(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Divide documentos en fragmentos de tamaño fijo por tokens.

    Args:
        documents: Lista de documentos LangChain.
        chunk_size: Número de caracteres por fragmento.
        chunk_overlap: Solapamiento entre fragmentos consecutivos.

    Returns:
        Lista de fragmentos como documentos LangChain.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
