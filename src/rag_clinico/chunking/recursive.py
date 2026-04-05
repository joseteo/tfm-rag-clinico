"""Chunking recursivo con separadores jerárquicos (C2)."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def recursive_chunker(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Document]:
    """Divide documentos respetando fronteras naturales del texto.

    Usa una jerarquía de separadores: doble salto de línea > salto simple >
    punto > espacio. Solo recurre a separadores menores cuando el fragmento
    supera chunk_size.

    Args:
        documents: Lista de documentos LangChain.
        chunk_size: Tamaño máximo de caracteres por fragmento.
        chunk_overlap: Solapamiento entre fragmentos consecutivos.
        separators: Jerarquía personalizada de separadores.

    Returns:
        Lista de fragmentos como documentos LangChain.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)
