"""Carga y preprocesamiento de guías clínicas ESC (PDF)."""
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

GUIDELINES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "guidelines"


def load_guideline(pdf_path: Path | str) -> Document:
    """Carga un PDF y devuelve un único Document con todo el texto."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    full_text = "\n\n".join(pages_text)
    return Document(
        page_content=full_text,
        metadata={"source": pdf_path.name, "pages": len(doc)},
    )


def load_all_guidelines(directory: Path | str | None = None) -> list[Document]:
    """Carga todos los PDFs del directorio de guías."""
    directory = Path(directory) if directory else GUIDELINES_DIR
    if not directory.exists():
        raise FileNotFoundError(
            f"Directorio de guías no encontrado: {directory}\n"
            f"Descarga las guías ESC y colócalas en {directory}"
        )
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No se encontraron PDFs en {directory}")
    return [load_guideline(pdf) for pdf in pdfs]
