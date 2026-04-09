"""Abstracción de modelos de embedding para experimentación."""
from langchain_community.embeddings import HuggingFaceEmbeddings

# Registro de modelos evaluados en el TFM
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "generalista",
        "dimensions": 384,
    },
    "bge-base-en-v1.5": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "type": "generalista",
        "dimensions": 768,
    },
    "PubMedBERT": {
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "type": "biomedico",
        "dimensions": 768,
    },
    "BioLORD-2023": {
        "model_name": "FremyCompany/BioLORD-2023",
        "type": "biomedico",
        "dimensions": 768,
    },
}


def get_embedding_model(name: str) -> HuggingFaceEmbeddings:
    """Devuelve un modelo de embedding por nombre corto.

    Args:
        name: Clave del modelo en EMBEDDING_MODELS.

    Returns:
        Instancia de HuggingFaceEmbeddings lista para usar.
    """
    if name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Modelo '{name}' no reconocido. "
            f"Opciones: {list(EMBEDDING_MODELS.keys())}"
        )
    config = EMBEDDING_MODELS[name]
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=config["model_name"],
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
