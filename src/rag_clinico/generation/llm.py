"""Configuración del LLM para generación (Ollama local)."""
from langchain_ollama import ChatOllama


def get_llm(
    model: str = "gemma3:12b",
    temperature: float = 0.0,
    base_url: str = "http://192.168.1.22:11434",
) -> ChatOllama:
    """Devuelve una instancia del LLM configurado.

    Args:
        model: Nombre del modelo en Ollama.
        temperature: Temperatura de generación (0 = determinista).
        base_url: URL del servidor Ollama.

    Returns:
        Instancia de ChatOllama.
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
    )
