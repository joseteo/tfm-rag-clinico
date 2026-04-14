"""Pipeline RAG end-to-end configurable para experimentación."""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .retrieval.vector_store import build_vector_store
from .generation.llm import get_llm

RAG_PROMPT_TEMPLATE = """Eres un asistente médico especializado. Responde la pregunta
basándote EXCLUSIVAMENTE en el contexto proporcionado de las guías clínicas ESC.
Si la información no está en el contexto, indica que no dispones de esa información.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""


class RAGPipeline:
    """Pipeline RAG configurable para experimentación."""

    def __init__(
        self,
        chunks: list[Document],
        embedding_model,
        llm=None,
        k: int = 4,
        collection_name: str = "experiment",
    ):
        self.embedding_model = embedding_model
        self.llm = llm or get_llm()
        self.k = k

        # Build vector store
        self.vector_store = build_vector_store(
            chunks=chunks,
            embedding_model=embedding_model,
            collection_name=collection_name,
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        # Build chain
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> str:
        """Ejecuta una consulta y devuelve la respuesta generada."""
        return self.chain.invoke(question)

    def query_with_sources(self, question: str) -> dict:
        """Ejecuta una consulta y devuelve respuesta + documentos fuente."""
        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return {
            "question": question,
            "answer": answer,
            "source_documents": retrieved_docs,
        }
