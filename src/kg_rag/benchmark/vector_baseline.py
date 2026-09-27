import logging

from kg_rag.answer.service import AnswerService
from kg_rag.retrieval.models import RetrievedChunk, RetrievalResult
from kg_rag.vector.embedder import OllamaEmbedder
from kg_rag.vector.pgvector_store import PgVectorStore


logger = logging.getLogger(__name__)


class VectorOnlyBaseline:
    """
    Vector-only RAG baseline.

    This intentionally bypasses:
        QueryRouter
        Neo4j
        GraphRetriever

    Pipeline:

        Question
            ↓
        Embedding
            ↓
        pgvector
            ↓
        Top-K chunks
            ↓
        AnswerGenerator
            ↓
        CitationValidator
            ↓
        Answer
    """

    def __init__(
        self,
        embedder: OllamaEmbedder | None = None,
        vector_store: PgVectorStore | None = None,
        answer_service: AnswerService | None = None,
    ):
        self.embedder = (
            embedder
            or OllamaEmbedder()
        )

        self.vector_store = (
            vector_store
            or PgVectorStore()
        )

        self.answer_service = (
            answer_service
            or AnswerService()
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Perform vector-only retrieval.

        The router is deliberately not used.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question must not be empty."
            )

        question = question.strip()

        query_embedding = self.embedder.embed(
            question
        )

        rows = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        chunks: list[RetrievedChunk] = []

        for row in rows:

            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    text=row["text"],
                    source_type="VECTOR",
                    score=row.get("similarity"),
                    section_path=row.get(
                        "section_path"
                    ),
                    entity_ids=row.get(
                        "entity_ids"
                    ) or [],
                    metadata={
                        "document_date": row.get(
                            "document_date"
                        ),
                        "similarity": row.get(
                            "similarity"
                        ),
                    },
                )
            )

        logger.info(
            "vector_baseline_retrieval "
            "question=%s chunks=%d",
            question,
            len(chunks),
        )

        return RetrievalResult(
            route="VECTOR_BASELINE",
            chunks=chunks,
            facts=[],
            query=question,
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
    ):
        """
        Complete vector-only RAG pipeline.
        """

        retrieval = self.retrieve(
            question=question,
            top_k=top_k,
        )

        return self.answer_service.answer(
            question=question,
            retrieval=retrieval,
        )