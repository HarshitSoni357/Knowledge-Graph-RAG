import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kg_rag.answer.service import AnswerService
from kg_rag.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Local Knowledge Graph RAG",
    description="Local Graph + Vector RAG for Enterprise Data",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to answer from the indexed corpus.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of vector chunks to retrieve.",
    )


class CitationResponse(BaseModel):
    chunk_id: str
    claim: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    citations: list[CitationResponse]


retrieval_service = RetrievalService()
answer_service = AnswerService()


@app.get("/")
def root():
    return {
        "service": "Local Knowledge Graph RAG",
        "status": "ok",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        retrieval = retrieval_service.retrieve(
            question=request.question,
            top_k=request.top_k,
        )

        answer = answer_service.answer(
            question=request.question,
            retrieval=retrieval,
        )

        return QueryResponse(
            answer=answer.answer,
            route=retrieval.route,
            citations=[
                CitationResponse(
                    chunk_id=citation.chunk_id,
                    claim=citation.claim,
                )
                for citation in answer.citations
            ],
        )

    except Exception as exc:
        logger.exception(
            "query_failed question=%s",
            request.question,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc