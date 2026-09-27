from kg_rag.answer.schema import GeneratedAnswer
from kg_rag.retrieval.models import RetrievalResult


class CitationValidationError(ValueError):
    pass


class CitationValidator:
    """
    Validates that every factual answer is grounded in retrieved
    chunks/facts.

    Exception:
    If retrieval contains no evidence and the generator explicitly
    returns the corpus-insufficiency response, citations are not
    required because there is nothing valid to cite.
    """

    OUT_OF_SCOPE_ANSWER = "Not available in corpus."

    def validate(
        self,
        answer: GeneratedAnswer,
        retrieval: RetrievalResult,
    ) -> GeneratedAnswer:

        valid_chunk_ids = set()

        for chunk in retrieval.chunks:
            if chunk.chunk_id:
                valid_chunk_ids.add(chunk.chunk_id)

        for fact in retrieval.facts:
            if fact.source_chunk_id:
                valid_chunk_ids.add(fact.source_chunk_id)

        # --------------------------------------------------------------
        # Explicit out-of-scope / insufficient-corpus response
        # --------------------------------------------------------------

        if (
            not retrieval.chunks
            and not retrieval.facts
            and answer.answer.strip() == self.OUT_OF_SCOPE_ANSWER
        ):
            # There is no evidence to cite, and the model correctly
            # stated that the corpus cannot answer the question.
            return answer

        # --------------------------------------------------------------
        # Normal grounded answer must contain citations
        # --------------------------------------------------------------

        if not answer.citations:
            raise CitationValidationError(
                "Generated answer contains no citations."
            )

        # --------------------------------------------------------------
        # Every citation must resolve to retrieved evidence
        # --------------------------------------------------------------

        invalid_ids = [
            citation.chunk_id
            for citation in answer.citations
            if citation.chunk_id not in valid_chunk_ids
        ]

        if invalid_ids:
            raise CitationValidationError(
                "Invalid citation chunk_id(s): "
                + ", ".join(invalid_ids)
            )

        return answer