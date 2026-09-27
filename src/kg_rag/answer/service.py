import logging

from kg_rag.answer.citation_validator import (
    CitationValidationError,
    CitationValidator,
)
from kg_rag.answer.generator import AnswerGenerator
from kg_rag.answer.schema import GeneratedAnswer
from kg_rag.retrieval.models import RetrievalResult


logger = logging.getLogger(__name__)


class AnswerService:

    def __init__(
        self,
        generator: AnswerGenerator | None = None,
        validator: CitationValidator | None = None,
    ):
        self.generator = generator or AnswerGenerator()
        self.validator = validator or CitationValidator()

    def answer(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> GeneratedAnswer:

        generated = self.generator.generate(
            question=question,
            retrieval=retrieval,
        )

        try:

            validated = self.validator.validate(
                answer=generated,
                retrieval=retrieval,
            )

            logger.info(
                "answer_validation status=VALID question=%s",
                question,
            )

            return validated

        except CitationValidationError as first_error:

            logger.warning(
                "answer_validation status=INVALID "
                "question=%s error=%s "
                "retrying=True",
                question,
                first_error,
            )

            # ----------------------------------------------------------
            # One regeneration attempt
            # ----------------------------------------------------------

            regenerated = self.generator.generate(
                question=question,
                retrieval=retrieval,
            )

            try:

                validated = self.validator.validate(
                    answer=regenerated,
                    retrieval=retrieval,
                )

                logger.info(
                    "answer_validation status=VALID_AFTER_RETRY "
                    "question=%s",
                    question,
                )

                return validated

            except CitationValidationError as second_error:

                logger.error(
                    "answer_validation status=FAILED "
                    "question=%s error=%s",
                    question,
                    second_error,
                )

                raise