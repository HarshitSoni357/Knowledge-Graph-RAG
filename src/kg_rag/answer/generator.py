import json
import logging

import ollama

from kg_rag.answer.schema import GeneratedAnswer
from kg_rag.config import settings
from kg_rag.retrieval.models import RetrievalResult


logger = logging.getLogger(__name__)


ANSWER_SYSTEM_PROMPT = """
You are the answer generation component of a Knowledge Graph RAG system.

Your job is to answer the user's question using ONLY the retrieved context
provided to you.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not infer facts that are not supported by the retrieved context.
4. Every factual claim in the answer must have at least one citation.
5. Every citation must use a chunk_id that appears in the retrieved context.
6. For GRAPH retrieval, source_chunk_id is the citation chunk_id.
7. For multi-hop GRAPH retrieval, use the evidence from all required hops.
8. Keep the answer concise and directly answer the question.
9. If the retrieved context is insufficient, say exactly:

"Not available in corpus."

10. Do not mention the retrieval system, prompts, models, or internal
implementation details in the answer.
11. Return only the structured GeneratedAnswer.

CITATION RULES:

Every factual claim must have a citation.

The citation chunk_id must be copied EXACTLY from the
VALID CITATION CHUNK IDS supplied in the user context.

Never generate a new chunk_id.
Never modify a chunk_id.
Never hash a chunk_id.
Never truncate a chunk_id.
Never use a chunk_id from memory or from outside the retrieved context.

For multi-hop graph questions, cite the evidence chunk for each
required relationship.

The "claim" field of each citation should briefly describe the
factual claim supported by that citation.
"""


class AnswerGenerator:

    def __init__(
        self,
        model: str | None = None,
    ):
        self.model = model or settings.llm_model

    def generate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> GeneratedAnswer:

        context, valid_chunk_ids = self._build_context(
            retrieval
        )

        allowed_ids = "\n".join(
            f"- {chunk_id}"
            for chunk_id in valid_chunk_ids
        )

        if not allowed_ids:
            allowed_ids = "- NONE"

        user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

VALID CITATION CHUNK IDS:
{allowed_ids}

Citation rules:

- You MUST cite factual claims.
- A citation chunk_id MUST be copied EXACTLY from the
  VALID CITATION CHUNK IDS list above.
- NEVER invent, modify, shorten, hash, truncate, or guess a chunk_id.
- Do not use any chunk_id that is not in that list.
- Each citation must support the claim it is attached to.
- For a multi-hop answer, cite the evidence for each required hop.
- If the context is insufficient, answer exactly:
  "Not available in corpus."

Return only the structured GeneratedAnswer.
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": ANSWER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=GeneratedAnswer.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        raw_content = response["message"]["content"]

        try:
            payload = json.loads(raw_content)
            answer = GeneratedAnswer.model_validate(payload)

        except (json.JSONDecodeError, ValueError) as exc:

            logger.error(
                "answer_generation_invalid_response "
                "question=%s response=%s",
                question,
                raw_content,
            )

            raise ValueError(
                f"Invalid answer response from LLM: {raw_content}"
            ) from exc

        logger.info(
            "answer_generated question=%s citations=%d",
            question,
            len(answer.citations),
        )

        return answer

    def _build_context(
        self,
        retrieval: RetrievalResult,
    ) -> tuple[str, list[str]]:

        sections: list[str] = []
        valid_chunk_ids: list[str] = []

        # --------------------------------------------------------------
        # Vector context
        # --------------------------------------------------------------

        for index, chunk in enumerate(
            retrieval.chunks,
            start=1,
        ):

            valid_chunk_ids.append(chunk.chunk_id)

            sections.append(
                "\n".join(
                    [
                        f"[VECTOR CHUNK {index}]",
                        f"chunk_id: {chunk.chunk_id}",
                        f"document_id: {chunk.document_id}",
                        f"section: {chunk.section_path or 'N/A'}",
                        f"text: {chunk.text}",
                    ]
                )
            )

        # --------------------------------------------------------------
        # Graph context
        # --------------------------------------------------------------

        for index, fact in enumerate(
            retrieval.facts,
            start=1,
        ):

            if fact.source_chunk_id:
                valid_chunk_ids.append(
                    fact.source_chunk_id
                )

            # ----------------------------------------------------------
            # Aggregation fact
            # ----------------------------------------------------------

            if fact.relationship_type == "COUNT":

                sections.append(
                    "\n".join(
                        [
                            f"[GRAPH FACT {index}]",
                            (
                                f"chunk_id: "
                                f"{fact.source_chunk_id or 'N/A'}"
                            ),
                            "relationship: COUNT",
                            f"evidence: {fact.evidence}",
                        ]
                    )
                )

                continue

            # ----------------------------------------------------------
            # Standard graph fact
            # ----------------------------------------------------------

            sections.append(
                "\n".join(
                    [
                        f"[GRAPH FACT {index}]",
                        f"chunk_id: {fact.source_chunk_id}",
                        f"source: {fact.source_entity}",
                        f"relationship: {fact.relationship_type}",
                        f"target: {fact.target_entity}",
                        f"evidence: {fact.evidence}",
                    ]
                )
            )

        if not sections:
            return (
                "NO RETRIEVED CONTEXT",
                [],
            )

        # Remove duplicate chunk IDs while preserving order.
        valid_chunk_ids = list(
            dict.fromkeys(valid_chunk_ids)
        )

        return (
            "\n\n".join(sections),
            valid_chunk_ids,
        )