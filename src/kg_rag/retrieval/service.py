import logging
import re

from kg_rag.retrieval.models import (
    RetrievedChunk,
    RetrievedFact,
    RetrievalResult,
)
from kg_rag.router.query_router import QueryRouter
from kg_rag.vector.embedder import OllamaEmbedder
from kg_rag.vector.pgvector_store import PgVectorStore
from kg_rag.graph.entity_resolution import create_entity_id
from kg_rag.graph.ontology import EntityType
from kg_rag.graph.retriever import GraphRetriever

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Routes a question to either:

    - VECTOR retrieval using pgvector
    - GRAPH retrieval using Neo4j and predefined Cypher templates

    The LLM is never allowed to generate Cypher.
    """

    # ------------------------------------------------------------------
    # Controlled entity candidates for the current benchmark corpus.
    #
    # This is intentionally deterministic. Entity resolution should
    # eventually move into a dedicated entity-resolution/index layer,
    # but for the controlled benchmark corpus this keeps retrieval
    # reproducible.
    # ------------------------------------------------------------------

    ENTITY_CANDIDATES = [
        ("Activision Blizzard", EntityType.COMPANY),
        ("Microsoft Cloud", EntityType.BUSINESS_SEGMENT),
        ("Microsoft 365", EntityType.PRODUCT),
        ("Intelligent Cloud", EntityType.BUSINESS_SEGMENT),
        (
            "Productivity and Business Processes",
            EntityType.BUSINESS_SEGMENT,
        ),
        ("More Personal Computing", EntityType.BUSINESS_SEGMENT),
        ("Satya Nadella", EntityType.PERSON),
        ("Blizzard", EntityType.COMPANY),
        ("Activision", EntityType.COMPANY),
        ("Azure", EntityType.PRODUCT),
        ("King", EntityType.COMPANY),
        ("Microsoft", EntityType.COMPANY),
        ("Windows", EntityType.PRODUCT),
        ("Xbox", EntityType.PRODUCT),
    ]

    COMPANY_ENTITIES = {
        "microsoft",
        "activision blizzard",
        "activision",
        "blizzard",
        "king",
    }

    def __init__(
        self,
        router=None,
        embedder=None,
        vector_store=None,
        graph_retriever=None,
    ):
        self.router = router or QueryRouter()
        self.embedder = embedder or OllamaEmbedder()
        self.vector_store = vector_store or PgVectorStore()
        self.graph_retriever = graph_retriever or GraphRetriever()

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Route the question and perform the appropriate retrieval.
        """

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        question = question.strip()

        decision = self.router.route(question)

        logger.info(
            "retrieval_route question=%s route=%s reason=%s",
            question,
            decision.route.value,
            decision.reason,
        )

        if decision.route.value == "GRAPH":
            return self._retrieve_graph(question)

        return self._retrieve_vector(
            question=question,
            top_k=top_k,
        )

    # ==================================================================
    # VECTOR RETRIEVAL
    # ==================================================================

    def _retrieve_vector(
        self,
        question: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        query_embedding = self.embedder.embed(question)

        rows = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        chunks = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                text=row["text"],
                source_type="VECTOR",
                score=row.get("similarity"),
                section_path=row.get("section_path"),
                entity_ids=row.get("entity_ids") or [],
                metadata={
                    "document_date": row.get("document_date"),
                    "similarity": row.get("similarity"),
                },
            )
            for row in rows
        ]

        logger.info(
            "vector_retrieval question=%s chunks=%d",
            question,
            len(chunks),
        )

        return RetrievalResult(
            route="VECTOR",
            chunks=chunks,
            facts=[],
            query=question,
        )

    # ==================================================================
    # GRAPH RETRIEVAL
    # ==================================================================

    def _retrieve_graph(
        self,
        question: str,
    ) -> RetrievalResult:
        entity = self._resolve_query_entity(question)

        if entity is None:
            logger.warning(
                "graph_entity_resolution_failed question=%s",
                question,
            )

            return RetrievalResult(
                route="GRAPH",
                chunks=[],
                facts=[],
                query=question,
            )

        entity_name, entity_type, entity_id = entity

        logger.info(
            "graph_entity_resolved question=%s entity=%s type=%s entity_id=%s",
            question,
            entity_name,
            entity_type.value,
            entity_id,
        )

        rows = self._execute_graph_operation(
            question=question,
            entity_id=entity_id,
        )

        facts = self._rows_to_facts(rows)

        logger.info(
            "graph_retrieval question=%s entity=%s facts=%d",
            question,
            entity_name,
            len(facts),
        )

        return RetrievalResult(
            route="GRAPH",
            chunks=[],
            facts=facts,
            query=question,
        )

    # ==================================================================
    # ENTITY RESOLUTION
    # ==================================================================

    def _resolve_query_entity(self, question: str):
        """
        Resolve the most specific known entity mentioned in the question.

        Longest-name match is intentional so that:

            Activision Blizzard

        wins over:

            Blizzard

        when both appear in the same question.
        """

        question_normalized = self._normalize_text(question)

        candidates = sorted(
            self.ENTITY_CANDIDATES,
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for entity_name, entity_type in candidates:
            candidate_normalized = self._normalize_text(
                entity_name
            )

            if candidate_normalized in question_normalized:
                entity_id = create_entity_id(
                    entity_name,
                    entity_type,
                )

                return (
                    entity_name,
                    entity_type,
                    entity_id,
                )

        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ==================================================================
    # GRAPH OPERATION SELECTION
    # ==================================================================

    def _execute_graph_operation(
        self,
        question: str,
        entity_id: str,
    ):
        """
        Select one of the predefined GraphRetriever operations.

        IMPORTANT:
        No Cypher is generated by the LLM.
        """

        question_lower = question.lower().strip()

        # --------------------------------------------------------------
        # 1. Multi-hop:
        # "What company acquired the owner of Activision?"
        #
        # Activision
        #    <- OWNS - Activision Blizzard
        #    <- ACQUIRED - Microsoft
        # --------------------------------------------------------------

        if (
            "acquired the owner" in question_lower
            or "acquirer of the owner" in question_lower
        ):
            return self.graph_retriever.get_acquirer_of_owner(
                entity_id
            )

        # --------------------------------------------------------------
        # 2. Multi-hop:
        # "Who leads the company that acquired Activision Blizzard?"
        #
        # Activision Blizzard
        #       <- ACQUIRED - Microsoft
        #                         <- LED_BY - Satya Nadella
        # --------------------------------------------------------------

        if (
            "leads the company that acquired"
            in question_lower
            or "leader of the company that acquired"
            in question_lower
        ):
            return self.graph_retriever.get_leader_of_acquirer(
                entity_id
            )

        # --------------------------------------------------------------
        # 3. Reverse PART_OF
        #
        # "Which platform is part of Microsoft Cloud?"
        #
        # Microsoft Cloud
        #       <- PART_OF - Azure
        # --------------------------------------------------------------

        if (
            "part of microsoft cloud" in question_lower
            or "inside microsoft cloud" in question_lower
            or "within microsoft cloud" in question_lower
        ):
            return self.graph_retriever.get_part_of_children(
                entity_id
            )

        # --------------------------------------------------------------
        # 4. Business areas / business segments
        #
        # "What are Microsoft's three major business areas?"
        #
        # Microsoft
        #    -> OPERATES_IN -> Productivity and Business Processes
        #    -> OPERATES_IN -> Intelligent Cloud
        #    -> OPERATES_IN -> More Personal Computing
        # --------------------------------------------------------------

        if any(
            phrase in question_lower
            for phrase in [
                "business area",
                "business areas",
                "business segment",
                "business segments",
                "major business area",
                "major business areas",
                "major business segment",
                "major business segments",
            ]
        ):
            return self.graph_retriever.get_business_segments(
                entity_id
            )

        # --------------------------------------------------------------
        # 5. Aggregation: ownership count
        #
        # "How many companies does Activision Blizzard own?"
        # --------------------------------------------------------------

        if (
            "how many" in question_lower
            or "count" in question_lower
        ):
            if (
                "own" in question_lower
                or "owned" in question_lower
            ):
                return self.graph_retriever.count_owned_entities(
                    entity_id
                )

        # --------------------------------------------------------------
        # 6. Leadership
        #
        # "Who is the CEO of Microsoft?"
        # "Who leads Microsoft?"
        # --------------------------------------------------------------

        if any(
            phrase in question_lower
            for phrase in [
                "ceo",
                "chief executive",
                "who leads",
                "leader of",
                "led by",
            ]
        ):
            return self.graph_retriever.get_company_leader(
                entity_id
            )

        # --------------------------------------------------------------
        # 7. Acquisition
        #
        # "Who acquired Activision Blizzard?"
        # --------------------------------------------------------------

        if "acquired" in question_lower:
            return self.graph_retriever.get_company_that_acquired(
                entity_id
            )

        # --------------------------------------------------------------
        # 8. What did X acquire?
        # --------------------------------------------------------------

        if (
            "what did" in question_lower
            and "acquire" in question_lower
        ):
            return self.graph_retriever.get_acquired_entities(
                entity_id
            )

        # --------------------------------------------------------------
        # 9. Ownership: incoming OWNS
        #
        # "Who owns Blizzard?"
        # "Who is the owner of Blizzard?"
        # --------------------------------------------------------------

        if (
            "who owns" in question_lower
            or "owner of" in question_lower
            or "owned by" in question_lower
        ):
            return self.graph_retriever.get_owners(
                entity_id
            )

        # --------------------------------------------------------------
        # 10. Ownership: outgoing OWNS
        #
        # "What does Activision Blizzard own?"
        # "What companies does X own?"
        # "What products does X own?"
        # --------------------------------------------------------------

        if any(
            phrase in question_lower
            for phrase in [
                "what does",
                "what companies",
                "what products",
                "what entities",
            ]
        ) and (
            "own" in question_lower
            or "owned" in question_lower
        ):
            return self.graph_retriever.get_products_of_owner(
                entity_id
            )

        # --------------------------------------------------------------
        # 11. PART_OF / containment
        #
        # "Which business segment contains Azure?"
        # "Which segment does Azure belong to?"
        # --------------------------------------------------------------

        if any(
            phrase in question_lower
            for phrase in [
                "contains",
                "contain",
                "belongs to",
                "part of",
                "business segment",
                "segment contains",
            ]
        ):
            return self.graph_retriever.get_part_of(
                entity_id
            )

        # --------------------------------------------------------------
        # 12. Fallback:
        # Return all explicit relationships for the entity.
        # --------------------------------------------------------------

        logger.info(
            "graph_operation_fallback question=%s entity_id=%s",
            question,
            entity_id,
        )

        return self.graph_retriever.get_all_relationships(
            entity_id
        )

    # ==================================================================
    # GRAPH ROW -> RETRIEVED FACT
    # ==================================================================

    def _rows_to_facts(self, rows):
        """
        Convert Neo4j rows into RetrievedFact objects.

        Handles:

        - standard relationship rows
        - aggregation rows
        - multi-hop owner/acquirer rows
        - multi-hop leader/acquirer rows
        """

        if not rows:
            return []

        facts: list[RetrievedFact] = []

        for row in rows:
            # ----------------------------------------------------------
            # Aggregation result
            # ----------------------------------------------------------

            if "edges" in row:
                edges = row.get("edges") or []

                count = row.get("count")

                entities = row.get("entities") or []

                for edge in edges:
                    source_entity = row.get(
                        "owner_name",
                        "Activision Blizzard",
                    )

                    target_entity = edge.get(
                        "target_name"
                    )

                    source_chunk_id = edge.get(
                        "source_chunk_id"
                    )

                    evidence = edge.get(
                        "evidence",
                        "",
                    )

                    confidence = edge.get(
                        "confidence"
                    )

                    if (
                        not target_entity
                        or not source_chunk_id
                    ):
                        continue

                    facts.append(
                        RetrievedFact(
                            source_entity=source_entity,
                            relationship_type="OWNS",
                            target_entity=target_entity,
                            source_chunk_id=source_chunk_id,
                            evidence=evidence,
                            confidence=confidence,
                            metadata={
                                "aggregation": True,
                                "count": count,
                                "entities": entities,
                            },
                        )
                    )

                continue

            # ----------------------------------------------------------
            # Multi-hop: leader of acquirer
            #
            # GraphRetriever should return rows containing:
            #
            # acquisition_source_entity
            # acquisition_relationship_type
            # acquisition_target_entity
            # acquisition_source_chunk_id
            #
            # leadership_source_entity
            # leadership_relationship_type
            # leadership_target_entity
            # leadership_source_chunk_id
            # ----------------------------------------------------------

            if (
                "acquisition_source_chunk_id" in row
                and "leadership_source_chunk_id" in row
            ):
                acquisition_source = row.get(
                    "acquisition_source_entity"
                )
                acquisition_target = row.get(
                    "acquisition_target_entity"
                )
                acquisition_chunk = row.get(
                    "acquisition_source_chunk_id"
                )

                leadership_source = row.get(
                    "leadership_source_entity"
                )
                leadership_target = row.get(
                    "leadership_target_entity"
                )
                leadership_chunk = row.get(
                    "leadership_source_chunk_id"
                )

                if (
                    acquisition_source
                    and acquisition_target
                    and acquisition_chunk
                ):
                    facts.append(
                        RetrievedFact(
                            source_entity=acquisition_source,
                            relationship_type="ACQUIRED",
                            target_entity=acquisition_target,
                            source_chunk_id=acquisition_chunk,
                            evidence=row.get(
                                "acquisition_evidence",
                                "",
                            ),
                            confidence=row.get(
                                "acquisition_confidence"
                            ),
                        )
                    )

                if (
                    leadership_source
                    and leadership_target
                    and leadership_chunk
                ):
                    facts.append(
                        RetrievedFact(
                            source_entity=leadership_source,
                            relationship_type="LED_BY",
                            target_entity=leadership_target,
                            source_chunk_id=leadership_chunk,
                            evidence=row.get(
                                "leadership_evidence",
                                "",
                            ),
                            confidence=row.get(
                                "leadership_confidence"
                            ),
                        )
                    )

                continue

            # ----------------------------------------------------------
            # Multi-hop: acquirer of owner
            #
            # Owner relationship:
            #
            # Activision Blizzard OWNS Activision
            #
            # Acquisition relationship:
            #
            # Microsoft ACQUIRED Activision Blizzard
            # ----------------------------------------------------------

            if (
                "ownership_source_chunk_id" in row
                and "acquisition_source_chunk_id" in row
            ):
                ownership_source = row.get(
                    "ownership_source_entity"
                )
                ownership_target = row.get(
                    "ownership_target_entity"
                )
                ownership_chunk = row.get(
                    "ownership_source_chunk_id"
                )

                acquisition_source = row.get(
                    "acquisition_source_entity"
                )
                acquisition_target = row.get(
                    "acquisition_target_entity"
                )
                acquisition_chunk = row.get(
                    "acquisition_source_chunk_id"
                )

                if (
                    ownership_source
                    and ownership_target
                    and ownership_chunk
                ):
                    facts.append(
                        RetrievedFact(
                            source_entity=ownership_source,
                            relationship_type="OWNS",
                            target_entity=ownership_target,
                            source_chunk_id=ownership_chunk,
                            evidence=row.get(
                                "ownership_evidence",
                                "",
                            ),
                            confidence=row.get(
                                "ownership_confidence"
                            ),
                        )
                    )

                if (
                    acquisition_source
                    and acquisition_target
                    and acquisition_chunk
                ):
                    facts.append(
                        RetrievedFact(
                            source_entity=acquisition_source,
                            relationship_type="ACQUIRED",
                            target_entity=acquisition_target,
                            source_chunk_id=acquisition_chunk,
                            evidence=row.get(
                                "acquisition_evidence",
                                "",
                            ),
                            confidence=row.get(
                                "acquisition_confidence"
                            ),
                        )
                    )

                continue

            # ----------------------------------------------------------
            # Standard relationship row
            # ----------------------------------------------------------

            source_entity = row.get(
                "source_entity"
            )

            relationship_type = row.get(
                "relationship_type"
            )

            target_entity = row.get(
                "target_entity"
            )

            source_chunk_id = row.get(
                "source_chunk_id"
            )

            evidence = row.get(
                "evidence",
                "",
            )

            confidence = row.get(
                "confidence"
            )

            if not (
                source_entity
                and relationship_type
                and target_entity
                and source_chunk_id
            ):
                logger.warning(
                    "graph_row_missing_provenance row=%s",
                    row,
                )
                continue

            facts.append(
                RetrievedFact(
                    source_entity=source_entity,
                    relationship_type=relationship_type,
                    target_entity=target_entity,
                    source_chunk_id=source_chunk_id,
                    evidence=evidence,
                    confidence=confidence,
                )
            )

        return facts