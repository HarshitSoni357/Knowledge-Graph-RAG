from __future__ import annotations

from typing import Iterable

from kg_rag.extraction.schema import ExtractionResult
from kg_rag.graph.entity_resolution import (
    ResolvedEntity,
    resolve_entity,
)
from kg_rag.graph.neo4j_client import Neo4jClient


class GraphWriter:
    """
    Writes validated entity and relationship extractions into Neo4j.

    Responsibilities:
    - Create the entity uniqueness constraint.
    - Resolve entities deterministically.
    - MERGE entities idempotently.
    - Resolve relationship endpoints across documents/chunks.
    - MERGE relationships idempotently.
    - Preserve source_chunk_id on every relationship.
    """

    def __init__(self, client: Neo4jClient):
        self.client = client

    # ------------------------------------------------------------------
    # Schema / constraints
    # ------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """
        Ensure entity IDs are unique in Neo4j.
        """

        query = """
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.entity_id IS UNIQUE
        """

        self.client.execute(query)

    # ------------------------------------------------------------------
    # Entity writing
    # ------------------------------------------------------------------

    def write_entities(
        self,
        entities: Iterable[ResolvedEntity],
    ) -> None:
        """
        Idempotently write entities to Neo4j.

        Entity identity is determined by the deterministic entity_id
        generated during entity resolution.
        """

        entity_data = [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type.value,
                "aliases": list(entity.aliases),
            }
            for entity in entities
        ]

        if not entity_data:
            return

        query = """
        UNWIND $entities AS entity

        MERGE (e:Entity {
            entity_id: entity.entity_id
        })

        SET
            e.name = entity.name,
            e.entity_type = entity.entity_type,
            e.aliases = CASE
                WHEN entity.aliases IS NULL THEN []
                ELSE entity.aliases
            END
        """

        self.client.execute(
            query,
            {
                "entities": entity_data,
            },
        )

    # ------------------------------------------------------------------
    # Extraction writing
    # ------------------------------------------------------------------

    def write_extraction(
        self,
        extraction: ExtractionResult,
        source_chunk_id: str,
    ) -> None:
        """
        Write one validated extraction into Neo4j.

        Entity resolution works at two levels:

        1. Entities extracted from the current chunk are resolved
           deterministically.

        2. Relationship endpoints that were not extracted from the
           current chunk are looked up against entities already
           present in the global graph.

        This allows relationships to span documents and chunks.
        """

        # --------------------------------------------------------------
        # Resolve and write entities extracted from this chunk.
        # --------------------------------------------------------------

        resolved_entities = [
            resolve_entity(entity)
            for entity in extraction.entities
        ]

        self.write_entities(
            resolved_entities
        )

        if not extraction.relationships:
            return

        # --------------------------------------------------------------
        # Local entity map.
        #
        # If a relationship endpoint was extracted in this same chunk,
        # use the already-resolved entity rather than querying Neo4j.
        # --------------------------------------------------------------

        local_entity_map = {
            (
                entity.name.strip().lower(),
                entity.entity_type,
            ): entity
            for entity in resolved_entities
        }

        # --------------------------------------------------------------
        # Resolve and write relationships.
        # --------------------------------------------------------------

        for relationship in extraction.relationships:

            source = local_entity_map.get(
                (
                    relationship.source_entity.strip().lower(),
                    relationship.source_type,
                )
            )

            target = local_entity_map.get(
                (
                    relationship.target_entity.strip().lower(),
                    relationship.target_type,
                )
            )

            # ----------------------------------------------------------
            # Cross-document resolution.
            #
            # Example:
            #
            # Document A:
            #   Microsoft ACQUIRED Activision Blizzard
            #
            # Document B:
            #   Activision Blizzard OWNS Blizzard
            #
            # If Microsoft isn't extracted from Document B, resolve
            # Microsoft from the existing global graph.
            # ----------------------------------------------------------

            if source is None:
                source = self._find_existing_entity(
                    name=relationship.source_entity,
                    entity_type=relationship.source_type,
                )

            if target is None:
                target = self._find_existing_entity(
                    name=relationship.target_entity,
                    entity_type=relationship.target_type,
                )

            # ----------------------------------------------------------
            # Never create a relationship with an unresolved endpoint.
            # ----------------------------------------------------------

            if source is None or target is None:

                missing = []

                if source is None:
                    missing.append(
                        f"source={relationship.source_entity}"
                    )

                if target is None:
                    missing.append(
                        f"target={relationship.target_entity}"
                    )

                print(
                    "  WARNING: Could not resolve "
                    "relationship endpoint(s): "
                    + ", ".join(missing)
                )

                continue

            # ----------------------------------------------------------
            # Write the relationship.
            # ----------------------------------------------------------

            self._write_relationships(
                [
                    {
                        "source_entity_id": source.entity_id,
                        "target_entity_id": target.entity_id,
                        "relationship_type": (
                            relationship.relationship_type.value
                        ),
                        "confidence": relationship.confidence,
                        "evidence": relationship.evidence,
                        "source_chunk_id": source_chunk_id,
                    }
                ]
            )

    # ------------------------------------------------------------------
    # Existing entity lookup
    # ------------------------------------------------------------------

    def _find_existing_entity(
        self,
        name: str,
        entity_type,
    ) -> ResolvedEntity | None:
        """
        Resolve an entity against entities already stored in Neo4j.

        Resolution is deterministic using:

            normalized name + entity type

        No fuzzy matching is performed here.
        """

        normalized_name = (
            name.strip().lower()
        )

        query = """
        MATCH (e:Entity)

        WHERE
            toLower(e.name) = $name
            AND e.entity_type = $entity_type

        RETURN
            e.entity_id AS entity_id,
            e.name AS name,
            e.entity_type AS entity_type,
            e.aliases AS aliases

        LIMIT 1
        """

        rows = self.client.execute(
            query,
            {
                "name": normalized_name,
                "entity_type": entity_type.value,
            },
        )

        if not rows:
            return None

        row = rows[0]

        return ResolvedEntity(
            entity_id=row["entity_id"],
            name=row["name"],
            entity_type=entity_type,
            aliases=tuple(
                row.get("aliases") or []
            ),
        )

    # ------------------------------------------------------------------
    # Relationship writing
    # ------------------------------------------------------------------

    def _write_relationships(
        self,
        relationships: Iterable[dict],
    ) -> None:
        """
        Idempotently write relationships.

        Relationship types are interpolated into Cypher only after
        deterministic Pydantic enum validation.

        Model-generated relationship types are therefore never
        directly inserted into Cypher.
        """

        for relationship in relationships:

            relationship_type = (
                relationship["relationship_type"]
            )

            query = f"""
            MATCH (
                source:Entity {{
                    entity_id: $source_entity_id
                }}
            )

            MATCH (
                target:Entity {{
                    entity_id: $target_entity_id
                }}
            )

            MERGE (
                source
            )-[r:{relationship_type} {{
                source_chunk_id: $source_chunk_id
            }}]->(
                target
            )

            SET
                r.confidence = $confidence,
                r.evidence = $evidence
            """

            self.client.execute(
                query,
                {
                    "source_entity_id": (
                        relationship[
                            "source_entity_id"
                        ]
                    ),
                    "target_entity_id": (
                        relationship[
                            "target_entity_id"
                        ]
                    ),
                    "source_chunk_id": (
                        relationship[
                            "source_chunk_id"
                        ]
                    ),
                    "confidence": (
                        relationship[
                            "confidence"
                        ]
                    ),
                    "evidence": (
                        relationship[
                            "evidence"
                        ]
                    ),
                },
            )