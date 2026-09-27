from kg_rag.extraction.schema import Entity, Relationship, ExtractionResult
from kg_rag.graph.graph_writer import GraphWriter
from kg_rag.graph.neo4j_client import Neo4jClient
from kg_rag.graph.ontology import EntityType, RelationshipType


def build_test_extraction() -> ExtractionResult:
    """
    Create a small deterministic extraction for testing the graph writer.
    """

    return ExtractionResult(
        entities=[
            Entity(
                name="Microsoft",
                entity_type=EntityType.COMPANY,
                aliases=[],
                evidence="Microsoft Corporation",
            ),
            Entity(
                name="Azure",
                entity_type=EntityType.PRODUCT,
                aliases=[],
                evidence="Microsoft Azure",
            ),
        ],
        relationships=[
            Relationship(
                source_entity="Microsoft",
                source_type=EntityType.COMPANY,
                relationship_type=RelationshipType.PRODUCES,
                target_entity="Azure",
                target_type=EntityType.PRODUCT,
                confidence=1.0,
                evidence="Microsoft Azure",
            )
        ],
    )


def count_graph(client: Neo4jClient) -> tuple[int, int]:
    """
    Return the number of Entity nodes and relationships.
    """

    node_result = client.execute(
        """
        MATCH (e:Entity)
        RETURN count(e) AS count
        """
    )

    relationship_result = client.execute(
        """
        MATCH ()-[r]->()
        RETURN count(r) AS count
        """
    )

    return (
        node_result[0]["count"],
        relationship_result[0]["count"],
    )


def main() -> None:

    extraction = build_test_extraction()

    source_chunk_id = "test-chunk-001"

    print("Connecting to Neo4j...")

    with Neo4jClient() as client:

        writer = GraphWriter(client)

        print("Creating Neo4j constraints...")
        writer.ensure_constraints()

        print("Writing test extraction...")
        writer.write_extraction(
            extraction=extraction,
            source_chunk_id=source_chunk_id,
        )

        nodes_after_first_run, relationships_after_first_run = (
            count_graph(client)
        )

        print("\n========== AFTER FIRST RUN ==========")

        print(
            f"Entity nodes: {nodes_after_first_run}"
        )

        print(
            f"Relationships: {relationships_after_first_run}"
        )

        print("\nWriting the SAME extraction again...")

        writer.write_extraction(
            extraction=extraction,
            source_chunk_id=source_chunk_id,
        )

        nodes_after_second_run, relationships_after_second_run = (
            count_graph(client)
        )

        print("\n========== AFTER SECOND RUN ==========")

        print(
            f"Entity nodes: {nodes_after_second_run}"
        )

        print(
            f"Relationships: {relationships_after_second_run}"
        )

        print("\n========== IDEMPOTENCY TEST ==========")

        if (
            nodes_after_first_run == nodes_after_second_run
            and relationships_after_first_run
            == relationships_after_second_run
        ):
            print("PASS: Graph ingestion is idempotent.")

        else:
            print("FAIL: Duplicate nodes or relationships detected.")


if __name__ == "__main__":
    main()