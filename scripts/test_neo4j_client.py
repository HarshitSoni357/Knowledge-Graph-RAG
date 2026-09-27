from kg_rag.graph.neo4j_client import Neo4jClient

def main() -> None:
    print("Connecting to Neo4j...")

    with Neo4jClient() as client:

        result = client.execute(
            "RETURN 'Neo4j connection successful' AS message"
        )

        print(result[0]["message"])


if __name__ == "__main__":
    main()