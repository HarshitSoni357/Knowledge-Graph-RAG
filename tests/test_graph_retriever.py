from kg_rag.graph.neo4j_client import Neo4jClient
from kg_rag.graph.retriever import GraphRetriever


def main():
    print("=" * 70)
    print("GRAPH RETRIEVER TEST")
    print("=" * 70)

    with Neo4jClient() as client:
        retriever = GraphRetriever(client)

        # ---------------------------------------------------------
        # 1. Who owns Blizzard?
        # ---------------------------------------------------------
        blizzard = retriever.find_entity("Blizzard")

        print("\n[1] Who owns Blizzard?")

        if blizzard:
            results = retriever.get_owner_of_product(
                blizzard["entity_id"]
            )

            for row in results:
                print(
                    f"{row['owner_name']} OWNS {row['product_name']}"
                )

        # ---------------------------------------------------------
        # 2. What does Activision Blizzard own?
        # ---------------------------------------------------------
        activision_blizzard = retriever.find_entity(
            "Activision Blizzard"
        )

        print("\n[2] What does Activision Blizzard own?")

        if activision_blizzard:
            results = retriever.get_products_of_owner(
                activision_blizzard["entity_id"]
            )

            for row in results:
                print(
                    f"{row['owner_name']} OWNS {row['product_name']}"
                )

        # ---------------------------------------------------------
        # 3. How many companies does Activision Blizzard own?
        # ---------------------------------------------------------
        print("\n[3] How many companies does Activision Blizzard own?")

        if activision_blizzard:
            result = retriever.count_owned_entities(
                activision_blizzard["entity_id"]
            )

            if result:
                print(f"Count: {result['owned_count']}")
                print(f"Entities: {result['owned_entities']}")

        # ---------------------------------------------------------
        # 4. Who acquired Activision Blizzard?
        # ---------------------------------------------------------
        print("\n[4] Who acquired Activision Blizzard?")

        if activision_blizzard:
            results = retriever.get_company_that_acquired(
                activision_blizzard["entity_id"]
            )

            for row in results:
                print(
                    f"{row['acquirer_name']} "
                    f"ACQUIRED "
                    f"{row['target_name']}"
                )

        # ---------------------------------------------------------
        # 5. Who leads the company that acquired Activision Blizzard?
        # ---------------------------------------------------------
        print(
            "\n[5] Who leads the company that acquired "
            "Activision Blizzard?"
        )

        if activision_blizzard:
            results = retriever.get_leader_of_acquirer(
                activision_blizzard["entity_id"]
            )

            for row in results:
                print(
                    f"{row['person_name']} leads "
                    f"{row['acquirer_name']}"
                )

        # ---------------------------------------------------------
        # 6. Which business segment contains Azure?
        # ---------------------------------------------------------
        azure = retriever.find_entity("Azure")

        print("\n[6] Which business segment contains Azure?")

        if azure:
            results = retriever.get_part_of(
                azure["entity_id"]
            )

            for row in results:
                print(
                    f"{row['source_name']} "
                    f"PART_OF "
                    f"{row['target_name']}"
                )


if __name__ == "__main__":
    main()