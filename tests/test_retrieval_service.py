from kg_rag.retrieval.service import RetrievalService


QUESTIONS = [
    "Who is the CEO of Microsoft?",
    "Who owns Blizzard?",
    "How many companies does Activision Blizzard own?",
    "Who acquired Activision Blizzard?",
    "Who leads the company that acquired Activision Blizzard?",
    "Which business segment contains Azure?",
]


def main():
    service = RetrievalService()

    print("=" * 70)
    print("UNIFIED RETRIEVAL SERVICE TEST")
    print("=" * 70)

    for question in QUESTIONS:

        print()
        print("-" * 70)
        print(f"QUESTION: {question}")

        result = service.retrieve(question)

        print(f"ROUTE: {result.route}")

        print(f"VECTOR CHUNKS: {len(result.chunks)}")
        print(f"GRAPH FACTS: {len(result.facts)}")

        for chunk in result.chunks:
            print(
                f"  [VECTOR] "
                f"{chunk.chunk_id[:12]}... "
                f"score={chunk.score:.4f}"
            )

        for fact in result.facts:
            print(
                f"  [GRAPH] "
                f"{fact.source_entity} "
                f"--[{fact.relationship_type}]--> "
                f"{fact.target_entity} "
                f"| chunk={fact.source_chunk_id[:12]}..."
            )


if __name__ == "__main__":
    main()