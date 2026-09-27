from kg_rag.vector.embedder import OllamaEmbedder
from kg_rag.vector.pgvector_store import PgVectorStore


def main():
    query = "Which company acquired Activision Blizzard?"

    print("=" * 70)
    print("VECTOR SEARCH TEST")
    print("=" * 70)
    print(f"Query: {query}")

    embedder = OllamaEmbedder()

    query_embedding = embedder.embed(query)

    with PgVectorStore() as store:

        results = store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

    print()
    print("RESULTS")
    print("-" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(f"[{index}]")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(
            f"Text: {result['text'][:300]}"
        )


if __name__ == "__main__":
    main()