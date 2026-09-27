from kg_rag.vector.embedder import OllamaEmbedder
from kg_rag.vector.pgvector_store import PgVectorStore


QUESTIONS = [
    {
        "question": "Who is the CEO of Microsoft?",
        "expected_keywords": ["Satya Nadella"],
    },
    {
        "question": "Which company did Microsoft acquire?",
        "expected_keywords": ["Activision Blizzard"],
    },
    {
        "question": "Who owns Blizzard?",
        "expected_keywords": ["Activision Blizzard"],
    },
    {
        "question": "Which platform is produced by Microsoft?",
        "expected_keywords": ["Azure"],
    },
    {
        "question": "Where is Microsoft headquartered?",
        "expected_keywords": ["Redmond", "Washington"],
    },
    {
        "question": "What companies are owned by Activision Blizzard?",
        "expected_keywords": ["Activision", "Blizzard", "King"],
    },
    {
        "question": "Which Microsoft business area contains Azure?",
        "expected_keywords": ["Intelligent Cloud"],
    },
    {
        "question": "What are Microsoft's three major business areas?",
        "expected_keywords": [
            "Productivity and Business Processes",
            "Intelligent Cloud",
            "More Personal Computing",
        ],
    },
]


def evaluate_recall(top_k: int) -> None:
    embedder = OllamaEmbedder()

    hits = 0

    with PgVectorStore() as store:
        for item in QUESTIONS:
            embedding = embedder.embed(item["question"])

            results = store.search(
                query_embedding=embedding,
                top_k=top_k,
            )

            retrieved_text = "\n".join(
                result["text"] for result in results
            ).lower()

            found = all(
                keyword.lower() in retrieved_text
                for keyword in item["expected_keywords"]
            )

            if found:
                hits += 1

            print()
            print(f"Question: {item['question']}")
            print(f"Recall@{top_k}: {'PASS' if found else 'MISS'}")

    recall = hits / len(QUESTIONS)

    print()
    print("=" * 60)
    print(f"Recall@{top_k}: {hits}/{len(QUESTIONS)} = {recall:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    for k in [1, 3, 5]:
        evaluate_recall(k)