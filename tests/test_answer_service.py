from kg_rag.answer.service import AnswerService
from kg_rag.retrieval.service import RetrievalService


QUESTIONS = [
    "Who is the CEO of Microsoft?",
    "Who owns Blizzard?",
    "How many companies does Activision Blizzard own?",
    "Who acquired Activision Blizzard?",
    "Who leads the company that acquired Activision Blizzard?",
    "Which business segment contains Azure?",
    "What company acquired the owner of Activision?",
    "What are Microsoft's three major business areas?",
]


def main():

    retrieval_service = RetrievalService()
    answer_service = AnswerService()

    for question in QUESTIONS:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        retrieval = retrieval_service.retrieve(
            question,
            top_k=5,
        )

        print(f"Route: {retrieval.route}")
        print(
            f"Retrieved chunks: {len(retrieval.chunks)}"
        )
        print(
            f"Retrieved facts: {len(retrieval.facts)}"
        )

        answer = answer_service.answer(
            question=question,
            retrieval=retrieval,
        )

        print(f"\nANSWER:\n{answer.answer}")

        print("\nCITATIONS:")

        for citation in answer.citations:
            print(
                f"- {citation.chunk_id} | "
                f"{citation.claim}"
            )


if __name__ == "__main__":
    main()