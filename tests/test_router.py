import logging

from kg_rag.router.query_router import QueryRouter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


QUESTIONS = [
    "Who is the CEO of Microsoft?",
    "What is Azure?",
    "Which company did Microsoft acquire?",
    "Who owns Blizzard?",
    "Which Microsoft business area contains Azure?",
    "What companies are owned by Activision Blizzard?",
    "Who leads the company that acquired Activision Blizzard?",
    "How many companies does Activision Blizzard own?",
    "What are Microsoft's three major business areas?",
]


def main():
    router = QueryRouter()

    print("=" * 70)
    print("QUERY ROUTER TEST")
    print("=" * 70)

    for question in QUESTIONS:
        decision = router.route(question)

        print()
        print(f"Question : {question}")
        print(f"Route    : {decision.route.value}")
        print(f"Reason   : {decision.reason}")


if __name__ == "__main__":
    main()