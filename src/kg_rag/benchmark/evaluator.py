import re
from statistics import mean, median


def normalize_answer(answer: str) -> str:
    """
    Normalize answers for deterministic benchmark comparison.
    """

    answer = answer.lower().strip()

    answer = re.sub(
        r"[^a-z0-9\s,.-]",
        " ",
        answer,
    )

    answer = re.sub(
        r"\s+",
        " ",
        answer,
    )

    return answer.strip()


def answer_matches(
    generated: str,
    expected: str,
) -> bool:

    generated_normalized = normalize_answer(
        generated
    )

    expected_normalized = normalize_answer(
        expected
    )

    # Exact match first.
    if generated_normalized == expected_normalized:
        return True

    # Numeric answers.
    if expected_normalized.isdigit():

        numbers = re.findall(
            r"\b\d+\b",
            generated_normalized,
        )

        return expected_normalized in numbers

    # Handle comma-separated entity lists.
    expected_items = [
        item.strip()
        for item in expected_normalized.split(",")
        if item.strip()
    ]

    if len(expected_items) > 1:

        return all(
            item in generated_normalized
            for item in expected_items
        )

    return expected_normalized in generated_normalized


def percentile(
    values: list[float],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = (
        (len(values) - 1)
        * p
    )

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def calculate_summary(results):

    total = len(results)

    vector_correct = sum(
        result.vector_correct
        for result in results
    )

    graph_correct = sum(
        result.graph_correct
        for result in results
    )

    vector_latencies = [
        result.vector_latency_ms
        for result in results
    ]

    graph_latencies = [
        result.graph_latency_ms
        for result in results
    ]

    categories = {}

    for result in results:

        category = result.category

        if category not in categories:
            categories[category] = {
                "total": 0,
                "vector_correct": 0,
                "graph_correct": 0,
            }

        categories[category]["total"] += 1

        if result.vector_correct:
            categories[category][
                "vector_correct"
            ] += 1

        if result.graph_correct:
            categories[category][
                "graph_correct"
            ] += 1

    for category, data in categories.items():

        total_category = data["total"]

        data["vector_accuracy"] = (
            data["vector_correct"]
            / total_category
        )

        data["graph_accuracy"] = (
            data["graph_correct"]
            / total_category
        )

    return {
        "total_questions": total,

        "vector_accuracy": (
            vector_correct / total
            if total
            else 0.0
        ),

        "graph_accuracy": (
            graph_correct / total
            if total
            else 0.0
        ),

        "vector_avg_latency_ms": (
            mean(vector_latencies)
            if vector_latencies
            else 0.0
        ),

        "graph_avg_latency_ms": (
            mean(graph_latencies)
            if graph_latencies
            else 0.0
        ),

        "vector_p50_latency_ms": percentile(
            vector_latencies,
            0.50,
        ),

        "graph_p50_latency_ms": percentile(
            graph_latencies,
            0.50,
        ),

        "vector_p95_latency_ms": percentile(
            vector_latencies,
            0.95,
        ),

        "graph_p95_latency_ms": percentile(
            graph_latencies,
            0.95,
        ),

        "by_category": categories,
    }