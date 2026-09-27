from kg_rag.benchmark.runner import BenchmarkRunner


def main():

    runner = BenchmarkRunner()

    results, summary = runner.run()

    assert len(results) > 0

    assert summary["total_questions"] == len(
        results
    )

    print(
        "\nBenchmark completed successfully."
    )


if __name__ == "__main__":
    main()