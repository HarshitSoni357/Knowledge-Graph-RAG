import json
import logging
import time
from pathlib import Path

from kg_rag.answer.service import AnswerService
from kg_rag.benchmark.evaluator import answer_matches, calculate_summary
from kg_rag.benchmark.models import BenchmarkQuestion
from kg_rag.benchmark.vector_baseline import VectorOnlyBaseline
from kg_rag.retrieval.service import RetrievalService
from kg_rag.benchmark.models import BenchmarkQuestion,BenchmarkResult


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(
        self,
        questions_path: Path | str = "data/benchmark/questions.json",
    ):
        self.questions_path = Path(questions_path)

        self.graph_retrieval = RetrievalService()
        self.graph_answer_service = AnswerService()

        self.vector_baseline = VectorOnlyBaseline(
            answer_service=AnswerService()
        )

    def load_questions(self) -> list[BenchmarkQuestion]:
        with self.questions_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        questions = []

        for item in data:
            # Support the current question file even if category is absent.
            category = item.get("category")

            if not category:
                question_id = item.get("id", "")

                # Temporary classification based on the current
                # controlled benchmark question IDs.
                if question_id in {"q01", "q02", "q03", "q04", "q05"}:
                    category = "single_hop"
                elif question_id in {"q06", "q07", "q08", "q09"}:
                    category = "two_hop"
                elif question_id in {"q10", "q11"}:
                    category = "three_hop"
                elif question_id in {"q12", "q13"}:
                    category = "aggregation"
                elif question_id in {"q14", "q15"}:
                    category = "out_of_scope"
                else:
                    raise ValueError(
                        f"Cannot determine category for question: {question_id}"
                    )

            questions.append(
                BenchmarkQuestion(
                    question_id=item["id"],
                    question=item["question"],
                    category=category,
                    expected_answer=item["expected_answer"],
                )
            )

        return questions
    def _run_question(
        self,
        question: BenchmarkQuestion,
    ) -> BenchmarkResult:
        result = BenchmarkResult(
            question_id=question.question_id,
            question=question.question,
            category=question.category,
            expected_answer=question.expected_answer,
        )

        try:
            # ----------------------------------------------------------
            # VECTOR-ONLY BASELINE
            # ----------------------------------------------------------

            start = time.perf_counter()

            vector_answer = self.vector_baseline.answer(
                question.question
            )

            result.vector_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            result.vector_answer = vector_answer.answer

            result.vector_correct = answer_matches(
                vector_answer.answer,
                question.expected_answer,
            )

            result.vector_citations = len(
                vector_answer.citations
            )

            # ----------------------------------------------------------
            # GRAPH RAG
            # ----------------------------------------------------------

            start = time.perf_counter()

            retrieval = self.graph_retrieval.retrieve(
                question.question
            )

            graph_answer = self.graph_answer_service.answer(
                question=question.question,
                retrieval=retrieval,
            )

            result.graph_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            result.graph_answer = graph_answer.answer

            result.graph_correct = answer_matches(
                graph_answer.answer,
                question.expected_answer,
            )

            result.graph_citations = len(
                graph_answer.citations
            )

        except Exception as exc:
            logger.exception(
                "benchmark_question_failed question_id=%s",
                question.question_id,
            )

            result.error = str(exc)

        return result

    def run(self):
        questions = self.load_questions()

        results = []

        for question in questions:
            print("\n" + "=" * 80)
            print(f"QUESTION {question.question_id}: {question.question}")
            print(f"CATEGORY: {question.category}")
            print("=" * 80)

            result = self._run_question(question)
            results.append(result)

            print(
                f"Vector: "
                f"{'PASS' if result.vector_correct else 'FAIL'} "
                f"({result.vector_latency_ms:.1f} ms)"
            )

            print(
                f"Graph:  "
                f"{'PASS' if result.graph_correct else 'FAIL'} "
                f"({result.graph_latency_ms:.1f} ms)"
            )

            if result.error:
                print(f"ERROR: {result.error}")

            summary = calculate_summary(results)

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"Questions: {summary['total_questions']}")

        print(
            f"Vector accuracy: "
            f"{summary['vector_accuracy']:.2%}"
        )

        print(
            f"Graph accuracy:  "
            f"{summary['graph_accuracy']:.2%}"
        )

        print(
            f"Vector avg latency: "
            f"{summary['vector_avg_latency_ms']:.1f} ms"
        )

        print(
            f"Graph avg latency:  "
            f"{summary['graph_avg_latency_ms']:.1f} ms"
        )

        print(
            f"Vector P50: "
            f"{summary['vector_p50_latency_ms']:.1f} ms"
        )

        print(
            f"Graph P50: "
            f"{summary['graph_p50_latency_ms']:.1f} ms"
        )

        print(
            f"Vector P95: "
            f"{summary['vector_p95_latency_ms']:.1f} ms"
        )

        print(
            f"Graph P95: "
            f"{summary['graph_p95_latency_ms']:.1f} ms"
        )

        print("\nAccuracy by category:")

        for category, metrics in summary["by_category"].items():
            print(
                f"  {category:15s} "
                f"Vector={metrics['vector_accuracy']:.2%} "
                f"Graph={metrics['graph_accuracy']:.2%}"
            )

        return results, summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    runner = BenchmarkRunner()
    runner.run()


if __name__ == "__main__":
    main()