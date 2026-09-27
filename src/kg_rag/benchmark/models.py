from dataclasses import dataclass, field


@dataclass
class BenchmarkQuestion:
    question_id: str
    question: str
    category: str
    expected_answer: str


@dataclass
class BenchmarkResult:
    question_id: str
    question: str
    category: str
    expected_answer: str

    vector_answer: str = ""
    graph_answer: str = ""

    vector_correct: bool = False
    graph_correct: bool = False

    vector_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0

    vector_citations: int = 0
    graph_citations: int = 0

    error: str | None = None


@dataclass
class BenchmarkSummary:
    total_questions: int

    vector_accuracy: float
    graph_accuracy: float

    vector_avg_latency_ms: float
    graph_avg_latency_ms: float

    vector_p50_latency_ms: float
    graph_p50_latency_ms: float

    vector_p95_latency_ms: float
    graph_p95_latency_ms: float

    by_category: dict[str, dict] = field(
        default_factory=dict
    )