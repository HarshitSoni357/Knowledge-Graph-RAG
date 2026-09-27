from pathlib import Path

from kg_rag.extraction.extractor import extract_from_chunk


DEMO_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "demo"
)


def test_demo_extraction():

    files = [
        "microsoft.md",
        "activision_blizzard.md",
        "azure.md",
        "leadership.md",
        "business_segments.md",
    ]

    for filename in files:

        path = DEMO_DIR / filename

        text = path.read_text(
            encoding="utf-8"
        )

        result = extract_from_chunk(text)

        assert result is not None

        assert isinstance(
            result.entities,
            list,
        )

        assert isinstance(
            result.relationships,
            list,
        )

        for relationship in result.relationships:

            assert (
                relationship.source_entity
            )

            assert (
                relationship.target_entity
            )

            assert (
                relationship.relationship_type
            )

            assert (
                relationship.evidence.lower()
                in text.lower()
            )