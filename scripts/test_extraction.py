from pathlib import Path
from unittest import result

from kg_rag.ingestion.html_loader import (
    load_html,
    extract_html_text,
)
from kg_rag.ingestion.chunker import chunk_text
from kg_rag.extraction.extractor import extract_from_chunk


SOURCE_PATH = Path(
    "data/raw/microsoft_2025_10k.html"
)


def main():

    print("Loading document...")

    document = load_html(SOURCE_PATH)

    text = extract_html_text(SOURCE_PATH)

    print("Chunking...")

    chunks = chunk_text(
        document,
        text,
    )

    print(f"Total chunks: {len(chunks)}")

    # Start with one relationship-rich chunk.
    chunk = chunks[5]

    test_text = chunk.text[:1200]

    print("\n================================")
    print("CHUNK")
    print("================================")
    print(chunk.text)

    print("\n================================")
    print("EXTRACTING")
    print("================================")

    result = extract_from_chunk(test_text)

    print("\n================================")
    print("TEST TEXT")
    print("================================")
    print(test_text)

    print("\n================================")
    print("EXTRACTING")
    print("================================")
    print("Calling local Qwen3...")

    for entity in result.entities:
        print(
            f"- {entity.name}"
            f" [{entity.entity_type.value}]"
        )

    print("\n================================")
    print("RELATIONSHIPS")
    print("================================")

    for relationship in result.relationships:
        print(
            f"- {relationship.source_entity}"
            f" --[{relationship.relationship_type.value}]--> "
            f"{relationship.target_entity}"
            f" ({relationship.confidence:.2f})"
        )


if __name__ == "__main__":
    main()