from __future__ import annotations

import sys
from pathlib import Path


# -------------------------------------------------------------------
# Project imports
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from kg_rag.extraction.extractor import extract_from_chunk
from kg_rag.graph.graph_writer import GraphWriter
from kg_rag.graph.neo4j_client import Neo4jClient
from kg_rag.ingestion.chunker import chunk_text
from kg_rag.ingestion.html_loader import create_document_id


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DEMO_DIR = PROJECT_ROOT / "data" / "raw" / "demo"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

DEMO_FILES = [
    DEMO_DIR / "microsoft.md",
    DEMO_DIR / "activision_blizzard.md",
    DEMO_DIR / "azure.md",
    DEMO_DIR / "leadership.md",
    DEMO_DIR / "business_segments.md",
]


# -------------------------------------------------------------------
# Markdown document loader
# -------------------------------------------------------------------

def load_demo_document(path: Path):
    """
    Load a small markdown document using the existing Document schema.
    """

    from kg_rag.ingestion.schema import Document

    content = path.read_bytes()

    document_id = create_document_id(content)

    return Document(
        document_id=document_id,
        filename=path.name,
        source=str(path),
    )


# -------------------------------------------------------------------
# Main ingestion
# -------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("LOCAL KNOWLEDGE GRAPH INGESTION — DEMO CORPUS")
    print("=" * 70)

    print()
    print(f"Corpus: {DEMO_DIR}")
    print(f"Documents: {len(DEMO_FILES)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    # ---------------------------------------------------------------
    # Connect to Neo4j
    # ---------------------------------------------------------------

    print()
    print("[1/3] Connecting to Neo4j...")

    with Neo4jClient() as client:

        client.verify_connectivity()

        print("Neo4j connection: OK")

        writer = GraphWriter(client)

        writer.ensure_constraints()

        # -----------------------------------------------------------
        # Process documents
        # -----------------------------------------------------------

        print()
        print("[2/3] Processing demo documents...")

        total_chunks = 0
        total_entities = 0
        total_relationships = 0

        for document_number, path in enumerate(
            DEMO_FILES,
            start=1,
        ):

            print()
            print("-" * 70)
            print(
                f"DOCUMENT {document_number}/{len(DEMO_FILES)}"
            )
            print("-" * 70)

            if not path.exists():

                raise FileNotFoundError(
                    f"Demo document not found: {path}"
                )

            document = load_demo_document(path)

            text = path.read_text(
                encoding="utf-8"
            )

            print(f"File: {document.filename}")
            print(f"Document ID: {document.document_id}")
            print(f"Characters: {len(text):,}")

            chunks = chunk_text(
                document=document,
                text=text,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            print(f"Chunks: {len(chunks)}")

            for chunk_number, chunk in enumerate(
                chunks,
                start=1,
            ):

                print()
                print(
                    f"  CHUNK {chunk_number}/{len(chunks)}"
                )

                print(
                    f"  Chunk ID: {chunk.chunk_id}"
                )

                print(
                    f"  Section: {chunk.section_path}"
                )

                print(
                    f"  Characters: {len(chunk.text)}"
                )

                extraction = extract_from_chunk(
                    chunk.text
                )

                print(
                    f"  Entities extracted: "
                    f"{len(extraction.entities)}"
                )

                print(
                    f"  Relationships extracted: "
                    f"{len(extraction.relationships)}"
                )

                for entity in extraction.entities:

                    print(
                        f"    ENTITY: "
                        f"{entity.name} "
                        f"[{entity.entity_type.value}]"
                    )

                for relationship in extraction.relationships:

                    print(
                        f"    RELATIONSHIP: "
                        f"{relationship.source_entity}"
                        f" --{relationship.relationship_type.value}--> "
                        f"{relationship.target_entity}"
                    )

                writer.write_extraction(
                    extraction,
                    source_chunk_id=chunk.chunk_id,
                )

                total_chunks += 1
                total_entities += len(
                    extraction.entities
                )
                total_relationships += len(
                    extraction.relationships
                )

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------

        print()
        print("[3/3] Ingestion complete.")

        print()
        print("=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)

        print(
            f"Documents processed: "
            f"{len(DEMO_FILES)}"
        )

        print(
            f"Chunks processed: "
            f"{total_chunks}"
        )

        print(
            f"Entity mentions extracted: "
            f"{total_entities}"
        )

        print(
            f"Relationship mentions extracted: "
            f"{total_relationships}"
        )

        print()
        print("Neo4j graph is ready for inspection.")


if __name__ == "__main__":
    main()