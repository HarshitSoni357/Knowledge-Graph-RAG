from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from kg_rag.ingestion.chunker import chunk_text
from kg_rag.ingestion.schema import Document
from kg_rag.ingestion.hashing import create_document_id
from kg_rag.vector.embedder import OllamaEmbedder
from kg_rag.vector.pgvector_store import PgVectorStore


DEMO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "demo"
)


DEMO_FILES = [
    DEMO_DIR / "microsoft.md",
    DEMO_DIR / "activision_blizzard.md",
    DEMO_DIR / "azure.md",
    DEMO_DIR / "leadership.md",
    DEMO_DIR / "business_segments.md",
]


CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def load_document(
    path: Path,
) -> tuple[Document, str]:

    content = path.read_bytes()

    document_id = create_document_id(
        content
    )

    document = Document(
        document_id=document_id,
        filename=path.name,
        source=str(path),
    )

    text = path.read_text(
        encoding="utf-8"
    )

    return document, text


def main():

    print("=" * 70)
    print("VECTOR INDEXING — DEMO CORPUS")
    print("=" * 70)

    embedder = OllamaEmbedder()

    with PgVectorStore() as store:

        # ------------------------------------------------------------
        # Determine embedding dimension using one test embedding.
        # ------------------------------------------------------------

        test_embedding = embedder.embed(
            "Knowledge Graph RAG test"
        )

        dimension = len(
            test_embedding
        )

        print(
            f"Embedding model: "
            f"{embedder.model}"
        )

        print(
            f"Embedding dimension: "
            f"{dimension}"
        )

        store.ensure_schema(
            embedding_dimension=dimension
        )

        total_chunks = 0

        for path in DEMO_FILES:

            print()
            print("-" * 70)
            print(f"DOCUMENT: {path.name}")
            print("-" * 70)

            document, text = load_document(
                path
            )

            chunks = chunk_text(
                document=document,
                text=text,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            print(
                f"Chunks: {len(chunks)}"
            )

            for chunk in chunks:

                print(
                    f"  Embedding chunk "
                    f"{chunk.chunk_index}: "
                    f"{chunk.chunk_id[:12]}..."
                )

                embedding = embedder.embed(
                    chunk.text
                )

                store.upsert_chunk(
                    chunk,
                    embedding,
                )

                total_chunks += 1

        print()
        print("=" * 70)
        print("VECTOR INDEXING COMPLETE")
        print("=" * 70)

        print(
            f"Chunks indexed: "
            f"{total_chunks}"
        )


if __name__ == "__main__":
    main()