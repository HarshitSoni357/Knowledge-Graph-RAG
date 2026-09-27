from pathlib import Path

from kg_rag.ingestion.html_loader import (
    load_html,
    extract_html_text,
)

from kg_rag.ingestion.chunker import chunk_text


SOURCE_PATH = Path(
    "data/raw/microsoft_2025_10k.html"
)


def main():

    print("Loading document...")

    document = load_html(SOURCE_PATH)

    print(f"Document ID: {document.document_id}")
    print(f"Filename:    {document.filename}")

    print("\nExtracting text...")

    text = extract_html_text(SOURCE_PATH)

    print(f"Characters: {len(text):,}")

    print("\nChunking...")

    chunks = chunk_text(
        document,
        text,
    )

    print(f"Chunks: {len(chunks)}")

    print("\n========== FIRST 10 CHUNKS ==========\n")

    for chunk in chunks[:10]:
        print(f"\n===== CHUNK {chunk.chunk_index} =====")
        print(f"Length: {len(chunk.text)}")
        print(f"Section: {chunk.section_path}")
        print()
        print(chunk.text[:1000])


if __name__ == "__main__":
    main()