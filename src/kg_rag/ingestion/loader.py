from pathlib import Path

import fitz  # PyMuPDF

from kg_rag.ingestion.schema import Document
from kg_rag.ingestion.hashing import create_document_id


def load_pdf(path: str | Path) -> Document:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {path}"
        )

    content = path.read_bytes()

    if not content.startswith(b"%PDF"):
        raise ValueError(
            f"{path} does not appear to be a valid PDF."
        )

    return Document(
        document_id=create_document_id(content),
        filename=path.name,
        source=str(path),
    )


def extract_pdf_text(path: str | Path) -> str:
    path = Path(path)

    document = fitz.open(path)

    try:
        text_parts: list[str] = []

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text("text")

            if text.strip():
                text_parts.append(
                    f"[PAGE {page_number}]\n{text}"
                )

        text = "\n\n".join(text_parts)

        if not text.strip():
            raise ValueError(
                f"No text could be extracted from {path}"
            )

        return text

    finally:
        document.close()