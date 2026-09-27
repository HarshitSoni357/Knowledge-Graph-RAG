from __future__ import annotations

import re

from kg_rag.ingestion.hashing import create_chunk_id
from kg_rag.ingestion.schema import Document, DocumentChunk


# SEC filing section headings we care about.
SECTION_PATTERN = re.compile(
    r"(?im)^("
    r"PART\s+[IVX]+"
    r"|ITEM\s+\d+[A-Z]?\."
    r")\s*(.*)$"
)


def _detect_section(line: str, current_section: str) -> str:
    """
    Update the current section based on an SEC filing heading.
    """

    match = SECTION_PATTERN.match(line.strip())

    if not match:
        return current_section

    prefix = match.group(1).strip()
    title = match.group(2).strip()

    if title:
        return f"{prefix} {title}"

    return prefix


def _split_paragraphs(text: str) -> list[str]:
    """
    Split document text into meaningful paragraphs.

    We prefer paragraph boundaries over arbitrary character boundaries.
    """

    paragraphs = re.split(
        r"\n\s*\n+",
        text,
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def _split_long_paragraph(
    paragraph: str,
    chunk_size: int,
) -> list[str]:
    """
    Split an unusually large paragraph without splitting words.

    This is a fallback for tables or unusually long SEC paragraphs.
    """

    if len(paragraph) <= chunk_size:
        return [paragraph]

    words = paragraph.split()

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:

        additional_length = (
            len(word)
            if not current
            else len(word) + 1
        )

        if (
            current
            and current_length + additional_length > chunk_size
        ):
            pieces.append(" ".join(current))
            current = []
            current_length = 0

        current.append(word)
        current_length += additional_length

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_text(
    document: Document,
    text: str,
    chunk_size: int = 5000,
    overlap: int = 500,
) -> list[DocumentChunk]:
    """
    Create paragraph-aware, section-aware document chunks.

    Rules:
    - Prefer paragraph boundaries.
    - Never split words during normal chunking.
    - Preserve SEC section information.
    - Use overlap between chunks where possible.
    - Generate deterministic chunk IDs.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    if not text.strip():
        return []

    # ---------------------------------------------------------------
    # Build paragraph + section records
    # ---------------------------------------------------------------

    raw_paragraphs = _split_paragraphs(text)

    records: list[tuple[str, str]] = []

    current_section = "Unknown"

    for paragraph in raw_paragraphs:

        # Inspect individual lines for SEC headings.
        lines = paragraph.splitlines()

        for line in lines:
            detected = _detect_section(
                line,
                current_section,
            )

            if detected != current_section:
                current_section = detected

        # Handle very large paragraphs.
        pieces = _split_long_paragraph(
            paragraph,
            chunk_size,
        )

        for piece in pieces:
            records.append(
                (
                    piece,
                    current_section,
                )
            )

    # ---------------------------------------------------------------
    # Build chunks
    # ---------------------------------------------------------------

    chunks: list[DocumentChunk] = []

    current_parts: list[str] = []
    current_sections: list[str] = []
    current_length = 0

    chunk_index = 0

    def flush_chunk() -> None:
        nonlocal current_parts
        nonlocal current_sections
        nonlocal current_length
        nonlocal chunk_index

        if not current_parts:
            return

        chunk_content = "\n\n".join(
            current_parts
        ).strip()

        if not chunk_content:
            return

        # Keep the most recent/current section represented in the chunk.
        section_path = " > ".join(
            dict.fromkeys(current_sections)
        )

        chunk_id = create_chunk_id(
            document.document_id,
            chunk_index,
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document.document_id,
                chunk_index=chunk_index,
                text=chunk_content,
                section_path=section_path,
            )
        )

        chunk_index += 1

        current_parts = []
        current_sections = []
        current_length = 0

    # ---------------------------------------------------------------
    # Pack paragraphs into ~chunk_size chunks
    # ---------------------------------------------------------------

    for paragraph, section in records:

        paragraph_length = len(paragraph)

        separator_length = (
            2 if current_parts else 0
        )

        proposed_length = (
            current_length
            + separator_length
            + paragraph_length
        )

        if (
            current_parts
            and proposed_length > chunk_size
        ):
            previous_parts = current_parts.copy()
            previous_sections = current_sections.copy()

            flush_chunk()

            # -------------------------------------------------------
            # Add lightweight overlap.
            #
            # We reuse complete previous paragraphs rather than
            # slicing arbitrary characters.
            # -------------------------------------------------------

            overlap_parts: list[str] = []
            overlap_sections: list[str] = []
            overlap_length = 0

            for prev_text, prev_section in reversed(
                list(zip(previous_parts, previous_sections))
            ):

                additional = (
                    len(prev_text)
                    if not overlap_parts
                    else len(prev_text) + 2
                )

                if (
                    overlap_parts
                    and overlap_length + additional > overlap
                ):
                    break

                if (
                    not overlap_parts
                    and len(prev_text) > overlap
                ):
                    break

                overlap_parts.insert(
                    0,
                    prev_text,
                )

                overlap_sections.insert(
                    0,
                    prev_section,
                )

                overlap_length += additional

            current_parts = overlap_parts
            current_sections = overlap_sections
            current_length = overlap_length

        separator_length = (
            2 if current_parts else 0
        )

        current_parts.append(paragraph)
        current_sections.append(section)

        current_length += (
            separator_length + paragraph_length
        )

    # Flush final chunk.
    flush_chunk()

    return chunks