from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning

from kg_rag.ingestion.hashing import create_document_id
from kg_rag.ingestion.schema import Document


# SEC inline-XBRL HTML can trigger this warning even though we intentionally
# parse it as HTML.
warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

ITEM_1_PATTERN = re.compile(
    r"\bITEM\s+1\.\s*BUSINESS\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _remove_non_content_elements(soup: BeautifulSoup) -> None:
    """
    Remove HTML elements that are not useful for RAG ingestion.

    We intentionally do NOT remove every inline-XBRL tag because some
    ix:* elements wrap actual visible financial values.
    """

    # Definitely non-content elements.
    for tag in soup.find_all(
        ["script", "style", "noscript", "template", "svg"]
    ):
        tag.decompose()

    # HTML comments.
    for comment in soup.find_all(
        string=lambda text: isinstance(text, Comment)
    ):
        comment.extract()

    # Hidden SEC/XBRL metadata.
    for tag in soup.find_all(True):

        attrs = getattr(tag, "attrs", None)

        if not isinstance(attrs, dict):
            continue

        style = str(attrs.get("style", "")).lower()
        aria_hidden = str(attrs.get("aria-hidden", "")).lower()

        if (
            "display:none" in style
            or "display: none" in style
            or "visibility:hidden" in style
            or "visibility: hidden" in style
            or "hidden" in attrs
            or aria_hidden == "true"
        ):
            tag.decompose()

    # Remove inline-XBRL metadata containers.
    for tag in soup.find_all(True):

        tag_name = str(
            getattr(tag, "name", "")
        ).lower()

        if tag_name in {
            "ix:header",
            "ix:hidden",
            "ix:resources",
        }:
            tag.decompose()


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize extracted HTML text while preserving line structure."""

    # Non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove zero-width characters.
    text = re.sub(
        r"[\u200b\u200c\u200d\ufeff]",
        "",
        text,
    )

    # Normalize horizontal whitespace.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Normalize whitespace around newlines.
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------------------------
# Filing boundary detection
# ---------------------------------------------------------------------------

def _find_actual_item_1(text: str) -> int:
    """
    Find the actual Item 1 Business section.

    SEC filings usually contain Item 1 Business in the Table of Contents
    before the real filing section.

    We use the final occurrence as the filing boundary.
    """

    matches = list(
        ITEM_1_PATTERN.finditer(text)
    )

    if not matches:
        raise ValueError(
            "Could not find 'Item 1. Business' in the SEC filing."
        )

    return matches[-1].start()


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------

def _extract_clean_text(html: str) -> str:
    """Convert SEC HTML into clean filing text."""

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    _remove_non_content_elements(soup)

    body = soup.body if soup.body else soup

    text = body.get_text("\n")

    text = _normalize_text(text)

    if not text:
        raise ValueError(
            "No visible text could be extracted from HTML."
        )

    # Remove Table of Contents / metadata before actual Item 1.
    start = _find_actual_item_1(text)

    filing_text = text[start:].strip()

    if len(filing_text) < 1000:
        raise ValueError(
            "Extracted filing text is suspiciously short "
            f"({len(filing_text)} characters)."
        )

    return filing_text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_html_text(path: str | Path) -> str:
    """
    Extract clean filing text from an SEC HTML file.

    This function returns the raw cleaned text.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"HTML file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but got: {path}"
        )

    html = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if not html.strip():
        raise ValueError(
            f"HTML file is empty: {path}"
        )

    return _extract_clean_text(html)


def load_html(path: str | Path) -> Document:
    """
    Load an SEC HTML filing into the project's Document model.

    This is the function consumed by inspect_document.py and the
    downstream ingestion pipeline.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"HTML file not found: {path}"
        )

    # Read bytes first so the document ID is based on the actual source file.
    content = path.read_bytes()

    document_id = create_document_id(content)

    text = _extract_clean_text(
        content.decode(
            "utf-8",
            errors="replace",
        )
    )

    return Document(
        document_id=document_id,
        filename=path.name,
        source=str(path),
    )