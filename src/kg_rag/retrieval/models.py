from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    source_type: str
    score: float | None = None
    section_path: str | None = None
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedFact:
    source_entity: str
    relationship_type: str
    target_entity: str
    source_chunk_id: str
    evidence: str
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    route: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    facts: list[RetrievedFact] = field(default_factory=list)
    query: str = ""