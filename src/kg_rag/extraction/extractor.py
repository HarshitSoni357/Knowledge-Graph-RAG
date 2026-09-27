from __future__ import annotations

import json
import logging

import ollama

from kg_rag.config import settings
from kg_rag.extraction.entity_validation import validate_entities
from kg_rag.extraction.schema import ExtractionResult
from kg_rag.extraction.validation import validate_relationship_type


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a conservative information extraction system for an enterprise
knowledge graph.

Your job is to extract concrete entities and explicitly stated relationships
from the supplied source text.

The extracted information will be stored in Neo4j and used for
retrieval-augmented generation.

============================================================
ONTOLOGY
============================================================

ENTITY TYPES:

Company
- A company, corporation, subsidiary, or other business organization.

Person
- A named individual.

Product
- A specifically named Microsoft product, platform, service, or offering.

BusinessSegment
- An explicitly named Microsoft business or reporting segment.

Industry
- A meaningful business or market domain in which Microsoft operates
  or competes.

Location
- A named geographic location such as a country, city, state, or region.

FinancialMetric
- A named financial measure such as revenue, operating income, margin,
  or earnings.

Event
- A named or explicitly described event such as an acquisition,
  announcement, launch, or restructuring.

============================================================
RELATIONSHIP TYPES
============================================================

ACQUIRED
SUBSIDIARY_OF
FOUNDED_BY
LED_BY
EXECUTIVE_OF
OWNS
PRODUCES
OPERATES_IN
COMPETES_WITH
PART_OF
REPORTED_METRIC
RELATED_TO
OCCURRED_IN

============================================================
ENTITY EXTRACTION RULES
============================================================

1. Be VERY CONSERVATIVE.

2. Extract only concrete entities explicitly mentioned in the source text.

3. Do NOT extract every noun, phrase, capability, technology, or concept.

4. Do NOT create entities simply because they could theoretically be useful.

5. Generic capabilities such as computing, networking, storage, software,
   hardware, services, and applications should normally NOT be extracted.

6. Do NOT use BusinessSegment for generic services or technologies.

7. Use BusinessSegment ONLY for explicitly identified Microsoft business
   or reporting segments.

8. Use Product ONLY for specifically named Microsoft products, platforms,
   services, or offerings.

9. Do NOT classify generic technology components as Products.

10. Do NOT extract generic phrases such as cloud services, AI offerings,
    cognitive services, machine learning, custom-built silicon,
    infrastructure-as-a-service, platform-as-a-service, or
    software-as-a-service unless clearly presented as a named offering.

11. Use Industry only for meaningful business or market domains.

12. Do NOT classify individual technologies, capabilities, architectures,
    service models, or technical features as Industry entities.

13. Do NOT extract extremely broad concepts such as AI, data, technology,
    software, hardware, cloud, computing, networking, storage,
    machine learning, or Internet of Things unless explicitly presented
    as a meaningful business or market domain.

14. Do not create duplicate entities with slightly different names.

15. Prefer the canonical name used in the source text.

16. Only include aliases when the source explicitly provides an alternative
    name or abbreviation.

17. If unsure whether something is a real entity, OMIT IT.

18. Precision is more important than recall.

============================================================
ENTITY EVIDENCE RULES
============================================================

1. Every entity MUST contain an "evidence" field.

2. Evidence MUST be a short, exact quote copied from the supplied source text.

3. Evidence must directly support the entity.

4. Do NOT paraphrase evidence.

5. Do NOT invent evidence.

6. If exact evidence cannot be provided, DO NOT extract the entity.

============================================================
RELATIONSHIP EXTRACTION RULES
============================================================

1. Be MORE conservative with relationships than entities.

2. A relationship must be explicitly supported by the source text.

3. Do NOT infer relationships merely because they are plausible.

4. Every relationship MUST have a short exact evidence quote.

5. Evidence must directly support the relationship.

6. If direct evidence does not exist, DO NOT create the relationship.

7. OWNS only when ownership is explicitly stated.

8. ACQUIRED only when acquisition is explicitly stated.

9. SUBSIDIARY_OF only when subsidiary status is explicit.

10. FOUNDED_BY only when founding is explicit.

11. LED_BY and EXECUTIVE_OF only when leadership/executive relationships
    are explicitly stated.

12. PRODUCES only when the source explicitly states that an entity
    produces, develops, offers, provides, manufactures, or sells the
    target in a way that clearly represents the relationship.

13. OPERATES_IN only when an organization explicitly operates in a
    geographic location or meaningful business/market domain.

14. PART_OF only when membership or organizational inclusion is explicit.

15. REPORTED_METRIC only when a company or business segment explicitly
    reports a financial metric.

16. OCCURRED_IN only when an event is explicitly associated with a location.

17. COMPETES_WITH only when competition is explicitly stated.

18. RELATED_TO should be used VERY sparingly.

19. Do NOT use RELATED_TO as a fallback.

20. A relationship endpoint may be referenced by the source text even when
    the endpoint is not explicitly introduced as an entity elsewhere in
    the same chunk. Extract the relationship if the source text directly
    supports it. Global entity resolution is handled downstream.

============================================================
QUALITY RULES
============================================================

1. Every relationship source entity MUST be represented in the entities list
   whenever it is explicitly identifiable in the source chunk.

2. Every relationship target entity MUST be represented in the entities list
   whenever it is explicitly identifiable in the source chunk.

3. Entity types MUST come from the ontology.

4. Relationship types MUST come from the ontology.

5. Confidence must be between 0.0 and 1.0.

6. Confidence reflects how directly the source supports the relationship.

7. Prefer fewer correct entities and relationships over questionable ones.

8. Never invent information from outside the supplied text.

9. Return ONLY valid JSON.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Every entity:

{
  "name": "string",
  "entity_type": "Company | Person | Product | BusinessSegment | Industry | Location | FinancialMetric | Event",
  "aliases": [],
  "evidence": "short exact quote from the source chunk"
}

Every relationship:

{
  "source_entity": "string",
  "source_type": "Company | Person | Product | BusinessSegment | Industry | Location | FinancialMetric | Event",
  "relationship_type": "ACQUIRED | SUBSIDIARY_OF | FOUNDED_BY | LED_BY | EXECUTIVE_OF | OWNS | PRODUCES | OPERATES_IN | COMPETES_WITH | PART_OF | REPORTED_METRIC | RELATED_TO | OCCURRED_IN",
  "target_entity": "string",
  "target_type": "Company | Person | Product | BusinessSegment | Industry | Location | FinancialMetric | Event",
  "confidence": 0.0,
  "evidence": "short exact quote from the source chunk"
}

Evidence MUST be copied exactly from the source text.

Do not invent information.
"""


def _clean_model_output(raw: str) -> str:
    """Remove accidental Markdown code fences from model output."""

    raw = raw.strip()

    if raw.startswith("```json"):
        raw = raw[len("```json"):].strip()
    elif raw.startswith("```"):
        raw = raw[len("```"):].strip()

    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return raw


def _build_extraction_prompt(text: str) -> str:
    """Build the initial extraction prompt."""

    return f"""
Extract entities and explicitly supported relationships from the following
source text.

IMPORTANT:

- Extract conservatively.
- Do not infer facts.
- Do not create generic entities unnecessarily.
- Every entity MUST contain exact source evidence.
- Every relationship requires direct textual evidence.
- Do not invent relationships.
- Follow the JSON schema exactly.

SOURCE TEXT:
--------------------
{text}
--------------------
"""


def _build_retry_prompt(text: str) -> str:
    """Build a smaller retry prompt after schema validation failure."""

    return f"""
Regenerate the extraction from the source text.

Return ONLY valid JSON matching the required schema.

Requirements:
- Top-level keys: entities and relationships.
- Every entity has: name, entity_type, aliases, evidence.
- Every relationship has: source_entity, source_type,
  relationship_type, target_entity, target_type, confidence, evidence.
- Use only ontology entity and relationship types.
- Evidence must be copied exactly from the source.
- Do not invent facts.
- Omit unsupported entities and relationships.

SOURCE TEXT:
--------------------
{text}
--------------------
"""


def extract_from_chunk(text: str) -> ExtractionResult:
    """
    Extract entities and explicitly supported relationships from one chunk.

    Uses the local Ollama model with structured output and one retry.
    """

    if not text.strip():
        raise ValueError("Cannot extract from an empty chunk.")

    prompt = _build_extraction_prompt(text)

    print("Calling local Qwen3...")

    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format=ExtractionResult.model_json_schema(),
        options={
            "temperature": 0,
        },
    )

    raw = _clean_model_output(
        response["message"]["content"]
    )

    # ---------------------------------------------------------------
    # First attempt
    # ---------------------------------------------------------------

    try:
        data = json.loads(raw)
        result = ExtractionResult.model_validate(data)

    except Exception as first_error:

        print(
            "\nWARNING: Qwen output failed validation."
        )

        print(
            "Retrying once with schema-correction prompt..."
        )

        retry_prompt = _build_retry_prompt(text)

        retry_response = ollama.chat(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": retry_prompt,
                },
            ],
            format=ExtractionResult.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        retry_raw = _clean_model_output(
            retry_response["message"]["content"]
        )

        try:
            retry_data = json.loads(retry_raw)

            result = ExtractionResult.model_validate(
                retry_data
            )

            print(
                "Retry succeeded: Qwen returned valid schema."
            )

        except Exception as retry_error:

            raise ValueError(
                "Qwen failed schema validation after one retry.\n\n"
                f"First validation error:\n{first_error}\n\n"
                f"Retry validation error:\n{retry_error}\n\n"
                f"Retry raw output:\n{retry_raw}"
            ) from retry_error

    # ---------------------------------------------------------------
    # Deterministic entity validation
    # ---------------------------------------------------------------

    valid_entities, rejected_entities = validate_entities(
        result.entities,
        text,
    )

    if rejected_entities:

        print(
            "\nWARNING: Rejected "
            f"{len(rejected_entities)} invalid entity/entities:"
        )

        for entity, reason in rejected_entities:
            print(
                "  - "
                f"{entity.name} "
                f"[{entity.entity_type.value}] "
                f"({reason})"
            )

    # ---------------------------------------------------------------
    # Build lookup of validated entities.
    # ---------------------------------------------------------------

    entity_keys = {
        (
            entity.name.strip().lower(),
            entity.entity_type,
        )
        for entity in valid_entities
    }

    # ---------------------------------------------------------------
    # Deterministic relationship validation
    # ---------------------------------------------------------------

    valid_relationships = []
    rejected_relationships = []

    for relationship in result.relationships:

        source_key = (
            relationship.source_entity.strip().lower(),
            relationship.source_type,
        )

        target_key = (
            relationship.target_entity.strip().lower(),
            relationship.target_type,
        )

        # Source endpoint must exist in the validated entity list.
        if source_key not in entity_keys:

            rejected_relationships.append(
                (
                    relationship,
                    "source entity missing after entity validation",
                )
            )

            continue

        # Target endpoint must exist in the validated entity list.
        if target_key not in entity_keys:

            rejected_relationships.append(
                (
                    relationship,
                    "target entity missing after entity validation",
                )
            )

            continue

        # Relationship evidence must exist.
        if not relationship.evidence.strip():

            rejected_relationships.append(
                (
                    relationship,
                    "evidence missing",
                )
            )

            continue

        # Relationship evidence must actually occur in source text.
        if relationship.evidence.lower() not in text.lower():

            rejected_relationships.append(
                (
                    relationship,
                    "relationship evidence not found in source text",
                )
            )

            continue

        # Validate entity-type / relationship compatibility.
        if not validate_relationship_type(relationship):

            rejected_relationships.append(
                (
                    relationship,
                    "incompatible entity types for relationship",
                )
            )

            continue

        # -----------------------------------------------------------
        # IMPORTANT:
        # Only valid relationships reach this point.
        # -----------------------------------------------------------

        valid_relationships.append(
            relationship
        )

    # ---------------------------------------------------------------
    # Construct final validated extraction result.
    # ---------------------------------------------------------------

    result = ExtractionResult(
        entities=valid_entities,
        relationships=valid_relationships,
    )

    # ---------------------------------------------------------------
    # Development-time diagnostics.
    # ---------------------------------------------------------------

    if rejected_relationships:

        print(
            "\nWARNING: Rejected "
            f"{len(rejected_relationships)} invalid relationship(s):"
        )

        for relationship, reason in rejected_relationships:

            print(
                "  - "
                f"{relationship.source_entity} "
                f"--[{relationship.relationship_type.value}]--> "
                f"{relationship.target_entity} "
                f"({reason})"
            )

    logger.info(
        "extraction_complete entities=%d relationships=%d",
        len(result.entities),
        len(result.relationships),
    )

    return result