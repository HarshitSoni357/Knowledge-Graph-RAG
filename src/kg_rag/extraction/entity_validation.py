from __future__ import annotations

import re

from kg_rag.extraction.schema import Entity
from kg_rag.graph.ontology import EntityType


# ------------------------------------------------------------------
# Generic concepts that should normally never become graph entities.
# These are intentionally conservative and focused on the current
# Microsoft annual-report corpus.
# ------------------------------------------------------------------

GENERIC_TERMS = {
    "ai",
    "artificial intelligence",
    "machine learning",
    "internet of things",
    "iot",
    "cognitive services",
    "computing",
    "networking",
    "storage",
    "software",
    "hardware",
    "data",
    "technology",
    "cloud",
    "cloud services",
    "services",
    "applications",
    "application",
    "digital",
    "security",
    "cybersecurity",
    "analytics",
}


# ------------------------------------------------------------------
# Generic service/business models.
# ------------------------------------------------------------------

GENERIC_SERVICE_MODELS = {
    "infrastructure-as-a-service",
    "platform-as-a-service",
    "software-as-a-service",
    "iaas",
    "paas",
    "saas",
}


# ------------------------------------------------------------------
# Generic technology/component phrases.
# ------------------------------------------------------------------

GENERIC_TECHNOLOGY_TERMS = {
    "custom-built silicon",
    "chip manufacturers",
    "chip manufacturer",
    "large workloads",
    "cloud platform",
    "cloud offering",
    "cloud offerings",
    "ai offerings",
    "ai services",
    "ai service",
    "machine learning services",
}


def _normalize(value: str) -> str:
    """
    Normalize entity names for deterministic comparisons.
    """

    value = value.strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value


def evidence_exists(
    entity: Entity,
    source_text: str,
) -> bool:
    """
    Check whether the entity evidence actually occurs in the source text.
    """

    evidence = entity.evidence.strip()

    if not evidence:
        return False

    return evidence.lower() in source_text.lower()


def is_generic_term(
    entity: Entity,
) -> bool:
    """
    Identify generic concepts that should normally not become
    knowledge-graph entities.
    """

    normalized = _normalize(entity.name)

    return (
        normalized in GENERIC_TERMS
        or normalized in GENERIC_SERVICE_MODELS
        or normalized in GENERIC_TECHNOLOGY_TERMS
    )


def is_valid_entity(
    entity: Entity,
    source_text: str,
) -> tuple[bool, str]:
    """
    Deterministically validate one extracted entity.

    Returns:
        (True, "") if valid
        (False, reason) if rejected
    """

    name = entity.name.strip()

    if not name:
        return False, "empty entity name"

    # --------------------------------------------------------------
    # Evidence must exist in source.
    # --------------------------------------------------------------

    if not evidence_exists(
        entity,
        source_text,
    ):
        return (
            False,
            "entity evidence not found in source text",
        )

    # --------------------------------------------------------------
    # Reject generic concepts.
    # --------------------------------------------------------------

    if is_generic_term(entity):

        return (
            False,
            "generic technology/concept/service-model term",
        )

    # --------------------------------------------------------------
    # Additional restrictions for Industry.
    # --------------------------------------------------------------

    if entity.entity_type == EntityType.INDUSTRY:

        normalized = _normalize(name)

        if normalized in GENERIC_TERMS:
            return (
                False,
                "too generic to be an Industry entity",
            )

        if normalized in GENERIC_SERVICE_MODELS:
            return (
                False,
                "service model cannot be an Industry entity",
            )

    return True, ""


def validate_entities(
    entities: list[Entity],
    source_text: str,
) -> tuple[list[Entity], list[tuple[Entity, str]]]:
    """
    Validate all extracted entities.

    Returns:
        valid_entities
        rejected_entities
    """

    valid_entities = []
    rejected_entities = []

    seen = set()

    for entity in entities:

        is_valid, reason = is_valid_entity(
            entity,
            source_text,
        )

        if not is_valid:
            rejected_entities.append(
                (
                    entity,
                    reason,
                )
            )
            continue

        # ----------------------------------------------------------
        # Deduplicate by canonical name + entity type.
        # ----------------------------------------------------------

        key = (
            _normalize(entity.name),
            entity.entity_type,
        )

        if key in seen:
            rejected_entities.append(
                (
                    entity,
                    "duplicate entity",
                )
            )
            continue

        seen.add(key)

        valid_entities.append(entity)

    return (
        valid_entities,
        rejected_entities,
    )