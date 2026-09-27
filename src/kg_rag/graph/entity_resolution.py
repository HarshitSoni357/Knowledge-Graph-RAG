from __future__ import annotations

import re
from dataclasses import dataclass

from kg_rag.extraction.schema import Entity
from kg_rag.graph.ontology import EntityType


@dataclass(frozen=True)
class ResolvedEntity:
    """
    Canonical representation of an entity that will be stored in Neo4j.
    """

    entity_id: str
    name: str
    entity_type: EntityType
    aliases: tuple[str, ...]


def normalize_entity_name(name: str) -> str:
    """
    Normalize an entity name for deterministic comparison.

    This intentionally performs only conservative normalization.
    """

    value = name.strip().lower()

    # Normalize common punctuation.
    value = value.replace("&", " and ")

    # Remove punctuation while preserving spaces.
    value = re.sub(
        r"[^\w\s-]",
        " ",
        value,
    )

    # Normalize whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def canonical_entity_key(
    name: str,
    entity_type: EntityType,
) -> str:
    """
    Create a deterministic key for entity identity.
    """

    normalized_name = normalize_entity_name(name)

    return f"{entity_type.value}:{normalized_name}"


def create_entity_id(
    name: str,
    entity_type: EntityType,
) -> str:
    """
    Create a deterministic entity ID.

    The same canonical entity name + type will always produce
    the same ID.
    """

    import hashlib

    key = canonical_entity_key(
        name,
        entity_type,
    )

    return hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()


def resolve_entity(
    entity: Entity,
) -> ResolvedEntity:
    """
    Resolve an extracted entity into a deterministic canonical entity.
    """

    canonical_name = entity.name.strip()

    entity_id = create_entity_id(
        canonical_name,
        entity.entity_type,
    )

    aliases = tuple(
        alias.strip()
        for alias in entity.aliases
        if alias.strip()
    )

    return ResolvedEntity(
        entity_id=entity_id,
        name=canonical_name,
        entity_type=entity.entity_type,
        aliases=aliases,
    )