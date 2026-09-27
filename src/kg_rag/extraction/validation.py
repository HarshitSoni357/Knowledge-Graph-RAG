from __future__ import annotations

from kg_rag.extraction.schema import Relationship
from kg_rag.graph.ontology import (
    EntityType,
    RelationshipType,
)


# ---------------------------------------------------------------------------
# Relationship compatibility rules
# ---------------------------------------------------------------------------
#
# These rules are intentionally conservative.
#
# None of these rules claim that a relationship MUST exist.
# They only prevent obviously nonsensical combinations from entering
# the graph.
# ---------------------------------------------------------------------------

ALLOWED_RELATIONSHIP_TYPES = {
    RelationshipType.ACQUIRED: {
        (EntityType.COMPANY, EntityType.COMPANY),
    },

    RelationshipType.SUBSIDIARY_OF: {
        (EntityType.COMPANY, EntityType.COMPANY),
    },

    RelationshipType.FOUNDED_BY: {
        (EntityType.COMPANY, EntityType.PERSON),
    },

    RelationshipType.LED_BY: {
        (EntityType.COMPANY, EntityType.PERSON),
        (EntityType.BUSINESS_SEGMENT, EntityType.PERSON),
    },

    RelationshipType.EXECUTIVE_OF: {
        (EntityType.PERSON, EntityType.COMPANY),
        (EntityType.PERSON, EntityType.BUSINESS_SEGMENT),
    },

    RelationshipType.OWNS: {
        (EntityType.COMPANY, EntityType.COMPANY),
        (EntityType.COMPANY, EntityType.PRODUCT),
    },

    RelationshipType.PRODUCES: {
        (EntityType.COMPANY, EntityType.PRODUCT),
        (EntityType.BUSINESS_SEGMENT, EntityType.PRODUCT),
    },

    RelationshipType.OPERATES_IN: {
        (EntityType.COMPANY, EntityType.INDUSTRY),
        (EntityType.COMPANY, EntityType.LOCATION),
        (EntityType.BUSINESS_SEGMENT, EntityType.INDUSTRY),
        (EntityType.BUSINESS_SEGMENT, EntityType.LOCATION),
    },

    RelationshipType.COMPETES_WITH: {
        (EntityType.COMPANY, EntityType.COMPANY),
        (EntityType.PRODUCT, EntityType.PRODUCT),
    },

    RelationshipType.PART_OF: {
        (EntityType.COMPANY, EntityType.COMPANY),
        (EntityType.PRODUCT, EntityType.BUSINESS_SEGMENT),
        (EntityType.BUSINESS_SEGMENT, EntityType.COMPANY),
        (EntityType.PRODUCT, EntityType.COMPANY),
    },

    RelationshipType.REPORTED_METRIC: {
        (EntityType.COMPANY, EntityType.FINANCIAL_METRIC),
        (EntityType.BUSINESS_SEGMENT, EntityType.FINANCIAL_METRIC),
    },

    RelationshipType.RELATED_TO: {
        # RELATED_TO is deliberately broad, but still restricted to
        # meaningful graph entities rather than arbitrary combinations.
        (EntityType.COMPANY, EntityType.PRODUCT),
        (EntityType.COMPANY, EntityType.INDUSTRY),
        (EntityType.COMPANY, EntityType.COMPANY),
        (EntityType.PRODUCT, EntityType.INDUSTRY),
        (EntityType.PRODUCT, EntityType.PRODUCT),
        (EntityType.BUSINESS_SEGMENT, EntityType.PRODUCT),
        (EntityType.BUSINESS_SEGMENT, EntityType.INDUSTRY),
    },

    RelationshipType.OCCURRED_IN: {
        (EntityType.EVENT, EntityType.LOCATION),
    },
}


def validate_relationship_type(
    relationship: Relationship,
) -> bool:
    """
    Check whether the entity-type combination is compatible with the
    relationship type.
    """

    allowed_pairs = ALLOWED_RELATIONSHIP_TYPES.get(
        relationship.relationship_type,
        set(),
    )

    pair = (
        relationship.source_type,
        relationship.target_type,
    )

    return pair in allowed_pairs