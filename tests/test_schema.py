import pytest

from kg_rag.extraction.schema import (
    Entity,
    Relationship,
    ExtractionResult,
)
from kg_rag.graph.ontology import (
    EntityType,
    RelationshipType,
)


def test_valid_extraction():

    result = ExtractionResult(
        entities=[
            Entity(
                name="Example Corp",
                entity_type=EntityType.COMPANY,
            ),
            Entity(
                name="Example Person",
                entity_type=EntityType.PERSON,
            ),
        ],
        relationships=[
            Relationship(
                source_entity="Example Person",
                source_type=EntityType.PERSON,
                relationship_type=RelationshipType.LED_BY,
                target_entity="Example Corp",
                target_type=EntityType.COMPANY,
                confidence=0.95,
            )
        ],
    )

    assert len(result.entities) == 2
    assert len(result.relationships) == 1


def test_invalid_relationship_is_rejected():

    with pytest.raises(ValueError):

        Relationship(
            source_entity="A",
            source_type=EntityType.COMPANY,
            relationship_type="INVENTED_RELATIONSHIP",
            target_entity="B",
            target_type=EntityType.COMPANY,
            confidence=0.9,
        )