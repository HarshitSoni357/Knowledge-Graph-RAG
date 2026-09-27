from pydantic import BaseModel, Field

from kg_rag.graph.ontology import EntityType, RelationshipType


class Entity(BaseModel):
    name: str = Field(min_length=1)
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    evidence: str = Field(min_length=1)


class Relationship(BaseModel):
    source_entity: str = Field(min_length=1)
    source_type: EntityType
    relationship_type: RelationshipType
    target_entity: str = Field(min_length=1)
    target_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1)


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)