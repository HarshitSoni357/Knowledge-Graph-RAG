from __future__ import annotations


GET_ENTITY_RELATIONSHIPS = """
MATCH (source:Entity {entity_id: $entity_id})
MATCH (source)-[r]->(target:Entity)
RETURN
    source.entity_id AS source_entity_id,
    source.name AS source_name,
    source.entity_type AS source_type,
    type(r) AS relationship_type,
    target.entity_id AS target_entity_id,
    target.name AS target_name,
    target.entity_type AS target_type,
    r.source_chunk_id AS source_chunk_id,
    r.confidence AS confidence,
    r.evidence AS evidence
"""


GET_OWNED_ENTITIES = """
MATCH (owner:Entity {entity_id: $entity_id})
MATCH (owner)-[r:OWNS]->(target:Entity)
RETURN
    owner.entity_id AS owner_entity_id,
    owner.name AS owner_name,
    target.entity_id AS target_entity_id,
    target.name AS target_name,
    target.entity_type AS target_type,
    r.source_chunk_id AS source_chunk_id,
    r.confidence AS confidence,
    r.evidence AS evidence
"""


GET_ACQUIRED_COMPANY = """
MATCH (acquirer:Entity {entity_id: $entity_id})
MATCH (acquirer)-[r:ACQUIRED]->(target:Entity)
RETURN
    acquirer.entity_id AS acquirer_entity_id,
    acquirer.name AS acquirer_name,
    target.entity_id AS target_entity_id,
    target.name AS target_name,
    target.entity_type AS target_type,
    r.source_chunk_id AS source_chunk_id,
    r.confidence AS confidence,
    r.evidence AS evidence
"""


GET_COMPANY_LEADER = """
MATCH (company:Entity {entity_id: $entity_id})
MATCH (company)-[r:LED_BY]->(person:Entity)
RETURN
    company.entity_id AS company_entity_id,
    company.name AS company_name,
    person.entity_id AS person_entity_id,
    person.name AS person_name,
    person.entity_type AS person_type,
    r.source_chunk_id AS source_chunk_id,
    r.confidence AS confidence,
    r.evidence AS evidence
"""


GET_PART_OF = """
MATCH (child:Entity {entity_id: $entity_id})
MATCH (child)-[r:PART_OF]->(parent:Entity)
RETURN
    child.entity_id AS child_entity_id,
    child.name AS child_name,
    parent.entity_id AS parent_entity_id,
    parent.name AS parent_name,
    parent.entity_type AS parent_type,
    r.source_chunk_id AS source_chunk_id,
    r.confidence AS confidence,
    r.evidence AS evidence
"""