from kg_rag.graph.neo4j_client import Neo4jClient


FIND_ENTITY = """
MATCH (e:Entity {entity_id: $entity_id})
RETURN
    e.entity_id AS entity_id,
    e.name AS name,
    e.entity_type AS entity_type
"""


GET_OWNED_ENTITIES = """
MATCH (source:Entity {entity_id: $entity_id})
      -[r:OWNS]->(target:Entity)
RETURN
    source.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY target.name
"""


GET_OWNERS = """
MATCH (target:Entity {entity_id: $entity_id})
      <-[r:OWNS]-(source:Entity)
RETURN
    source.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY source.name
"""


GET_ACQUIRED_ENTITIES = """
MATCH (source:Entity {entity_id: $entity_id})
      -[r:ACQUIRED]->(target:Entity)
RETURN
    source.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY target.name
"""


GET_COMPANY_LEADER = """
MATCH (company:Entity {entity_id: $entity_id})
      -[r:LED_BY]->(person:Entity)
RETURN
    company.name AS source_entity,
    type(r) AS relationship_type,
    person.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY person.name
"""


GET_PART_OF = """
MATCH (source:Entity {entity_id: $entity_id})
      -[r:PART_OF]->(target:Entity)
RETURN
    source.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY target.name
"""

GET_BUSINESS_SEGMENTS = """
MATCH (company:Entity {entity_id: $entity_id})
      -[r:OPERATES_IN]->(segment:Entity)
WHERE segment.entity_type = "BusinessSegment"
RETURN
    company.name AS source_entity,
    type(r) AS relationship_type,
    segment.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY segment.name
"""


GET_ALL_RELATIONSHIPS = """
MATCH (source:Entity {entity_id: $entity_id})
      -[r]-(target:Entity)
RETURN
    source.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY type(r), target.name
"""


COMPANY_THAT_ACQUIRED = """
MATCH (acquirer:Entity)
      -[r:ACQUIRED]->(target:Entity {entity_id: $entity_id})
RETURN
    acquirer.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY acquirer.name
"""


LEADER_OF_ACQUIRER = """
MATCH (acquirer:Entity)
      -[acq:ACQUIRED]->(target:Entity {entity_id: $entity_id}),
      (acquirer)-[lead:LED_BY]->(leader:Entity)
RETURN
    acquirer.name AS acquirer_entity,
    target.name AS acquired_entity,
    leader.name AS leader_entity,

    acq.source_chunk_id AS acquisition_chunk_id,
    acq.evidence AS acquisition_evidence,
    acq.confidence AS acquisition_confidence,

    lead.source_chunk_id AS leadership_chunk_id,
    lead.evidence AS leadership_evidence,
    lead.confidence AS leadership_confidence
ORDER BY leader.name
"""


ACQUIRER_OF_OWNER = """
MATCH (target:Entity {entity_id: $entity_id})
      <-[owns:OWNS]-(owner:Entity)
      <-[acq:ACQUIRED]-(acquirer:Entity)
RETURN
    target.name AS target_entity,
    owner.name AS owner_entity,
    acquirer.name AS acquirer_entity,

    owns.source_chunk_id AS ownership_chunk_id,
    owns.evidence AS ownership_evidence,
    owns.confidence AS ownership_confidence,

    acq.source_chunk_id AS acquisition_chunk_id,
    acq.evidence AS acquisition_evidence,
    acq.confidence AS acquisition_confidence
ORDER BY acquirer.name
"""


OWNER_OF_PRODUCT = """
MATCH (owner:Entity)
      -[r:OWNS]->(target:Entity {entity_id: $entity_id})
RETURN
    owner.name AS source_entity,
    type(r) AS relationship_type,
    target.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY owner.name
"""


PRODUCTS_OF_OWNER = """
MATCH (owner:Entity {entity_id: $entity_id})
      -[r:OWNS]->(product:Entity)
RETURN
    owner.name AS source_entity,
    type(r) AS relationship_type,
    product.name AS target_entity,
    r.source_chunk_id AS source_chunk_id,
    r.evidence AS evidence,
    r.confidence AS confidence
ORDER BY product.name
"""


COUNT_OWNED_ENTITIES = """
MATCH (owner:Entity {entity_id: $entity_id})
      -[r:OWNS]->(target:Entity)
WITH owner,
     collect({
         target_name: target.name,
         source_chunk_id: r.source_chunk_id,
         evidence: r.evidence,
         confidence: r.confidence
     }) AS ownership_edges
RETURN
    size(ownership_edges) AS count,
    [edge IN ownership_edges | edge.target_name] AS entities,
    ownership_edges AS edges
"""


class GraphRetriever:

    def __init__(self, client: Neo4jClient | None = None):
        self.client = client or Neo4jClient()

    def find_entity(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            FIND_ENTITY,
            {"entity_id": entity_id},
        )

    def get_owned_entities(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_OWNED_ENTITIES,
            {"entity_id": entity_id},
        )

    def get_owners(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_OWNERS,
            {"entity_id": entity_id},
        )

    def get_acquired_entities(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_ACQUIRED_ENTITIES,
            {"entity_id": entity_id},
        )

    def get_company_leader(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_COMPANY_LEADER,
            {"entity_id": entity_id},
        )

    def get_part_of(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_PART_OF,
            {"entity_id": entity_id},
        )

    def get_all_relationships(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            GET_ALL_RELATIONSHIPS,
            {"entity_id": entity_id},
        )

    def get_company_that_acquired(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            COMPANY_THAT_ACQUIRED,
            {"entity_id": entity_id},
        )

    def get_leader_of_acquirer(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            LEADER_OF_ACQUIRER,
            {"entity_id": entity_id},
        )

    def get_acquirer_of_owner(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            ACQUIRER_OF_OWNER,
            {"entity_id": entity_id},
        )

    def get_owner_of_product(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            OWNER_OF_PRODUCT,
            {"entity_id": entity_id},
        )

    def get_products_of_owner(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            PRODUCTS_OF_OWNER,
            {"entity_id": entity_id},
        )

    def count_owned_entities(self, entity_id: str) -> list[dict]:
        return self.client.execute(
            COUNT_OWNED_ENTITIES,
            {"entity_id": entity_id},
        )
    def get_business_segments(self, entity_id: str):
        return self.client.execute(
         GET_BUSINESS_SEGMENTS,
            {"entity_id": entity_id},
        )   