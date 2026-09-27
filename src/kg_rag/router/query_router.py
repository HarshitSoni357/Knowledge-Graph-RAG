import json
import logging

import ollama

from kg_rag.config import settings
from kg_rag.router.schema import RouteDecision


logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """
You are a query router for a Knowledge Graph RAG system.

Choose exactly one route:

GRAPH
VECTOR

Choose GRAPH whenever the answer depends on traversing, querying,
counting, comparing, or reasoning over explicit relationships between
entities in the knowledge graph.

GRAPH examples:
- Who owns X?
- What does X own?
- Who acquired X?
- What did X acquire?
- Who leads X?
- Who is the CEO of X?
- Who leads the company that acquired X?
- Which company acquired the owner of X?
- Which business segment contains X?
- Which business area does X belong to?
- What entities are part of X?
- What entities are connected to X?
- List the relationships of X.
- How many companies does X own?
- How many entities are related to X?
- Compare entities based on their graph relationships.
- Any question requiring one or more relationship hops.
- What are the business segments/areas of X?
- What are X's major business areas?
- List the business segments of X.
- Which segments does X operate?

VECTOR examples:
- What is X?
- Describe X.
- What does X do?
- What is Microsoft's cloud strategy?
- What policy does the company have?
- What was stated about X?
- A standalone factual statement or definition contained in a passage.
- What does X provide?
- What capabilities does X have?
- Describe the strategy, purpose, or characteristics of X.

IMPORTANT ROUTING RULES:

1. Choose GRAPH whenever the answer requires retrieving a set of
   entities connected to another entity through an explicit relationship.

2. Choose GRAPH for:
   - ownership
   - acquisition
   - leadership
   - containment / membership
   - business segments / business areas
   - products belonging to a segment
   - entities belonging to a company
   - relationship lists
   - counts and aggregations
   - comparisons based on relationships
   - multi-hop connections

3. Even if the same information appears verbatim in a document,
   choose GRAPH when the question asks for entities connected through
   an explicit relationship.

4. Choose VECTOR when the question can be answered from a standalone
   document passage without traversing relationships in the graph.

5. Do not generate Cypher.

6. Do not invent facts.

7. Return only the structured RouteDecision.
"""


class QueryRouter:
    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model

    def route(self, question: str) -> RouteDecision:
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": ROUTER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            format=RouteDecision.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        raw_content = response["message"]["content"]

        try:
            payload = json.loads(raw_content)
            decision = RouteDecision.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "query_route_invalid_response question=%s response=%s",
                question,
                raw_content,
            )
            raise ValueError(
                f"Invalid router response from LLM: {raw_content}"
            ) from exc

        logger.info(
            "query_route route=%s reason=%s question=%s",
            decision.route.value,
            decision.reason,
            question,
        )

        return decision