Local Knowledge Graph RAG for Enterprise Data

A local Retrieval-Augmented Generation (RAG) engineering project that combines knowledge-graph retrieval with vector search. It is designed to explore questions that require entity relationships, multi-hop traversal, aggregation, or document-grounded facts.

The project demonstrates how graph retrieval can complement vector-only retrieval. It is a portfolio and architecture-validation project, not a production deployment. LLM inference and embeddings use Ollama; PostgreSQL with pgvector and Neo4j provide the vector and graph stores. No paid LLM API is required.

At a Glance

Graph retrieval: Neo4j and predefined, parameterized Cypher templates.

Vector retrieval: PostgreSQL with pgvector and an HNSW cosine index.

Local models: Qwen3 for extraction, routing, and answer generation; EmbeddingGemma for embeddings.

Grounding: Retrieved graph facts and passages retain chunk provenance; generated citations are validated against retrieved evidence.

Evaluation: A controlled 15-question benchmark compares graph-plus-vector RAG with a vector-only baseline.

Why Graph and Vector Retrieval?

Vector search is useful for finding passages that are semantically similar to a question. Relationship questions can require more than passage similarity, for example:

Who owns a particular company?

Which company acquired the company that owns X?

Who leads the company that acquired X?

Which business segment contains a particular product?

How many companies does X own?

What relationships exist between two entities?

The query router selects graph or vector retrieval. Retrieved facts and passages are assembled into context for answer generation, and citations are checked before an answer is accepted.

flowchart TD
    Q[User question] --> R[Query router]

    R -->|VECTOR| V[PostgreSQL + pgvector]
    R -->|GRAPH| G[Neo4j knowledge graph]

    V --> C[Context assembly: passages and facts]
    G --> C

    C --> L[Local LLM: Qwen3]
    L --> CV[Citation validation]
    CV --> A[Final answer]

The project is organized around five goals: knowledge-graph construction, vector indexing, query routing, answer generation with citation validation, and benchmarking vector-only against graph-plus-vector RAG. Deterministic application code controls graph structure and database queries; the LLM does not generate arbitrary Cypher.

Technology and Models

Component

Technology

Language

Python 3.11 / 3.12

LLM runtime

Ollama

LLM

qwen3:4b-instruct-2507-q4_K_M

Embedding model

embeddinggemma:300m

Knowledge graph

Neo4j 5 Community

Vector database

PostgreSQL with pgvector

Vector index

HNSW with cosine distance

API

FastAPI

Data validation

Pydantic

Document processing

BeautifulSoup, PyMuPDF, pypdf

Tests

pytest

Local infrastructure

Docker Compose

Graph queries

Predefined parameterized Cypher

The LLM is used for entity and relationship extraction, query routing, and answer generation. The embedding model creates document-chunk embeddings used for vector similarity retrieval. All model inference is intended to run locally through Ollama.

Data

The primary source document is the Microsoft FY2025 Annual Report / 10-K, represented in the project as data/raw/microsoft_2025_10k.html.

A smaller controlled corpus is used for deterministic development and benchmarking. It contains Markdown documents about Microsoft, Activision Blizzard, Azure, Microsoft leadership, and Microsoft's business segments.

The controlled corpus makes it possible to exercise graph traversal, entity resolution, routing, citations, and benchmark behavior against a bounded set of facts. The benchmark questions are in data/benchmark/questions.json.

Architecture

1. Knowledge-Graph Construction

The ingestion pipeline turns documents into graph data:

Document
  -> document hashing
  -> text extraction
  -> paragraph/section chunking
  -> LLM entity and relationship extraction
  -> Pydantic schema validation
  -> entity resolution
  -> Neo4j MERGE
  -> knowledge graph

Controlled ontology

The graph uses a fixed ontology rather than allowing the LLM to invent entity or relationship types.

Entity types:

Company

Person

Product

BusinessSegment

Industry

Location

FinancialMetric

Event

Relationship types:

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

The controlled schema is intended to make graph writes and downstream retrieval predictable and testable.

Entity resolution and idempotency

Entity resolution is deterministic. The canonical identity is based on entity type and normalized entity name:

SHA256(entity_type + normalized_name)

For example, Company:Microsoft and Company: microsoft resolve to the same canonical entity. The graph writer also checks for existing entities in Neo4j. Writes use Neo4j MERGE, so rerunning ingestion is intended not to create duplicate entities or relationships.

Graph relationships retain a source chunk ID, evidence, and confidence. This provenance allows graph facts to be traced back to the document chunk from which they were extracted.

Example relationships in the controlled corpus:

Microsoft --ACQUIRED--> Activision Blizzard

Activision Blizzard --OWNS--> Activision
Activision Blizzard --OWNS--> Blizzard
Activision Blizzard --OWNS--> King

2. Vector Retrieval

The same document chunks used for graph extraction are embedded and stored in PostgreSQL with pgvector. Chunk metadata includes:

chunk_id

document_id

chunk_index

text

section_path

document_date

entity_ids

embedding

Chunk IDs are deterministic and shared across graph and vector data, providing a common provenance identifier for graph facts and vector passages.

The vector store uses an HNSW index with cosine distance:

CREATE INDEX document_chunks_embedding_hnsw_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops);

Vector retrieval returns chunks with high semantic similarity to the question.

3. Query Routing and Graph Retrieval

The router chooses VECTOR or GRAPH using structured output validated by Pydantic. It does not generate Cypher.

Graph retrieval is intended for ownership, acquisition, leadership, business-segment membership, entity connections, relationship traversal, multi-hop questions, and aggregations. Vector retrieval is intended for definitions, descriptions, policies, strategies, capabilities, standalone factual statements, and semantic document lookup.

In short:

Vector search    -> finds relevant passages
Graph retrieval  -> traverses explicit relationships

Graph retrieval uses predefined query templates, including:

FIND_ENTITY

GET_OWNED_ENTITIES

GET_OWNERS

GET_ACQUIRED_ENTITIES

GET_COMPANY_LEADER

GET_PART_OF

GET_PART_OF_CHILDREN

GET_BUSINESS_SEGMENTS

COMPANY_THAT_ACQUIRED

LEADER_OF_ACQUIRER

ACQUIRER_OF_OWNER

COUNT_OWNED_ENTITIES

The LLM helps determine the retrieval strategy, while application code selects and parameterizes the actual graph query:

LLM output / routing decision
  -> application logic
  -> parameterized Cypher template
  -> Neo4j

Example graph questions:

Who owns Blizzard?

Which company acquired the company that owns Blizzard?

Who leads the company that acquired Activision Blizzard?

How many companies does Activision Blizzard own?

Which Microsoft business segment contains Azure?

For the question:

Which company acquired the company that owns Blizzard?

the intended relationship path is:

Blizzard <-OWNS- Activision Blizzard <-ACQUIRED- Microsoft

For:

Who leads the company that acquired Activision Blizzard?

the intended path is:

Activision Blizzard <-ACQUIRED- Microsoft -LED_BY-> Satya Nadella

These examples illustrate why explicit relationship traversal can be useful where semantic similarity alone does not guarantee the required multi-hop path.

4. Answer Generation and Citation Validation

The answer generator receives the user question, retrieved evidence, and source chunk IDs. It is instructed to answer only from retrieved context, avoid outside knowledge, cite factual claims with valid retrieved chunk IDs, and cite the evidence for each required hop in multi-hop graph answers.

If the required information is not available, the required response is:

Not available in corpus.

Citation validation checks that each citation refers to a chunk ID present in the retrieval result. For graph facts, the source is relationship.source_chunk_id; for vector results, it is retrieved_chunk.chunk_id.

If citation validation fails, the answer is regenerated once using the same retrieved context and validated again.

Retrieve evidence
  -> generate answer
  -> validate citations
       -> valid: return answer
       -> invalid: regenerate once, then validate again

5. Benchmarking

The benchmark compares a vector-only baseline with graph-plus-vector retrieval. The vector baseline intentionally bypasses the query router so the comparison focuses on retrieval strategy rather than routing behavior.

The current benchmark contains 15 controlled questions across these categories:

Category

Intended question type

Single-hop

Direct entity or relationship lookup

Two-hop

Two relationship traversals

Three-hop

Multi-step graph reasoning

Aggregation

Count or grouped relationship queries

Out-of-scope

Questions not answerable from the corpus

Examples from the benchmark include:

Who is the CEO of Microsoft?

Who owns Blizzard?

Which company acquired Activision Blizzard?

Which company acquired the company that owns Blizzard?

Who leads the company that acquired Activision Blizzard?

How many companies does Activision Blizzard own?

Which Microsoft business segment contains Azure?

What is Microsoft's stock price today?

For each question, the current result model records each approach's answer, correctness, latency, citation count, and any error. The summary reports overall and per-category accuracy, average latency, and P50/P95 latency. Latency is measured end-to-end for retrieval plus answer generation. The benchmark does not currently record a per-question retrieval-route field.

The evaluation is intended to examine whether graph retrieval provides increasing value as relationship depth and aggregation difficulty increase. It does not assume graph retrieval is universally superior: vector retrieval may perform well for some single-hop or passage-oriented questions, while explicit traversal may help with deeper relationship questions.

The current 15-question corpus is small and intended for architecture validation, not statistically strong conclusions. A larger stratified benchmark with repeated measurements would be needed for stronger evaluation.

Example expected graph results:

Question

Expected relationship or result

Who owns Blizzard?

Activision Blizzard OWNS Blizzard

Who acquired Activision Blizzard?

Microsoft ACQUIRED Activision Blizzard

Who leads the company that acquired Activision Blizzard?

Microsoft acquired Activision Blizzard; Microsoft is led by Satya Nadella

How many companies does Activision Blizzard own?

Count its OWNS relationships to Activision, Blizzard, and King

API

The FastAPI application exposes a root status endpoint, a health endpoint, and a query endpoint.

Method and path

Purpose

GET /

Returns service name, status, and version

GET /health

Returns {"status": "healthy"}

POST /query

Retrieves evidence and generates a cited answer

Example request:

{
  "question": "Who owns Blizzard?",
  "top_k": 5
}

Example response shape:

{
  "answer": "Activision Blizzard owns Blizzard.",
  "route": "GRAPH",
  "citations": [
    {
      "chunk_id": "<retrieved chunk id>",
      "claim": "Activision Blizzard owns Blizzard."
    }
  ]
}

The query's top_k defaults to 5 and must be between 1 and 20. Exact answer wording and citation IDs depend on the indexed evidence and model output.

Run Locally

Prerequisites

Python 3.11 or 3.12

Docker with the Docker Compose plugin

Ollama installed and running

Enough local memory and compute for the selected Ollama models

The project was developed with an Intel i5-10300H, 8 GB RAM, NVIDIA GTX 1650 with 4 GB VRAM, and Windows 11. Inference latency can be significant on consumer hardware.

1. Start PostgreSQL and Neo4j

docker compose up -d
docker compose ps

The configured container names are kg-rag-postgres and kg-rag-neo4j. PostgreSQL is published on port 5432; Neo4j Bolt and browser interfaces are published on ports 7687 and 7474.

The Compose file includes development credentials. Do not reuse them in a production deployment.

2. Configure Python and dependencies

Create and activate a virtual environment in PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install the required packages:

python -m pip install fastapi uvicorn pydantic pydantic-settings ollama neo4j "psycopg[binary]" pgvector beautifulsoup4 pymupdf pypdf pytest
python -m pip install -e .

3. Install the Ollama models

ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull embeddinggemma:300m
ollama list

4. Configure environment variables

Create a local .env file at the repository root:

OLLAMA_BASE_URL=http://localhost:11434

LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
EMBEDDING_MODEL=embeddinggemma:300m

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=kg_rag_dev_password

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=kg_rag
POSTGRES_USER=kg_rag
POSTGRES_PASSWORD=kg_rag_dev_password

.env should remain out of version control. Do not commit local secrets or production credentials.

5. Build the knowledge graph

The demo-corpus ingestion script processes the five Markdown documents in data/raw/demo/, chunks them, extracts entities and relationships through Ollama, validates the extraction, resolves entities, and writes graph data to Neo4j:

python scripts/ingest_document.py

The script uses 2,000-character chunks with 200-character overlap. It creates the graph constraints and uses MERGE-based writes intended to make repeated ingestion idempotent.

Additional development utilities include:

scripts/test_extraction.py
scripts/verify_connections.py
scripts/inspect_document.py
scripts/test_graph_writer.py
scripts/test_neo4j_client.py
scripts/ingest_one_chunk.py

These utilities may require the corresponding local services, models, environment variables, or source files.

6. Build the vector index

The vector indexing script chunks the same five demo documents, embeds each chunk with Ollama, and upserts it to PostgreSQL:

python scripts/index_vectors.py

The script determines the embedding dimension from a test embedding and ensures the database schema and HNSW index.

7. Run tests

python -m pytest

Some tests exercise components that connect to local services or call Ollama, so the required services, environment variables, and models may need to be available.

8. Start and call the API

Start the API:

uvicorn kg_rag.api:app --host 127.0.0.1 --port 8001

Open Swagger UI:

http://127.0.0.1:8001/docs

Example query using PowerShell:

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8001/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"Who acquired Activision Blizzard?"}'

9. Run the benchmark

After the services, models, and indexes are ready:

python -m kg_rag.benchmark.runner

The benchmark reads data/benchmark/questions.json and runs both the vector-only baseline and graph-plus-vector path.

Project Structure

Knowledge Graph Rag/
|
|-- data/
|   |-- benchmark/
|   |   `-- questions.json
|   |-- processed/
|   `-- raw/
|       |-- demo/
|       |   |-- activision_blizzard.md
|       |   |-- azure.md
|       |   |-- business_segments.md
|       |   |-- leadership.md
|       |   `-- microsoft.md
|       `-- microsoft_2025_10k.html
|
|-- scripts/
|   |-- index_vectors.py
|   |-- ingest_document.py
|   |-- ingest_one_chunk.py
|   |-- inspect_document.py
|   |-- test_extraction.py
|   |-- test_graph_writer.py
|   |-- test_neo4j_client.py
|   `-- verify_connections.py
|
|-- src/
|   `-- kg_rag/
|       |-- answer/
|       |-- api/
|       |-- benchmark/
|       |-- extraction/
|       |-- graph/
|       |-- ingestion/
|       |-- retrieval/
|       |-- router/
|       `-- vector/
|
|-- tests/
|-- docker-compose.yml
|-- pyproject.toml
`-- README.md

Key Engineering Decisions

Controlled ontology: The LLM does not create arbitrary graph schemas. A fixed ontology makes graph retrieval predictable, Cypher templates manageable, entity resolution deterministic, and benchmarking reproducible.

Deterministic entity IDs: Entity type and normalized name define identity, reducing duplicate logical entities.

Deterministic chunk IDs: Document ID and chunk index produce stable chunk identifiers shared by documents, graph facts, vector records, citations, and benchmark results.

No LLM-generated Cypher: The model selects a retrieval strategy; application code selects the query template. This avoids executing arbitrary model-generated graph queries.

Provenance first: Graph relationships retain source_chunk_id, evidence, and confidence so graph-derived facts can be traced to their source.

Citation validation: Generated citations are checked against retrieved evidence before an answer is accepted, with one regeneration attempt after invalid citations.

Current Status

The project implements an end-to-end local path for document extraction, graph ingestion, vector indexing, query routing, retrieval, cited answer generation, citation validation, API serving, and benchmark execution.

Implemented capabilities include:

Local LLM inference and local embeddings

Controlled ontology, entity extraction, relationship extraction, and entity resolution

Idempotent Neo4j ingestion

pgvector indexing and HNSW vector search

Query routing and parameterized graph retrieval

Multi-hop graph retrieval and aggregation queries

Citation-aware answer generation and citation validation

A vector-only baseline and controlled benchmark framework

FastAPI serving

An idempotency check is included to verify that repeated ingestion does not create duplicate entities or relationships.

Known Limitations

Portfolio scope: This is an engineering and architecture-validation project, not a production deployment.

Small benchmark: The current benchmark has 15 controlled questions. A larger, stratified 50–100-question set with repeated measurements would support stronger evaluation.

Local hardware: The project was designed around consumer hardware (Intel i5-10300H, 8 GB RAM, NVIDIA GTX 1650 4 GB, Windows 11). LLM inference can be slow.

Entity resolution: Resolution is deterministic and conservative. A production system could add alias tables, cross-document identity resolution, embedding-assisted matching, and human review for ambiguous entities.

Extraction quality: Small local LLMs can return invalid or incomplete structured extraction. The pipeline uses structured output, Pydantic validation, relationship validation, evidence validation, retry logic, and downstream entity resolution, but these do not guarantee perfect extraction.

Dependency metadata: Runtime dependencies are currently installed explicitly as part of the local setup.

Future Improvements

Area

Possible improvements

Retrieval

Hybrid BM25 plus vector retrieval; Reciprocal Rank Fusion; graph-aware vector filtering; entity-aware retrieval; community detection; graph summarization

Graph

Better cross-document entity resolution; alias management; temporal relationships; relationship-confidence calibration; graph versioning

LLM

Model routing across local models; a smaller classification model; a larger complex-reasoning model; structured reasoning traces; better extraction retries

Evaluation

50–100+ questions; Recall@K; Precision@K; citation accuracy; faithfulness; answer completeness; latency percentiles; cost per query; graph-ingestion cost; retrieval-level evaluation independent of generation

Productionization

Async API; background ingestion workers; observability and metrics; request tracing; caching; authentication; rate limiting; containerized application deployment; CI/CD

Design Philosophy

Let the LLM reason about language. Use it for extraction, classification, routing, and answer generation.

Let deterministic systems handle structure. Use application code and databases for identity, graph schema, Cypher, storage, provenance, and validation.

Keep answers grounded. Preserve the path from document to chunk to retrieved evidence to answer claim.

Evaluate retrieval strategies separately. Vector and graph retrieval address different question types; the benchmark compares them rather than assuming one approach is universally better.

End-to-End Example

Question:

Who leads the company that acquired Activision Blizzard?

The router selects GRAPH.

Entity resolution identifies Activision Blizzard.

Graph traversal follows:

Activision Blizzard <-ACQUIRED- Microsoft -LED_BY-> Satya Nadella

Context assembly retains the source chunk IDs for the graph facts.

The LLM answers using retrieved evidence only.

Citation validation checks citation IDs against that evidence.

The API returns an answer and its supporting citations.

Example response shape:

{
  "answer": "Satya Nadella leads Microsoft, which acquired Activision Blizzard.",
  "route": "GRAPH",
  "citations": [
    {
      "chunk_id": "<retrieved chunk id>",
      "claim": "Microsoft acquired Activision Blizzard."
    },
    {
      "chunk_id": "<retrieved chunk id>",
      "claim": "Microsoft is led by Satya Nadella."
    }
  ]
}

What This Project Demonstrates

The project demonstrates practical work with:

RAG architecture

Knowledge graphs

Graph-based retrieval

Vector databases

pgvector

Neo4j

Structured LLM output

Entity resolution

Ontology design

Multi-hop reasoning

Query routing

Retrieval provenance

Citation validation

Benchmark design

FastAPI

Docker

Local LLM deployment

Retrieval-strategy evaluation

Its central architectural idea is to use different retrieval mechanisms for different question types rather than treating every RAG problem as semantic similarity search.

License and Distribution

This is a personal engineering and portfolio project. No license is specified here; add an appropriate license file before publicly distributing or permitting reuse of the code.

Before publishing, keep .env and credentials out of version control. Also review the repository's .gitignore rules and confirm that the intended demo corpus and source documents are included or excluded appropriately.