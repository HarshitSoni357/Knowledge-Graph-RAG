from ollama import Client
from neo4j import GraphDatabase
import psycopg

from kg_rag.config import settings


def verify_ollama():
    print("\n[1/3] Testing Ollama...")

    client = Client(host=settings.ollama_base_url)

    response = client.chat(
        model=settings.llm_model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OLLAMA_OK",
            }
        ],
    )

    print("Response:", response["message"]["content"])


def verify_neo4j():
    print("\n[2/3] Testing Neo4j...")

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    with driver.session() as session:
        result = session.run(
            "RETURN 'NEO4J_OK' AS status"
        )
        print("Response:", result.single()["status"])

    driver.close()


def verify_postgres():
    print("\n[3/3] Testing PostgreSQL + pgvector...")

    with psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    ) as conn:

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector';
                """
            )

            row = cur.fetchone()

            if row:
                print("pgvector version:", row[0])
            else:
                raise RuntimeError(
                    "pgvector extension is not installed."
                )


if __name__ == "__main__":
    verify_ollama()
    verify_neo4j()
    verify_postgres()

    print("\n==============================")
    print("ALL CONNECTIONS VERIFIED")
    print("==============================")