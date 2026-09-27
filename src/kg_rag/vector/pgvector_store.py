from __future__ import annotations

from datetime import date
from typing import Iterable

import psycopg
from pgvector.psycopg import register_vector

from kg_rag.config import settings
from kg_rag.ingestion.schema import DocumentChunk


class PgVectorStore:
    def __init__(self):
        self.connection = psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        register_vector(self.connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()

        self.connection.close()

    def ensure_schema(self, embedding_dimension: int) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section_path TEXT,
                    document_date DATE,
                    entity_ids TEXT[] NOT NULL DEFAULT '{{}}',
                    embedding VECTOR({embedding_dimension}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            # Supports upgrading an already-created table.
            cursor.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS document_date DATE;
                """
            )

            cursor.execute(
                """
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS entity_ids TEXT[] NOT NULL DEFAULT '{}';
                """
            )

        self.connection.commit()

    def upsert_chunk(
        self,
        chunk: DocumentChunk,
        embedding: list[float],
        entity_ids: list[str] | None = None,
        document_date: date | None = None,
    ) -> None:
        self._validate_embedding(embedding)

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    chunk_index,
                    text,
                    section_path,
                    document_date,
                    entity_ids,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    chunk_index = EXCLUDED.chunk_index,
                    text = EXCLUDED.text,
                    section_path = EXCLUDED.section_path,
                    document_date = EXCLUDED.document_date,
                    entity_ids = EXCLUDED.entity_ids,
                    embedding = EXCLUDED.embedding;
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.section_path,
                    document_date,
                    entity_ids or [],
                    self._to_pgvector(embedding),
                ),
            )

        self.connection.commit()

    def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        embeddings: Iterable[list[float]],
        entity_ids_by_chunk: dict[str, list[str]] | None = None,
        document_date: date | None = None,
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            entity_ids = (
                entity_ids_by_chunk.get(chunk.chunk_id, [])
                if entity_ids_by_chunk
                else []
            )

            self.upsert_chunk(
                chunk=chunk,
                embedding=embedding,
                entity_ids=entity_ids,
                document_date=document_date,
            )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        self._validate_embedding(query_embedding)

        vector = self._to_pgvector(query_embedding)

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    chunk_index,
                    text,
                    section_path,
                    document_date,
                    entity_ids,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector, vector, top_k),
            )

            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "chunk_index": row[2],
                "text": row[3],
                "section_path": row[4],
                "document_date": row[5],
                "entity_ids": row[6],
                "similarity": float(row[7]),
            }
            for row in rows
        ]

    def ensure_hnsw_index(self) -> None:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )

        self.connection.commit()

    def count(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            return cursor.fetchone()[0]

    def count_with_embeddings(self) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM document_chunks
                WHERE embedding IS NOT NULL;
                """
            )
            return cursor.fetchone()[0]

    @staticmethod
    def _validate_embedding(embedding: list[float]) -> None:
        if not embedding:
            raise ValueError("Embedding cannot be empty.")

        if not all(isinstance(value, (int, float)) for value in embedding):
            raise TypeError("Embedding must contain only numeric values.")

    @staticmethod
    def _to_pgvector(embedding: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in embedding) + "]"