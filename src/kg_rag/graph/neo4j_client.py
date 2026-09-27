from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import Driver, GraphDatabase

from kg_rag.config import settings


class Neo4jClient:
    """
    Small wrapper around the Neo4j Python driver.

    Responsible only for:
    - creating the driver
    - verifying connectivity
    - providing sessions
    - executing Cypher queries
    - closing the driver
    """

    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(
                settings.neo4j_user,
                settings.neo4j_password,
            ),
        )

    def verify_connectivity(self) -> None:
        """
        Verify that Neo4j is reachable using the configured credentials.
        """

        self._driver.verify_connectivity()

    @contextmanager
    def session(self) -> Iterator[Any]:
        """
        Provide a Neo4j session and ensure it is closed afterwards.
        """

        session = self._driver.session()

        try:
            yield session
        finally:
            session.close()

    def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return records as dictionaries.
        """

        parameters = parameters or {}

        with self.session() as session:

            result = session.run(
                query,
                parameters,
            )

            return [
                record.data()
                for record in result
            ]

    def close(self) -> None:
        """
        Close the Neo4j driver.
        """

        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        self.verify_connectivity()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()