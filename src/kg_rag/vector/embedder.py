from __future__ import annotations

import ollama

from kg_rag.config import settings


class OllamaEmbedder:

    def __init__(self) -> None:

        self.model = settings.embedding_model

    def embed(
        self,
        text: str,
    ) -> list[float]:

        response = ollama.embed(
            model=self.model,
            input=text,
        )

        return response["embeddings"][0]