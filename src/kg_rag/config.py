from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"

    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    embedding_model: str = "embeddinggemma:300m"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kg_rag_dev_password"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "kg_rag"
    postgres_user: str = "kg_rag"
    postgres_password: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()