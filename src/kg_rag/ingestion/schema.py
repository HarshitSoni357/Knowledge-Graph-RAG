from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    filename: str
    source: str


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str

    chunk_index: int

    text: str = Field(min_length=1)

    section_path: str | None = None