from pydantic import BaseModel, Field


class AnswerCitation(BaseModel):
    chunk_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[AnswerCitation] = Field(default_factory=list)