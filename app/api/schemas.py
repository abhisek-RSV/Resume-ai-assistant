from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    include_sources: bool = False


class Source(BaseModel):
    source: str | None
    page: int | None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    ragInitialized: bool
    chunks: int


class IngestResponse(BaseModel):
    ingested: list[str]
    skipped: list[str]
    removed: list[str]
    total_chunks: int
