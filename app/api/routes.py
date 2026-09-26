import json
import logging
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    Source,
)
from app.core.prompts import FALLBACK_REPLY
from app.rag.pipeline import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_rag(request: Request) -> RAGService:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG service is not initialized")
    return rag


Rag = Annotated[RAGService, Depends(get_rag)]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    rag: RAGService | None = getattr(request.app.state, "rag", None)
    chunks = rag.store.count() if rag else 0
    return HealthResponse(status="ok", ragInitialized=chunks > 0, chunks=chunks)


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(body: ChatRequest, rag: Rag) -> ChatResponse:
    history = [m.model_dump() for m in body.history]
    try:
        reply, chunks = rag.answer(body.message, history)
    except Exception:
        logger.exception("Error while answering /chat")
        return ChatResponse(reply=FALLBACK_REPLY)

    sources = None
    if body.include_sources:
        sources = [
            Source(
                source=c.metadata.get("source"),
                page=c.metadata.get("page"),
                score=round(1 - c.distance, 4),
                snippet=c.text[:200],
            )
            for c in chunks
        ]
    return ChatResponse(reply=reply, sources=sources)


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, rag: Rag) -> StreamingResponse:
    """Server-Sent Events: `data: {"token": "..."}` per chunk, then `data: [DONE]`."""
    history = [m.model_dump() for m in body.history]

    def events() -> Iterator[str]:
        try:
            for token in rag.stream_answer(body.message, history):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception:
            logger.exception("Error while streaming /chat/stream")
            yield f"data: {json.dumps({'error': FALLBACK_REPLY})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/ingest", response_model=IngestResponse)
def ingest(rag: Rag, force: bool = False) -> IngestResponse:
    report = rag.ingest(force=force)
    return IngestResponse(**report.__dict__)
