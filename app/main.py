import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.pipeline import RAGService

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.ollama_api_key and "ollama.com" in settings.ollama_chat_host:
        logger.warning("OLLAMA_API_KEY is not set; chat requests to Ollama Cloud will fail.")

    app.state.rag = RAGService.from_settings(settings)
    try:
        report = app.state.rag.ingest()
        logger.info(
            "Knowledge base ready: %d chunks (ingested=%s, skipped=%s, removed=%s)",
            report.total_chunks,
            report.ingested,
            report.skipped,
            report.removed,
        )
    except Exception:
        # Keep serving so /health reports the problem instead of crash-looping
        logger.exception(
            "Initial ingestion failed; is Ollama running at %s?", settings.ollama_embed_host
        )
    yield


app = FastAPI(title="Resume AI Assistant", version="1.0.0", lifespan=lifespan)

origins = [o.strip() for o in settings.frontend_origin.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    run()
