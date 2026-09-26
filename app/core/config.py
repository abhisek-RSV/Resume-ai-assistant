from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Server
    host: str = "0.0.0.0"
    port: int = 4000
    frontend_origin: str = "*"
    log_level: str = "INFO"

    # Chat LLM — Ollama Cloud (https://ollama.com). Point at http://localhost:11434
    # and use a "-cloud" model tag instead if you prefer routing through the local daemon.
    ollama_chat_host: str = "https://ollama.com"
    ollama_api_key: str = ""
    ollama_chat_model: str = "gpt-oss:120b"
    chat_temperature: float = 0.5
    chat_top_p: float = 0.9
    chat_max_tokens: int = 600

    # Embeddings — local Ollama, so documents never leave the machine
    ollama_embed_host: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding:0.6b"

    # Vector store
    chroma_path: Path = BASE_DIR / "chroma_db"
    chroma_collection: str = "resume_knowledge"

    # Ingestion / retrieval
    documents_dir: Path = BASE_DIR / "data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    # Cosine distance (0 = identical, 2 = opposite); chunks above this are dropped
    max_distance: float = 0.75
    max_history_messages: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
