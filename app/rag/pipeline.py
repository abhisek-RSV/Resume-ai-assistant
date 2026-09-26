import logging
from collections.abc import Iterator
from dataclasses import dataclass

from app.core.config import Settings
from app.core.prompts import OUT_OF_SCOPE_REPLY, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.rag.chunker import chunk_documents
from app.rag.embeddings import OllamaEmbedder
from app.rag.llm import OllamaChatLLM
from app.rag.loader import discover_files, file_sha256, load_file
from app.rag.vector_store import ChromaVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    ingested: list[str]
    skipped: list[str]
    removed: list[str]
    total_chunks: int


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embedder: OllamaEmbedder,
        store: ChromaVectorStore,
        llm: OllamaChatLLM,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGService":
        return cls(
            settings=settings,
            embedder=OllamaEmbedder(settings.ollama_embed_host, settings.ollama_embed_model),
            store=ChromaVectorStore(settings.chroma_path, settings.chroma_collection),
            llm=OllamaChatLLM(
                host=settings.ollama_chat_host,
                model=settings.ollama_chat_model,
                api_key=settings.ollama_api_key,
                temperature=settings.chat_temperature,
                top_p=settings.chat_top_p,
                max_tokens=settings.chat_max_tokens,
            ),
        )

    # ---------- Ingestion ----------

    def ingest(self, force: bool = False) -> IngestReport:
        """Sync the documents directory into Chroma. Unchanged files are skipped by hash."""
        files = discover_files(self.settings.documents_dir)
        ingested, skipped = [], []

        for path in files:
            file_hash = file_sha256(path)
            if not force and self.store.source_hash(path.name) == file_hash:
                skipped.append(path.name)
                continue

            chunks = chunk_documents(
                load_file(path), self.settings.chunk_size, self.settings.chunk_overlap
            )
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({"file_hash": file_hash, "chunk_index": i})

            embeddings = self.embedder.embed([c.text for c in chunks]) if chunks else []
            self.store.replace_source(path.name, chunks, embeddings)
            ingested.append(path.name)
            logger.info("Ingested %s (%d chunks)", path.name, len(chunks))

        # Drop chunks whose source file was deleted from the documents directory
        current = {p.name for p in files}
        removed = sorted(self.store.list_sources() - current)
        for source in removed:
            self.store.delete_source(source)
            logger.info("Removed stale source %s", source)

        return IngestReport(ingested, skipped, removed, self.store.count())

    # ---------- Retrieval + generation ----------

    def retrieve(self, question: str, history: list[dict]) -> list[RetrievedChunk]:
        # Include the previous user turn so follow-ups like "tell me more" still retrieve
        # the right context
        previous_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
        )
        query = f"{previous_user}\n{question}".strip()
        results = self.store.query(self.embedder.embed_query(query), self.settings.top_k)
        return [r for r in results if r.distance <= self.settings.max_distance]

    def build_messages(
        self, question: str, history: list[dict], chunks: list[RetrievedChunk]
    ) -> list[dict]:
        context = "\n\n".join(
            f"[{i}] (source: {c.metadata.get('source')}, page {c.metadata.get('page')})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in history[-self.settings.max_history_messages :]:
            role = "user" if m.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": str(m.get("content", ""))})
        messages.append(
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(context=context, question=question),
            }
        )
        return messages

    def answer(self, question: str, history: list[dict]) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retrieve(question, history)
        # Nothing relevant in the knowledge base: refuse without spending an LLM call
        if not chunks:
            return OUT_OF_SCOPE_REPLY, []
        reply = self.llm.chat(self.build_messages(question, history, chunks))
        return reply, chunks

    def stream_answer(self, question: str, history: list[dict]) -> Iterator[str]:
        chunks = self.retrieve(question, history)
        if not chunks:
            yield OUT_OF_SCOPE_REPLY
            return
        yield from self.llm.stream(self.build_messages(question, history, chunks))
