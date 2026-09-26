from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.loader import Document


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    distance: float


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Embeddings are computed by OllamaEmbedder, so no Chroma embedding function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def count(self) -> int:
        return self._collection.count()

    def source_hash(self, source: str) -> str | None:
        """Return the content hash stored for a source file, if it has been ingested."""
        result = self._collection.get(where={"source": source}, limit=1, include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return metadatas[0].get("file_hash") if metadatas else None

    def replace_source(
        self, source: str, chunks: list[Document], embeddings: list[list[float]]
    ) -> None:
        self.delete_source(source)
        if not chunks:
            return
        file_hash = chunks[0].metadata["file_hash"]
        self._collection.add(
            ids=[f"{source}:{file_hash[:12]}:{i}" for i in range(len(chunks))],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=embeddings,
        )

    def delete_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def list_sources(self) -> set[str]:
        result = self._collection.get(include=["metadatas"])
        return {m["source"] for m in result.get("metadatas") or [] if m and "source" in m}

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(text=doc, metadata=meta or {}, distance=dist)
            for doc, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
