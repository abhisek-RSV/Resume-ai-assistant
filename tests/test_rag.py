import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api.routes import router
from app.core.config import Settings
from app.core.prompts import OUT_OF_SCOPE_REPLY
from app.rag.pipeline import RAGService
from app.rag.vector_store import ChromaVectorStore


class FakeEmbedder:
    """Deterministic bag-of-words embedding so similar texts land close together."""

    dim = 64

    def embed(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        vec = [0.0] * self.dim
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vec if any(vec) else [1.0] + [0.0] * (self.dim - 1)


class FakeLLM:
    def __init__(self):
        self.last_messages = None

    def chat(self, messages):
        self.last_messages = messages
        return "fake reply"

    def stream(self, messages):
        self.last_messages = messages
        yield from ["fake ", "reply"]


@pytest.fixture
def rag(tmp_path):
    docs = tmp_path / "data"
    docs.mkdir()
    (docs / "profile.md").write_text(
        "Skilled in Python FastAPI and distributed systems.\n\n"
        "Built a Kafka streaming pipeline processing events."
    )
    settings = Settings(
        _env_file=None,
        documents_dir=docs,
        chroma_path=tmp_path / "chroma",
        chunk_size=60,
        chunk_overlap=0,
        max_distance=2.0,
    )
    store = ChromaVectorStore(settings.chroma_path, "test")
    return RAGService(settings, FakeEmbedder(), store, FakeLLM())


def test_ingest_is_idempotent_and_tracks_changes(rag):
    first = rag.ingest()
    assert first.ingested == ["profile.md"]
    assert first.total_chunks >= 2

    second = rag.ingest()
    assert second.skipped == ["profile.md"]
    assert second.total_chunks == first.total_chunks

    (rag.settings.documents_dir / "profile.md").write_text("Only one short line now.")
    third = rag.ingest()
    assert third.ingested == ["profile.md"]
    assert third.total_chunks == 1

    (rag.settings.documents_dir / "profile.md").unlink()
    fourth = rag.ingest()
    assert fourth.removed == ["profile.md"]
    assert fourth.total_chunks == 0


def test_retrieval_ranks_relevant_chunk_first(rag):
    rag.ingest()
    top = rag.retrieve("kafka streaming pipeline", history=[])[0]
    assert "Kafka" in top.text


def test_prompt_contains_context_history_and_question(rag):
    rag.ingest()
    history = [
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello"},
    ]
    reply, _ = rag.answer("What about kafka?", history)
    assert reply == "fake reply"

    messages = rag.llm.last_messages
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:3]] == ["user", "assistant"]
    assert "<context>" in messages[-1]["content"]
    assert "What about kafka?" in messages[-1]["content"]


def test_off_topic_question_skips_llm(rag):
    rag.ingest()
    rag.settings.max_distance = 0.0
    reply, chunks = rag.answer("capital of france", [])
    assert chunks == []
    assert reply == OUT_OF_SCOPE_REPLY
    assert rag.llm.last_messages is None


def test_api_endpoints(rag):
    from fastapi import FastAPI

    rag.ingest()
    app = FastAPI()
    app.include_router(router)
    app.state.rag = rag
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["ragInitialized"] is True

    res = client.post("/chat", json={"message": "python", "history": []})
    assert res.status_code == 200
    assert res.json() == {"reply": "fake reply"}

    res = client.post("/chat", json={"message": "python", "include_sources": True})
    assert res.json()["sources"][0]["source"] == "profile.md"

    assert client.post("/chat", json={"message": ""}).status_code == 422

    stream = client.post("/chat/stream", json={"message": "python"}).text
    assert '"token": "fake "' in stream
    assert stream.rstrip().endswith("data: [DONE]")
