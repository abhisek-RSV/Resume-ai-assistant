# Resume AI Assistant

A Python RAG backend that answers questions about a professional profile, grounded in your own documents.

- **API**: FastAPI
- **Vector DB**: ChromaDB (persistent, cosine similarity)
- **Embeddings**: local Ollama (`qwen3-embedding:0.6b`), so documents never leave your machine
- **Chat LLM**: Ollama Cloud (`gpt-oss:120b` by default)

## Architecture

```
data/*.pdf|md|txt ──► loader ──► recursive chunker ──► Ollama embeddings ──► ChromaDB
                                                                              │
POST /chat ──► embed query (+ previous user turn) ──► top-k search ──► distance filter
                                                                              │
             no relevant chunks? ──► out-of-scope reply (no LLM call)         │
                                                                              ▼
                     system prompt + history + <context> ──► Ollama Cloud ──► reply
```

```
app/
  main.py              FastAPI app, lifespan (auto-ingest on startup), CORS
  api/routes.py        /health, /chat, /chat/stream, /ingest
  api/schemas.py       request/response models
  core/config.py       settings from environment / .env
  core/prompts.py      system prompt and templates
  rag/loader.py        PDF/text loading (page-level metadata)
  rag/chunker.py       RecursiveCharacterTextSplitter
  rag/embeddings.py    Ollama embedding client (batched)
  rag/vector_store.py  ChromaDB wrapper
  rag/llm.py           Ollama chat client (sync + streaming)
  rag/pipeline.py      RAGService: ingest, retrieve, answer
scripts/ingest.py      CLI to build/refresh the index
tests/                 offline tests (fake embedder/LLM, real Chroma)
```

Ingestion runs incrementally. Each file's SHA-256 hash is stored with its chunks, so unchanged files are skipped, changed files are re-embedded, and chunks from deleted files are removed.

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Pull the embedding model (local Ollama must be running)
ollama pull qwen3-embedding:0.6b

# 3. Configure
cp .env.example .env
# set OLLAMA_API_KEY (create one at https://ollama.com/settings/keys)

# 4. Add your documents
cp /path/to/profile.pdf data/

# 5. Run (ingests automatically on startup)
uv run python -m app.main
```

To use the local daemon instead of an API key, run `ollama signin` and set `OLLAMA_CHAT_HOST=http://localhost:11434` and `OLLAMA_CHAT_MODEL=gpt-oss:120b-cloud`.

## API

The request and response format matches the earlier Node backend, so existing frontends keep working.

| Method | Path           | Body / Query                                        | Response                                   |
|--------|----------------|-----------------------------------------------------|--------------------------------------------|
| GET    | `/health`      |                                                     | `{status, ragInitialized, chunks}`         |
| POST   | `/chat`        | `{message, history?: [{role, content}], include_sources?}` | `{reply, sources?}`                 |
| POST   | `/chat/stream` | same as `/chat`                                     | SSE: `data: {"token": "..."}` … `data: [DONE]` |
| POST   | `/ingest`      | `?force=true` to re-embed everything                | `{ingested, skipped, removed, total_chunks}` |

Interactive docs: `http://localhost:4000/docs`.

```bash
curl -X POST localhost:4000/chat -H 'content-type: application/json' \
  -d '{"message": "What are his main skills?", "history": []}'
```

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run python -m scripts.ingest [--force]   # rebuild index manually
```

## Tuning

| Variable        | Default | Notes |
|-----------------|---------|-------|
| `CHUNK_SIZE`    | 1000    | characters per chunk |
| `CHUNK_OVERLAP` | 200     | |
| `TOP_K`         | 5       | chunks retrieved per question |
| `MAX_DISTANCE`  | 0.75    | cosine distance cutoff; lower is stricter. On-topic questions typically score 0.5–0.7, off-topic 0.8+ |

If you change the embedding model, rebuild the index with `--force`, or delete `chroma_db/`.
