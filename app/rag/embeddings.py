import ollama


class OllamaEmbedder:
    def __init__(self, host: str, model: str, batch_size: int = 32):
        self._client = ollama.Client(host=host)
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embed(model=self.model, input=batch)
            vectors.extend(list(v) for v in response.embeddings)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
