from collections.abc import Iterator

import ollama


class OllamaChatLLM:
    def __init__(
        self,
        host: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 600,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = ollama.Client(host=host, headers=headers)
        self.model = model
        self._options = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens}

    def chat(self, messages: list[dict]) -> str:
        response = self._client.chat(model=self.model, messages=messages, options=self._options)
        return response.message.content or ""

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for part in self._client.chat(
            model=self.model, messages=messages, options=self._options, stream=True
        ):
            if part.message.content:
                yield part.message.content
