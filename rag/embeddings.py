from __future__ import annotations

import httpx

from .config import Settings


class OpenAICompatEmbedder:
    tag: str

    def __init__(self, base_url: str, api_key: str | None, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.tag = f"openai:{model}"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": batch},
                headers=headers,
                timeout=120.0,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding request failed HTTP {response.status_code}: {response.text[:300]}"
                )
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(item["embedding"] for item in data)
        return vectors


class DefaultEmbedder:
    tag = "default:onnx-minilm-l6-v2"

    def __init__(self) -> None:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        except ImportError as exc:
            raise RuntimeError(
                "Default embedder unavailable. Install onnxruntime "
                "(pip install chromadb[onnx]) or configure OPENAI_API_KEY."
            ) from exc
        self._fn = DefaultEmbeddingFunction()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]


def get_embedder(settings: Settings) -> object:
    provider = settings.embed_provider.lower()
    if provider == "openai":
        if not settings.openai_api_key and "api.openai.com" not in settings.openai_base_url:
            pass
        return OpenAICompatEmbedder(
            settings.openai_base_url, settings.openai_api_key, settings.embed_model
        )
    if provider == "default":
        return DefaultEmbedder()
    if settings.openai_api_key:
        return OpenAICompatEmbedder(
            settings.openai_base_url, settings.openai_api_key, settings.embed_model
        )
    return DefaultEmbedder()
