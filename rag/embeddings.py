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


class E5MultilingualEmbedder:
    """Cross-lingual local embedder: intfloat/multilingual-e5-small via ONNX.

    Uses HuggingFace Hub to download the ONNX-exported model + tokenizer on
    first use (cached under ~/.cache/huggingface), then runs it directly with
    `onnxruntime` + the `tokenizers` Rust tokenizer — no torch needed.

    e5 models require a task prefix: "query: "/"passage: " (or an equivalent).
    Queries and documents are prefixed so the query/document direction is
    disambiguated, which is what makes cross-lingual retrieval work.

    Tag is stable so the Chroma store's embedder-tag guard auto-rebuilds the
    collection on first switch (re-index is automatic).
    """

    MODEL = "Xenova/multilingual-e5-small"
    ONNX_FILE = "model.onnx"
    tag = f"e5:{MODEL.lower()}"

    def __init__(self, model: str | None = None, cache_dir: str | None = None) -> None:
        import onnxruntime as ort

        self.model_name = model or self.MODEL
        self.cache_dir = cache_dir
        self._session: ort.InferenceSession | None = None
        self._tokenizer = None
        self._tokenizer_loaded = False

    # lazily build the model once; all batches share the same session
    def _ensure_loaded(self) -> None:
        if self._session is not None and self._tokenizer_loaded:
            return
        from pathlib import Path

        import numpy as np  # noqa: F401  (numpy is a hard dep of onnxruntime)
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer

        cache = self.cache_dir
        # download (or load from cache) model repo
        repo = snapshot_download(
            repo_id=self.model_name,
            allow_patterns=[
                "onnx/model.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
            ],
            local_dir=cache or None,
        )

        onnx_path = None
        for cand in (
            Path(repo) / self.ONNX_FILE,
            Path(repo) / "onnx" / self.ONNX_FILE,
        ):
            if cand.exists():
                onnx_path = cand
                break
        if onnx_path is None:
            raise RuntimeError(
                f"Could not find {self.ONNX_FILE} in model {self.model_name} "
                f"(downloaded to {repo})"
            )

        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tok_path = Path(repo) / "tokenizer.json"
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer_loaded = True

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        self._ensure_loaded()
        if self._tokenizer is None or self._session is None:  # pragma: no cover
            raise RuntimeError("e5 embedder failed to initialise")

        # e5 prefix: documents/table chunks are "passage:", search queries are "query:"
        prefixed = [
            (("query: " if t.startswith("query: ") or t.startswith("passage: ") else "passage: ") + t)
            for t in texts
        ]
        enc = self._tokenizer.encode_batch(prefixed)

        max_len = max(len(x.ids) for x in enc) if enc else 1
        max_len = min(max_len, 512)
        ids = np.zeros((len(prefixed), max_len), dtype=np.int64)
        mask = np.zeros((len(prefixed), max_len), dtype=np.int64)
        for i, x in enumerate(enc):
            n = min(len(x.ids), max_len)
            ids[i, :n] = x.ids[:n]
            mask[i, :n] = x.attention_mask[:n]

        result = self._session.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids, dtype=np.int64),
        })
        # output is [batch, seq, hidden] -> mean-pool over masked tokens
        token_emb = result[0]
        mask3 = mask[..., None].astype(token_emb.dtype)
        summed = (token_emb * mask3).sum(axis=1)
        counts = mask.sum(axis=1, keepdims=True).astype(token_emb.dtype)
        counts = np.clip(counts, 1, None)
        pooled = summed / counts
        # L2-normalise
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-8, None)
        return [[float(x) for x in row] for row in pooled]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)


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
    # auto + no API key: use a local model. If EMBED_MODEL names a
    # multilingual/e5 model, use the ONNX cross-lingual embedder; otherwise
    # fall back to the bundled MiniLM (English-only).
    model = (settings.embed_model or "").strip().lower() or "default"
    if model in ("default", "", "minilm", "minilm-l6-v2", "text-embedding-3-small"):
        # text-embedding-3-small is an API model, but without a key we can't
        # call it — fall back to the local MiniLM for read-only offline use.
        return DefaultEmbedder()
    return E5MultilingualEmbedder(model=settings.embed_model)
