from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking import Chunk


class VectorStore:
    def __init__(self, path: str, collection: str, embedder) -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self.embedder = embedder
        expected_tag = getattr(embedder, "tag", "unknown")

        existing = None
        try:
            existing = self._client.get_collection(collection)
        except Exception:
            existing = None

        if existing is not None and existing.metadata.get("embedder") != expected_tag:
            self._client.delete_collection(collection)
            existing = None

        if existing is None:
            self.collection = self._client.create_collection(
                name=collection,
                metadata={"embedder": expected_tag},
                configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            self.collection = existing

    @property
    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.collection.name
        meta = self.collection.metadata
        self._client.delete_collection(name)
        self.collection = self._client.create_collection(
            name=name, metadata=meta, configuration={"hnsw": {"space": "cosine"}}
        )

    def upsert(self, chunks: list[Chunk], batch_size: int = 128, progress=None) -> int:
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [c.text for c in batch]
            embeddings = self.embedder(documents)
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=documents,
                metadatas=[c.metadata for c in batch],
                embeddings=embeddings,
            )
            total += len(batch)
            if progress:
                progress(total, len(chunks))
        return total

    def query_text(self, text: str, top_k: int) -> list[dict[str, Any]]:
        embedding = self.embedder([text])[0]
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, max(self.count, 1)),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i in range(len(ids)):
            hits.append(
                {
                    "id": ids[i],
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else 1.0,
                }
            )
        return hits

    def peek_tables(self, limit: int = 20) -> list[str]:
        got = self.collection.get(limit=limit, include=["metadatas"])
        names = []
        for m in got.get("metadatas") or []:
            t = m.get("table")
            if t and t not in names:
                names.append(t)
        return names
