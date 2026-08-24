from __future__ import annotations

import hashlib
import json
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings


def example_id(collection: str, question: str) -> str:
    normalized = " ".join(question.lower().split())
    digest = hashlib.sha256(f"{collection}::{normalized}".encode()).hexdigest()
    return f"ex_{digest[:28]}"


class ExampleStore:
    def __init__(self, path: str, base_collection: str, embedder) -> None:
        self.name = f"{base_collection}-examples"
        client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        try:
            existing = client.get_collection(self.name)
            if existing.metadata.get("embedder") != getattr(embedder, "tag", "unknown"):
                client.delete_collection(self.name)
                existing = None
        except Exception:
            existing = None
        if existing is None:
            self.collection = client.create_collection(
                name=self.name,
                metadata={"embedder": getattr(embedder, "tag", "unknown"), "kind": "examples"},
            )
        else:
            self.collection = existing
        self.embedder = embedder

    def add(self, question: str, sql: str, notes: str = "", extra: dict[str, Any] | None = None) -> str:
        ex_id = example_id(self.name.split("-examples")[0], question)
        doc_parts = [f"Q: {question}", f"SQL:\n{sql}"]
        if notes:
            doc_parts.append(f"Notes: {notes}")
        meta = {
            "question": question,
            "sql": sql,
            "notes": notes or "",
            **(extra or {}),
        }
        meta = {k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v)) for k, v in meta.items()}
        self.collection.upsert(
            ids=[ex_id],
            documents=["\n".join(doc_parts)],
            metadatas=[meta],
            embeddings=self.embedder(["\n".join(doc_parts)]),
        )
        return ex_id

    def remove(self, question: str) -> bool:
        ex_id = example_id(self.name.split("-examples")[0], question)
        try:
            self.collection.delete(ids=[ex_id])
            return True
        except Exception:
            return False

    def search(self, question: str, k: int = 2) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        embedding = self.embedder([question])[0]
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.collection.count()),
            include=["metadatas", "distances"],
        )
        hits = []
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, meta in enumerate(metas):
            hits.append(
                {
                    "question": meta.get("question", ""),
                    "sql": meta.get("sql", ""),
                    "notes": meta.get("notes", ""),
                    "distance": dists[i] if i < len(dists) else 1.0,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
