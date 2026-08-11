import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from book_loader import TextChunk


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float


class VectorStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[TextChunk] = []
        self.book_titles: set[str] = set()

    def build(self, chunks: list[TextChunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.book_titles = {c.book_title for c in chunks}

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        print(f"  Index built: {len(chunks)} chunks, {len(self.book_titles)} books")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, book_title: str | None = None) -> list[SearchResult]:
        assert self.index is not None, "Index not built. Call build() or load() first."

        scores, indices = self.index.search(query_embedding, top_k * 3 if book_title else top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if book_title and chunk.book_title.lower() != book_title.lower():
                continue
            results.append(SearchResult(chunk=chunk, score=float(score)))
            if len(results) >= top_k:
                break

        return results

    def save(self):
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))

        metadata = {
            "dim": self.dim,
            "num_chunks": len(self.chunks),
            "book_titles": list(self.book_titles),
            "chunks": [
                {
                    "text": c.text,
                    "book_title": c.book_title,
                    "chunk_index": c.chunk_index,
                    "page": c.page,
                }
                for c in self.chunks
            ],
        }
        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"  Index saved to {self.index_dir}")

    def load(self) -> bool:
        index_path = self.index_dir / "index.faiss"
        metadata_path = self.index_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.chunks = [
            TextChunk(
                text=c["text"],
                book_title=c["book_title"],
                chunk_index=c["chunk_index"],
                page=c.get("page"),
            )
            for c in metadata["chunks"]
        ]
        self.book_titles = set(metadata["book_titles"])
        print(f"  Index loaded: {len(self.chunks)} chunks, {len(self.book_titles)} books")
        return True
