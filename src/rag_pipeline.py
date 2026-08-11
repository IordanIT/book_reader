import ollama

from book_loader import TextChunk
from embedder import Embedder
from vector_store import SearchResult, VectorStore


class RAGPipeline:
    SYSTEM_PROMPT = """You are a literature assistant. Answer user questions using ONLY the provided book text fragments.
Give accurate answers with references to the source. If the answer is not in the context, say so."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        model: str = "mistral",
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.model = model
        self.top_k = top_k

    def query(self, question: str, book_title: str | None = None) -> str:
        query_emb = self.embedder.encode_query(question)
        results: list[SearchResult] = self.vector_store.search(query_emb, top_k=self.top_k, book_title=book_title)

        if not results:
            return "Could not find relevant fragments in the book."

        context = self._format_context(results)
        prompt = self._build_prompt(question, context)

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )

        answer = response["message"]["content"]
        sources = self._format_sources(results)

        return f"{answer}\n\n---\nSources:\n{sources}"

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            header = f"[Fragment {i} | {r.chunk.book_title}"
            if r.chunk.page:
                header += f", p. {r.chunk.page}"
            header += f"]"
            parts.append(f"{header}\n{r.chunk.text}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Book context:
{context}

User question: {question}

Answer based on the context provided."""

    def _format_sources(self, results: list[SearchResult]) -> str:
        lines = []
        for r in results:
            info = f"* {r.chunk.book_title}"
            if r.chunk.page:
                info += f", page {r.chunk.page}"
            info += f" (relevance: {r.score:.2f})"
            preview = r.chunk.text[:100].replace("\n", " ")
            info += f"\n  \"{preview}...\""
            lines.append(info)
        return "\n".join(lines)
