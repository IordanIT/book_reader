"""RAG-пайплайн: поиск + генерация через Ollama."""

import ollama

from book_loader import TextChunk
from embedder import Embedder
from vector_store import SearchResult, VectorStore


class RAGPipeline:
    """Полный RAG-пайплайн для вопросов по книгам."""

    SYSTEM_PROMPT = """Ты — помощник по литературе. Отвечай на вопросы пользователя,
используя ТОЛЬКО предоставленные фрагменты текста книги. Давай точные ответы со ссылками на источник.
Если в контексте нет ответа — честно скажи об этом."""

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
        """Обрабатывает вопрос и возвращает ответ с источниками."""
        query_emb = self.embedder.encode_query(question)
        results: list[SearchResult] = self.vector_store.search(query_emb, top_k=self.top_k, book_title=book_title)

        if not results:
            return "К сожалению, не удалось найти релевантные фрагменты в книге."

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

        return f"{answer}\n\n---\n📚 Источники:\n{sources}"

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            header = f"[Фрагмент {i} | {r.chunk.book_title}"
            if r.chunk.page:
                header += f", стр. {r.chunk.page}"
            header += f"]"
            parts.append(f"{header}\n{r.chunk.text}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Контекст из книги:
{context}

Вопрос пользователя: {question}

Ответь на русском языке, опираясь на контекст."""

    def _format_sources(self, results: list[SearchResult]) -> str:
        lines = []
        for r in results:
            info = f"• {r.chunk.book_title}"
            if r.chunk.page:
                info += f", страница {r.chunk.page}"
            info += f" (релевантность: {r.score:.2f})"
            preview = r.chunk.text[:100].replace("\n", " ")
            info += f"\n  «{preview}...»"
            lines.append(info)
        return "\n".join(lines)
