"""ML-модули: дообученные модели для RAG-пайплайна.

Архитектура расширений:
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline v2                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────┐ │
│  │ Question │   │Embedding │   │ Re-ranker │   │  LLM   │ │
│  │Classifier│   │  Model   │   │(cross-    │   │(custom │ │
│  │(fine-    │   │(fine-    │   │ encoder)  │   │ fine-  │ │
│  │ tuned)   │   │ tuned)   │   │           │   │ tuned) │ │
│  └──────────┘   └──────────┘   └───────────┘   └────────┘ │
│       ↓              ↓              ↓              ↓        │
│  тип вопроса   векторизация   переранжирование   генерация  │
└─────────────────────────────────────────────────────────────┘

Как дообучить и подключить каждую модель — см. README_finetune.md
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import ollama
from sentence_transformers import CrossEncoder, SentenceTransformer

from book_loader import TextChunk
from vector_store import SearchResult


# ──────────────────────────────────────────────────────────────
# 1. КЛАССИФИКАТОР ВОПРОСОВ
# ──────────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    PLOT = "plot"            # сюжет, что произошло
    CHARACTER = "character"  # герои, характеристики
    QUOTE = "quote"          # цитаты, кто сказал
    THEME = "theme"          # темы, анализ, смысл
    CONTEXT = "context"      # исторический контекст
    UNKNOWN = "unknown"


@dataclass
class ClassifiedQuestion:
    question: str
    qtype: QuestionType
    confidence: float
    reformulated: str  # переформулировка для лучшего поиска


class BaseQuestionClassifier(ABC):
    """Интерфейс классификатора вопросов."""

    @abstractmethod
    def classify(self, question: str) -> ClassifiedQuestion:
        ...


class RuleBasedClassifier(BaseQuestionClassifier):
    """Простой классификатор на правилах (без ML)."""

    KEYWORDS = {
        QuestionType.CHARACTER: ["герой", "персонаж", "кто такой", "кто такая", "описание", "характер"],
        QuestionType.QUOTE: ["цитат", "сказал", "говорить", "фраза", "слова"],
        QuestionType.THEME: ["тема", "смысл", "идея", "проблема", "анализ", "символ"],
        QuestionType.PLOT: ["что случилось", "почему", "как", "произошло", "событие", "сюжет"],
        QuestionType.CONTEXT: ["историческ", "эпоха", "когда написан", "автор", "фон"],
    }

    def classify(self, question: str) -> ClassifiedQuestion:
        q_lower = question.lower()
        scores: dict[QuestionType, float] = {}

        for qtype, keywords in self.KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q_lower) / len(keywords)
            scores[qtype] = score

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score == 0:
            best_type = QuestionType.UNKNOWN
            best_score = 0.0

        reformulated = self._reformulate(question, best_type)

        return ClassifiedQuestion(
            question=question,
            qtype=best_type,
            confidence=best_score,
            reformulated=reformulated,
        )

    def _reformulate(self, question: str, qtype: QuestionType) -> str:
        """Добавляет контекст для лучшего поиска по эмбеддингам."""
        prefixes = {
            QuestionType.CHARACTER: "описание персонажа характеристика ",
            QuestionType.QUOTE: "цитата слова персонажа диалог ",
            QuestionType.THEME: "тема идея смысл анализ произведения ",
            QuestionType.PLOT: "событие сюжет действие развитие ",
            QuestionType.CONTEXT: "исторический контекст эпоха автор ",
        }
        prefix = prefixes.get(qtype, "")
        return prefix + question


class FineTunedClassifier(BaseQuestionClassifier):
    """Дообученный классификатор на базе sentence-transformers.

    Обучение: см. src/ml/classifier_train.py
    Использует близость к эмбеддингам эталонных вопросов каждого класса.
    """

    EXAMPLES = {
        QuestionType.PLOT: [
            "Что произошло в этой главе?",
            "Как развивался сюжет?",
            "Почему герой принял это решение?",
            "Чем закончилась книга?",
        ],
        QuestionType.CHARACTER: [
            "Кто такой Раскольников?",
            "Опиши характер Наташи Ростовой",
            "Какие герои главные?",
            "Кто антагонист произведения?",
        ],
        QuestionType.QUOTE: [
            "Кто сказал «все счастливы одинаково»?",
            "Найди цитату про честь",
            "Какие слова произнёс герой?",
            "Какая фраза повторяется в книге?",
        ],
        QuestionType.THEME: [
            "Какова главная тема произведения?",
            "О чем эта книга?",
            "Какой смысл вложил автор?",
            "Какие проблемы поднимаются?",
        ],
        QuestionType.CONTEXT: [
            "Когда было написано произведение?",
            "В какую эпоху происходит действие?",
            "Что известно об авторе?",
            "Каков исторический фон?",
        ],
    }

    def __init__(self, model_path: str | None = None):
        """Если model_path указан — загружает дообученную модель."""
        if model_path and Path(model_path).exists():
            self.model = SentenceTransformer(model_path)
            self.use_finetuned = True
        else:
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.use_finetuned = False

        # Предвычисляем эмбеддинги эталонных вопросов
        self.example_embeddings: dict[QuestionType, np.ndarray] = {}
        self._precompute_examples()

    def _precompute_examples(self):
        for qtype, examples in self.EXAMPLES.items():
            embs = self.model.encode(examples, normalize_embeddings=True)
            self.example_embeddings[qtype] = embs

    def classify(self, question: str) -> ClassifiedQuestion:
        q_emb = self.model.encode([question], normalize_embeddings=True)[0]

        best_type = QuestionType.UNKNOWN
        best_score = -1.0

        for qtype, embs in self.example_embeddings.items():
            # Средний косинус с эталонами класса
            similarities = embs @ q_emb
            score = float(np.mean(similarities))

            if score > best_score:
                best_score = score
                best_type = qtype

        reformulated = self._reformulate(question, best_type)

        return ClassifiedQuestion(
            question=question,
            qtype=best_type,
            confidence=min(best_score, 1.0),
            reformulated=reformulated,
        )

    def _reformulate(self, question: str, qtype: QuestionType) -> str:
        prefixes = {
            QuestionType.CHARACTER: "описание персонажа характеристика ",
            QuestionType.QUOTE: "цитата слова персонажа диалог ",
            QuestionType.THEME: "тема идея смысл анализ ",
            QuestionType.PLOT: "событие сюжет действие ",
            QuestionType.CONTEXT: "исторический контекст эпоха ",
        }
        return prefixes.get(qtype, "") + question


# ──────────────────────────────────────────────────────────────
# 2. EMBEDDING МОДЕЛЬ (fine-tuned)
# ──────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Интерфейс для эмбеддинг-моделей."""

    @abstractmethod
    def encode_chunks(self, chunks: list[TextChunk], batch_size: int = 32) -> np.ndarray:
        ...

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class DefaultEmbedder(BaseEmbedder):
    """Стандартная многоязычная модель."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode_chunks(self, chunks: list[TextChunk], batch_size: int = 32) -> np.ndarray:
        texts = [c.text for c in chunks]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)


class FineTunedEmbedder(BaseEmbedder):
    """Дообученная на литературных текстах модель эмбеддингов.

    Обучение: src/ml/embedding_train.py
    Использует contrastive learning на парах (query, relevant_chunk).
    """

    def __init__(self, model_path: str, base_model: str | None = None):
        """
        Args:
            model_path: путь к дообученной модели
            base_model: базовая модель (если использовался LoRA/Adapter)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        if base_model:
            # Если использовался PEFT/LoRA — загружаем базу + адаптер
            from peft import PeftModel
            base = SentenceTransformer(base_model)
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            # Полностью дообученная модель
            self.model = SentenceTransformer(model_path)

        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"  ✅ Fine-tuned embedder загружен: {model_path} (dim={self._dim})")

    @property
    def dim(self) -> int:
        return self._dim

    def encode_chunks(self, chunks: list[TextChunk], batch_size: int = 32) -> np.ndarray:
        texts = [c.text for c in chunks]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)


# ──────────────────────────────────────────────────────────────
# 3. RE-RANKER (кросс-энкодер)
# ──────────────────────────────────────────────────────────────

class BaseReranker(ABC):
    """Интерфейс для переранжирования."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        ...


class NoReranker(BaseReranker):
    """Без переранжирования (пропускаем шаг)."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Кросс-энкодер для переранжирования результатов.

    Обучение: src/ml/reranker_train.py
    Берёт query и каждый чанк, вычисляет точечную релевантность.
    """

    def __init__(self, model_path: str | None = None):
        if model_path and Path(model_path).exists():
            self.model = CrossEncoder(model_path)
            self.use_finetuned = True
            print(f"  ✅ Fine-tuned reranker загружен: {model_path}")
        else:
            # Мультиязычный кросс-энкодер по умолчанию
            self.model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
            self.use_finetuned = False

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return results

        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)

        for r, score in zip(results, scores):
            r.score = float(score)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ──────────────────────────────────────────────────────────────
# 4. LLM ГЕНЕРАЦИЯ (fine-tuned)
# ──────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Интерфейс для генерации ответов."""

    @abstractmethod
    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        ...


class OllamaLLM(BaseLLM):
    """Генерация через Ollama (можно указать свою дообученную модель)."""

    SYSTEM_PROMPT = """Ты — литературный ассистент. Отвай на вопросы, используя предоставленный контекст из книги.
Давай точные ответы со ссылками на текст. Если ответа в контексте нет — скажи честно.
Отвечай на русском языке, доступно и структурированно."""

    def __init__(self, model: str = "mistral", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        prompt = f"""Контекст из книги:
{context}

Вопрос: {question}"""

        if question_type and question_type != QuestionType.UNKNOWN:
            prompt += f"\n(Тип вопроса: {question_type.value})"

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]


class OllamaLLMWithFallback(BaseLLM):
    """С фоллбэком на другую модель при неудачных ответах."""

    def __init__(self, primary_model: str = "mistral-literary", fallback_model: str = "mistral"):
        self.primary = OllamaLLM(primary_model)
        self.fallback = OllamaLLM(fallback_model)

    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        try:
            return self.primary.generate(question, context, question_type)
        except Exception as e:
            print(f"  ⚠️ Primary model error: {e}, using fallback")
            return self.fallback.generate(question, context, question_type)
