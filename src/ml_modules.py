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


class QuestionType(str, Enum):
    PLOT = "plot"
    CHARACTER = "character"
    QUOTE = "quote"
    THEME = "theme"
    CONTEXT = "context"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedQuestion:
    question: str
    qtype: QuestionType
    confidence: float
    reformulated: str


class BaseQuestionClassifier(ABC):
    @abstractmethod
    def classify(self, question: str) -> ClassifiedQuestion:
        ...


class RuleBasedClassifier(BaseQuestionClassifier):
    KEYWORDS = {
        QuestionType.CHARACTER: ["geroy", "personazh", "kto takoy", "kto takaya", "opisanie", "kharakter"],
        QuestionType.QUOTE: ["tsitat", "skazal", "govorit", "fraza", "slova"],
        QuestionType.THEME: ["tema", "smysl", "ideya", "problema", "analiz", "simvol"],
        QuestionType.PLOT: ["chto sluchilos", "pochemu", "kak", "proizoshlo", "sobytie", "syuzhet"],
        QuestionType.CONTEXT: ["istorichesk", "epokha", "kogda napisan", "avtor", "fon"],
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
        prefixes = {
            QuestionType.CHARACTER: "opisanie personazha kharakteristika ",
            QuestionType.QUOTE: "tsitata slova personazha dialog ",
            QuestionType.THEME: "tema ideya smysl analiz proizvedeniya ",
            QuestionType.PLOT: "sobytie syuzhet deystvit razvitie ",
            QuestionType.CONTEXT: "istoricheskiy kontekst epokha avtor ",
        }
        prefix = prefixes.get(qtype, "")
        return prefix + question


class FineTunedClassifier(BaseQuestionClassifier):
    EXAMPLES = {
        QuestionType.PLOT: [
            "Chto proizoshlo etoy glave?",
            "Kak razvivalsya syuzhet?",
            "Pochemu geroy prinyal eto reshenie?",
            "Chem zakonchilas kniga?",
        ],
        QuestionType.CHARACTER: [
            "Kto takoy Raskolnikov?",
            "Opishi kharakter Natasha Rostovoy",
            "Kakie geroi glavnye?",
            "Kto antagonist proizvedeniya?",
        ],
        QuestionType.QUOTE: [
            "Kto skazal «vse schastlivy odinakovo»?",
            "Naydi tsitu pro chest'",
            "Kakie slova proiznes geroy?",
            "Kakaya fraza povtoryaetsya v knige?",
        ],
        QuestionType.THEME: [
            "Kakova glavnaya tema proizvedeniya?",
            "O chem eta kniga?",
            "Kakoy smysl vlozhil avtor?",
            "Kakie problemy podnimayutsya?",
        ],
        QuestionType.CONTEXT: [
            "Kogda bylo napisano proizvedenie?",
            "V kakuyu epokhu proiskhodit deystvie?",
            "Chto izvestno ob avtore?",
            "Kakov istoricheskiy fon?",
        ],
    }

    def __init__(self, model_path: str | None = None):
        if model_path and Path(model_path).exists():
            self.model = SentenceTransformer(model_path)
            self.use_finetuned = True
        else:
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.use_finetuned = False

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
            QuestionType.CHARACTER: "opisanie personazha kharakteristika ",
            QuestionType.QUOTE: "tsitata slova personazha dialog ",
            QuestionType.THEME: "tema ideya smysl analiz ",
            QuestionType.PLOT: "sobytie syuzhet deystvie ",
            QuestionType.CONTEXT: "istoricheskiy kontekst epokha ",
        }
        return prefixes.get(qtype, "") + question


class BaseEmbedder(ABC):
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
    def __init__(self, model_path: str, base_model: str | None = None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if base_model:
            from peft import PeftModel
            base = SentenceTransformer(base_model)
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
            self.model = SentenceTransformer(model_path)

        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"  Fine-tuned embedder loaded: {model_path} (dim={self._dim})")

    @property
    def dim(self) -> int:
        return self._dim

    def encode_chunks(self, chunks: list[TextChunk], batch_size: int = 32) -> np.ndarray:
        texts = [c.text for c in chunks]
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        ...


class NoReranker(BaseReranker):
    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_path: str | None = None):
        if model_path and Path(model_path).exists():
            self.model = CrossEncoder(model_path)
            self.use_finetuned = True
            print(f"  Fine-tuned reranker loaded: {model_path}")
        else:
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


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        ...


class OllamaLLM(BaseLLM):
    SYSTEM_PROMPT = """You are a literature assistant. Answer questions using the provided book context.
Give accurate answers with text references. If the answer is not in context, say so.
Answer in Russian, clearly and structured."""

    def __init__(self, model: str = "mistral", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature

    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        prompt = f"""Book context:
{context}

Question: {question}"""

        if question_type and question_type != QuestionType.UNKNOWN:
            prompt += f"\n(Question type: {question_type.value})"

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
    def __init__(self, primary_model: str = "mistral-literary", fallback_model: str = "mistral"):
        self.primary = OllamaLLM(primary_model)
        self.fallback = OllamaLLM(fallback_model)

    def generate(self, question: str, context: str, question_type: QuestionType | None = None) -> str:
        try:
            return self.primary.generate(question, context, question_type)
        except Exception as e:
            print(f"  Primary model error: {e}, using fallback")
            return self.fallback.generate(question, context, question_type)
