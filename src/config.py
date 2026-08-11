"""Конфигурация RAG-пайплайна — выбираем какие ML-модели использовать."""

from pathlib import Path

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
BOOKS_DIR = Path(__file__).parent.parent / "data" / "books"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

# ──────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ — переключайте модули здесь
# ──────────────────────────────────────────────────────────────

CONFIG = {
    # 1. Классификатор вопросов
    # "use_rules" — на правилах, "use_finetuned" — дообученная ML
    "classifier": {
        "use_rules": True,
        # "use_finetuned": True,
        # "model_path": str(MODELS_DIR / "fine-tuned-classifier"),
    },

    # 2. Embedding модель
    # "model_name" — стандартная, "use_finetuned" — дообученная
    "embedder": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # "use_finetuned": True,
        # "model_path": str(MODELS_DIR / "fine-tuned-embedder"),
        # "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # для LoRA
    },

    # 3. Re-ranker
    # "use_default" — кросс-энкодер, "use_finetuned" — дообученный
    "reranker": {
        "use_default": True,
        # "use_finetuned": True,
        # "model_path": str(MODELS_DIR / "fine-tuned-reranker"),
    },

    # 4. LLM для генерации
    # "model" — название модели в Ollama
    "llm": {
        "model": "mistral",
        "temperature": 0.3,
        # "use_fallback": True,
        # "primary_model": "mistral-literary",  # дообученная версия
        # "fallback_model": "mistral",
    },

    # Количество чанков для контекста
    "top_k": 5,
}
