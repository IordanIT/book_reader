from pathlib import Path

INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
BOOKS_DIR = Path(__file__).parent.parent / "data" / "books"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

CONFIG = {
    "classifier": {
        "use_rules": True,
    },
    "embedder": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "reranker": {
        "use_default": True,
    },
    "llm": {
        "model": "mistral",
        "temperature": 0.3,
    },
    "top_k": 5,
}
