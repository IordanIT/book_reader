"""Дообучение LLM для литературных QA через Ollama Modelfile.

Не требует прямого обучения Python'ом — использует fine-tuning пайплайн
через Unsloth/Axolotl или создание кастомной модели в Ollama.

Этот скрипт генерирует Modelfile и обучающие данные.

Запуск:
    python src/ml/llm_train_prepare.py
    # Затем:
    ollama create mistral-literary -f data/models/Modelfile
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


MODELFILE = """
FROM mistral

# Системный промпт специализированный на литературе
SYSTEM """Ты — литературный ассистент для школьников. Отвечай на вопросы по книгам,
используя предоставленный контекст. Давай точные ответы со ссылками на текст.
Объясняй сложные моменты простым языком. Отвечай на русском."""

# Параметры генерации
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
"""


def generate_qa_dataset():
    """Создаёт датасет Q&A из книг для fine-tuning."""
    books_dir = Path(__file__).parent.parent.parent / "data" / "books"
    dataset = []

    # Шаблоны вопросов по типам
    templates = {
        "plot": [
            ("Что произошло в этом фрагменте?", "summary"),
            ("Как развиваются события?", "events"),
        ],
        "character": [
            ("Кто этот персонаж?", "description"),
            ("Какие слова используются для описания?", "portrait"),
        ],
        "quote": [
            ("Найди цитату о {theme}", "quote_search"),
            ("Что означают эти слова?", "quote_meaning"),
        ],
    }

    for book_file in books_dir.glob("*.txt"):
        text = book_file.read_text(encoding="utf-8")
        words = text.split()
        chunk_size = 500
        title = book_file.stem

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk) < 200:
                continue

            # Генерируем Q&A пары
            dataset.append({
                "instruction": f"Книга: «{title}». Ответь на вопрос по тексту.",
                "input": f"Текст: {chunk}\nВопрос: О чём этот фрагмент?",
                "output": f"Этот фрагмент книги «{title}» рассказывает о: {chunk[:100]}...",
            })

            dataset.append({
                "instruction": f"Книга: «{title}». Ответь на вопрос по тексту.",
                "input": f"Текст: {chunk}\nВопрос: Какие главные мысли здесь?",
                "output": f"В этом фрагменте поднимаются темы, связанные с содержанием: «{chunk[:150]}...»",
            })

    return dataset


def prepare():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("📊 Генерация датасета...")
    dataset = generate_qa_dataset()
    print(f"  Создано {len(dataset)} Q&A пар")

    # Сохраняем в формате, совместимом с Axolotl/Unsloth
    dataset_path = DATA_DIR / "literary_qa.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Датасет: {dataset_path}")

    # Создаём Modelfile
    modelfile_path = MODEL_DIR / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(MODELFILE)
    print(f"  Modelfile: {modelfile_path}")

    print("\n✅ Данные подготовлены!")
    print("\n--- Вариант 1: Быстрый (Ollama create) ---")
    print("  ollama create mistral-literary -f data/models/Modelfile")
    print("\n--- Вариант 2: Полное дообучение (Unsloth) ---")
    print("  Смотри: src/ml/README_unsloth.md")
    print("\n--- Вариант 3: QLoRA дообучение ---")
    print("  Установите: pip install unsloth")
    print("  Используйте ноутбук: src/ml/finetune_unsloth.ipynb")


if __name__ == "__main__":
    prepare()
