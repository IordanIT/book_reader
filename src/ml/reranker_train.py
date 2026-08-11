"""Обучение cross-encoder re-ranker для переранжирования результатов.

Input: (query, chunk) → score релевантности
Output: перезаупорядоченный список чанков

Запуск:
    python src/ml/reranker_train.py
"""

import random
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training"
MODEL_OUTPUT = Path(__file__).parent.parent.parent / "data" / "models" / "fine-tuned-reranker"


def generate_reranker_data():
    """Генерирует данные для re-ranker: (query, chunk, score)."""
    books_dir = Path(__file__).parent.parent.parent / "data" / "books"
    examples = []

    for book_file in books_dir.glob("*.txt"):
        text = book_file.read_text(encoding="utf-8")
        words = text.split()
        chunk_size = 500
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk) > 100:
                chunks.append(chunk)

        for chunk in chunks:
            # Высокий score для семантически связанных
            key_phrase = " ".join(chunk.split()[:15])
            examples.append(InputExample(texts=[key_phrase, chunk], label=1.0))

            # Низкий score для случайных
            if len(chunks) > 1:
                random_chunk = random.choice([c for c in chunks if c != chunk])
                random_phrase = " ".join(random_chunk.split()[:15])
                examples.append(InputExample(texts=[random_phrase, chunk], label=0.1))

    return examples


def train():
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("📊 Подготовка данных...")
    train_data = generate_reranker_data()
    print(f"  Сгенерировано {len(train_data)} примеров")

    if len(train_data) < 10:
        print("  ⚠️ Мало данных. Добавьте книги в data/books/")
        return

    print("🧮 Загрузка базового cross-encoder...")
    base_model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    model = CrossEncoder(base_model, num_labels=1)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)

    print("🚀 Начинаем обучение...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=50,
        output_path=str(MODEL_OUTPUT),
        show_progress_bar=True,
    )

    model.save(str(MODEL_OUTPUT))
    print(f"\n✅ Re-ranker сохранён: {MODEL_OUTPUT}")
    print("\nИспользование в RAG:")
    print(f'  reranker = CrossEncoderReranker(model_path="{MODEL_OUTPUT}")')


if __name__ == "__main__":
    train()
