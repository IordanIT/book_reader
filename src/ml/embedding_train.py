"""Обучение/дообучение embedding модели на литературных данных.

Contrastive Learning: учим модель близко кодировать (query, релевантный_чанк)
и далеко — (query, нерелевантный_чанк).

Запуск:
    python src/ml/embedding_train.py
"""

import json
import random
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training"
MODEL_OUTPUT = Path(__file__).parent.parent.parent / "data" / "models" / "fine-tuned-embedder"


def generate_training_pairs():
    """Генерирует пары (query, positive_chunk) для обучения.

    В реальности — загрузите размеченные данные.
    Здесь: синтетическая генерация из чанков.
    """
    books_dir = Path(__file__).parent.parent.parent / "data" / "books"
    pairs = []

    for book_file in books_dir.glob("*.txt"):
        text = book_file.read_text(encoding="utf-8")
        words = text.split()
        title = book_file.stem

        # Разбиваем на чанки
        chunk_size = 500
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk) > 100:
                chunks.append(chunk)

        # Для каждого чанка генерируем "запросы"
        for chunk in chunks:
            # Положительный: запрос из ключевых слов чанка
            key_words = " ".join(chunk.split()[:20])
            pairs.append(InputExample(texts=[key_words, chunk], label=1.0))

            # Отрицательный: запрос из другого чанка
            if len(chunks) > 1:
                other = random.choice([c for c in chunks if c != chunk])
                negative_words = " ".join(other.split()[:20])
                pairs.append(InputExample(texts=[key_words, negative_words], label=0.0))

    return pairs


def train():
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("📊 Подготовка данных...")
    train_data = generate_training_pairs()
    print(f"  Сгенерировано {len(train_data)} пар")

    if len(train_data) < 10:
        print("  ⚠️ Мало данных. Добавьте книги в data/books/")
        return

    print("🧮 Загрузка базовой модели...")
    base_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(base_model)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    print("🚀 Начинаем обучение...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=50,
        output_path=str(MODEL_OUTPUT),
        show_progress_bar=True,
    )

    print(f"\n✅ Модель сохранена: {MODEL_OUTPUT}")
    print("\nИспользование в RAG:")
    print(f'  embedder = FineTunedEmbedder(model_path="{MODEL_OUTPUT}")')


if __name__ == "__main__":
    train()
