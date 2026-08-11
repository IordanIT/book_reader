import random
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

MODEL_OUTPUT = Path(__file__).parent.parent.parent / "data" / "models" / "fine-tuned-reranker"


def generate_reranker_data():
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
            key_phrase = " ".join(chunk.split()[:15])
            examples.append(InputExample(texts=[key_phrase, chunk], label=1.0))

            if len(chunks) > 1:
                random_chunk = random.choice([c for c in chunks if c != chunk])
                random_phrase = " ".join(random_chunk.split()[:15])
                examples.append(InputExample(texts=[random_phrase, chunk], label=0.1))

    return examples


def train():
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    train_data = generate_reranker_data()
    if len(train_data) < 10:
        print("Not enough data. Add books to data/books/")
        return

    model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", num_labels=1)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=50,
        output_path=str(MODEL_OUTPUT),
        show_progress_bar=True,
    )

    model.save(str(MODEL_OUTPUT))


if __name__ == "__main__":
    train()
