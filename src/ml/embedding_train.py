import random
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

MODEL_OUTPUT = Path(__file__).parent.parent.parent / "data" / "models" / "fine-tuned-embedder"


def generate_training_pairs():
    books_dir = Path(__file__).parent.parent.parent / "data" / "books"
    pairs = []

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
            key_words = " ".join(chunk.split()[:20])
            pairs.append(InputExample(texts=[key_words, chunk], label=1.0))

            if len(chunks) > 1:
                other = random.choice([c for c in chunks if c != chunk])
                negative_words = " ".join(other.split()[:20])
                pairs.append(InputExample(texts=[key_words, negative_words], label=0.0))

    return pairs


def train():
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    train_data = generate_training_pairs()
    if len(train_data) < 10:
        print("Not enough data. Add books to data/books/")
        return

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=50,
        output_path=str(MODEL_OUTPUT),
        show_progress_bar=True,
    )


if __name__ == "__main__":
    train()
