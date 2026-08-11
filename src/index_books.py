"""Индексация книг — запускать при добавлении новых книг."""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent))

from book_loader import BookLoader
from embedder import Embedder
from vector_store import VectorStore

BOOKS_DIR = Path(__file__).parent.parent / "data" / "books"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


def main():
    print("📖 Загрузка книг...")
    loader = BookLoader(str(BOOKS_DIR), chunk_size=500, chunk_overlap=50)
    chunks = loader.load_all_books()

    if not chunks:
        print("  ⚠️  Книги не найдены. Добавьте .txt файлы в data/books/")
        return

    print(f"\n🧮 Векторизация ({len(chunks)} чанков)...")
    embedder = Embedder()
    embeddings = embedder.encode_chunks(chunks)

    print("\n💾 Построение индекса...")
    store = VectorStore(dim=embedder.dim, index_dir=str(INDEX_DIR))
    store.build(chunks, embeddings)
    store.save()

    print("\n✅ Готово! Теперь можно запустить: python src/chat.py")


if __name__ == "__main__":
    main()
