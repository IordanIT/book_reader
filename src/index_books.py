import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from book_loader import BookLoader
from embedder import Embedder
from vector_store import VectorStore

BOOKS_DIR = Path(__file__).parent.parent / "data" / "books"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"


def main():
    loader = BookLoader(str(BOOKS_DIR), chunk_size=500, chunk_overlap=50)
    chunks = loader.load_all_books()

    if not chunks:
        print("No books found. Add .txt files to data/books/")
        return

    embedder = Embedder()
    embeddings = embedder.encode_chunks(chunks)

    store = VectorStore(dim=embedder.dim, index_dir=str(INDEX_DIR))
    store.build(chunks, embeddings)
    store.save()


if __name__ == "__main__":
    main()
