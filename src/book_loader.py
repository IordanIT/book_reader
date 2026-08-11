from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class TextChunk:
    text: str
    book_title: str
    chunk_index: int
    page: int | None = None


class BookLoader:
    def __init__(self, books_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.books_dir = Path(books_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_all_books(self) -> List[TextChunk]:
        all_chunks = []
        for file_path in self.books_dir.iterdir():
            if file_path.suffix.lower() in (".txt", ".pdf"):
                chunks = self._load_file(file_path)
                all_chunks.extend(chunks)
                print(f"  Loaded: {file_path.name} - {len(chunks)} chunks")
        return all_chunks

    def _load_file(self, file_path: Path) -> List[TextChunk]:
        title = file_path.stem
        if file_path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path, title)
        return self._load_txt(file_path, title)

    def _load_txt(self, file_path: Path, title: str) -> List[TextChunk]:
        text = file_path.read_text(encoding="utf-8")
        return self._split_text(text, title)

    def _load_pdf(self, file_path: Path, title: str) -> List[TextChunk]:
        reader = PdfReader(str(file_path))
        chunks = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                page_chunks = self._split_text(page_text, title, page=i + 1)
                chunks.extend(page_chunks)
        return chunks

    def _split_text(self, text: str, title: str, page: int | None = None) -> List[TextChunk]:
        words = text.split()
        chunks = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append(TextChunk(
                    text=chunk_text,
                    book_title=title,
                    chunk_index=idx,
                    page=page,
                ))
            idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks
