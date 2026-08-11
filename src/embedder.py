import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from book_loader import TextChunk


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_chunks(self, chunks: list[TextChunk], batch_size: int = 32) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)

        for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding"):
            batch = texts[i : i + batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings[i : i + len(batch)] = batch_emb

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)
