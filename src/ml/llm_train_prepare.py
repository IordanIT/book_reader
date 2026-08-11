import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


MODELFILE = """
FROM mistral

SYSTEM """You are a literature assistant for students. Answer questions about books using the provided context.
Give accurate answers with text references. Explain complex points in simple terms. Answer in Russian."""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
"""


def generate_qa_dataset():
    books_dir = Path(__file__).parent.parent.parent / "data" / "books"
    dataset = []

    for book_file in books_dir.glob("*.txt"):
        text = book_file.read_text(encoding="utf-8")
        words = text.split()
        chunk_size = 500
        title = book_file.stem

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk) < 200:
                continue

            dataset.append({
                "instruction": f"Book: {title}. Answer the question based on the text.",
                "input": f"Text: {chunk}\nQuestion: What is this fragment about?",
                "output": f"This fragment of {title} is about: {chunk[:100]}...",
            })

            dataset.append({
                "instruction": f"Book: {title}. Answer the question based on the text.",
                "input": f"Text: {chunk}\nQuestion: What are the main ideas here?",
                "output": f"This fragment raises themes related to: {chunk[:150]}...",
            })

    return dataset


def prepare():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset = generate_qa_dataset()

    dataset_path = MODEL_DIR.parent / "training" / "literary_qa.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    modelfile_path = MODEL_DIR / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(MODELFILE)


if __name__ == "__main__":
    prepare()
