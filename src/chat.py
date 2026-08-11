import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ollama

from advanced_rag import AdvancedRAGPipeline
from config import CONFIG, INDEX_DIR


def check_ollama():
    try:
        models = ollama.list()
        available = [m["name"].split(":")[0] for m in models["models"]]
        if CONFIG["llm"]["model"] not in available:
            print(f"Model {CONFIG['llm']['model']} not found. Run: ollama pull {CONFIG['llm']['model']}")
            return False
        return True
    except Exception:
        print("Ollama not running. Install: https://ollama.ai")
        return False


def print_header(store):
    print(f"\nBookTalk v2\n")
    print(f"Books ({len(store.book_titles)}):")
    for title in sorted(store.book_titles):
        print(f"  - {title}")

    c = CONFIG["classifier"]
    e = CONFIG["embedder"]
    r = CONFIG["reranker"]
    l = CONFIG["llm"]

    clf_type = "fine-tuned" if c.get("use_finetuned") else "rules"
    emb_type = "fine-tuned" if e.get("use_finetuned") else e["model_name"].split("/")[-1]
    rnk_type = "fine-tuned" if r.get("use_finetuned") else ("cross-encoder" if r.get("use_default") else "off")

    print(f"\nModules: classifier={clf_type}, embedder={emb_type}, reranker={rnk_type}, llm={l.get('model')}")
    print(f"\nCommands: /book <name>, /all, /debug, /help, /quit\n")


def main():
    from vector_store import VectorStore

    store = VectorStore(dim=384, index_dir=str(INDEX_DIR))

    if not store.load():
        print("Index not found. Run: python src/index_books.py")
        return

    if not check_ollama():
        return

    pipeline = AdvancedRAGPipeline.from_config(store, CONFIG)

    current_book = None
    show_debug = False
    print_header(store)

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            break

        if user_input.lower() == "/help":
            print_header(store)
            continue

        if user_input.lower() == "/debug":
            show_debug = not show_debug
            continue

        if user_input.lower().startswith("/book "):
            book_name = user_input[6:].strip()
            matches = [t for t in store.book_titles if book_name.lower() in t.lower()]
            if matches:
                current_book = matches[0]
            else:
                print(f"'{book_name}' not found")
            continue

        if user_input.lower() == "/all":
            current_book = None
            continue

        result = pipeline.query(user_input, book_title=current_book)

        print(f"\n{result['answer']}\n")

        if result["sources"]:
            for s in result["sources"]:
                loc = f" | p.{s['page']}" if s['page'] else ""
                print(f"  {s['book']}{loc} [{s['score']:.3f}]")

        if show_debug:
            d = result["debug"]
            print(f"[debug] {d['question_type']} | conf={d['confidence']:.2f} | {d['initial_results']}->{d['reranked_results']}")

        print()


if __name__ == "__main__":
    main()
