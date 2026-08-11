"""Чат-интерфейс с расширенным RAG-пайплайном."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import ollama

from advanced_rag import AdvancedRAGPipeline
from config import CONFIG, INDEX_DIR, OLLAMA_MODEL


def check_ollama():
    try:
        models = ollama.list()
        available = [m["name"].split(":")[0] for m in models["models"]]
        if CONFIG["llm"]["model"] not in available:
            model_name = CONFIG["llm"]["model"]
            print(f"  ⚠️  Модель {model_name} не найдена. Запустите: ollama pull {model_name}")
            return False
        return True
    except Exception:
        print("  ❌ Ollama не запущена. Установите: https://ollama.ai")
        return False


def print_header(store):
    print("\n" + "═" * 65)
    print("  📚 BookTalk v2 — Поговори с книгой (ML-powered RAG)")
    print("═" * 65)
    print(f"\n  Доступные книги ({len(store.book_titles)}):")
    for title in sorted(store.book_titles):
        print(f"    • {title}")

    print(f"\n  ML-модули:")
    c = CONFIG["classifier"]
    e = CONFIG["embedder"]
    r = CONFIG["reranker"]
    l = CONFIG["llm"]

    clf_type = "fine-tuned" if c.get("use_finetuned") else "правила"
    emb_type = "fine-tuned" if e.get("use_finetuned") else e["model_name"].split("/")[-1]
    rnk_type = "fine-tuned" if r.get("use_finetuned") else ("кросс-энкодер" if r.get("use_default") else "выкл")
    llm_type = l.get("model", "mistral")

    print(f"    🧠 Классификатор: {clf_type}")
    print(f"    🔢 Embeddings:    {emb_type}")
    print(f"    📊 Re-ranker:     {rnk_type}")
    print(f"    💬 LLM:           {llm_type}")

    print(f"\n  Команды:")
    print(f"    /book <назв>  — выбрать книгу")
    print(f"    /all          — поиск по всем")
    print(f"    /debug        — показать ML-диагностику")
    print(f"    /help         — помощь")
    print(f"    /quit         — выход")
    print()


def main():
    from vector_store import VectorStore

    print("🔍 Загрузка...")
    store = VectorStore(dim=384, index_dir=str(INDEX_DIR))

    if not store.load():
        print("  ❌ Индекс не найден. Запустите: python src/index_books.py")
        return

    if not check_ollama():
        return

    pipeline = AdvancedRAGPipeline.from_config(store, CONFIG)

    current_book = None
    show_debug = False
    print_header(store)

    while True:
        try:
            user_input = input("🔎 Вопрос> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 До встречи!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("👋 До встречи!")
            break

        if user_input.lower() == "/help":
            print_header(store)
            continue

        if user_input.lower() == "/debug":
            show_debug = not show_debug
            print(f"  {'✅' if show_debug else '❌'} Диагностика {'включена' if show_debug else 'выключена'}")
            continue

        if user_input.lower().startswith("/book "):
            book_name = user_input[6:].strip()
            matches = [t for t in store.book_titles if book_name.lower() in t.lower()]
            if matches:
                current_book = matches[0]
                print(f"  📖 {current_book}")
            else:
                print(f"  ⚠️  «{book_name}» не найдена")
            continue

        if user_input.lower() == "/all":
            current_book = None
            print("  🔍 Поиск по всем книгам")
            continue

        # Запуск RAG-пайплайна
        result = pipeline.query(user_input, book_title=current_book)

        print(f"\n💬 {result['answer']}\n")

        if result["sources"]:
            print("📚 Источники:")
            for s in result["sources"]:
                loc = f" | стр. {s['page']}" if s['page'] else ""
                print(f"  • {s['book']}{loc} (score: {s['score']:.3f})")
                print(f"    «{s['preview']}»")

        if show_debug:
            d = result["debug"]
            print(f"\n🔬 [debug] Тип: {d['question_type']} | "
                  f"Уверенность: {d['confidence']:.2f} | "
                  f"Найдено: {d['initial_results']} → {d['reranked_results']}")

        print()


if __name__ == "__main__":
    main()
