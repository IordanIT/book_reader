# Как дообучить и подключить свои ML-модели

## Обзор архитектуры

```
Пользовательский вопрос
        │
        ▼
┌──────────────────┐
│ 1. Классификатор  │ ← дообучить: FineTunedClassifier
│    типа вопроса   │   (определяет: сюжет/герои/цитаты/темы)
└───────┬──────────┘
        │ reformulated query
        ▼
┌──────────────────┐
│ 2. Embedding      │ ← дообучить: FineTunedEmbedder
│    векторизация    │   (contrastive learning на книжных данных)
└───────┬──────────┘
        │ top-15 результатов
        ▼
┌──────────────────┐
│ 3. Re-ranker      │ ← дообучить: CrossEncoderReranker
│    переранжирование│   (кросс-энкодер точнее биэнкодера)
└───────┬──────────┘
        │ top-5 лучших чанков
        ▼
┌──────────────────┐
│ 4. LLM генерация  │ │ дообучить: QLoRA fine-tuning
│    финальный ответ │   (Mistral/Llama на литературных QA)
└──────────────────┘
```

---

## 1. Дообучение Embedding модели

**Когда:** хотите лучше находить релевантные фрагменты

```bash
# Подготовка: добавьте книги в data/books/
python src/ml/embedding_train.py
```

**Свои данные:** создайте `data/training/embedding_pairs.jsonl`:
```jsonl
{"query": "Кто такой Обломов?", "positive": "Илья Ильич Обломов — помещик...", "negative": "Собакевич сидел в кресле..."}
```

**Подключение в config.py:**
```python
"embedder": {
    "use_finetuned": True,
    "model_path": "data/models/fine-tuned-embedder",
}
```

---

## 2. Дообучение Re-ranker

**Когда:** хотите точнее упорядочивать найденные фрагменты

```bash
python src/ml/reranker_train.py
```

**Свои данные:** создайте `data/training/reranker_data.jsonl`:
```jsonl
{"query": "Опиши характер Печорина", "passage": "Печорин был эгоист...", "label": 1.0}
{"query": "Опиши характер Печорина", "passage": "Погода была пасмурной...", "label": 0.1}
```

**Подключение:**
```python
"reranker": {
    "use_finetuned": True,
    "model_path": "data/models/fine-tuned-reranker",
}
```

---

## 3. Дообучение LLM (QLoRA)

**Когда:** хотите, чтобы модель лучше понимала литературный анализ

### Вариант А: Быстрый (Ollama Modelfile)
```bash
# Подготовка данных
python src/ml/llm_train_prepare.py

# Создание кастомной модели в Ollama
ollama create mistral-literary -f data/models/Modelfile
```

### Вариант Б: Полный (Unsloth)
```bash
pip install unsloth datasets transformers peft
python src/ml/finetune_unsloth.py
```

### Подключение:
```python
"llm": {
    "model": "mistral-literary",  # ваша дообученная модель
    "temperature": 0.2,            # ниже — фактуальнее
}
```

---

## 4. Дообучение классификатора

**Когда:** хотите определять сложные типы вопросов

```bash
# Создайте размеченный датасет
python src/ml/classifier_train.py
```

**Формат данных:** `data/training/classifier_data.jsonl`
```jsonl
{"text": "Какова главная тема романа?", "label": "theme"}
{"text": "Что случилось с Раскольниковым?", "label": "plot"}
{"text": "Кто сказал «человек человеку волк»?", "label": "quote"}
```

---

## Порядок внедрения (рекомендация)

1. **Начните со стандарта** — протестируйте RAG без ML
2. **Добавьте re-ranker** — максимальный эффект за минимум усилий
3. **Дообучите embedder** — если поиск плохо находит
4. **Дообучите LLM** — если генерация недостаточно хороша

---

## Структура директорий обучения

```
data/
├── training/
│   ├── embedding_pairs.jsonl
│   ├── reranker_data.jsonl
│   ├── classifier_data.jsonl
│   └── literary_qa.jsonl
└── models/
    ├── fine-tuned-embedder/
    ├── fine-tuned-reranker/
    ├── fine-tuned-classifier/
    └── Modelfile
```
