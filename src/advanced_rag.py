from ml_modules import (
    BaseLLM,
    BaseQuestionClassifier,
    BaseReranker,
    ClassifiedQuestion,
    CrossEncoderReranker,
    DefaultEmbedder,
    FineTunedClassifier,
    FineTunedEmbedder,
    NoReranker,
    OllamaLLM,
    OllamaLLMWithFallback,
    QuestionType,
    RuleBasedClassifier,
)
from vector_store import SearchResult, VectorStore


class AdvancedRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        classifier: BaseQuestionClassifier | None = None,
        embedder: DefaultEmbedder | FineTunedEmbedder | None = None,
        reranker: BaseReranker | None = None,
        llm: BaseLLM | None = None,
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.classifier = classifier or RuleBasedClassifier()
        self.embedder = embedder or DefaultEmbedder()
        self.reranker = reranker or NoReranker()
        self.llm = llm or OllamaLLM()
        self.top_k = top_k

    @classmethod
    def default(cls, vector_store: VectorStore, ollama_model: str = "mistral") -> "AdvancedRAGPipeline":
        return cls(
            vector_store=vector_store,
            classifier=RuleBasedClassifier(),
            embedder=DefaultEmbedder(),
            reranker=NoReranker(),
            llm=OllamaLLM(model=ollama_model),
        )

    @classmethod
    def from_config(cls, vector_store: VectorStore, config: dict) -> "AdvancedRAGPipeline":
        classifier = None
        embedder = None
        reranker = None
        llm = None

        if config.get("classifier", {}).get("use_finetuned"):
            classifier = FineTunedClassifier(config["classifier"]["model_path"])
        elif config.get("classifier", {}).get("use_rules", True):
            classifier = RuleBasedClassifier()

        if config.get("embedder", {}).get("use_finetuned"):
            embedder = FineTunedEmbedder(
                model_path=config["embedder"]["model_path"],
                base_model=config["embedder"].get("base_model"),
            )
        else:
            embedder = DefaultEmbedder(config.get("embedder", {}).get("model_name"))

        if config.get("reranker", {}).get("use_finetuned"):
            reranker = CrossEncoderReranker(config["reranker"]["model_path"])
        elif config.get("reranker", {}).get("use_default", False):
            reranker = CrossEncoderReranker()
        else:
            reranker = NoReranker()

        llm_config = config.get("llm", {})
        if llm_config.get("use_fallback"):
            llm = OllamaLLMWithFallback(
                primary_model=llm_config.get("primary_model", "mistral-literary"),
                fallback_model=llm_config.get("fallback_model", "mistral"),
            )
        else:
            llm = OllamaLLM(
                model=llm_config.get("model", "mistral"),
                temperature=llm_config.get("temperature", 0.3),
            )

        return cls(
            vector_store=vector_store,
            classifier=classifier,
            embedder=embedder,
            reranker=reranker,
            llm=llm,
            top_k=config.get("top_k", 5),
        )

    def query(self, question: str, book_title: str | None = None) -> dict:
        debug = {}

        classified: ClassifiedQuestion = self.classifier.classify(question)
        debug["question_type"] = classified.qtype.value
        debug["confidence"] = classified.confidence
        debug["search_query"] = classified.reformulated

        query_emb = self.embedder.encode_query(classified.reformulated)

        initial_results: list[SearchResult] = self.vector_store.search(
            query_emb, top_k=self.top_k * 3, book_title=book_title
        )
        debug["initial_results"] = len(initial_results)

        reranked_results = self.reranker.rerank(
            classified.reformulated, initial_results, top_k=self.top_k
        )
        debug["reranked_results"] = len(reranked_results)

        if not reranked_results:
            return {
                "answer": "Could not find relevant fragments.",
                "sources": [],
                "debug": debug,
            }

        context = self._format_context(reranked_results)

        answer = self.llm.generate(
            question=question,
            context=context,
            question_type=classified.qtype,
        )

        sources = self._format_sources(reranked_results)

        return {
            "answer": answer,
            "sources": sources,
            "debug": debug,
        }

    def _format_context(self, results: list[SearchResult]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            header = f"[Fragment {i} | {r.chunk.book_title}"
            if r.chunk.page:
                header += f", p. {r.chunk.page}"
            header += f"]"
            parts.append(f"{header}\n{r.chunk.text}")
        return "\n\n".join(parts)

    def _format_sources(self, results: list[SearchResult]) -> list[dict]:
        return [
            {
                "book": r.chunk.book_title,
                "page": r.chunk.page,
                "score": round(r.score, 3),
                "preview": r.chunk.text[:120].replace("\n", " ") + "...",
            }
            for r in results
        ]
