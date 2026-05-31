from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, RERANK_TOP_K, RERANKER_BATCH_SIZE


class Reranker:
    """
    CrossEncoder reranker với:
    - Batch predict (vectorized, nhanh hơn loop)
    - Score normalize qua sigmoid (→ range [0,1] nhất quán)
    - apply_softmax=False để tránh distort relative scores
    """

    _instance: "Reranker | None" = None  

    def __new__(cls) -> "Reranker":
        """Singleton: chỉ load model 1 lần dù khởi tạo nhiều lần."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        print(f"[Reranker] Loading: {RERANKER_MODEL}")
        
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,      
            device=None,         
        )
        print("[Reranker] Ready")
        self._initialized = True

    # core

    def _predict_batch(self, pairs: List[List[str]]) -> List[float]:
        """Chạy predict theo batch, trả list[float] scores."""
        all_scores: List[float] = []
        for start in range(0, len(pairs), RERANKER_BATCH_SIZE):
            batch = pairs[start : start + RERANKER_BATCH_SIZE]
            scores = self.model.predict(
                batch,
                apply_softmax=False,  
                show_progress_bar=False,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            all_scores.extend(scores)
        return all_scores

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = RERANK_TOP_K,
    ) -> List[Tuple[int, float]]:
        """
        Trả về list[(original_index, score)] đã sort descending.
        """
        if not docs:
            return []

        pairs = [[query, doc] for doc in docs]
        scores = self._predict_batch(pairs)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def rerank_documents(self, query: str, docs: List) -> List:
        """
        Nhận list LangChain Document, trả về list đã rerank.
        Score được ghi vào metadata['reranker_score'].
        """
        if not docs:
            return []

        contents = [doc.page_content for doc in docs]
        ranked = self.rerank(query, contents)

        results = []
        for idx, score in ranked:
            doc = docs[idx]
            doc.metadata["reranker_score"] = round(float(score), 4)
            results.append(doc)

        return results
