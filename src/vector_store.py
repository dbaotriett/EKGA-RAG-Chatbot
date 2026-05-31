from __future__ import annotations

from typing import List, Set

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.config import (
    EMBED_MODEL,
    OLLAMA_URL,
    CHROMA_DIR,
    COLLECTION_NAME,
    BATCH_SIZE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA,
)


class VectorStoreManager:
    """
    Quản lý ChromaDB với:
    - Dedup theo file_hash trước khi add (tránh index trùng)
    - Batch add với progress log
    - Retriever dùng similarity_score_threshold thay vì MMR thuần
      khi cần precision cao; giữ MMR khi cần diversity (mặc định)
    """

    def __init__(self) -> None:
        self.embedding = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_URL,
        )
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding,
            persist_directory=CHROMA_DIR,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _existing_hashes(self) -> Set[str]:
        """Lấy tập file_hash đã có trong DB để dedup."""
        try:
            data = self.db.get(include=["metadatas"])
            return {
                m.get("file_hash", "")
                for m in (data.get("metadatas") or [])
                if m.get("file_hash")
            }
        except (ValueError, RuntimeError):
            return set()

    # public API

    def add_documents(self, docs: List[Document], dedup: bool = True) -> int:
        """
        Thêm docs vào DB.
        - dedup=True: bỏ qua chunk có file_hash đã tồn tại.
        Trả về số chunk thực sự được thêm.
        """
        if not docs:
            return 0

        if dedup:
            existing = self._existing_hashes()
            docs = [
                d for d in docs
                if d.metadata.get("file_hash", "__no_hash__") not in existing
            ]
            if not docs:
                print("[VectorStore] Tất cả documents đã được index, bỏ qua.")
                return 0

        total = len(docs)
        added = 0
        for i in range(0, total, BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            self.db.add_documents(batch)
            added += len(batch)
            batch_num = i // BATCH_SIZE + 1
            total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[VectorStore] Batch {batch_num}/{total_batches} → {added}/{total} chunks")

        print(f"[VectorStore] Đã index {added} chunks")
        return added

    def get_retriever(self, search_type: str = "mmr"):
        """
        Trả retriever.
        search_type:
          - "mmr"        : Max Marginal Relevance (diversity, mặc định)
          - "similarity" : Pure cosine similarity (precision cao hơn, nhanh hơn)
          - "threshold"  : Similarity với score threshold lọc noise
        """
        if search_type == "similarity":
            return self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVER_K},
            )

        if search_type == "threshold":
            return self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.4,   
                    "k": RETRIEVER_K,
                },
            )

        # default: MMR
        return self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA,
            },
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.db.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> List[tuple[Document, float]]:
        """Trả kèm cosine score để debug."""
        return self.db.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        """Số chunk hiện có trong DB."""
        try:
            data = self.db.get()
            return len(data.get("ids") or [])
        except (ValueError, RuntimeError):
            return -1

    def clear(self) -> None:
        ids = self.db.get().get("ids", [])
        if ids:
            self.db.delete(ids=ids)
            print(f"[VectorStore] Đã xóa {len(ids)} chunks")
        else:
            print("[VectorStore] Collection rỗng, không cần xóa")
