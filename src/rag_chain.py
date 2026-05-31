from __future__ import annotations

from typing import Iterator, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import (
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    OLLAMA_URL,
    RERANK_SCORE_THRESHOLD,
    USE_RERANKER,
)
from src.reranker import Reranker
from src.vector_store import VectorStoreManager

# Prompt
_PROMPT_TEMPLATE = """\
Bạn là trợ lý AI. Chỉ trả lời dựa trên ngữ cảnh bên dưới.
Nếu không có thông tin, hãy nói: "Tôi không tìm thấy thông tin này trong tài liệu."
Không được bịa đặt hoặc suy luận ngoài tài liệu.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Yêu cầu: Trả lời tiếng Việt, ngắn gọn, cite nguồn [Doc X].
Trả lời:"""
# ─────────────────────────────────────────────────────────────────────────────


class RAGChain:
    """
    RAG pipeline với:
    - Prompt ngắn gọn hơn (ít token hơn ~30%)
    - build_context tối ưu: pre-compute block length, early-exit
    - Hỗ trợ streaming qua ask_stream()
    - Search type tunable (mmr / similarity / threshold)
    """

    def __init__(self, search_type: str = "mmr") -> None:
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
        )

        self.store = VectorStoreManager()
        self.retriever = self.store.get_retriever(search_type=search_type)
        self.reranker = Reranker() if USE_RERANKER else None

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=_PROMPT_TEMPLATE,
        )

        self.chain = (
            {"context": self._build_context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # context builder

    def _build_context(self, question: str) -> str:
        """
        Retrieve → rerank → format context.
        Tối ưu:
        - Tính len(block) 1 lần, không tính lại
        - Skip doc có score thấp ngay trong vòng lặp
        - Giới hạn MAX_CONTEXT_CHARS để tránh vượt context window LLM
        """
        docs = self.retriever.invoke(question)

        if self.reranker and docs:
            docs = self.reranker.rerank_documents(question, docs)

        if not docs:
            return "Không có thông tin phù hợp."

        blocks: list[str] = []
        total = 0

        for i, doc in enumerate(docs, 1):
            score: float = doc.metadata.get("reranker_score", 1.0)

            if score < RERANK_SCORE_THRESHOLD:
                continue  # lọc ngay, không build block

            source = doc.metadata.get("file_name", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "N/A")

            header = f"[Doc {i}] {source} | chunk={chunk_id} | score={score:.2f}"
            block = f"{header}\n{doc.page_content}"
            blen = len(block)

            if total + blen > MAX_CONTEXT_CHARS:
                break  # early-exit thay vì append rồi check

            blocks.append(block)
            total += blen

        if not blocks:
            return "Không có thông tin đủ liên quan."

        return "\n\n---\n\n".join(blocks)

    # public API

    def ask(self, question: str) -> str:
        """Trả lời đồng bộ."""
        return self.chain.invoke(question)

    def ask_stream(self, question: str) -> Iterator[str]:
        context = self._build_context(question)
        prompt_val = self.prompt.invoke({"context": context, "question": question})
        for chunk in self.llm.stream(prompt_val):
            # chunk là AIMessageChunk, content thường là str
            content = chunk.content
            if isinstance(content, str):
                yield content
            else:
                # fallback nếu không phải string
                yield str(content)

    def debug(self, question: str) -> dict:
        """
        Trả về dict chứa context + answer để debug pipeline.
        """
        context = self._build_context(question)
        prompt_val = self.prompt.invoke({"context": context, "question": question})
        answer = (self.llm | StrOutputParser()).invoke(prompt_val)
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "context_chars": len(context),
        }

    def retrieve_contexts(self, question: str) -> List[str]:
        """Chỉ lấy các đoạn context đã được retrieve + rerank (dạng text thuần)"""
        docs = self.retriever.invoke(question)
        if self.reranker and docs:
            docs = self.reranker.rerank_documents(question, docs)
        # Trả về list nội dung text của các doc, để đánh giá retrieval riêng với LLM response
        return [doc.page_content for doc in docs]