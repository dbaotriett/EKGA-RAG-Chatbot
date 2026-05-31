from __future__ import annotations

import os
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_WORKERS

# Separators ưu tiên theo cấu trúc markdown → tránh cắt giữa câu
_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", "! ", "? ", " ", ""]


class DocumentProcessor:
    """
    Xử lý tài liệu với:
    - DocumentConverter (docling) để parse PDF/DOCX/HTML/PPTX
    - Clean text tốt hơn: xử lý thêm form-feed, tab, unicode spaces
    - Parallel load nhiều file (ThreadPoolExecutor)
    - Global chunk_id duy nhất (file_hash prefix) tránh collision
    """

    def __init__(self) -> None:
        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            strip_whitespace=True,
            separators=_SEPARATORS,
        )

    # ── text utils ────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Làm sạch text: chuẩn hóa whitespace, xóa ký tự rác."""
        # form-feed, vertical tab → newline
        text = re.sub(r"[\x0c\x0b]", "\n", text)
        # nhiều dòng trống liên tiếp → tối đa 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # nhiều space/tab liên tiếp → 1 space
        text = re.sub(r"[ \t]+", " ", text)
        # unicode spaces, non-breaking space
        text = text.replace("\xa0", " ").replace("\u200b", "")
        return text.strip()

    def hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]  # 16 chars sufficient for file-level dedup

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, path: str) -> List[Document]:
        """Load và parse 1 file, trả về [Document]."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File không tồn tại: {path}")
        try:
            result  = self.converter.convert(path)
            content = result.document.export_to_markdown()
            content = self.clean_text(content)

            if not content:
                print(f"[Processor] Cảnh báo: {path} cho ra nội dung rỗng")
                return []

            file_hash = self.hash_text(content)
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source":    path,
                        "file_name": os.path.basename(path),
                        "file_hash": file_hash,
                        "file_size": os.path.getsize(path),
                    },
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Lỗi xử lý {path}: {e}") from e

    def load_many(self, paths: List[str]) -> List[Document]:
        """
        Load nhiều file song song (ThreadPoolExecutor).
        File lỗi sẽ in cảnh báo và bỏ qua, không crash toàn bộ.
        """
        all_docs: List[Document] = []
        workers = min(MAX_WORKERS, len(paths))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(self.load, p): p for p in paths}
            for future in as_completed(future_map):
                path = future_map[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    print(f"[Processor] ✓ {os.path.basename(path)}")
                except (RuntimeError, FileNotFoundError, OSError) as e:
                    print(f"[Processor] ✗ {os.path.basename(path)}: {e}")

        return all_docs

    # split

    def split(self, docs: List[Document]) -> List[Document]:
        """
        Split documents thành chunks.
        chunk_id = {file_hash}_{global_idx} → unique kể cả khi merge nhiều file.
        """
        if not docs:
            return []

        chunks = self.splitter.split_documents(docs)

        for idx, chunk in enumerate(chunks):
            file_name = chunk.metadata.get("file_name", "unknown")
            file_hash = chunk.metadata.get("file_hash", "0000000000000000")
            chunk.metadata.update(
                {
                    "chunk_id":    f"{file_hash}_{idx}",
                    "chunk_index": idx,
                    "chunk_size":  len(chunk.page_content),
                    "file_name":   file_name,
                }
            )

        # Lọc chunk quá ngắn 
        chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]

        print(f"[Processor] Split → {len(chunks)} chunks")
        return chunks
