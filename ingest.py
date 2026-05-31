"""
ingest.py  –  Nhập tài liệu vào ChromaDB
Usage:
  python ingest.py              # nhập ./data, hỏi clear trước
  python ingest.py --clear      # xóa DB rồi nhập
  python ingest.py --no-dedup   # nhập kể cả file đã có
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".html", ".pptx", ".md"}


def collect_files(data_dir: str) -> list[str]:
    """Quét tất cả file được hỗ trợ trong data_dir."""
    files = []
    for fname in sorted(os.listdir(data_dir)):
        if any(fname.lower().endswith(ext) for ext in SUPPORTED_EXTS):
            files.append(os.path.join(data_dir, fname))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Ingest")
    parser.add_argument("--data-dir", default="./data", help="Thư mục chứa tài liệu")
    parser.add_argument("--clear",    action="store_true", help="Xóa DB trước khi nhập")
    parser.add_argument("--no-dedup", action="store_true", help="Tắt kiểm tra trùng lặp")
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"[Ingest] Không tìm thấy thư mục: {data_dir}")
        sys.exit(1)

    files = collect_files(data_dir)
    if not files:
        print(f"[Ingest] Không tìm thấy tài liệu trong {data_dir}")
        sys.exit(0)

    print(f"\n[Ingest] Tìm thấy {len(files)} tài liệu:")
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        print(f"  - {os.path.basename(f)}  ({size_kb:.1f} KB)")

    store = VectorStoreManager()
    print(f"\n[Ingest] Số chunk hiện có trong DB: {store.count()}")

    # clear DB nếu cần
    if args.clear:
        store.clear()
    else:
        ans = input("\nXóa DB cũ trước khi nhập? (y/N): ").strip().lower()
        if ans in ("y", "yes"):
            store.clear()

    # load 
    print("\n[Ingest] Đang load tài liệu (song song)...")
    processor = DocumentProcessor()
    docs      = processor.load_many(files)

    if not docs:
        print("[Ingest] Không có tài liệu nào được load thành công.")
        sys.exit(1)

    # split thành chunks 
    print("\n[Ingest] Đang split thành chunks...")
    chunks = processor.split(docs)

    if not chunks:
        print("[Ingest] Không tạo được chunk nào.")
        sys.exit(1)

    # add vào DB 
    print(f"\n[Ingest] Đang index {len(chunks)} chunks vào ChromaDB...")
    dedup = not args.no_dedup
    added = store.add_documents(chunks, dedup=dedup)

    print(f"\n[Ingest] ✓ Hoàn thành. Đã thêm {added}/{len(chunks)} chunks.")
    print(f"[Ingest]   Tổng trong DB: {store.count()} chunks")


if __name__ == "__main__":
    main()
