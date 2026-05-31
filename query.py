"""
query.py  –  CLI query RAG
Usage:
  python query.py                     # mặc định MMR + streaming
  python query.py --search similarity # dùng pure cosine similarity
  python query.py --search threshold  # dùng score threshold
  python query.py --no-stream         # output 1 lần (non-streaming)
  python query.py --debug             # in cả context + answer
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_chain import RAGChain

QUIT_WORDS = {"exit", "quit", "q", "thoát", "bye"}


def run_interactive(rag: RAGChain, stream: bool, debug: bool) -> None:
    print("\nRAG sẵn sàng. Gõ 'exit' để thoát.\n")
    print("=" * 60)

    while True:
        try:
            question = input("\nCâu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Thoát]")
            break

        if not question:
            continue
        if question.lower() in QUIT_WORDS:
            print("[Thoát]")
            break

        print()

        try:
            if debug:
                result = rag.debug(question)
                print("── CONTEXT ──────────────────────────────────────────")
                print(result["context"])
                print(f"\n[Context: {result['context_chars']} chars]")
                print("── ANSWER ───────────────────────────────────────────")
                print(result["answer"])
                print("─────────────────────────────────────────────────────")

            elif stream:
                for token in rag.ask_stream(question):
                    print(token, end="", flush=True)
                print()  # newline cuối

            else:
                answer = rag.ask(question)
                print(answer)

        except (RuntimeError, ValueError, ConnectionError) as e:
            print(f"[Lỗi] {e}")
            print("→ Kiểm tra Ollama đang chạy và models đã pull.")

        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Query CLI")
    parser.add_argument(
        "--search",
        choices=["mmr", "similarity", "threshold"],
        default="mmr",
        help="Kiểu retrieval (mặc định: mmr)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Tắt streaming, in toàn bộ câu trả lời 1 lần",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="In cả context và câu trả lời",
    )
    args = parser.parse_args()

    print(f"[Query] Khởi tạo RAG (search={args.search})...")
    rag = RAGChain(search_type=args.search)
    print("[Query] Sẵn sàng.")

    stream = not args.no_stream
    run_interactive(rag, stream=stream, debug=args.debug)


if __name__ == "__main__":
    main()
