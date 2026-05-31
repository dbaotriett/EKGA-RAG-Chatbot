import sys
import types
import json
import warnings


dummy_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertexai
warnings.filterwarnings("ignore")

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.rag_chain import RAGChain

print(" Đang khởi tạo mô hình đánh giá và embeddings...")
judge_llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b-instruct", temperature=0.0))
judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

rag = RAGChain(search_type="mmr")

with open("NovaTech_RAG_Benchmark_v1.0.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f).get("qa_pairs", [])

dataset = []
print(f" Đang chạy RAG cho {len(qa_pairs)} câu hỏi...")

for item in qa_pairs:
    q = item["user_input"]
    
    dataset.append({
        "user_input": q,
        "response": rag.ask(q),
        "retrieved_contexts": rag.retrieve_contexts(q),
        "reference": item["reference"]
    })

eval_dataset = EvaluationDataset.from_list(dataset)

print("Đang đánh giá metrics")
run_config = RunConfig(timeout=600, max_workers=1)

result = evaluate(
    dataset=eval_dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=run_config  
)

print("\n results:")
print(result)

result.to_pandas().to_csv("rag_evaluation_results.csv", index=False, encoding="utf-8-sig") # type: ignore
print("\n Đã lưu file: rag_evaluation_results.csv")