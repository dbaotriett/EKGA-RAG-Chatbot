import os
from dotenv import load_dotenv

load_dotenv()

# Ollama 
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL   = "qwen2.5:3b-instruct"

# ChromaDB 
CHROMA_DIR       = "./chroma_db"
COLLECTION_NAME  = "rag_documents"

# Chunking 
CHUNK_SIZE    = 512   
CHUNK_OVERLAP = 64    

# Retrieval
# similarity + mmr

RETRIEVER_K        = 8
RETRIEVER_FETCH_K  = 16   
RETRIEVER_LAMBDA   = 0.75  

# Reranker 
USE_RERANKER           = True
RERANKER_MODEL         = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K           = 4    
RERANK_SCORE_THRESHOLD = 0.3  
RERANKER_BATCH_SIZE    = 16   

# Context / LLM 
MAX_CONTEXT_CHARS = 8000  
LLM_NUM_CTX       = 4096  
LLM_TEMPERATURE   = 0.0   

# Ingest 
BATCH_SIZE        = 64    
MAX_WORKERS       = 4     
