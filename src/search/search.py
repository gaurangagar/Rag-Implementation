import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from langchain_groq import ChatGroq

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from src.vectordb.vectorStore import FaissVectorStore

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
VECTOR_DB_PATH="faiss_store"


class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.3-70b-versatile"):
        self.vectorStore=FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorStore.load()
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")
    
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorStore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content
    
if __name__ == "__main__":
    rag_search = RAGSearch()
    query="What is minimum spanning tree"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
