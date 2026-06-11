from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any

class EmbeddingPipeline:
    def __init__(self, model_name:str="all-MiniLM-L6-v2", chunk_size:int=1000, chunk_overlap:int=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents:list[Any])->List[Any]:
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )
        chunked_docs=splitter.split_documents(documents)
        return chunked_docs
    
    def generate_embeddings(self, chunked_docs:list[Any])->list[list[float]]:
        texts=[chunk.page_content for chunk in chunked_docs]
        embeddings=self.model.encode(texts, show_progress_bar=True)
        return embeddings

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.loaders.loader import load_documents


if __name__=='__main__':
    pipeline=EmbeddingPipeline()
    docs=load_documents('data')
    chunked_docs=pipeline.chunk_documents(docs)
    embeddings=pipeline.generate_embeddings(chunked_docs)
    print(len(embeddings))
    print(embeddings)
