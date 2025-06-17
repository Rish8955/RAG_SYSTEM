import os
import pickle
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.schema import Document as LangchainDocument
from ..config import settings

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.index: Optional[faiss.Index] = None
        self.documents: List[LangchainDocument] = []
        self.dimension = 384 
    
    async def create_embeddings(self, documents: List[LangchainDocument]) -> np.ndarray:
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    async def build_index(self, documents: List[LangchainDocument]) -> None:
        self.documents = documents        
        embeddings = await self.create_embeddings(documents)        
        self.index = faiss.IndexFlatIP(self.dimension)         
        faiss.normalize_L2(embeddings)        
        self.index.add(embeddings.astype('float32'))
    
    async def similarity_search(self, query: str, k: int = 3) -> List[LangchainDocument]:
        if self.index is None or not self.documents:
            raise ValueError("Index not built. Call build_index first.")
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def save_index(self, path: str = None) -> None:
        if path is None:
            path = settings.vector_store_path
        
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
    
    def load_index(self, path: str = None) -> bool:
        if path is None:
            path = settings.vector_store_path
        
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.pkl")
        
        if not (os.path.exists(index_path) and os.path.exists(docs_path)):
            return False
        
        try:
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            return True
        except Exception:
            return False
