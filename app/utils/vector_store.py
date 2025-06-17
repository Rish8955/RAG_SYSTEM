import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.metadatas = []
        self.chunks = [] 
        self.index = None

    def add_embeddings(self, embeddings, metadatas, chunks):
        self.embeddings = np.array(embeddings).astype('float32')
        self.metadatas = metadatas
        self.chunks = chunks
            
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, query_embedding, top_k=5):
        if self.index is None:
            return [], [], []
            
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        
        retrieved_metadatas = [self.metadatas[i] for i in I[0]]
        retrieved_chunks = [self.chunks[i] for i in I[0]]
        
        return retrieved_metadatas, retrieved_chunks, I[0]