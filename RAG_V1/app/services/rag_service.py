import os
from app.utils.document_loader import load_and_chunk_documents
from app.utils.embedding import embed_chunks
from app.utils.vector_store import VectorStore
from app.services.llm import LLMService
import numpy as np
from app.utils.embedding import model as embed_model


class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = LLMService()
        self._ingest_documents()

    def _ingest_documents(self):
        docs = load_and_chunk_documents("data/")
        
        formatted_chunks = []
        for doc in docs:
            if isinstance(doc, dict):
                if 'text' in doc and 'metadata' in doc:
                    formatted_chunks.append(doc)
                else:
                    text = doc.get('text', doc.get('content', doc.get('chunk', str(doc))))
                    metadata = doc.get('metadata', doc.copy())
                    formatted_chunks.append({'text': text, 'metadata': metadata})
            else:
                formatted_chunks.append({
                    'text': str(doc),
                    'metadata': {'source': 'unknown', 'chunk_id': len(formatted_chunks)}
                })
        
        embeddings, metadatas, texts = embed_chunks(formatted_chunks)
        self.vector_store.add_embeddings(embeddings, metadatas, texts)

    async def answer_question(self, question, conversation):
        query_embedding = embed_model.encode([question])[0]        
        metadatas, chunks, indices = self.vector_store.query(query_embedding, top_k=3)        
        if not chunks or all(not str(chunk).strip() for chunk in chunks):
            return "No relevant context found in the documents.", conversation, []        
        context_parts = []
        for i, chunk in enumerate(chunks):
            chunk_text = str(chunk).strip()
            if chunk_text and len(chunk_text) > 20:  
                context_parts.append(f"[Chunk {i+1}]: {chunk_text}")
        
        if not context_parts:
            return "No relevant context found in the documents.", conversation, []        
        context = "\n\n".join(context_parts)        
        answer = await self.llm.generate_answer(question, context, conversation)        
        print(f"Question: {question}")
        print(f"Number of chunks retrieved: {len(chunks)}")
        print(f"Context length: {len(context)} characters")
        print(f"Context preview: {context[:300]}...")
        print(f"Generated answer: {answer}")
        print(f"Answer length: {len(answer)} characters")
        
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            if isinstance(obj, (list, tuple, set)):
                return tuple(make_hashable(x) for x in obj)
            return obj
        
        seen = set()
        unique_references = []
        for meta in metadatas:
            meta_tuple = make_hashable(meta)
            if meta_tuple not in seen:
                seen.add(meta_tuple)
                unique_references.append(meta)
        
        return answer, conversation + [(question, answer)], unique_references