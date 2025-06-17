from typing import List, Dict, Any, Optional, Tuple
from ..services.document_processor import DocumentProcessor
from ..services.vector_store import VectorStore
from ..services.llm_service import LLMService
from ..models.conversation import ConversationMemory
from ..models.schemas import QueryResponse, DocumentInfo
from datetime import datetime

class RAGService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.conversation_memory = ConversationMemory()
        self.is_initialized = False
    
    async def initialize(self, force_rebuild: bool = False) -> List[DocumentInfo]:
        if not force_rebuild and self.vector_store.load_index():
            self.is_initialized = True
            return []
        
        documents, doc_info = await self.document_processor.process_all_documents()
        
        await self.vector_store.build_index(documents)
        
        self.vector_store.save_index()
        
        self.is_initialized = True
        return doc_info
    
    async def query(self, question: str, session_id: Optional[str] = None, k: int = 3) -> Tuple[QueryResponse, str]:
        if not self.is_initialized:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        if session_id is None:
            session_id = self.conversation_memory.create_session()
        
        conversation_context = self.conversation_memory.format_history_for_context(session_id)
        
        relevant_docs = await self.vector_store.similarity_search(question, k=k)
        context_parts = []
        sources = []
        
        for doc in relevant_docs:
            page = doc.metadata.get('page', '?')
            chunk = doc.metadata.get('chunk', '?')
            source = doc.metadata.get('source', 'Unknown')
            
            context_parts.append(f"[Source: {source}, Page {page}, Chunk {chunk}]: {doc.page_content}")
            sources.append({
                "source": source,
                "page": page,
                "chunk": chunk,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
        
        context = "\n\n".join(context_parts)        
        answer = await self.llm_service.create_rag_response(question, context, conversation_context)
        
        self.conversation_memory.add_message(session_id, "user", question)
        self.conversation_memory.add_message(session_id, "assistant", answer)
        
        response = QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
        return response, session_id
