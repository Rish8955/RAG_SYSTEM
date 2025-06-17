from fastapi import APIRouter, HTTPException, status
from typing import List
from ..models.schemas import QueryRequest, QueryResponse, DocumentInfo
from ..services.rag_service import RAGService

router = APIRouter()
rag_service = RAGService()

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if not rag_service.is_initialized:
            await rag_service.initialize()
        
        response, session_id = await rag_service.query(
            question=request.question,
            session_id=request.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.post("/initialize", response_model=List[DocumentInfo])
async def initialize_system(force_rebuild: bool = False):
    try:
        doc_info = await rag_service.initialize(force_rebuild=force_rebuild)
        return doc_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initializing system: {str(e)}"
        )

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "initialized": rag_service.is_initialized
    }
