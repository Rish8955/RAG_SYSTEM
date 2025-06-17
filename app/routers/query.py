from fastapi import APIRouter, Request
from app.services.rag_service import RAGService
from app.services.memory import ConversationMemory
from pydantic import BaseModel

router = APIRouter()
rag_service = RAGService()
memory = ConversationMemory()

class QueryRequest(BaseModel):
    question: str
    session_id: str

@router.post("/query")
async def query_endpoint(request: QueryRequest):
    conversation = memory.get(request.session_id)
    answer, updated_memory, references = await rag_service.answer_question(request.question, conversation)
    memory.set(request.session_id, updated_memory)
    return {"answer": answer, "references": references}
