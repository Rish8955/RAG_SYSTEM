import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.rag_service import RAGService
from app.models.schemas import QueryResponse
import asyncio

@pytest.mark.asyncio
async def test_initialize_loads_index(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(rag.vector_store, 'load_index', MagicMock(return_value=True))
    result = await rag.initialize()
    assert rag.is_initialized is True
    assert result == []

@pytest.mark.asyncio
async def test_initialize_builds_index(monkeypatch):
    rag = RAGService()
    monkeypatch.setattr(rag.vector_store, 'load_index', MagicMock(return_value=False))
    monkeypatch.setattr(rag.document_processor, 'process_all_documents', AsyncMock(return_value=(['doc'], ['info'])))
    monkeypatch.setattr(rag.vector_store, 'build_index', AsyncMock())
    monkeypatch.setattr(rag.vector_store, 'save_index', MagicMock())
    result = await rag.initialize(force_rebuild=True)
    assert rag.is_initialized is True
    assert result == ['info']

@pytest.mark.asyncio
async def test_query_returns_response(monkeypatch):
    rag = RAGService()
    rag.is_initialized = True
    session_id = 'test-session'
    monkeypatch.setattr(rag.conversation_memory, 'create_session', MagicMock(return_value=session_id))
    monkeypatch.setattr(rag.conversation_memory, 'format_history_for_context', MagicMock(return_value='history'))
    mock_doc = MagicMock()
    mock_doc.page_content = 'content'
    mock_doc.metadata = {'page': 1, 'chunk': 0, 'source': 'test.pdf'}
    monkeypatch.setattr(rag.vector_store, 'similarity_search', AsyncMock(return_value=[mock_doc]))
    monkeypatch.setattr(rag.llm_service, 'create_rag_response', AsyncMock(return_value='answer'))
    monkeypatch.setattr(rag.conversation_memory, 'add_message', MagicMock())
    response, sid = await rag.query('question', session_id=session_id)
    assert isinstance(response, QueryResponse)
    assert response.answer == 'answer'
    assert sid == session_id
    assert response.sources[0]['source'] == 'test.pdf'

@pytest.mark.asyncio
async def test_query_raises_if_not_initialized():
    rag = RAGService()
    rag.is_initialized = False
    with pytest.raises(ValueError):
        await rag.query('question')
