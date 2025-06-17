from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router, rag_service
from .config import settings
import os
from contextlib import asynccontextmanager

if not settings.groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

@asynccontextmanager
async def lifespan(app):
    try:
        await rag_service.initialize()
    except Exception as e:
        print(f"Warning: Could not initialize RAG service: {e}")
    yield

app = FastAPI(
    title="RAG System API",
    description="A RAG-based system for document Q&A using Groq LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])

@app.get("/")
async def root():
    return {"message": "RAG System API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
