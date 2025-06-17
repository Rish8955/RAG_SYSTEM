from fastapi import FastAPI, HTTPException
from app.routers import query

app = FastAPI(title="RAG-based QA System")

app.include_router(query.router)
