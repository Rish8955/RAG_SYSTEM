import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    vector_store_path: str = "./data/vector_store"
    documents_path: str = "./documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_conversation_history: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"
    groq_model: str = "mistral-saba-24b" 
    
    class Config:
        env_file = ".env"

settings = Settings()
