# RAG System

A Retrieval-Augmented Generation (RAG) system for document Q&A using Groq LLM and FastAPI.

## Setup
1. Copy `.env` and fill in your Groq API key.
2. Place your PDF files in the `documents/` folder.
3. Build and run with Docker:
   ```sh
   docker-compose up --build
   ```
4. Access the API at `http://localhost:8000/api/v1/`

## Endpoints
- `POST /api/v1/query` — Query the system with a question
- `POST /api/v1/initialize` — (Re)build the vector store
- `GET /api/v1/health` — Health check

## Testing
Run tests with:
```sh
pytest
```
