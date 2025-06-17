import os
import asyncio
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from ..config import settings
from ..models.schemas import DocumentInfo
from datetime import datetime

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    async def process_pdf(self, file_path: str) -> tuple[List[LangchainDocument], DocumentInfo]:
        try:
            reader = PdfReader(file_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    chunks = self.text_splitter.split_text(text)
                    for chunk_idx, chunk in enumerate(chunks):
                        metadata = {
                            "source": os.path.basename(file_path),
                            "page": page_num + 1,
                            "chunk": chunk_idx,
                            "total_pages": len(reader.pages)
                        }
                        
                        documents.append(LangchainDocument(
                            page_content=chunk,
                            metadata=metadata
                        ))
            
            doc_info = DocumentInfo(
                filename=os.path.basename(file_path),
                page_count=len(reader.pages),
                chunk_count=len(documents),
                processed_at=datetime.now()
            )
            
            return documents, doc_info
            
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
    
    async def process_all_documents(self, documents_path: str = None) -> tuple[List[LangchainDocument], List[DocumentInfo]]:
        if documents_path is None:
            documents_path = settings.documents_path
        
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents directory not found: {documents_path}")
        
        all_documents = []
        all_doc_info = []
        
        pdf_files = [f for f in os.listdir(documents_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {documents_path}")
        
        for filename in pdf_files:
            file_path = os.path.join(documents_path, filename)
            documents, doc_info = await self.process_pdf(file_path)
            all_documents.extend(documents)
            all_doc_info.append(doc_info)
        
        return all_documents, all_doc_info
