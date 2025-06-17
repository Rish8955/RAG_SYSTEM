from groq import Groq
from typing import List, Dict, Any
from ..config import settings

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    async def create_rag_response(self, query: str, context: str, conversation_context: str = "") -> str:
        system_prompt = """You are a helpful assistant that answers questions based on the provided context from documents. 
        Always base your answers on the given context. If the context doesn't contain enough information to answer the question, 
        say so clearly. Provide specific references to page numbers when available."""
        
        user_prompt = f"""Context from documents:
                        {context}

                        {conversation_context}

                        Question: {query}

                        Please provide a comprehensive answer based on the context above. Include page references when available."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return await self.generate_response(messages)
