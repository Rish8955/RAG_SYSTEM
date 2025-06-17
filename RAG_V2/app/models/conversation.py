from typing import List, Dict
from datetime import datetime
from .schemas import ConversationMessage
import uuid

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        
        self.conversations[session_id].append(message)
        
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
    
    def get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        return self.conversations.get(session_id, [])
    
    def create_session(self) -> str:
        return str(uuid.uuid4())
    
    def format_history_for_context(self, session_id: str) -> str:
        history = self.get_conversation_history(session_id)
        if not history:
            return ""
        
        formatted = "\n\nPrevious conversation:\n"
        for msg in history[-6:]:  # Last 3 exchanges
            formatted += f"{msg.role.title()}: {msg.content}\n"
        
        return formatted
