class ConversationMemory:
    def __init__(self):
        self.memory = {}

    def get(self, session_id):
        return self.memory.get(session_id, [])

    def set(self, session_id, conversation):
        self.memory[session_id] = conversation
