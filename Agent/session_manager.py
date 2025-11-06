from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class SessionManager:
    def __init__(self):
        self.store = {}

    def get_session_history(self , session_id:str)-> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def clear_history(self , session_id:str):
        if session_id in self.store:
            del self.store[session_id]


session_manager = SessionManager()