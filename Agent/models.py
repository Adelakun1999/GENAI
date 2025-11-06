from pydantic import BaseModel
from typing import List , Optional

class ChatRequest(BaseModel):
    message:str

class ChatResponse(BaseModel):
    session_id : str
    answer:str
    sources : Optional[List[str]] = None


class HistoryResponse(BaseModel):
    session_id : str
    messages : list 