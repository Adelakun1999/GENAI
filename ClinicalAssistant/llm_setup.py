from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from config import settings
import openai

available_model  = [
    "llama-3.1-8b-instant",
    "gpt-4o-mini",
    'gpt-4'
]

def get_llm(model):
    if model not in available_model:
        raise ValueError(f"Invalid model. Available models : {available_model}")
    
    if model == "llama-3.1-8b-instant":
        return ChatGroq(
            model_name = "llama-3.1-8b-instant",
            api_key = settings.GROQ_API_KEY
        )
    elif model in ['gpt-4o-mini','gpt-4']:
        return ChatOpenAI(
            model_name = "gpt-4o-mini"
        )
    
def get_embedding_model():
    openai.api_key = settings.OPENAI_API_KEY
    embeddings = OpenAIEmbeddings()
    return embeddings


