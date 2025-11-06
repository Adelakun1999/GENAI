from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from config import Config

groq_api_key = Config.GROQ_API_KEY


def get_llm():
    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name = "llama-3.1-8b-instant"
    )

    return model


def get_embedding_model():
    embeddings = OpenAIEmbeddings(

    )

    return embeddings

