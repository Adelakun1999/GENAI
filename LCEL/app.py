from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from langchain_groq import ChatGroq
import os 
from streamlit_chat import message
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model = "Gemma2-9b-It" , groq_api_key=groq_api_key
)

system_template = "Translate the following text to {language}"
prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)

parser = StrOutputParser()

chain =prompt_template | model | parser

class TranslationRequest(BaseModel):
    text: str
    language: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Hello, how are you?",
                "language": "French"
            }
        }
    }

app = FastAPI()

@app.post("/translate")
async def translate(request: TranslationRequest):
    if request.text:
        result = chain.invoke({"text": request.text, "language": request.language})
        return {"translation": result}
    else:
        return {"error": "Please enter text to translate"}
    


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
