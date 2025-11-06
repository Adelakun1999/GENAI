from fastapi import APIRouter , Depends
from pydantic import BaseModel
from typing import List
from utils.logger import logger
from app.services.llm_options import fetch_llm_options
from app.api.deps.auth import verify_api_key

router = APIRouter()

class LLMOptionsResponse(BaseModel):
    Groq: List[str]
    Ollama : List[str]
    Openai : List[str]

@router.get('/llm_option' , tags=['llm_options'], 
            response_model=LLMOptionsResponse, 
            dependencies=[Depends(verify_api_key)])
async def llm_options():
    logger.info("Fetching available llm options")
    try : 
        llm_details = fetch_llm_options()
        logger.info("llm details fetched succesfully")
        return llm_details
    except Exception as e:
        logger.exception("Failed to fetch llm options")


