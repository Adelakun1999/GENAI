import os 
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import json
from pydantic_settings import BaseSettings 
from utils.logger import logger
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

try:
    with open(CONFIG_PATH) as f:
        app_config = json.load(f)
        logger.info("Loaded app configuration from config.json successfully.")
except FileNotFoundError:
    logger.error("config.json not found at : %s" , CONFIG_PATH)
    raise
except json.JSONDecodeError as e :
    logger.exception("Invalid JSON format in config.json: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error loading config.json.")
    raise 


class AppConfig(BaseSettings):
    GROQ_API_KEY :str
    OPENAI_API_KEY : str
    APP_API_KEY  : str
    

    class Config:
        env_file = ".env"

try:
    env_config = AppConfig()
    logger.info("Environment variables loaded succesfully from .env")

except Exception as e:
    logger.exception("Error loading environment variables with Pydantic.")
    raise
