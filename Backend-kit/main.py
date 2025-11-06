from utils.config import app_config
from utils.logger import logger
from fastapi import FastAPI
from app.api.routes import health , select_llm , qa
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")

app.include_router(health.router)
app.include_router(select_llm.router)
app.include_router(qa.router)
logger.info("API routes registered succesfully")






