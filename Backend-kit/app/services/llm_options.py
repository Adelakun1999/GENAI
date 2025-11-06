from utils.config import app_config
from utils.logger import logger 
def fetch_llm_options():
    llm_config = app_config['llm']
    return llm_config