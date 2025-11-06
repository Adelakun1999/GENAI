from utils.logger import logger 
from app.llm.llm_factory import load_llm

def get_answer(provider , model , question):
    logger.info(f"Received request for answer generation. provider : {provider}, model: {model}, question: {question}")

    try:
        llm = load_llm(provider = provider , model=model)
        logger.debug(f"LLM loaded succesfulluy for provider : {provider} , Model : {model}")

        response = llm.invoke(question)
        answer = response.content

        logger.info(f"Answer generated succesully")
        logger.debug(f"Answer: {answer}")

        return {'answer' : answer}
    
    except Exception as e:
        logger.exception(f"Error occured while generating answer: {str(e)}")
        return {'error' : "failed to generate answer . please try again later"}
    
    