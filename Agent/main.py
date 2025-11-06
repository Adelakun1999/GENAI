from fastapi import FastAPI , UploadFile , File , Path , HTTPException
from fastapi.responses import JSONResponse
import os 
import uuid
from contextlib import asynccontextmanager

from models import ChatRequest , ChatResponse , HistoryResponse
from vectorstore import process_pdfs , get_vectorstore_path
from rag_chain import get_rag_chain
from session_manager import session_manager
from llm_setup import get_llm
import shutil
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage



@asynccontextmanager
async def lifespan(app:FastAPI):
    os.makedirs("temp_pdfs" , exist_ok=True)
    yield
    shutil.rmtree('temp_pdfs' , ignore_errors=True)

app = FastAPI(
    title= "Conversational RAG with PDF",
    description="Chat with uploaded PDFs using RAG and session-based memory",
    version = "0.1.0",
    lifespan= lifespan
)

active_sessions = set()

@app.post("/upload_pdf/{session_id}")
async def upload_pdf(session_id:str = Path() , files:list[UploadFile] = File):
    if not files:
        raise HTTPException(status_code=400 , detail='No files upload')
    
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir , exist_ok = True)
    file_paths = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400 , 
                detail = f"Invalid file type : {file.filename}"
            )
        
        temp_path =os.path.join(temp_dir , f"{uuid.uuid4()}_{file.filename}")
        with open(temp_path , "wb") as f:
            f.write(await file.read())

        file_paths.append(temp_path)

        try :
            retriever = process_pdfs(file_paths , session_id)
            rag_chain = get_rag_chain(retriever)
            
            if not hasattr(app.state , "rag_chains"):
                app.state.rag_chains = {}
            app.state.rag_chains[session_id] = rag_chain

            active_sessions.add(session_id)

            for path in file_paths:
                if os.path.exists(path):
                    os.remove(path)
            return {"session_id" : session_id , "processed_files" : [f.filename for f in files]}
        
        except Exception as e:
            raise HTTPException(status_code=500 ,
                              detail = f"Error processing PDFs : {str(e)}")
        


@app.post("/chat/{session_id}" , response_model= ChatResponse)
async def chat(session_id : str , request : ChatRequest):
    if not hasattr(app.state,"rag_chains") or session_id not in app.state.rag_chains:
        raise HTTPException(status_code=404 , detail="Session not found or no PDFs uploaded")
    
    rag_chain = app.state.rag_chains[session_id]
    history = session_manager.get_session_history(session_id)

    try:
        response = rag_chain.invoke(
            {"input" : request.message , 'chat_history' : history.messages} ,
            config = {"configurable" : {"session_id" : session_id}}
        )

        sources = list(set(doc.metadata.get('source', 'unknown') for doc in response.get("context", [])))
        return ChatResponse(
            session_id = session_id,
            answer = response['answer'],
            sources = sources
        )
    except Exception as e:
        raise HTTPException(status_code=500 , detail = f"Error generating response : {str(e)}")


@app.get("/history/{session_id}" , response_model= HistoryResponse)
async def get_history(session_id: str):
    if session_id not in session_manager.store:
        raise HTTPException(status_code=404, detail="No chat history found for this session")

    messages = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in session_manager.store[session_id].messages
    ]
    return HistoryResponse(session_id=session_id, messages=messages)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in session_manager.store:
        del session_manager.store[session_id]
    if hasattr(app.state, "rag_chains") and session_id in app.state.rag_chains:
        del app.state.rag_chains[session_id]

    active_sessions.discard(session_id)
    return {"status": "deleted", "session_id": session_id}
        


        
