import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_chroma import Chroma
from llm_setup import get_embedding_model

VECTORSTORE_DIR = "storage/vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

def get_vectorstore_path(session_id : str):
    return os.path.join(VECTORSTORE_DIR , f"vs_{session_id}")

def process_pdfs(uploaded_files_paths : list , session_id:str):
    documents = []
    for path in uploaded_files_paths:
        loader  = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 5000 , 
        chunk_overlap = 500
    )
    splits = text_splitter.split_documents(documents)

    persist_dir = get_vectorstore_path(session_id)
    vectorstore = Chroma.from_documents(
        documents = splits , 
        embedding = get_embedding_model(),
        persist_directory = persist_dir
    )

    return vectorstore.as_retriever()
