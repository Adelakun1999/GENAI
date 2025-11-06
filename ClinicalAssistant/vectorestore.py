import os 
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_setup import get_embedding_model
from langchain_chroma import Chroma

def process_pdf():
    uploaded_files = st.sidebar.file_uploader("choose a pdf file" ,  type="pdf" , accept_multiple_files=True)
    if uploaded_files:
        document = []
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            document.extend(docs)


        text_spliiter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        splits = text_spliiter.split_documents(document)
    
        vectorstore = Chroma.from_documents(
           splits , get_embedding_model()
        )

        return vectorstore.as_retriever()
    




if __name__ == "__main__":
    process_pdf()
    


    

