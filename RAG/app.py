import streamlit as st 
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
"""
)

def create_vector_embedding():
    if "vector_store" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.final_docs, 
            st.session_state.embeddings
        )

st.title("RAG Document Q&A With Groq And Lama3")
user_input = st.chat_input("Enter your query from the research paper")
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retrieval , document_chain)
    response = retrieval_chain.invoke({"input": user_input})
    with st.chat_message("assistant"):
        st.write(response['answer'])

    with st.expander('Document similarity search'):
        for i , doc in enumerate(response['context']):
            st.write(doc.page_content)