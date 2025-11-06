import os 
from dotenv import load_dotenv
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#load the env variables
load_dotenv()

#streamlit page set up
st.set_page_config(
    page_title= "Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title('🤖 Generative AI Chatbot')

#Initiate chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#show chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ("Groq","OpenAI")
)

#llm initiate 

if provider == "Groq":
    model = st.sidebar.selectbox(
        "Select OpenAI Model",
        ("llama-3.1-8b-instant","llama-3.3-70b-versatile")
    )
    llm = ChatGroq(
    model = model,
    temperature=0
)

#user input 
user_prompt = st.chat_input('Ask Chatbot...')

if user_prompt:
    #display user message
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user" , "content" : user_prompt})

    #get response from llm
    response = llm.invoke(
        input = [
            {"role":"system", "content":"You are a helpful assistant.Be concise and accurate"},
            *st.session_state.chat_history,
        ]
    )
    answer = response.content
    st.session_state.chat_history.append({"role":"assistant" , "content" : answer})

    #display llm response 
    with st.chat_message("assistant"):
        st.markdown(answer)