import streamlit as st 
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import openai

# Load .env
load_dotenv()

# Set LangChain environment variables
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Simple Q&A Chatbot With OPENAI'

# Streamlit config
st.set_page_config(page_title="Dark Q&A Chatbot", layout="centered")

# Session state for message storage
if "store" not in st.session_state:
    st.session_state.store = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = "session1"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """ You are an autonomous clinical assistant. 
        You must assist doctors in reasoning about diagnoses and treatment plans using medical knowledge. 
        Always cite your reasoning from retrieved sources and include a safety disclaimer. 
        Never make final clinical decisions."""),
        
        ("user", "Question: {input}"),
    ]
)

# Get message history
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Create the RAG chain
def generate_response(api_key, engine, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)
    chain = prompt | llm
    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input"
    )
    return conversational_chain

# Sidebar for configs
with st.sidebar:
    api_key = st.text_input('Enter your OpenAI API Key', type='password')
    engine = st.selectbox('OpenAI Model', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])
    temperature = st.slider('Temperature', 0.0, 1.0, 0.7)
    max_tokens = st.slider('Max tokens', 100, 2000, 500)
    st.session_state.session_id = st.selectbox('Session ID', ['session1', 'session2', 'session3'])

# Display conversation
history = get_session_history(st.session_state.session_id)
for msg in history.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Chat input box (fixed at bottom)
if api_key:
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Get response
        chain = generate_response(api_key, engine, temperature, max_tokens)
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )

        # Display assistant message
        with st.chat_message("assistant"):
            st.write(response.content)
else:
    st.warning("Please enter your API key in the sidebar.")
