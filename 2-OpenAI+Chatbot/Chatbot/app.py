import streamlit as st 
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
import openai

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Simple Q&A Chatbot With OPENAI'

if "store" not in st.session_state:
    st.session_state.store = {}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please respond to the user queries"),
        ('user', 'Question: {input}'),
    ]
)

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


def generate_response(api_key , engine , temperature, max_tokens, question):
    openai.api_key = api_key
    llm = ChatOpenAI(model=engine , temperature=temperature, max_tokens=max_tokens)
    chain = prompt | llm 
    #answer = chain.invoke({'input': question})

    conversational_rag_chain = RunnableWithMessageHistory(
        chain , get_session_history,
        input_messages_key= "input"
    )
    return conversational_rag_chain

st.title('Enhanced Q&A Chatbot with OpenAI')

api_key = st.sidebar.text_input('Enter your OpenAI API Key', type='password')
engine = st.sidebar.selectbox('Select the OpenAI Model', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o' ,'gpt-5'])
temperature = st.sidebar.slider('select the temperature' , min_value = 0.0 , max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider('Select the maximum number of tokens', min_value=100, max_value=2000, value=500)

question = st.text_input('Enter your question')
config_text = st.sidebar.selectbox(
    'Select a session ID',
    ['session1', 'session2', 'session3'],
    index=0
)


if question and api_key:
    response = generate_response(api_key, engine, temperature, max_tokens, question)
    if st.button('Submit'):
        response = response.invoke(
        {"input": question},
        config= {"configurable" : {"session_id": config_text} }  # constructs a key "session1" in `store`.
    )
        session_history = get_session_history(config_text)
        st.success(response.content)
        st.write("Chat History:", session_history.messages)

elif not api_key:
    st.warning('Please enter your OpenAI API Key to get started.')

else : 
    st.warning('Please enter a question to get a response.')

