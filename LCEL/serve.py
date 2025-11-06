import streamlit as st
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model = "Gemma2-9b-It" , groq_api_key=groq_api_key
)

st.set_page_config(page_title="Q&A Chatbot", page_icon= '🤖' , layout='wide')
st.title("LangChain Groq")

system_template = "Translate the following text to {language}"
prompt_template =  ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}"),
    ]
)

parser = StrOutputParser()

chain =prompt_template | model | parser
st.subheader("Translate text to different languages")
text = st.text_input("Enter text to translate")
language = st.selectbox(
    "Select the language to translate to",
    ("French", "Spanish", "German", "Italian", "Chinese", "Japanese"),
)

if st.button("Translate"):
    if text:
        with st.spinner("Translating..."):
            result = chain.invoke({"text": text, "language": language})
            st.success(result)
    else:
        st.error("Please enter text to translate")


