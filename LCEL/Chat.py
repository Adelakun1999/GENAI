import streamlit as st 
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage , HumanMessage ,AIMessage
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")



# ——— Page config ———
st.set_page_config(page_title="Q&A Chatbot", layout="wide")



# ——— Initialize chat history ———
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

#render chat
st.markdown("## 🤖 Ask me anything (via ChatGroq)!")
for i, msg in enumerate(st.session_state.history):
    if msg["role"] == "user":
        message(msg["content"], is_user=True, key=f"user_{i}")
    elif msg["role"] == "assistant":
        message(msg["content"], is_user=False, key=f"bot_{i}")

# ——— User input ———
user_input = st.text_input("You:", key="input", placeholder="Type your question…")
send = st.button("Send")

if send and user_input:
    # append user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # build LangChain messages
    lc_messages = []
    for msg in st.session_state.history:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # instantiate ChatGroq
    chat = ChatGroq(
        model = "Gemma2-9b-It" , groq_api_key=groq_api_key
    )

    # get a response
    with st.spinner("🤔 thinking…"):
        resp = chat.invoke(lc_messages)
        answer = resp.content.strip()

    # append assistant reply and rerun
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()