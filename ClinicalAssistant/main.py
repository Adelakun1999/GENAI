import streamlit as st 
from vectorestore import process_pdf
from rag_engine import get_rag_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")


if 'store' not in st.session_state:
    st.session_state.store={}

if "session_id" not in st.session_state:
    st.session_state.session_id = "session1"




st.session_state.session_id = st.sidebar.selectbox("Select session id", options=["session1", "session2", "session3"], index=0)


def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


retriever = process_pdf()
if retriever is None:
    st.warning("Please upload at least one PDF to start chatting.")
else:
    rag_chain = get_rag_chain(retriever)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    history = get_session_history(st.session_state.session_id)
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

    user_input = st.chat_input("Your question:")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": st.session_state.session_id}
            },
        )

        with st.chat_message("assistant"):
            st.write(response['answer'])


