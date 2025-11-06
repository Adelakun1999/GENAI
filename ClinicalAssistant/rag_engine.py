from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.chains import create_history_aware_retriever , create_retrieval_chain 
from llm_setup import get_llm

def get_rag_chain(retriever):
    llm = get_llm('gpt-4')
    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
        ("system" , contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human" , "{input}")

        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm , retriever , contextualize_q_prompt
    )

    system_prompt = (
        """ You are an autonomous clinical assistant. 
        You must assist doctors in reasoning about diagnoses and treatment plans using medical knowledge. 
        Always cite your reasoning from retrieved sources and include a safety disclaimer. 
        Never make final clinical decisions."""
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm ,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever , question_answer_chain)

    return rag_chain
