# 🚀 Conversational RAG Chatbot with PDF Support

A FastAPI-based backend that lets users upload PDFs and chat with their content using **Retrieval-Augmented Generation (RAG)**, **session-based memory**, and **vector search**.

Perfect for building AI assistants that answer questions from private documents — no frontend needed. Just REST APIs.

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🌟 Features

✅ **Upload PDFs** and chat with their content  
✅ **Session-based memory**: follow-up questions work naturally  
✅ **Semantic search** via ChromaDB and OpenAI embeddings  
✅ **RAG pipeline** with history-aware rewriting and grounded answers  
✅ **Persistent vector storage** — no reprocessing on restart  
✅ **Clean, modular code** — easy to extend or learn from  

---

## 🛠️ Tech Stack

- **FastAPI** – High-performance backend
- **LangChain** – RAG, prompts, and chains
- **ChromaDB** – Vector database for document search
- **OpenAI Embeddings** – Semantic chunk embedding
- **Groq** – Fast LLM inference (Gemma2-9b)
- **PyPDFLoader** – PDF parsing

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
https://github.com/Adelakun1999/pdf-rag-api.git
cd pdf-rag-api
```
#### Install Dependencies: Ensure you have Python 3.8+ installed. Then run:

```pip install -r requirements.txt```

### Set up environment
Edit .env and add your API keys:
```bash
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

 Get your keys: 

Groq: https://console.groq.com
OpenAI: https://platform.openai.com/api-keys
```
### Run the App 
```bash
uvicorn main:app --reload
```
### Open API Docs

👉 Visit: http://localhost:8000/docs

You’ll see an interactive Swagger UI to test all endpoints.


## 📡 API Endpoints

| Endpoint | Method | Description |
|--------|--------|-------------|
| `/upload_pdf/{session_id}` | `POST` | Upload one or more PDFs for a session |
| `/chat/{session_id}` | `POST` | Send a message and get a RAG response |
| `/history/{session_id}` | `GET` | Retrieve the full chat history |
| `/session/{session_id}` | `DELETE` | Delete session data (chat) |

### 🧪 Example Usage
**1 upload a pdf**
```bash
curl -X POST "http://localhost:8000/upload_pdf/user123" \
  -F "files=@sample.pdf"
```
**2 Chat with it**
```bash
curl -X POST "http://localhost:8000/chat/user123" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is this document about?"}'
```
**3 Chat History**
```bash
curl "http://localhost:8000/history/user123"
```

🙌 Feedback Welcome!

Have a suggestion? Found a bug? Want to contribute?

👉 Open an issue or PR — I’d love to hear from you!


