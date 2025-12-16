Application Name: RAG PDF Chatbot

--------------------------------------------------
DEVELOPED BY
--------------------------------------------------
- Utkarsh Ojha
- Shivansh Pokhriyal
- Siddharth Pundir

-Shivansh Pokhriyal - Shivansh is currently pursuing his bachelors from VMSB UTU Dehradun and is an honest guy
--------------------------------------------------
APPLICATION DESCRIPTION
--------------------------------------------------
This application is a Retrieval-Augmented Generation (RAG) based chatbot
that allows users to upload a PDF document and ask questions strictly
based on the content of that document.

The system combines document retrieval with a Large Language Model (LLM)
to generate accurate, context-aware responses.

--------------------------------------------------
HOW TO USE
--------------------------------------------------
1. Upload a PDF document using the upload section.
2. Ask questions related to the uploaded document.
3. If a new document is uploaded, the previous knowledge base is cleared
   automatically and replaced with the new document.

--------------------------------------------------
KEY DESIGN PRINCIPLE
--------------------------------------------------
This application uses a session-based RAG approach.

Only one document is active at a time. Whenever a new PDF is uploaded,
the existing vector database is reset to ensure:
- No mixing of documents
- No retention of previous user data
- Clean and predictable responses

--------------------------------------------------
TECHNOLOGY STACK
--------------------------------------------------
Backend:
- FastAPI

Frontend:
- Streamlit

RAG Components:
- ChromaDB (Vector Store)
- Sentence Transformers (Embeddings)
- Groq LLM

--------------------------------------------------
END OF DOCUMENT
--------------------------------------------------
