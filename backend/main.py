from pathlib import Path
import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File
from api_schemas import QueryRequest, QueryResponse
from rag_engine import RAGEngine

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RAG Backend API")

# Initialize RAG engine (singleton pattern)
rag_engine = None

@app.on_event("startup")
def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    try:
        print("Initializing RAG Engine...")
        rag_engine = RAGEngine()
        rag_engine.reset_knowledge_base()
        rag_engine.ingest_initial_document()
        print("RAG Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")
        raise

@app.get("/")
def health():
    """Health check endpoint"""
    return {
        "status": "RAG backend running",
        "documents_indexed": rag_engine.vector_store.collection.count() if rag_engine else 0
    }

@app.get("/health")
def detailed_health():
    """Detailed health check"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return {
        "status": "healthy",
        "documents_indexed": rag_engine.vector_store.collection.count(),
        "embedding_model": rag_engine.embedding_manager.model.get_sentence_embedding_dimension(),
        "collection_name": rag_engine.vector_store.collection.name
    }

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the RAG system with conversation history
    
    Args:
        request: QueryRequest containing question, optional history, and top_k
        
    Returns:
        QueryResponse with the answer
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        # Validate question
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Query with history support
        answer = rag_engine.query(
            question=request.question,
            history=request.history if request.history else [],
            top_k=request.top_k if request.top_k else 3
        )
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

# Save uploaded PDF
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Ingest ONLY new PDFs
        rag_engine.ingest_uploaded_pdf(file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"{file.filename} uploaded and indexed successfully"}

@app.post("/reset")
def reset_session():
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    rag_engine.reset_knowledge_base()
    return {"status": "new session started"}



@app.post("/clear-history")
def clear_history():
    """Endpoint to clear conversation history (if needed for future enhancements)"""
    return {"status": "History management handled client-side"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)