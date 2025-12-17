from pathlib import Path
import os
import shutil
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from backend.api_schemas import QueryRequest, QueryResponse
from backend.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info("Initializing RAG Engine...")
        rag_engine = RAGEngine()
        # Don't reset on startup - just ingest initial document
        rag_engine.ingest_initial_document()
        doc_count = rag_engine.vector_store.collection.count()
        logger.info(f"RAG Engine initialized successfully ({doc_count} chunks from README_APP.txt)")
    except Exception as e:
        logger.error(f"Error initializing RAG Engine: {e}")
        raise

@app.get("/")
def health():
    """Health check endpoint"""
    doc_count = rag_engine.vector_store.collection.count() if rag_engine else 0
    logger.info(f"Health check: {doc_count} documents indexed")
    return {
        "status": "RAG backend running",
        "documents_indexed": doc_count
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
        "collection_name": rag_engine.vector_store.collection.name,
        "indexed_sources": list(rag_engine.vector_store.get_indexed_sources())
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
        
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Query with history support
        answer = rag_engine.query(
            question=request.question,
            history=request.history if request.history else [],
            top_k=request.top_k if request.top_k else 3
        )
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """FIX 4: Add deduplication check and better error handling"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Check if already indexed
    existing_sources = rag_engine.vector_store.get_indexed_sources()
    if file.filename in existing_sources:
        logger.info(f"PDF already indexed: {file.filename}")
        return {
            "message": f"{file.filename} is already indexed",
            "already_indexed": True
        }

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded PDF
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file: {file.filename}")
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        # Ingest uploaded PDF (already has internal deduplication)
        rag_engine.ingest_uploaded_pdf(file_path)
        logger.info(f"Successfully indexed: {file.filename}")
        
        return {
            "message": f"{file.filename} uploaded and indexed successfully",
            "already_indexed": False
        }

    except Exception as e:
        logger.error(f"Error ingesting {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing file: {str(e)}")
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file.filename}")
        except Exception as e:
            logger.warning(f"Error cleaning up {file_path}: {e}")

@app.post("/reset")
def reset_session():
    """FIX 5: Reset clears everything and re-ingests initial document"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    try:
        logger.info("Resetting knowledge base...")
        rag_engine.reset_knowledge_base()  # Now includes initial document re-ingestion
        doc_count = rag_engine.vector_store.collection.count()
        logger.info(f"Knowledge base reset complete. Documents: {doc_count}")
        
        return {
            "status": "new session started",
            "documents_indexed": doc_count
        }
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting: {str(e)}")

@app.post("/clear-history")
def clear_history():
    """Endpoint to clear conversation history (handled client-side)"""
    return {"status": "History management handled client-side"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)