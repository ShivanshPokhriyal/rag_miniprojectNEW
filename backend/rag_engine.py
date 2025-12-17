from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(os.path.join(BASE_DIR, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Embedding Manager
# =========================

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)


# =========================
# Vector Store (ChromaDB)
# =========================

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents"):
        self.client = chromadb.Client()  # IN-MEMORY ONLY
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Session-based RAG embeddings"},
        )
        logger.info(f"Initialized ChromaDB collection: {collection_name}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids, texts, metas, emb_list = [], [], [], []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            texts.append(doc.page_content)
            metas.append(doc.metadata)
            emb_list.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=emb_list,
        )
        logger.info(f"Added {len(documents)} document chunks to vector store")

    def get_indexed_sources(self) -> set:
        """Get set of all source files currently indexed"""
        try:
            existing = self.collection.get(include=["metadatas"])
            metas = existing.get("metadatas") or []
            sources = set()
            for m in metas:
                if m and isinstance(m, dict):
                    src = m.get("source_file")
                    if src:
                        sources.add(src)
            return sources
        except Exception as e:
            logger.warning(f"Error getting indexed sources: {e}")
            return set()


# =========================
# PDF Processing
# =========================

class PDFProcessor:
    @staticmethod
    def load_pdfs(directory: str) -> List[Any]:
        documents = []
        pdf_files = Path(directory).glob("**/*.pdf")

        for pdf in pdf_files:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for page in pages:
                page.metadata["source_file"] = pdf.name
            documents.extend(pages)

        logger.info(f"Loaded {len(documents)} pages from {directory}")
        return documents

    @staticmethod
    def split_documents(
        documents: List[Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks


# =========================
# Retriever
# =========================

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_docs = []

        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                retrieved_docs.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": 1 - dist,
                })

        logger.info(f"Retrieved {len(retrieved_docs)} documents for query")
        return retrieved_docs


# =========================
# RAG Engine (MAIN CLASS)
# =========================

class RAGEngine:
    def __init__(self):
        # Load environment variables from .env file in project root
        load_dotenv(os.path.join(BASE_DIR, ".env"))
        
        # Get API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        logger.info(f"GROQ_API_KEY value: {repr(groq_api_key)}")
        logger.info(f"GROQ_API_KEY length: {len(groq_api_key) if groq_api_key else 'None'}")

        # FIX 1: Use environment variable instead of hardcoded key
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            logger.error(f"Looking for .env file at: {BASE_DIR / '.env'}")
            raise ValueError("GROQ_API_KEY not found in environment variables. Please create a .env file with GROQ_API_KEY=your_key")

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,  # Use the loaded key
            model_name="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=1024,
        )
        logger.info("RAG Engine initialized successfully")
    
    def ingest_initial_document(self):
        """
        CRITICAL: Ingest README_APP.txt with developer details
        This MUST always be present in the knowledge base
        """
        doc_path = BASE_DIR / "content" / "README_APP.txt"
        if not doc_path.exists():
            logger.error("CRITICAL: README_APP.txt not found! Developer details missing.")
            raise FileNotFoundError(f"README_APP.txt must exist at {doc_path}")

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            logger.error("CRITICAL: README_APP.txt is empty!")
            raise ValueError("README_APP.txt cannot be empty - it contains developer details")

        from langchain_core.documents import Document

        # Check if already indexed to prevent duplicate ingestion
        existing_sources = self.vector_store.get_indexed_sources()
        if "README_APP.txt" in existing_sources:
            logger.info("README_APP.txt already indexed, skipping duplicate ingestion")
            return

        doc = Document(
            page_content=text,
            metadata={"source_file": "README_APP.txt", "document_type": "developer_info"}
        )

        chunks = PDFProcessor.split_documents([doc])

        if not chunks:
            logger.error("CRITICAL: No chunks created from README_APP.txt")
            raise ValueError("Failed to chunk README_APP.txt")

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        if embeddings is None or len(embeddings) == 0:
            logger.error("CRITICAL: Embedding generation failed for README_APP.txt")
            raise RuntimeError("Failed to generate embeddings for developer document")

        self.vector_store.add_documents(chunks, embeddings)
        logger.info(f"✓ README_APP.txt ingested successfully ({len(chunks)} chunks)")

    def reset_knowledge_base(self):
        """
        Completely reset vector store and ALWAYS re-ingest README_APP.txt
        This ensures developer details are always available after reset
        """
        try:
            self.vector_store.client.delete_collection(
                self.vector_store.collection.name
            )
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")

        # Recreate vector store and retriever
        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_manager
        )

        # CRITICAL: Always re-ingest developer document after reset
        self.ingest_initial_document()
        logger.info("✓ Knowledge base reset complete with README_APP.txt restored")

    def ingest_pdfs(self):
        """Ingest PDFs from content directory (avoiding duplicates)"""
        pdf_dir = BASE_DIR / "content"

        if not pdf_dir.exists():
            raise RuntimeError(f"PDF directory not found: {pdf_dir}")

        docs = PDFProcessor.load_pdfs(str(pdf_dir))
        chunks = PDFProcessor.split_documents(docs)
        if not chunks:
            logger.info("No document chunks found to ingest")
            return

        # Get existing sources to avoid re-ingestion
        existing_sources = self.vector_store.get_indexed_sources()

        # Filter to only new chunks
        new_chunks = [c for c in chunks if c.metadata.get("source_file") not in existing_sources]

        if not new_chunks:
            logger.info("No new PDFs to ingest (all files already indexed)")
            return

        texts = [c.page_content for c in new_chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        self.vector_store.add_documents(new_chunks, embeddings)
        
        new_files = len(set([c.metadata.get('source_file') for c in new_chunks]))
        logger.info(f"Ingested {len(new_chunks)} chunks from {new_files} new file(s)")

    def _format_history(self, history: List[Dict[str, str]], max_turns: int = 3) -> str:
        """Format conversation history for context"""
        if not history:
            return ""
        
        # Only use last N turns to avoid context overflow
        recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
        
        formatted = "Previous conversation:\n"
        for msg in recent_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        
        return formatted + "\n"

    def ingest_uploaded_pdf(self, pdf_path: str):
        """FIX 3: Check if file already indexed before ingesting"""
        filename = os.path.basename(pdf_path)
        
        # Check if already indexed
        existing_sources = self.vector_store.get_indexed_sources()
        if filename in existing_sources:
            logger.info(f"PDF already indexed: {filename}, skipping")
            return

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for page in docs:
            page.metadata["source_file"] = filename

        chunks = PDFProcessor.split_documents(docs)

        if not chunks:
            logger.warning(f"No chunks found in uploaded PDF: {filename}")
            return

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        self.vector_store.add_documents(chunks, embeddings)
        logger.info(f"Ingested uploaded PDF: {filename} ({len(chunks)} chunks)")

    def query(self, question: str, history: List[Dict[str, str]] = None, top_k: int = 3) -> str:
        """
        Query the RAG system with conversation history
        
        Args:
            question: Current user question
            history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            top_k: Number of documents to retrieve
        """
        if history is None:
            history = []
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k)

        if not results:
            logger.warning("No relevant context found for query")
            return "No relevant context found in the documents."

        context = "\n\n".join([r["content"] for r in results])
        
        # Format conversation history
        history_text = self._format_history(history)

        # Build prompt with history
        prompt = f"""You are a helpful assistant answering questions based on provided documents.

{history_text}Context from documents:
{context}

Current question: {question}

Instructions:
- You are an expert teacher explaining concepts clearly to a student.
- Use the provided context as the primary source of information.
- If the context partially covers the topic, explain it clearly and logically using that information.
- Do NOT talk about the context itself unless the information is truly missing.
- Avoid meta statements like "the context does not define" or "based on the context".
- If essential information is missing, state it briefly and then give a reasonable high-level explanation.
- Write in a direct, confident, and exam-oriented style.
- Be concise, clear, and structured.

Answer:"""

        response = self.llm.invoke(prompt)
        logger.info(f"Generated response for query: {question[:50]}...")
        return response.content