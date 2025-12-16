from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import os
import uuid
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

load_dotenv(os.path.join(BASE_DIR, ".env"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Embedding Manager
# =========================

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

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
        return splitter.split_documents(documents)


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

        return retrieved_docs


# =========================
# RAG Engine (MAIN CLASS)
# =========================

class RAGEngine:
    def __init__(self):
        load_dotenv()

        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)

        self.llm = ChatGroq(
            groq_api_key="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",
            model_name="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=1024,
        )
    
    def ingest_initial_document(self):
        doc_path = BASE_DIR / "content" / "README_APP.txt"
        if not doc_path.exists():
            print("README_APP.txt not found, skipping initial ingestion")
            return

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print("README_APP.txt is empty, skipping ingestion")
            return

        from langchain_core.documents import Document

        doc = Document(
            page_content=text,
            metadata={"source_file": "README_APP.txt"}
        )

        chunks = PDFProcessor.split_documents([doc])

        if not chunks:
            print("No chunks created from README")
            return

    # 🔹 DO NOT filter chunks
        texts = [c.page_content for c in chunks]

        embeddings = self.embedding_manager.generate_embeddings(texts)

        if embeddings is None or len(embeddings) == 0:
            print("Embedding generation failed")
            return

        self.vector_store.add_documents(chunks, embeddings)

        print(f"Initial developer document ingested ({len(chunks)} chunks)")


    def reset_knowledge_base(self):
        self.vector_store.client.delete_collection(
            self.vector_store.collection.name
        )

        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_manager
        )

        print("Vector store reset")


    def ingest_pdfs(self):
        pdf_dir = BASE_DIR / "content"

        if not pdf_dir.exists():
            raise RuntimeError(f"PDF directory not found: {pdf_dir}")

        docs = PDFProcessor.load_pdfs(str(pdf_dir))
        chunks = PDFProcessor.split_documents(docs)
        if not chunks:
            print("No document chunks found to ingest")
            return

        # Try to detect already-ingested source files and only ingest new PDFs
        existing_sources = set()
        try:
            # request existing metadatas from the collection
            existing = self.vector_store.collection.get(include=["metadatas"])
            metas = existing.get("metadatas") or []
            for m in metas:
                if not m:
                    continue
                src = m.get("source_file") if isinstance(m, dict) else None
                if src:
                    existing_sources.add(src)
        except Exception:
            # If the collection is empty or the get call fails, fall back to count check
            existing_sources = set()

        # Filter to only chunks coming from source files not already present
        new_chunks = [c for c in chunks if c.metadata.get("source_file") not in existing_sources]

        if not new_chunks:
            print("No new PDFs to ingest (all files already indexed), skipping ingestion")
            return

        texts = [c.page_content for c in new_chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        self.vector_store.add_documents(new_chunks, embeddings)
        print(f"Ingested {len(new_chunks)} chunks from {len(set([c.metadata.get('source_file') for c in new_chunks]))} new file(s)")

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
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for page in docs:
            page.metadata["source_file"] = os.path.basename(pdf_path)

        chunks = PDFProcessor.split_documents(docs)

        if not chunks:
            print("No chunks found in uploaded PDF")
            return

        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        self.vector_store.add_documents(chunks, embeddings)

        print(f"Ingested uploaded PDF: {os.path.basename(pdf_path)}")


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
        return response.content