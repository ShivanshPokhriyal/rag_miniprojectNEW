from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Message(BaseModel):
    """Single conversation message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str = Field(..., description="User's question")
    history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Conversation history as list of {role, content} dicts"
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant documents to retrieve (1-10)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the company's revenue?",
                "history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi! How can I help?"}
                ],
                "top_k": 3
            }
        }


class QueryResponse(BaseModel):
    """Response model from the RAG system"""
    answer: str = Field(..., description="Generated answer from RAG system")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "According to the documents, the company's revenue is $10M."
            }
        }