"""
api/app.py - FastAPI server for RankNet search engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import SearchEngine
from src.model import Document

# Initialize FastAPI
app = FastAPI(
    title="RankNet Search API",
    description="Learning to Rank Search Engine",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global search engine
search_engine = None


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    title: str
    content: str
    url: str
    author: str
    score: float
    views: int
    timestamp: str


class ClickFeedback(BaseModel):
    query_id: str
    doc_id: str
    position: int
    dwell_time: float


class DocumentInput(BaseModel):
    title: str
    content: str
    url: str
    author: str
    tags: List[str] = []


class TrainingRequest(BaseModel):
    epochs: int = 50
    batch_size: int = 32


# API Endpoints
@app.on_event("startup")
async def startup():
    """Initialize search engine on startup"""
    global search_engine
    
    print("Initializing search engine...")
    search_engine = SearchEngine(use_ranknet=True)
    
    # Try to load saved model
    try:
        search_engine.load("data/models/")
        print("Loaded saved model")
    except:
        # Create sample data if no saved model
        search_engine.create_sample_data(n_docs=100)
        print("Created sample data")
    
    print("Search engine ready!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RankNet Search API",
        "endpoints": [
            "/search",
            "/feedback",
            "/documents",
            "/train",
            "/evaluate",
            "/stats"
        ]
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Search endpoint"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    results = search_engine.search(request.query, request.top_k)
    
    return {
        "query": request.query,
        "results": results,
        "total": len(results)
    }


@app.post("/feedback")
async def feedback(feedback: ClickFeedback):
    """Log click feedback"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    search_engine.log_click(
        feedback.query_id,
        feedback.doc_id,
        feedback.position,
        feedback.dwell_time
    )
    
    return {"message": "Feedback logged"}


@app.post("/documents")
async def add_document(doc: DocumentInput):
    """Add new document"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    doc_id = f"doc_{datetime.now().timestamp()}"
    document = Document(
        doc_id=doc_id,
        title=doc.title,
        content=doc.content,
        url=doc.url,
        author=doc.author,
        timestamp=datetime.now(),
        tags=doc.tags
    )
    
    search_engine.add_document(document)
    
    return {"doc_id": doc_id, "message": "Document added"}


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document by ID"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if doc_id not in search_engine.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = search_engine.documents[doc_id]
    return doc.to_dict()


@app.post("/train")
async def train(request: TrainingRequest):
    """Train RankNet model"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    history = search_engine.train(
        epochs=request.epochs,
        batch_size=request.batch_size
    )
    
    # Save model
    search_engine.save("data/models/")
    
    return {
        "message": "Training completed",
        "epochs": request.epochs,
        "history": history
    }


@app.get("/evaluate")
async def evaluate():
    """Evaluate search engine"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    metrics = search_engine.evaluate()
    
    return metrics


@app.get("/stats")
async def stats():
    """Get statistics"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    return {
        "num_documents": len(search_engine.documents),
        "num_queries": len(search_engine.query_log),
        "num_clicks": len(search_engine.click_log),
        "has_model": search_engine.ranknet is not None,
        "corpus_stats": {
            "n_docs": search_engine.corpus_stats.get('n_docs', 0),
            "avg_doc_len": search_engine.corpus_stats.get('avg_doc_len', 0)
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)