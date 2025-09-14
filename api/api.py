from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import os
import sys

# Add the parent directory to Python path to import rag module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from rag.rag import TinyRAG

# Global RAG instance
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_system
    try:
        rag_system = TinyRAG(model_name="tinyllama")
        print("✅ TinyLlama RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        rag_system = None
    yield
    # Shutdown
    # Add any cleanup code here

app = FastAPI(
    title="TinyLlama RAG API",
    description="RAG system with TinyLlama for testing",
    version="1.0.0",
    lifespan=lifespan  # Add lifespan manager
)

# Get the absolute path to the api directory
api_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(api_dir, "templates")
static_dir = os.path.join(api_dir, "static")

# Create directories if they don't exist
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# Setup templates with absolute path
templates = Jinja2Templates(directory=templates_dir)

# Mount static files directory with absolute path
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    question: str

class DocumentRequest(BaseModel):
    content: str
    filename: str = "document.txt"

# Global RAG instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = TinyRAG(model_name="tinyllama")
        print("✅ TinyLlama RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        rag_system = None

@app.get("/")
async def root():
    return {
        "message": "TinyLlama RAG API is running", 
        "model": "tinyllama",
        "endpoints": ["/health", "/load-text", "/query"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_system else "unhealthy",
        "model": "tinyllama",
        "rag_initialized": rag_system is not None
    }

@app.post("/load-text")
async def load_text(request: DocumentRequest):
    """Load text content and process it"""
    if not rag_system:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        # Ensure data directory exists
        os.makedirs("../data", exist_ok=True)
        filepath = f"../data/{request.filename}"
        
        # Save content to file
        with open(filepath, "w") as f:
            f.write(request.content)
        
        # Process document
        texts = rag_system.load_documents(filepath)
        rag_system.setup_vectorstore(texts)
        rag_system.create_qa_chain()
        
        return {
            "status": "success", 
            "chunks_processed": len(texts),
            "filename": request.filename,
            "model": "tinyllama"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG system"""
    if not rag_system:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        result = rag_system.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/models")
async def get_models():
    """Get available Ollama models"""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        return {"models": result.stdout.splitlines() if result.stdout else "No models found"}
    except Exception as e:
        return {"models": f"Error fetching models: {str(e)}"}

@app.get("/ui", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main UI page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Serve chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 