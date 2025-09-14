import requests
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api_connection():
    """Test if API is running"""
    base_url = "http://localhost:8000"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print("✅ API is running:", response.json())
        return True
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Start it with: python -m api.api")
        return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def test_rag_functionality():
    """Test the RAG functionality"""
    base_url = "http://localhost:8000"
    
    if not test_api_connection():
        return
    
    print("\n🧪 Testing RAG functionality...")
    
    # Sample content about the project itself
    sample_content = """ZDD AI is a project for testing RAG systems with TinyLlama.
    RAG stands for Retrieval Augmented Generation.
    This system combines document retrieval with language model generation.
    TinyLlama is a small language model good for testing and development.
    The project uses LangChain for the RAG pipeline and FastAPI for the web interface."""
    
    # Load document
    print("📝 Loading test document...")
    doc_data = {
        "content": sample_content,
        "filename": "zddai_test.txt"
    }
    
    try:
        response = requests.post(f"{base_url}/load-text", json=doc_data, timeout=30)
        load_result = response.json()
        print("✅ Document loaded:", load_result)
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return
    
    # Test queries
    test_questions = [
        "What is ZDD AI?",
        "What does RAG stand for?",
        "What is TinyLlama used for?",
        "What technologies does this project use?"
    ]
    
    print("\n🤖 Testing queries...")
    for question in test_questions:
        print(f"\n   Q: {question}")
        try:
            response = requests.post(f"{base_url}/query", json={"question": question}, timeout=30)
            result = response.json()
            
            if "answer" in result:
                print(f"   A: {result['answer']}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        time.sleep(2)  # Be nice to TinyLlama

def test_direct_rag():
    """Test RAG directly without API"""
    print("\n🔧 Testing RAG directly...")
    
    try:
        from rag.rag import TinyRAG
        
        # Create test document
        test_content = """Direct testing of the RAG system.
        This test bypasses the API and tests the core functionality.
        The RAG system should be able to answer questions about this content."""
        
        test_file = "../data/direct_test.txt"
        os.makedirs("../data", exist_ok=True)
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        # Initialize and test
        rag = TinyRAG(model_name="tinyllama")
        texts = rag.load_documents(test_file)
        
        if texts:
            rag.setup_vectorstore(texts)
            rag.create_qa_chain()
            
            result = rag.query("What is being tested?")
            print("✅ Direct test result:", result.get('answer', 'No answer'))
        else:
            print("❌ No texts were loaded")
            
    except Exception as e:
        print(f"❌ Direct test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting ZDD AI RAG System Tests")
    print("=" * 50)
    
    # Test API connection
    test_api_connection()
    
    # Test through API
    test_rag_functionality()
    
    # Test directly
    test_direct_rag()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")