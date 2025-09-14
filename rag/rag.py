from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class TinyRAG:
    def __init__(self, model_name="tinyllama"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vectorstore = None
        self.qa_chain = None
        
        self.prompt_template = """Answer the question based only on the following context:
        
        {context}
        
        Question: {question}
        
        Answer clearly and concisely:"""
    
    def load_documents(self, file_path):
        """Load and split documents"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            loader = TextLoader(file_path)
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=30,
                separator="\n"
            )
            texts = text_splitter.split_documents(documents)
            return texts
        except Exception as e:
            print(f"Error loading document: {e}")
            return []
    
    def setup_vectorstore(self, texts):
        """Create vector store with documents"""
        os.makedirs("../vector_db", exist_ok=True)
        
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="../vector_db"
        )
    
    def create_qa_chain(self):
        """Create QA chain with optimized prompt"""
        if not self.vectorstore:
            raise ValueError("Please setup vectorstore first")
        
        PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
            
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question, max_length=150):
        """Query the RAG system"""
        if not self.qa_chain:
            return {"error": "Please initialize QA chain first"}
        
        try:
            result = self.qa_chain({"query": question})
            
            answer = result["result"].strip()
            if len(answer) > max_length:
                answer = answer[:max_length] + "..."
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "source": doc.metadata.get('source', 'unknown')
                    } for doc in result["source_documents"]
                ],
                "model": self.model_name
            }
        except Exception as e:
            return {"error": str(e), "model": self.model_name}

__all__ = ['TinyRAG']