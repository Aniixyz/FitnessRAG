import os
import time
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load the API key from the .env file
load_dotenv()

def run_ingestion():
    print("1. Loading PDFs from Data folder...")
    data_dir = "Data" if os.path.exists("Data") else "data"
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs.")
    
    print("2. Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("3. Building Vector Database in 'chroma_db'...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Save the database locally in a folder named 'chroma_db'
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )
    
    # Ingest in batches of 50 to adhere to Gemini free tier rate limits (100 req/min)
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Ingesting batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        for attempt in range(5):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 35 + attempt * 10
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
                    
        if i + batch_size < len(chunks):
            print("Pausing 30s to stay within Gemini API rate limits...")
            time.sleep(30)
            
    print("Database built successfully!")

if __name__ == "__main__":
    run_ingestion()