import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_pdfs_from_folder(folder_path):
    """Loads all PDFs from a given folder and extracts text."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    documents = []
    
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
    
    return documents

def process_and_store_documents(folder_path, persist_directory="./vector_db"):
    """Processes PDFs, splits text, and stores embeddings in ChromaDB."""
    # Load and split documents with improved chunking strategy
    documents = load_pdfs_from_folder(folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=216, chunk_overlap=70)
    split_docs = text_splitter.split_documents(documents)
    
    # Generate embeddings and store in ChromaDB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_directory)
    
    print("Documents processed and stored in vector database.")
    return vector_store

def query_rag(vector_store, query):
    """Retrieves the top 5 most relevant document chunks based on semantic similarity."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 matches
    results = retriever.get_relevant_documents(query)
    
    # Format the retrieved results
    answer_text = "\n\n".join([f"Match {i+1}: {doc.page_content}" for i, doc in enumerate(results)])
    
    # Save output to file
    formatted_output = f"Query: {query}\nTop 5 Matches:\n{answer_text}\n{'-'*50}\n"
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(formatted_output)
    
    return formatted_output

if __name__ == "__main__":
    folder_path = "Pdfs"
    vector_store = process_and_store_documents(folder_path)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = query_rag(vector_store, query)
        print("\n", answer)