import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

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

def generate_vector_store():
    folder_path = "Pdfs"
    vector_store = process_and_store_documents(folder_path)
    return vector_store


def query_rag(vector_store, query):
    """Retrieves the top 10 most relevant document chunks and uses Ollama LLM to generate an answer."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Retrieve top 10 matches
    results = retriever.get_relevant_documents(query)
    
    # Format the retrieved results
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Query Ollama LLM
    prompt = f"Based on the following retrieved information, provide a well-structured answer:\n\n{context}\n\nUser Query: {query}"
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    answer_text = response['message']['content']
    
    # Save output to file
    formatted_output = f"Query: {query}\nAnswer: {answer_text}\n{'-'*50}\n"
    with open("output_final.txt", "a", encoding="utf-8") as f:
        f.write(formatted_output)
    
    return formatted_output

if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    ask=input("Have u added anything to you knowledge repository after the last vectorization...? (yes/no): ")
    if(ask=='yes'):
        vector_store=generate_vector_store()
    else:   
        PERSIST_DIRECTORY = "./vector_db"
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)

    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = query_rag(vector_store, query)
        print("\n", answer)