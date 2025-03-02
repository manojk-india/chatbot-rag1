import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    # Generate embeddings and store in ChromaDB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(split_docs, embedding_model, persist_directory=persist_directory)
    
    print("Documents processed and stored in vector database.")
    return vector_store

def query_rag(vector_store, query):
    """Retrieves relevant documents and generates an answer using LLM."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
    
    # Load model and tokenizer
    model_name = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    answer = qa_chain.invoke(query)
    
    # Extract answer text from the dictionary
    answer_text = answer.get("result", "").strip()
    if not answer_text or "use the following pieces of context" in answer_text.lower():
        answer_text = "I don't know the answer to that question."
    
    # Format output and save to file
    formatted_output = f"Query: {query}\nAnswer: {answer_text}\n{'-'*50}\n"
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(formatted_output)
    
    return formatted_output

if __name__ == "__main__":
    folder_path ="Pdfs"
    vector_store = process_and_store_documents(folder_path)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        answer = query_rag(vector_store, query)
        print("\n", answer)
