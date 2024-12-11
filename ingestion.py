import os
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
import fitz  # PyMuPDF for extracting text from PDF files

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment='us-east1-aws')
index = pc.Index("health")

# Initialize embeddings
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Set up text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file using PyMuPDF.
    """
    pdf_document = fitz.open(pdf_path)
    text = ''
    for page in range(pdf_document.page_count):
        text += pdf_document.load_page(page).get_text()
    pdf_document.close()
    return text


# Function to ingest a document
def ingest_document(document_path):
    """
    Handles ingestion of a document by extracting text, splitting it, embedding chunks, 
    and upserting into Pinecone.
    """
    # Determine the type of document
    if document_path.endswith('.pdf'):
        # Extract text from PDF
        text = extract_text_from_pdf(document_path)
    else:
        # For other formats, simply read the text
        with open(document_path, 'r') as file:
            text = file.read()

    # Split the document into chunks
    texts = text_splitter.split_text(text)

    # Prepare embeddings and upsert with metadata into Pinecone index
    upsert_data = []
    for i, text_chunk in enumerate(texts):
        unique_id = f"{os.path.basename(document_path)}_{i}"  # Create a unique ID for each chunk
        vector = embeddings.embed_query(text_chunk)  # Generate the embedding for each chunk
        # Append embedding and metadata to the upsert list
        upsert_data.append({
            "id": unique_id,
            "values": vector,
            "metadata": {"text": text_chunk}  # Map the chunk's content as metadata
        })

    # Upload all the embeddings and metadata at once
    index.upsert(upsert_data)
    print(f"Successfully upserted {len(upsert_data)} chunks into Pinecone.")

   


if __name__ == "__main__":
    # Directory containing documents
    directory_path = "document"  # Path to the directory containing files

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Ensure it's a file and not a subdirectory
        if os.path.isfile(file_path):
            try:
                # Ingest the document
                ingest_document(file_path)
                print(f"Document '{filename}' ingested successfully.")
            except Exception as e:
                print(f"Error ingesting '{filename}': {e}")
