import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# ------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "health"
PINECONE_ENV = "us-east-1"     # Your serverless region
NAMESPACE = None               # Optional: set to a string

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment.")

# ------------------------------------------------------
# INITIALIZE PINECONE v2 CLIENT
# ------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to existing index
index = pc.Index(name=PINECONE_INDEX_NAME)
print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

# ------------------------------------------------------
# EMBEDDING MODEL
# ------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ------------------------------------------------------
# TEXT SPLITTER
# ------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# ------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    output = ""
    for page in doc:
        output += page.get_text()
    doc.close()
    return output

# ------------------------------------------------------
# DOCUMENT INGESTION
# ------------------------------------------------------
def ingest_document(document_path):
    """
    Extracts text, splits into chunks, embeds, and upserts into Pinecone.
    """
    print(f"Processing: {document_path}")

    # Extract text depending on file type
    if document_path.lower().endswith(".pdf"):
        text = extract_text_from_pdf(document_path)
    else:
        with open(document_path, "r", encoding="utf-8") as f:
            text = f.read()

    # Split into chunks
    chunks = text_splitter.split_text(text)
    print(f"- Created {len(chunks)} text chunks")

    # Embed chunks
    vectors = embeddings.embed_documents(chunks)

    # Prepare upserts
    upserts = []
    file_id = os.path.basename(document_path)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        upserts.append({
            "id": f"{file_id}_{i}",
            "values": vector,
            "metadata": {"text": chunk, "source": file_id}
        })

    # Upsert to Pinecone
    index.upsert(vectors=upserts, namespace=NAMESPACE)
    print(f"✓ Upserted {len(upserts)} vectors into Pinecone.\n")


# ------------------------------------------------------
# BATCH INGESTION
# ------------------------------------------------------
if __name__ == "__main__":
    directory = "document"

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found.")

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            try:
                ingest_document(file_path)
            except Exception as e:
                print(f"❌ Error ingesting {filename}: {e}")
        else:
            print(f"Skipping directory: {filename}")
