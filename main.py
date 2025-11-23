# -------------------------
# File: main.py
# -------------------------
# ChatBot class: RAG pipeline using Pinecone v2 + HuggingFace InferenceClient
# Features:
# - Pinecone v2 client with LangChain
# - HuggingFaceEmbeddings for document vectors
# - InferenceClient for LLM calls
# - LRU caching for repeated prompts
# - Robust error handling

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

import pinecone
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# Logging & env
# -------------------------
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "health")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUG_TOKEN_1")
HF_RAG_MODEL = os.getenv("HF_RAG_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


# -------------------------
# Initialize Pinecone v2 client
# -------------------------
pinecone_client = None
if PINECONE_API_KEY:
    try:
        pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        logging.info(f"Pinecone client ready. Index: {PINECONE_INDEX}")
    except Exception as e:
        logging.warning(f"Pinecone init failed: {e}")
else:
    logging.warning("PINECONE_API_KEY not found; Pinecone vector DB disabled.")


# -------------------------
# ChatBot class
# -------------------------
class ChatBot:
    def __init__(self, index_name: str = PINECONE_INDEX, hf_model: str = HF_RAG_MODEL, hf_token: str = HUGGINGFACE_TOKEN):
        self.index_name = index_name
        self.hf_model = hf_model
        self.hf_token = hf_token

        # Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            self.embeddings = None
            logging.warning(f"Failed to initialize HuggingFaceEmbeddings: {e}")

        # Connect to Pinecone index (v2)
        if pinecone_client and self.embeddings:
            try:
                self.docsearch = LC_Pinecone.from_existing_index(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    client=pinecone_client  # v2 client
                )
                self.retriever = self.docsearch.as_retriever()
                logging.info(f"Connected to Pinecone index '{self.index_name}'")
            except Exception as e:
                logging.warning(f"Failed to connect to Pinecone index '{self.index_name}': {e}")
                self.docsearch = None
                self.retriever = None
        else:
            self.docsearch = None
            self.retriever = None

        # HuggingFace InferenceClient
        if self.hf_token:
            try:
                self.hf_client = InferenceClient(model=self.hf_model, token=self.hf_token)
            except Exception as e:
                self.hf_client = None
                logging.warning(f"Failed to create HuggingFace InferenceClient: {e}")
        else:
            self.hf_client = None
            logging.warning("HuggingFace token missing; LLM calls disabled.")

        # Prompt template
        self.prompt_template = (
            "You are a political scholar who relies strictly on verifiable, evidence-based information
