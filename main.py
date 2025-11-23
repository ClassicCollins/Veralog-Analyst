# -------------------------
# File: main.py
# -------------------------
# ChatBot class: RAG pipeline using Pinecone + Hugging Face Router (InferenceClient)
# - Uses huggingface_hub InferenceClient (router.huggingface.co)
# - Uses langchain_huggingface embeddings for Pinecone
# - Robust error handling and simple in-memory caching for recent queries


import os
import time
import logging
from functools import lru_cache
from typing import List


import pinecone
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings


# Optional: reduce noisy logs
logging.getLogger("urllib3").setLevel(logging.WARNING)


load_dotenv()


# Basic configuration
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "health")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUG_TOKEN_1")
HF_RAG_MODEL = os.getenv("HF_RAG_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


# Initialize Pinecone client on import (safe no-op if keys missing)
if PINECONE_API_KEY:
try:
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
except Exception as e:
pinecone_client = None
logging.warning(f"Pinecone init failed: {e}")
else:
pinecone_client = None
logging.warning("PINECONE_API_KEY not found in environment; vector DB disabled.")




class ChatBot:
def __init__(self, index_name: str = PINECONE_INDEX, hf_model: str = HF_RAG_MODEL, hf_token: str = HUGGINGFACE_TOKEN):
# Load and validate environment variables
self.index_name = index_name
self.hf_model = hf_model
self.hf_token = hf_token


# Embeddings for Pinecone (used by LangChain interface)
try:
self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
self.embeddings = None
logging.warning(f"Failed to initialize HuggingFaceEmbeddings: {e}")


# Connect to Pinecone/LangChain vectorstore if Pinecone is ready
if pinecone_client and self.embeddings:
try:
self.docsearch = LC_Pinecone.from_existing_index(index_name=self.index_name, embedding=self.embeddings)
return "An internal error occurred. Please try again later."
