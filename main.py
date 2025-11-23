from typing import List
# -------------------------
# File: main.py
# -------------------------
# ChatBot class: RAG pipeline using Pinecone + Hugging Face Router (InferenceClient)
# - Uses huggingface_hub InferenceClient (router.huggingface.co)
# - Uses langchain_huggingface embeddings for Pinecone
# - Robust error handling and simple in-memory caching for recent queries

import os
from typing import List
import time
import logging
from functools import lru_cache
from typing import List

List

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
                self.retriever = self.docsearch.as_retriever()
            except Exception as e:
                logging.warning(f"Failed to connect to Pinecone index '{self.index_name}': {e}")
                self.docsearch = None
                self.retriever = None
        else:
            self.docsearch = None
            self.retriever = None

        # Create direct HuggingFace router client (InferenceClient) for stable calls
        if self.hf_token:
            try:
                self.hf_client = InferenceClient(model=self.hf_model, token=self.hf_token)
            except Exception as e:
                self.hf_client = None
                logging.warning(f"Failed to create InferenceClient: {e}")
        else:
            self.hf_client = None
            logging.warning("HuggingFace token missing; LLM calls disabled.")

        # Prompt template (kept concise for LLM)
        self.prompt_template = (
            "You are a political scholar who adheres strictly to factual information from reliable sources.\n"
            "Assess whether the user's post is VERIFIED or UNVERIFIED based only on the context below.\n"
            "If insufficient information, reply: 'Please be informed I am limited to the content in my database. Kindly check back for an update.'\n"
            "Context: {context} Question: {question} Answer:")

        # Simple LRU cache for recent queries (helps reduce duplicate LLM calls)
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        """Fetch top_k documents from Pinecone and return concatenated page_content.
        If Pinecone/retriever is not available, return an informative empty string.
        """
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.get_relevant_documents(question)
            # Limit to top_k if retriever returns many
            docs = docs[:top_k]
            context_text = "

".join(d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip())
            return context_text
        except Exception as e:
            logging.exception(f"retrieve_context error: {e}")
            return ""

    def _call_llm(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6):
        """Internal method to call the HF router via InferenceClient. This is cached via lru_cache.
        Returns the generated text or raises an exception which the caller handles.
        """
        if not self.hf_client:
            raise RuntimeError("HuggingFace Inference client not available. Set HUG_TOKEN_1.")

        try:
            response = self.hf_client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            if isinstance(response, str):
                return response
            if isinstance(response, list) and len(response) > 0:
                first = response[0]
                if isinstance(first, dict):
                    return first.get("generated_text") or first.get("text") or str(first)
                return str(first)

            if isinstance(response, dict):
                return response.get("generated_text") or response.get("text") or str(response)

            return str(response)

        except Exception as e:
            logging.exception(f"LLM call failed: {e}")
            raise

    def ask_question(self, question: str, use_cache: bool = True) -> str:
        """Main public method: retrieve context, build prompt and call LLM.
        Returns a clean string answer and logs errors gracefully.
        """
        try:
            context = self.retrieve_context(question)

            if not context.strip():
                return "Please be informed I am limited to the content in my database. Kindly check back for an update."

            full_prompt = self.prompt_template.format(context=context, question=question)

            if use_cache:
                result = self._ask_cache(full_prompt)
            else:
                result = self._call_llm(full_prompt)

            if isinstance(result, str):
                return result.strip()

            return str(result)

        except RuntimeError as e:
            logging.error(f"Runtime error in ask_question: {e}")
            return "LLM is not available. Check your HuggingFace token."
        except Exception as e:
            logging.exception(f"ask_question failed: {e}")
            return "An internal error occurred. Please try again later."
