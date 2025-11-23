# -------------------------
# File: main.py
# -------------------------
# ChatBot class: RAG pipeline using Pinecone v2 + HuggingFace InferenceClient
# Fully fixed for Pinecone v2
# -------------------------

import os
import logging
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone

# Pinecone v2 imports
from pinecone import Pinecone, ServerlessSpec

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------------------------
# Load environment
# -------------------------
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
pinecone_index_obj = None
try:
    if PINECONE_API_KEY:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        # Ensure the index exists
        if PINECONE_INDEX not in [idx.name for idx in pinecone_client.list_indexes()]:
            pinecone_client.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # match sentence-transformers/all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        # Correct Pinecone v2 usage: Index() capital I
        pinecone_index_obj = pinecone_client.Index(PINECONE_INDEX)
        logging.info(f"Pinecone client ready. Index: {PINECONE_INDEX}")
except Exception as e:
    logging.warning(f"Pinecone init failed: {e}")
    pinecone_client = None
    pinecone_index_obj = None

# -------------------------
# ChatBot class
# -------------------------
class ChatBot:
    def __init__(self):
        # Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logging.warning(f"Failed to load embeddings: {e}")
            self.embeddings = None

        # Pinecone/LangChain vectorstore
        if pinecone_index_obj and self.embeddings:
            try:
                self.docsearch = LC_Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=self.embeddings)
                self.retriever = self.docsearch.as_retriever()
            except Exception as e:
                logging.warning(f"Failed to connect to Pinecone index: {e}")
                self.docsearch = None
                self.retriever = None
        else:
            self.docsearch = None
            self.retriever = None

        # HuggingFace LLM client
        try:
            if HUGGINGFACE_TOKEN:
                self.hf_client = InferenceClient(model=HF_RAG_MODEL, token=HUGGINGFACE_TOKEN)
            else:
                self.hf_client = None
                logging.warning("HuggingFace token missing; LLM calls disabled.")
        except Exception as e:
            self.hf_client = None
            logging.warning(f"Failed to initialize HuggingFace InferenceClient: {e}")

        # Prompt template
        self.prompt_template = (
            "You are a political scholar who relies strictly on verifiable, evidence-based information from credible sources.\n"
            "Evaluate the user's post using ONLY the context provided.\n"
            "Determine if the claim is VERIFIED or UNVERIFIED.\n"
            "If insufficient context, respond with: 'Please be informed I am limited to the content in my database. Kindly check back for an update.'\n\n"
            "Context:\n{context}\n\n"
            "Claim:\n{question}\n\n"
            "Answer:"
        )

        # LRU cache for repeated prompts
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        """Fetch top_k documents from Pinecone and concatenate page_content."""
        if not self.retriever:
            return ""
        try:
            docs = self.retriever.get_relevant_documents(question)
            docs = docs[:top_k]
            context_text = "\n\n".join(d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip())
            return context_text
        except Exception as e:
            logging.exception(f"retrieve_context error: {e}")
            return ""

    def _call_llm(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6):
        """Internal LLM call using HF InferenceClient."""
        if not self.hf_client:
            raise RuntimeError("HuggingFace Inference client not available.")

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
        """Main method: retrieve context and call LLM."""
        try:
            context = self.retrieve_context(question)
            if not context.strip():
                return "Please be informed I am limited to the content in my database. Kindly check back for an update."

            full_prompt = self.prompt_template.format(context=context, question=question)

            if use_cache:
                result = self._ask_cache(full_prompt)
            else:
                result = self._call_llm(full_prompt)

            return result.strip() if isinstance(result, str) else str(result)
        except Exception as e:
            logging.exception(f"ask_question failed: {e}")
            return "An internal error occurred. Please try again later."


# -------------------------
# Test run
# -------------------------
if __name__ == "__main__":
    chatbot = ChatBot()
    sample_question = "Is Nigeria's GDP growth positive this year?"
    answer = chatbot.ask_question(sample_question)
    print("Answer:", answer)
