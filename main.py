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
            "You are a political scholar who relies strictly on verifiable, evidence-based information from credible sources.\n"
            "Evaluate the user's post using ONLY the context provided.\n"
            "Your task is to determine whether the claim is VERIFIED or UNVERIFIED.\n"
            "If the context does not contain enough evidence to reach a conclusion, respond with:\n"
            "'Please be informed I am limited to the content in my database. Kindly check back for an update.'\n\n"
            "Context:\n{context}\n\n"
            "Claim:\n{question}\n\n"
            "Answer:"
        )

        # LRU cache for repeated prompts
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    # -------------------------
    # Retrieve top_k docs
    # -------------------------
    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.get_relevant_documents(question)
            docs = docs[:top_k]
            context_text = "\n\n".join(
                d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip()
            )
            return context_text
        except Exception as e:
            logging.exception(f"retrieve_context error: {e}")
            return ""

    # -------------------------
    # LLM call
    # -------------------------
    def _call_llm(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6):
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

    # -------------------------
    # Main public method
    # -------------------------
    def ask_question(self, question: str, use_cache: bool = True) -> str:
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


# -------------------------
# For testing
# -------------------------
if __name__ == "__main__":
    bot = ChatBot()
    question = "Is the current president of Nigeria verified to have approved the new economic policy?"
    answer = bot.ask_question(question)
    print("Answer:", answer)
