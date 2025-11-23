# -------------------------
# File: main.py
# -------------------------
# ChatBot class: RAG pipeline using Pinecone v2 + HuggingFace InferenceClient
# Uses langchain_huggingface embeddings and Pinecone v2 properly
# Robust error handling and simple in-memory caching

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec

# Reduce noisy logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# -------------------------
# Environment / Config
# -------------------------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "health")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUG_TOKEN_1")
HF_RAG_MODEL = os.getenv("HF_RAG_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMENSION = 384  # dimension for MiniLM-L6-v2

# -------------------------
# Pinecone v2 client init
# -------------------------
pc = None
pinecone_index_obj = None

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        logging.info("Pinecone client ready.")

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            logging.info(f"Creating Pinecone index: {PINECONE_INDEX}")
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Connect to the index
        pinecone_index_obj = pc.index(PINECONE_INDEX)
        logging.info(f"Connected to Pinecone index: {PINECONE_INDEX}")

    except Exception as e:
        logging.warning(f"Pinecone init failed: {e}")
        pc = None
        pinecone_index_obj = None
else:
    logging.warning("PINECONE_API_KEY not found; Pinecone disabled.")

# -------------------------
# ChatBot class
# -------------------------
class ChatBot:
    def __init__(self):
        # Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        except Exception as e:
            self.embeddings = None
            logging.warning(f"Failed to initialize embeddings: {e}")

        # LangChain Pinecone wrapper
        if pinecone_index_obj and self.embeddings:
            try:
                self.docsearch = LC_Pinecone.from_existing_index(
                    index_name=PINECONE_INDEX,
                    embedding=self.embeddings,
                    namespace="",  # optional
                )
                self.retriever = self.docsearch.as_retriever()
                logging.info("Retriever connected to Pinecone index.")
            except Exception as e:
                logging.warning(f"Failed to connect LangChain Pinecone wrapper: {e}")
                self.docsearch = None
                self.retriever = None
        else:
            self.docsearch = None
            self.retriever = None

        # HuggingFace InferenceClient
        if HUGGINGFACE_TOKEN:
            try:
                self.hf_client = InferenceClient(model=HF_RAG_MODEL, token=HUGGINGFACE_TOKEN)
            except Exception as e:
                logging.warning(f"Failed to create HF InferenceClient: {e}")
                self.hf_client = None
        else:
            logging.warning("HuggingFace token missing; LLM disabled.")
            self.hf_client = None

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

        # Cache recent LLM calls
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    # -------------------------
    # Retrieve top documents
    # -------------------------
    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.get_relevant_documents(question)
            docs = docs[:top_k]
            context_text = "\n\n".join(d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip())
            return context_text
        except Exception as e:
            logging.warning(f"retrieve_context error: {e}")
            return ""

    # -------------------------
    # Internal LLM call
    # -------------------------
    def _call_llm(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6):
        if not self.hf_client:
            raise RuntimeError("HF InferenceClient not available.")

        try:
            response = self.hf_client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

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
            logging.warning(f"LLM call failed: {e}")
            raise

    # -------------------------
    # Public method
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
            return result.strip() if isinstance(result, str) else str(result)
        except Exception as e:
            logging.warning(f"ask_question failed: {e}")
            return "An internal error occurred. Please try again later."


# -------------------------
# Optional test
# -------------------------
if __name__ == "__main__":
    bot = ChatBot()
    question = "Who is the current president of Nigeria?"
    answer = bot.ask_question(question)
    print(f"Answer: {answer}")
