# -------------------------
# File: main.py
# -------------------------
import os
import logging
from functools import lru_cache
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone

# Reduce noisy logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Env vars
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "health")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUG_TOKEN_1")
HF_RAG_MODEL = os.getenv("HF_RAG_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Initialize Pinecone
if PINECONE_API_KEY:
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        pinecone_client = pinecone.Index(PINECONE_INDEX)
        logging.info(f"Pinecone initialized with index: {PINECONE_INDEX}")
    except Exception as e:
        logging.warning(f"Pinecone init failed: {e}")
        pinecone_client = None
else:
    logging.warning("PINECONE_API_KEY not found; vector DB disabled.")
    pinecone_client = None

class ChatBot:
    def __init__(self):
        # Embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            self.embeddings = None
            logging.warning(f"Embeddings init failed: {e}")

        # Connect to Pinecone / LangChain retriever
        if pinecone_client and self.embeddings:
            try:
                self.docsearch = LC_Pinecone.from_existing_index(
                    index_name=PINECONE_INDEX,
                    embedding=self.embeddings
                )
                self.retriever = self.docsearch.as_retriever()
            except Exception as e:
                logging.warning(f"Failed to connect to Pinecone index: {e}")
                self.retriever = None
        else:
            self.retriever = None

        # HuggingFace Inference Client
        try:
            self.hf_client = InferenceClient(model=HF_RAG_MODEL, token=HUGGINGFACE_TOKEN)
        except Exception as e:
            self.hf_client = None
            logging.warning(f"HuggingFace InferenceClient failed: {e}")

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

        # LRU cache
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        if not self.retriever:
            logging.info("Retriever not initialized.")
            return ""

        try:
            docs = self.retriever.get_relevant_documents(question)
            docs = docs[:top_k]
            context_chunks = []

            for d in docs:
                # Prioritize page_content, then text, then metadata
                if hasattr(d, "page_content") and d.page_content:
                    context_chunks.append(d.page_content)
                elif hasattr(d, "text") and d.text:
                    context_chunks.append(d.text)
                elif "text" in getattr(d, "metadata", {}):
                    context_chunks.append(d.metadata["text"])

            logging.info(f"Retrieved {len(context_chunks)} context chunks")
            return "\n\n".join(context_chunks)

        except Exception as e:
            logging.exception(f"retrieve_context error: {e}")
            return ""

    def _call_llm(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6):
        if not self.hf_client:
            raise RuntimeError("HuggingFace Inference client not available.")

        try:
            response = self.hf_client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            # Normalize response
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

    def ask_question(self, question: str) -> str:
        context = self.retrieve_context(question)
        if not context.strip():
            return "Please be informed I am limited to the content in my database. Kindly check back for an update."

        full_prompt = self.prompt_template.format(context=context, question=question)
        try:
            return self._ask_cache(full_prompt).strip()
        except Exception:
            return "An internal error occurred. Please try again later."
