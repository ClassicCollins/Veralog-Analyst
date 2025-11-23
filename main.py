# -------------------------
# File: main.py  (Pinecone v1 Compatible)
# -------------------------

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

import pinecone
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)

# -------------------------
# ENV CONFIG
# -------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "health")

HUGGINGFACE_TOKEN = os.getenv("HUG_TOKEN_1")
HF_RAG_MODEL = os.getenv("HF_RAG_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------
# INIT PINECONE v1
# -------------------------
pinecone_ready = False
if PINECONE_API_KEY and PINECONE_ENV:
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        pinecone_index_obj = pinecone.Index(PINECONE_INDEX)
        pinecone_ready = True
        logging.info("Connected to Pinecone v1 index successfully.")
    except Exception as e:
        pinecone_index_obj = None
        logging.warning(f"Failed to connect to Pinecone v1: {e}")
else:
    pinecone_index_obj = None
    logging.warning("Pinecone environment variables missing.")


class ChatBot:
    def __init__(self):

        # Embeddings model
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        except Exception as e:
            logging.warning(f"Embedding init failed: {e}")
            self.embeddings = None

        # LangChain retriever
        if pinecone_ready and self.embeddings:
            try:
                self.docsearch = LC_Pinecone.from_existing_index(
                    index_name=PINECONE_INDEX,
                    embedding=self.embeddings
                )
                self.retriever = self.docsearch.as_retriever()
                logging.info("Retriever connected.")
            except Exception as e:
                logging.warning(f"Retriever init failed: {e}")
                self.retriever = None
        else:
            self.retriever = None

        # HuggingFace LLM
        if HUGGINGFACE_TOKEN:
            try:
                self.hf_client = InferenceClient(model=HF_RAG_MODEL, token=HUGGINGFACE_TOKEN)
            except:
                self.hf_client = None
                logging.warning("Failed to initialize HF client.")
        else:
            self.hf_client = None
            logging.warning("Missing HuggingFace token.")

        # Prompt
        self.prompt_template = (
            "You are a political scholar who relies strictly on verifiable, evidence-based information.\n"
            "Evaluate the user's post using ONLY the provided context.\n"
            "If there is not enough evidence, respond exactly:\n"
            "'Please be informed I am limited to the content in my database. Kindly check back for an update.'\n\n"
            "Context:\n{context}\n\n"
            "Claim:\n{question}\n\n"
            "Answer:"
        )

        # Cache LLM calls
        self._ask_cache = lru_cache(maxsize=128)(self._call_llm)

    # ---------------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # ---------------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        if not self.retriever:
            return ""

        try:
            docs = self.retriever.get_relevant_documents(query)
            docs = docs[:top_k]
            return "\n\n".join([d.page_content for d in docs])
        except Exception as e:
            logging.warning(f"Context retrieval error: {e}")
            return ""

    # ---------------------------------------------------------------
    # LLM CALL
    # ---------------------------------------------------------------
    def _call_llm(self, prompt: str):
        if not self.hf_client:
            raise RuntimeError("HF client unavailable.")

        try:
            response = self.hf_client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.5
            )
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logging.warning(f"LLM error: {e}")
            return "Model error."

    # ---------------------------------------------------------------
    # PUBLIC METHOD
    # ---------------------------------------------------------------
    def ask_question(self, question: str) -> str:
        context = self.retrieve_context(question)

        if not context.strip():
            return "Please be informed I am limited to the content in my database. Kindly check back for an update."

        prompt = self.prompt_template.format(context=context, question=question)
        return self._ask_cache(prompt).strip()


if __name__ == "__main__":
    bot = ChatBot()
    print(bot.ask_question("Who is the Senate President of Nigeria?"))
