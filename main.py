import os
import logging
from typing import List, Dict, Any

import pinecone                     # Pinecone v2 client
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Load Embedding Models
# ------------------------------------------------------------
try:
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rerank_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Failed loading embedding models: {e}")
    raise

# ------------------------------------------------------------
# Pinecone v2 Initialization
# ------------------------------------------------------------
pc = None
pc_index = None

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "health")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # region like "us-east-1"
NAMESPACE = os.getenv("PINECONE_NAMESPACE", None)

if not PINECONE_API_KEY or not PINECONE_ENV:
    logger.warning("Pinecone environment variables missing.")
else:
    try:
        logger.info("Initializing Pinecone (v2)...")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

        # v2 correct index connection
        pc_index = pc.Index(PINECONE_INDEX)

        # test
        stats = pc_index.describe_index_stats()
        logger.info(f"Pinecone connected. Stats: {stats}")

    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        pc_index = None


# ------------------------------------------------------------
# Embedding Function
# ------------------------------------------------------------
def embed_text(text: str):
    return embed_model.encode(text).tolist()


# ------------------------------------------------------------
# RAG Retrieval
# ------------------------------------------------------------
def retrieve_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Query Pinecone and return top_k results with metadata.
    """
    if not pc_index:
        logger.warning("Pinecone index not available — retrieval skipped.")
        return []

    try:
        q_embed = embed_text(query)

        result = pc_index.query(
            vector=q_embed,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE
        )

        matches = result.get("matches", [])
        docs = [
            {
                "id": m["id"],
                "score": m["score"],
                "text": m["metadata"].get("text", "")
            }
            for m in matches
        ]
        return docs

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []


# ------------------------------------------------------------
# Rerank
# ------------------------------------------------------------
def rerank_results(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not docs:
        return []

    try:
        q_vec = rerank_model.encode(query)

        for doc in docs:
            d_vec = rerank_model.encode(doc["text"])
            score = float(util.cos_sim(q_vec, d_vec))
            doc["rerank"] = score

        return sorted(docs, key=lambda x: x["rerank"], reverse=True)

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return docs


# ------------------------------------------------------------
# Main Answer Function (used by Streamlit)
# ------------------------------------------------------------
def answer_query(query: str) -> str:
    docs = retrieve_documents(query)

    if not docs:
        return (
            "No relevant documents were found in the database.\n"
            "Ensure your Pinecone index is populated."
        )

    ranked = rerank_results(query, docs)

    best = ranked[0]
    return f"Top result:\n\n{best['text']}\n\nScore: {best['rerank']}"
