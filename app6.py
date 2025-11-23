# -------------------------
# File: app6.py (Streamlit UI)
# Updated to work with the new main.py RAG system
# -------------------------

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textwrap import shorten
from typing import List

from main import ChatBot

# ------------------
# Cached resources
# ------------------
@st.cache_resource
def get_chatbot():
    return ChatBot()

@st.cache_resource
def get_local_embedder():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data(ttl=300)
def embed_texts(embedder, texts: List[str]):
    return embedder.encode(texts, convert_to_numpy=True)

# ------------------
# Simple sentiment helper
# ------------------
def quick_sentiment(text: str):
    lowered = text.lower()
    if any(w in lowered for w in ["happy", "good", "positive", "approve", "support"]):
        return "POSITIVE", 0.9
    if any(w in lowered for w in ["sad", "bad", "angry", "oppose", "concern"]):
        return "NEGATIVE", 0.85
    return "NEUTRAL", 0.6

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(page_title="VeraLog Analyst RAG", layout="wide")

st.sidebar.image("images/bot.png", use_container_width=True)
st.sidebar.title("VeraLog - Fact Checker")
st.sidebar.markdown(
    "Verify statements about Nigeria's Politics, Economy and Leadership. Powered by Pinecone + HF Router."
)

chatbot = get_chatbot()
embedder = get_local_embedder()

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("VeraLog — Nigerian Fact Checker (RAG)")
st.write("Paste a political or economic claim to check against the database.")

# User input
user_input = st.chat_input(placeholder="Paste the claim you want verified...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating verification..."):
            answer = chatbot.ask_question(user_input)

            # Similarity index
            try:
                vecs = embed_texts(embedder, [user_input, answer])
                sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
            except Exception:
                sim = 0.0

            # Sentiment
            sentiment_label, sentiment_score = quick_sentiment(user_input)

            # Output block
            st.markdown(f"**Verification Result:** {answer}")
            st.markdown(f"**Context-Fact Index:** {round(sim * 100, 2)}%")
            st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")

            # Show DB sources
            if chatbot.retriever:
                docs = chatbot.retriever.get_relevant_documents(user_input)[:4]
                if docs:
                    st.markdown("---")
                    st.markdown("### 🔎 Top Supporting Documents from DB:")
                    for i, d in enumerate(docs, start=1):
                        with st.expander(f"Source {i}: {d.metadata.get('source', 'Unknown')}"):
                            st.write(shorten(d.page_content, width=1000))

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------
# Buttons
# ------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate Insight"):
        last_msg = next((m for m in reversed(st.session_state.messages) if m['role']=="assistant"), None)
        if last_msg:
            st.write(last_msg['content'])
        else:
            st.warning("No insight available. Verify a statement first.")

with col2:
    if st.button("Context Fact Index"):
        last_user = next((m for m in reversed(st.session_state.messages) if m['role']=="user"), None)
        last_assistant = next((m for m in reversed(st.session_state.messages) if m['role']=="assistant"), None)
        if last_user and last_assistant:
            try:
                vecs = embed_texts(embedder, [last_user['content'], last_assistant['content']])
                sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
                st.write(f"**Context Fact Index:** {round(sim * 100, 2)}%")
            except:
                st.error("Similarity computation failed.")
        else:
            st.warning("Verify a claim first.")

with col3:
    if st.button("Analyze Sentiment"):
        last_user = next((m for m in reversed(st.session_state.messages) if m['role']=="user"), None)
        if last_user:
            label, score = quick_sentiment(last_user['content'])
            st.write(f"**Sentiment:** {label} ({score:.2f})")
        else:
            st.warning("Enter a claim first.")

st.caption("Ensure PINECONE_API_KEY and HUG_TOKEN_1 are configured correctly.")
