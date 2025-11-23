# -------------------------
# File: app6.py
# -------------------------
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textwrap import shorten
from main import ChatBot  # Import the working ChatBot from main.py

# ------------------
# Streamlit caching
# ------------------
@st.cache_resource
def get_chatbot():
    return ChatBot()

@st.cache_resource
def get_local_embedder():
    # small local model for similarity checks
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data(ttl=300)
def embed_texts(embedder, texts: List[str]):
    return embedder.encode(texts, convert_to_numpy=True)

# ------------------
# Quick sentiment helper
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
st.set_page_config(page_title="VeraLog Analyst", layout="wide")
st.sidebar.image("images/bot.png", use_container_width=True)
st.sidebar.title("VeraLog - Nigeria Fact Checker")
st.sidebar.markdown(
    "Verify posts about Politics, Economy, Leadership in Nigeria. Powered by RAG (Pinecone + HF Router)."
)

chatbot = get_chatbot()
embedder = get_local_embedder()

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title("VeraLog — Fact Check Posts on Nigerian Politics & Economy")
st.write("A Retrieval-Augmented Fact-Checker using Pinecone and the HuggingFace router. Use responsibly.")

# ------------------
# User input
# ------------------
user_input = st.chat_input(placeholder="Paste the claim or post you want verified...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Checking database & generating response..."):
            answer = chatbot.ask_question(user_input)

            # Similarity / context fact index
            try:
                vecs = embed_texts(embedder, [user_input, answer])
                sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
            except Exception:
                sim = 0.0

            # Sentiment
            sentiment_label, sentiment_score = quick_sentiment(user_input)

            # Show results
            st.markdown(f"**Verification Result:** {shorten(answer, width=120, placeholder='...')}")
            st.markdown(f"**Context-Fact Index:** {round(sim * 100, 2)}%")
            st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")

            # Display top documents from DB
            if chatbot.retriever:
                docs = chatbot.retriever.get_relevant_documents(user_input)[:4]
                if docs:
                    st.markdown("---")
                    st.markdown("**Top supporting documents from DB:**")
                    for i, d in enumerate(docs, start=1):
                        src = getattr(d, "metadata", {}).get("source", f"Doc {i}")
                        with st.expander(f"Source {i}: {shorten(str(src), width=40)}"):
                            st.write(shorten(getattr(d, "page_content", ""), width=1000))

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------
# Action buttons
# ------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate Insight"):
        if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
            st.write(st.session_state.messages[-1]['content'])
        else:
            st.warning("No generated insight yet. Enter a claim to verify first.")

with col2:
    if st.button("Context Fact Index"):
        if len(st.session_state.messages) >= 2:
            last_user = next((m for m in reversed(st.session_state.messages) if m['role']=='user'), None)
            last_assistant = next((m for m in reversed(st.session_state.messages) if m['role']=='assistant'), None)
            if last_user and last_assistant:
                try:
                    vecs = embed_texts(embedder, [last_user['content'], last_assistant['content']])
                    sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
                    st.write(f"**Context Fact Index:** {round(sim * 100,2)}%")
                except Exception:
                    st.error("Could not compute similarity.")
            else:
                st.warning("Please generate a response first.")
        else:
            st.warning("Please enter and verify a claim first.")

with col3:
    if st.button("Analyze Sentiment"):
        last_user = next((m for m in reversed(st.session_state.messages) if m['role']=='user'), None)
        if last_user:
            label, score = quick_sentiment(last_user['content'])
            st.write(f"**Sentiment:** {label} ({score:.2f})")
        else:
            st.warning("Please enter a claim to analyze.")

# Footer
st.caption("Ensure environment variables are set: PINECONE_API_KEY, HUG_TOKEN_1, PINECONE_INDEX (optional).")
