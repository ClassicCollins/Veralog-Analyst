# -------------------------
# File: app.py (Streamlit)
# -------------------------
# Streamlit front-end that uses ChatBot from main.py
# Features:
# - st.cache_resource for ChatBot instance
# - SentenceTransformer local model for quick similarity scoring
# - Improved UI: expandable related articles, credibility score, sentiment, caching

from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textwrap import shorten

# Import ChatBot from the same file (if split, replace with: from main import ChatBot)
# If you keep the files separate, change the import accordingly.
# from main import ChatBot

# For this combined file, ChatBot is already defined above.

# ------------------
# Helpers & caching
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

# Simple sentiment helper using a tiny heuristic if HF pipeline not available in this environment
# (You can swap to transformers.pipeline('sentiment-analysis') if you have the package.)

def quick_sentiment(text: str):
    # naive polarity based on keywords — replace with transformer pipeline for production
    lowered = text.lower()
    if any(w in lowered for w in ["happy", "good", "positive", "approve", "support"]):
        return "POSITIVE", 0.9
    if any(w in lowered for w in ["sad", "bad", "angry", "oppose", "concern"]):
        return "NEGATIVE", 0.85
    return "NEUTRAL", 0.6

# ------------------
# Streamlit UI
# ------------------
st.set_page_config(page_title="VeraLog Analyst (Optimized)", layout="wide")

st.sidebar.image("images/bot.png", use_column_width=True)
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

# Input area
user_input = st.chat_input(placeholder="Paste the claim or post you want verified...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Checking database & generating response..."):
            answer = chatbot.ask_question(user_input)

            # similarity / context fact index (best-effort)
            try:
                # create small similarity between user input and returned answer
                vecs = embed_texts(embedder, [user_input, answer])
                sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
            except Exception:
                sim = 0.0

            # sentiment
            sentiment_label, sentiment_score = quick_sentiment(user_input)

            # Show results
            st.markdown(f"**Verification Result:** {shorten(answer, width=120, placeholder='...')}")
            st.markdown(f"**Context-Fact Index:** {round(sim * 100, 2)}%")
            st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")

            # If retriever is available, show top docs (best-effort)
            if chatbot.retriever:
                docs = chatbot.retriever.get_relevant_documents(user_input)[:4]
                if docs:
                    st.markdown("---")
                    st.markdown("**Top supporting documents from DB:**")
                    for i, d in enumerate(docs, start=1):
                        with st.expander(f"Source {i}: {shorten(d.metadata.get('source', 'doc'), width=40)}"):
                            st.write(shorten(d.page_content, width=1000))

            # Record assistant message
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Buttons for additional operations
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate Insight"):
        if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
            st.write(st.session_state.messages[-1]['content'])
        else:
            st.warning("No generated insight yet. Enter a claim to verify first.")

with col2:
    if st.button("Context Fact Index"):
        if 'messages' in st.session_state and len(st.session_state.messages) >= 2:
            last_user = next((m for m in reversed(st.session_state.messages) if m['role']=='user'), None)
            last_assistant = next((m for m in reversed(st.session_state.messages) if m['role']=='assistant'), None)
            if last_user and last_assistant:
                try:
                    vecs = embed_texts(embedder, [last_user['content'], last_assistant['content']])
                    sim = float(cosine_similarity([vecs[0]], [vecs[1]])[0][0])
                    st.write(f"**Context Fact Index:** {round(sim * 100,2)}%")
                except Exception as e:
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

# End of file
