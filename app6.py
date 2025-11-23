import streamlit as st
from main import answer_query  # <-- Uses your working RAG pipeline
from datetime import datetime

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(
    page_title="Veralog Analyst – Fact Checker",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------
# SESSION STATE SETUP
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("🔍 Veralog Fact Checker")
    st.markdown("This tool checks whether political claims are *verified or unverified* based solely on your curated RAG dataset.")

    if st.button("Clear Chat"):
        st.session_state.history = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.caption("Built with Pinecone v2 + Streamlit + OpenAI")

# ----------------------------
# MAIN UI
# ----------------------------
st.title("🗳️ Political Fact Checker Assistant")

st.write(
    "Enter a political claim or question. The assistant retrieves supporting evidence "
    "from your Pinecone database and evaluates whether the claim is **verified** or **unverified**."
)

# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.text_area(
    "Enter your political claim or statement:",
    placeholder="Example: 'The health budget increased by 25% in 2024'…",
    height=140,
)

submit = st.button("Check Claim")

# ----------------------------
# PROCESS QUERY
# ----------------------------
if submit and user_input.strip():

    with st.spinner("Checking factual accuracy… please wait."):

        try:
            response = answer_query(user_input)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            response = None

    if response:

        # Store in conversation history
        st.session_state.history.append({
            "question": user_input,
            "answer": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Display immediate result
        st.markdown("### ✅ Fact-Check Result")
        st.write(response)

    else:
        st.warning("No response generated. Check your Pinecone index or embeddings.")

# ----------------------------
# SHOW CHAT HISTORY
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🧠 Conversation History")

    for item in reversed(st.session_state.history):
        st.markdown(
            f"""
            **You:** {item['question']}  
            **Assistant:** {item['answer']}  
            <div style='font-size:12px;color:gray;'>({item['timestamp']})</div>
            ---
            """,
            unsafe_allow_html=True
        )
