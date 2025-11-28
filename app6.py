import streamlit as st
from main import answer_query  
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
    st.markdown("This tool fact-checks whether claims are **verified or unverified**")

    if st.button("Clear Chat"):
        st.session_state.history = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.caption("Built With Veegil Technologies")

# ----------------------------
# MAIN UI
# ----------------------------
st.title("🗳️ VeraLog")

st.write(
    "Veegil Fact Checker Assistant. Enter a post or question. The fact-checker retrieves supporting evidence "
    "from our database and evaluates whether the claim is **verified** (score above 0.6 fact-index) or **unverified** (Score below 0.6 fact-index)"
)

# ----------------------------
# USER INPUT
# ----------------------------
user_input = st.text_area(
    "Enter your claim or statement:",
    placeholder="Example: 'Federal Government’s allocation decreased by N41.44bn'…",
    height=140,
)

submit = st.button("Fact-Check Claim")

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
        st.markdown("### Fact-Check Result")
        st.write(response)

    else:
        st.warning("No response generated. Check index or embeddings.")

# ----------------------------
# SHOW CHAT HISTORY
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Conversation History")

    for item in reversed(st.session_state.history):
        st.markdown(
            f"""
            **You:** {item['question']}  
            **VeraLog:** {item['answer']}  
            <div style='font-size:12px;color:gray;'>({item['timestamp']})</div>
            ---
            """,
            unsafe_allow_html=True
        )
