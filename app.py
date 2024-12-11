import streamlit as st
from main import ChatBot

# Initialize ChatBot instance
chatbot = ChatBot()

# Set the page title
st.set_page_config(
    page_title="Mental Health Chatbot",
    layout="centered",
)

# Sidebar Section
with st.sidebar:
    st.image("images/bot.png", use_container_width=True)
    st.title("Mental Health Diagnosis Bot")
    st.write(
        """
        Ask me questions about:
        - Mental Health Disorders & diagnoses.
        - DSM 5 classification.
        - Therapeutic Techniques.

        I'll answer concisely based on the information in my database. If the information isn't available, I'll let you know.
        """
    )

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize messages as an empty list

# Streamlit UI
st.title("🧠 Mental Health Diagnosis Bot")
st.write("""
Welcome to the Mental Health Assistant. Ask me any question related to mental health disorders, and I'll try to assist you. 
If I cannot find relevant information, I'll let you know. 
""")


@st.cache_data
def generate_response(user_input):
    try:
        result = chatbot.qa_chain.invoke(user_input)
        if result:
            # Split by '**Question:**' and focus on the answer to the user's question
            if "**Question:**" in result:
                sections = result.split("**Question:**")
                for section in sections:
                    if user_input.lower() in section.lower():  # Match the user's question
                        if "Answer:" in section:
                            return section.split("Answer:")[-1].strip()
            elif "result" in result:
                return result["result"].strip()  # If no **Question:** pattern, fallback to result key
            else:
                return "I'm sorry. My response is currently limited to the content in my Database."
        else:
            return "I'm sorry. My response is currently limited to the content in my Database."
    except Exception as e:
        return f"Error: {e}"

# Display chat history
st.markdown("#### Chat with Bot:")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided input via chat box
if user_input := st.chat_input(placeholder="Ask me a question about mental health disorders:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Fetching insights..."):
                response = generate_response(user_input)
                st.write(response)

        # Store the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
