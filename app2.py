import streamlit as st
from main import ChatBot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ChatBot instance
chatbot = ChatBot()

# Load local pre-trained model from SentenceTransformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Set the page title
st.set_page_config(
    page_title="Veralog Analyst",
    layout="centered",
)

# Sidebar Section
with st.sidebar:
    st.image("images/bot.png", use_container_width=True)
    st.title("VeraLog \nNigerian Politics and Economics Analyst")
    st.write(
        """
        Ask me questions about:
        - Political Development Across Nigeria.
        - Government.
        - Leadership and Economy.

        I'll answer concisely based on the information in my database. If the information isn't available, I'll let you know.
        """
    )

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize messages as an empty list

# Streamlit UI
st.title("🗺️ Research on Nigerian Politics and Economy 📊")
st.write("""Welcome to the Veegil Media Platform. A place to get every political analysis in Nigerian well-detailed to you without any partisan bias. Please, ask questions. If I cannot find relevant information, I'll let you know.""")

# Function to calculate embeddings using the local pre-trained model
def get_embeddings(text_list):
    embeddings = model.encode(text_list)
    return embeddings

# Function to calculate cosine similarity between two text embeddings
def calculate_similarity(query, response):
    query_embedding = get_embeddings([query])
    response_embedding = get_embeddings([response])
    similarity_score = cosine_similarity(query_embedding, response_embedding)[0][0]
    return similarity_score

# Function to categorize the similarity score
def get_verification_status(similarity_score):
    if similarity_score > 0.9:
        return "Verified"
    elif 0.5 <= similarity_score <= 0.9:
        return "Partly Verified"
    elif similarity_score < 0.5:
        return "Not Verified"
    else:
        return "Cannot substantiate this post at this time, check back later."

# Streamlit interaction with the user
@st.cache_data
def generate_response(user_input):
    try:
        # Retrieve the response using the chatbot QA chain
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
                return "Please, be informed that my response is currently limited to the content in my Database."
        else:
            return "I'm sorry. My response is currently limited to the content in my Database."
    except Exception as e:
        return f"Error: {e}"

# Display chat history
st.markdown("#### Chat with VeraLog:")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided input via chat box
if user_input := st.chat_input(placeholder="Ask me a question about politics in Nigeria:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Fetching insights..."):
                # Generate response
                response = generate_response(user_input)

                # Calculate similarity score using the local pre-trained model
                similarity_score = calculate_similarity(user_input, response)

                # Get the verification status based on the similarity score
                verification_status = get_verification_status(similarity_score)

                # Display the response and verification status
                st.write(response)
                st.write(f"Fact Index: {round(similarity_score * 100, 2)}%")
                st.write(f"Verification Status: {verification_status}")

        # Store the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
