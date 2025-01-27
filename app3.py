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
    st.title("VeraLog \nNigeria's Economic and Political Fact-Checker")
    st.write(
        """
        Verify Posts on:
        - Political Development Across Nigeria.
        - Government.
        - Leadership and Economy.

        I'll give feedback based on the information in my database. If the information isn't available, I'll let you know.
        """
    )

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize messages as an empty list

# Initialize session state for response
if "response" not in st.session_state:
    st.session_state.response = None  # Initialize response as None

# Streamlit UI
st.title("🗺️ Fact Check Posts on Nigerian Politics and Economy 📊verify☑️")
st.write("""Welcome to the Veegil Media Platform. A place to get every political analysis in Nigerian well-detailed to you without any partisan bias. Please, verify posts with VeraLog. If I cannot find relevant information, I'll let you know.I verify based on what I have in my database""")

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
    if similarity_score > 0.7:
        return "Verified"
    elif 0.4 <= similarity_score <= 0.7:
        return "Partly Verified"
    elif 0.2 <= similarity_score < 0.4:
        return "Not Verified based on information in my Database"
    else:
        return "Cannot substantiate this post at this time, check back later."

# Streamlit interaction with the user
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
st.markdown("#### Verify Post with VeraLog:")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided input via chat box
if user_input := st.chat_input(placeholder="Verify posts on politics, Leadership and economy in Nigeria:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Buttons to trigger actions
    if st.button("Generate Response"):
        with st.chat_message("assistant"):
            with st.spinner("Fetching insights..."):
                try:
                    # Attempt to generate the response
                    st.session_state.response = generate_response(user_input)  # Store the response in session state
                    
                    # Debugging: Output the raw result
                    if st.session_state.response:
                        st.session_state.messages.append({"role": "assistant", "content": st.session_state.response})
                        st.write(st.session_state.response)  # Display the response
                    else:
                        st.write("No response generated. Please check the input or chatbot settings.")
                except Exception as e:
                    st.write(f"Error while generating response: {e}")

    if st.button("Calculate Fact Index"):
        if st.session_state.response:  # Ensure the response exists before calculating similarity
            similarity_score = calculate_similarity(user_input, st.session_state.response)
            st.write(f"Context Fact Index: {round(similarity_score * 100, 2)}%")
        else:
            st.warning("Please generate a response first.")

    if st.button("Get Verification Status"):
        if st.session_state.response:  # Ensure the response exists before calculating verification status
            similarity_score = calculate_similarity(user_input, st.session_state.response)
            verification_status = get_verification_status(similarity_score)
            st.write(f"Verification Status: {verification_status}")
        else:
            st.warning("Please generate a response first.")
