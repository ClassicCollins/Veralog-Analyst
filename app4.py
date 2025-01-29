import streamlit as st
import requests
from bs4 import BeautifulSoup
from main import ChatBot
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline  # Added for sentiment analysis

# Initialize ChatBot instance
chatbot = ChatBot()

# Load local pre-trained model from SentenceTransformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

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

# Streamlit UI
st.title("🗺️ Fact Check Posts on Nigerian Politics and Economy 📊verify☑️")
st.write("""Welcome to the Veegil Media Platform. A place to get every political analysis in Nigerian well-detailed to you without any partisan bias. Please, verify posts with VeraLog. If I cannot find relevant information, I'll let you know. I verify based on what I have in my database""")

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

# Function to fetch articles from trusted news sources
def fetch_articles_from_sources(query):
    sources = [
        {"name": "Punch", "url": "https://punchng.com/"},
        {"name": "Guardian", "url": "https://guardian.ng/"},
        {"name": "Vanguard", "url": "https://www.vanguardngr.com/"}
    ]
    
    relevant_articles = []
    for source in sources:
        try:
            response = requests.get(source["url"])
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for all links or news articles related to the query
            articles = soup.find_all('a', href=True)
            for article in articles:
                if query.lower() in article.get_text().lower():
                    relevant_articles.append({
                        "source": source["name"],
                        "title": article.get_text(),
                        "url": article['href']
                    })
        except Exception as e:
            print(f"Error fetching from {source['name']}: {e}")
    
    return relevant_articles

# Function to calculate source credibility
def calculate_credibility_score(relevant_articles):
    # The more relevant articles, the higher the credibility
    unique_sources = set(article["source"] for article in relevant_articles)
    return len(unique_sources)

# Function to generate the chatbot response
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

# Function for sentiment analysis
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment = result[0]['label']
    score = result[0]['score']
    return sentiment, score

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

    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Fetching insights..."):
                # Generate response
                response = generate_response(user_input)

                # Fetch related articles based on the query
                relevant_articles = fetch_articles_from_sources(user_input)

                # Calculate similarity score using the local pre-trained model
                similarity_score = calculate_similarity(user_input, response)

                # Get the verification status based on the similarity score
                verification_status = get_verification_status(similarity_score)

                # Calculate source credibility score
                credibility_score = calculate_credibility_score(relevant_articles)

                # Store the response and details in session state for later use
                st.session_state.generated_response = response
                st.session_state.similarity_score = similarity_score
                st.session_state.verification_status = verification_status
                st.session_state.relevant_articles = relevant_articles
                st.session_state.credibility_score = credibility_score

# Add buttons to display various information

if st.button("Generate Insight"):
    if 'response' in st.session_state:
        st.write(f"**Response:** {st.session_state.response}")
    else:
        st.warning("Please generate a response first.")

if st.button("Context Fact Index"):
    if 'similarity_score' in st.session_state:
        st.write(f"**Context Fact Index:** {round(st.session_state.similarity_score * 100, 2)}%")
    else:
        st.warning("Please generate a response first.")

if st.button("Verification Status"):
    if 'verification_status' in st.session_state:
        st.write(f"**Verification Status:** {st.session_state.verification_status}")
    else:
        st.warning("Please generate a response first.")

if st.button("Related Articles"):
    if 'relevant_articles' in st.session_state and st.session_state.relevant_articles:
        st.write("**Related Articles:**")
        for article in st.session_state.relevant_articles:
            st.write(f"- **{article['source']}**: [{article['title']}]({article['url']})")
    else:
        st.warning("No related articles found. Please try again later.")

if st.button("Source Credibility Score"):
    if 'credibility_score' in st.session_state:
        st.write(f"**Source Credibility Score:** {st.session_state.credibility_score}/3 (Higher score indicates more corroboration)")
    else:
        st.warning("Please generate a response first.")

# Add a button for sentiment analysis
if st.button("Analyze Sentiment"):
    if 'messages' in st.session_state and st.session_state.messages:
        user_input = st.session_state.messages[-1]["content"]  # Get the latest user input
        sentiment, score = analyze_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Sentiment Score:** {score:.2f}")
    else:
        st.warning("Please enter some text to analyze sentiment.")
