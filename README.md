# VeraLog - Nigeria's Economic and Political Fact-Checker

VeraLog is an AI-powered platform that helps verify posts related to political and economic developments in Nigeria. Powered by **Streamlit**, **Sentence Transformers**, and **Sklearn**, VeraLog can compare user queries with a pre-trained chatbot model and assess the accuracy of posts by calculating similarity scores and providing a verification status.

## Features

- Verify political posts and news in Nigeria.
- Assess the credibility of posts related to the economy and leadership.
- Instant feedback based on similarity analysis of the user input.
- Provides a verification status with labels: `Verified`, `Partly Verified`, `Not Verified`, or `Cannot substantiate this post at this time`.
- Uses a local pre-trained model from **SentenceTransformers** and cosine similarity to measure the relevance of responses.

## Requirements

To run this project locally, you'll need the following dependencies:

- Python 3.8 or higher
- Streamlit
- SentenceTransformers
- Sklearn

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```
Clone this repository to your local machine:

```bash
git clone https://gitlab.com/your-repository/veralog.git

cd veralog
```
start the application by running the following command:
```bash
streamlit run app.py
```

The application will start and will be accessible in your browser at http://localhost:8501

### How It Works
The application uses Streamlit to provide an interactive web interface where users can input political or economic posts to verify.
The ChatBot is invoked to retrieve responses, which are then compared to the user input using SentenceTransformers to generate embeddings.
Cosine Similarity is used to calculate how close the response is to the user input, and based on the similarity score, a Verification Status is provided.
The possible verification statuses are:
- `Verified:` When the similarity score is above 0.7.
- `Partly Verified:` When the similarity score is between 0.4 and 0.7.
- `Not Verified:` When the similarity score is below 0.4.
- `Cannot substantiate this post at this time:` If the model cannot provide relevant information.
### Example Workflow:
- The user enters a post or query about Nigerian politics or economy.
- The chatbot retrieves a response based on its [trained database](https://github.com/ClassicCollins/Veralog-Analyst/blob/classic/document/Veegil_Post.pdf).
- The system calculates the similarity score between the query and the response.
- The verification status is displayed based on the calculated similarity score.
- Users receive insights with a Context Fact Index showing the relevance of the response.



