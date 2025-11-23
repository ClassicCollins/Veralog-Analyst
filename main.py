import os
import pinecone
from dotenv import load_dotenv

# Correct imports (use langchain_huggingface)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()


class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize Pinecone client
        pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="us-east1-aws"
        )
        index_name = "health"

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Connect LangChain to Pinecone index
        self.docsearch = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        # Create retriever
        self.retriever = self.docsearch.as_retriever()

        # Initialize HuggingFace LLM using HuggingFace Router
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.6,
            top_k=40,
            top_p=0.8,
            max_new_tokens=200,
            huggingfacehub_api_token=os.getenv("HUG_TOKEN_1")
        )

        # Prompt
        template = """
        You are a political scholar who adheres strictly to factual information from reliable sources.
        Assess and confirm whether a user's post is VERIFIED or UNVERIFIED based solely on the provided context.

        If context does not support an answer, reply:
        "Please be informed I am limited to the content in my database. Kindly check back for an update."

        Context:
        {context}

        **Question:** {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def retrieve_context(self, question):
        """Retrieve only the page_content safely."""
        try:
            results = self.retriever.get_relevant_documents(question)
            context_text = " ".join([doc.page_content for doc in results])

            if not context_text.strip():
                return "No relevant information found in the database."

            return context_text

        except Exception as e:
            print(f"Retrieval Error: {e}")
            return "No relevant information found in the database."

    def ask_question(self, question):
        """Pass correct variables into the QA chain."""
        try:
            context = self.retrieve_context(question)

            response = self.qa_chain.run({
                "context": context,
                "question": question
            })

            return response

        except Exception as e:
            print(f"Error during question handling: {e}")
            return "Please be informed I am limited to the content in my database. Kindly check back for an update."


# Test
if __name__ == "__main__":
    chatbot = ChatBot()
    question = "Hello, what can you help me with?"
    answer = chatbot.ask_question(question)
    print(f"Answer: {answer}")
