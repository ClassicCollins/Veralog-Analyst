import os
import pinecone
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()


class ChatBot:
    def __init__(self):
        # Initialize Pinecone
        pc = pinecone.Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="us-east1-aws"
        )
        index_name = "health"

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vectorstore
        self.docsearch = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        self.retriever = self.docsearch.as_retriever()

        # DIRECT HuggingFace Router client (NEW)
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            token=os.getenv("HUG_TOKEN_1")
        )

        template = """
        You are a political scholar who adheres strictly to factual information from reliable sources.
        Assess whether the user's post is VERIFIED or UNVERIFIED based only on the context.

        If you cannot verify, reply:
        "Please be informed I am limited to the content in my database. Kindly check back for an update."

        Context:
        {context}

        **Question:** {question}

        Answer:
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def retrieve_context(self, question):
        """Retrieve relevant documents."""
        try:
            docs = self.retriever.get_relevant_documents(question)
            context_text = " ".join([doc.page_content for doc in docs])
            return context_text if context_text.strip() else "No relevant information found."
        except:
            return "No relevant information found."

    def ask_question(self, question):
        """Send prompt to HuggingFace using the NEW router."""
        try:
            context = self.retrieve_context(question)
            full_prompt = self.prompt.format(context=context, question=question)

            # NEW API CALL → This uses router.huggingface.co automatically
            response = self.client.text_generation(
                full_prompt,
                max_new_tokens=200,
                temperature=0.6
            )

            return response

        except Exception as e:
            return f"Error contacting HuggingFace router: {e}"


if __name__ == "__main__":
    bot = ChatBot()
    output = bot.ask_question("What is the role of the CBN governor?")
    print(output)
