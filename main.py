import os
import pinecone
from dotenv import load_dotenv 
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint


# Load environment variables
load_dotenv()


class ChatBot:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize Pinecone client
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment='us-east1-aws')
        index_name = "health"

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Connect Langchain to Pinecone
        self.docsearch = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Set up retriever properly
        self.retriever = self.docsearch.as_retriever()  # Use retriever interface

        # Initialize HuggingFaceEndpoint LLM
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            top_k=50,
            top_p=0.8,
            temperature=0.7,
            max_new_tokens=200,
            huggingfacehub_api_token=os.getenv("HUG_TOKEN_1")
        )

        # Define prompt template
        template = """
        You are a highly knowledgeable and empathetic mental health consultant with expertise in clinical psychology, therapy techniques, mental health assessment, and psychological well-being.
        Humans will ask you questions about mental health challenges, therapeutic approaches, and strategies for improving psychological well-being.

        Use the provided context to answer their questions accurately and compassionately.
        You MUST ONLY use the provided context to answer the question. If the context does NOT contain sufficient information about the question, respond with:
        "I'm sorry. My response is currently limited to the content in my Database."

        You MUST ONLY RESPOND and COMMUNICATE in ENGLISH LANGUAGE at ALL TIMES

        Context: {context}

        **Question:** {question}

        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Set up the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def retrieve_context(self, question):
        """
        Custom retrieval logic: Only extract page content safely without metadata dependencies.
        """
        try:
            # Use retriever to fetch relevant documents
            retriever_results = self.retriever.get_relevant_documents(question)

            # Only focus on 'page_content' directly
            context_text = " ".join([result.page_content for result in retriever_results])

            # Handle case where no relevant content is found
            if len(context_text.strip()) == 0:
                return "I'm sorry. My response is currently limited to the content in my Database."

            return context_text
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return "I'm sorry. My response is currently limited to the content in my Database."

    def ask_question(self, question):
        """
        Safe invocation of the RetrievalQA chain while passing safe and filtered context data.
        """
        try:
            # Fetch context safely without metadata assumptions
            context = self.retrieve_context(question)
            response = self.qa_chain.run({"context": context, "query": question})
            return response
        except Exception as e:
            print(f"Error during question handling: {e}")
            return "I'm sorry. My response is currently limited to the content in my Database."


if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ChatBot()

    # Ask a sample question
    question = "What are the symptoms of PTSD?"
    answer = chatbot.ask_question(question)
    print(f"Answer: {answer}")