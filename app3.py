import streamlit as st
from main import ChatBot

# Initialize the ChatBot instance
chatbot = ChatBot()

# User input
user_input = st.text_input("Ask a question", "")

if user_input:
    try:
        # Get the response from qa_chain
        response = chatbot.qa_chain.invoke(user_input)
        
        # Debugging: print the raw response to ensure it's correct
        st.write("Raw Response:")
        st.write(response)
        
        # Clean up and display just the relevant answer
        answer = response.get('result', "Sorry, no result found.")
        st.write("Answer:")
        st.write(answer)
        
    except Exception as e:
        st.error(f"Error: {e}")
