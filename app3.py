# Test script to inspect chatbot and qa_chain
from main import ChatBot

# Initialize the ChatBot instance
chatbot = ChatBot()

# Test if qa_chain is initialized correctly
def test_chatbot():
    try:
        # Inspect the qa_chain object
        print("Inspecting qa_chain:", chatbot.qa_chain)
        
        # Test invoking qa_chain with sample input
        user_input = "What is the current state of the Nigerian economy?"
        print("User input:", user_input)

        # Test if qa_chain.invoke() produces output
        result = chatbot.qa_chain.invoke(user_input)
        print("Result from qa_chain.invoke():", result)

    except Exception as e:
        print(f"Error: {e}")

# Run the test
test_chatbot()
