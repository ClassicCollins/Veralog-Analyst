import streamlit as st

# Simple mock function for testing
def simple_response(user_input):
    if user_input:
        return f"Response to: {user_input}"
    return "No input provided."

# Streamlit interaction with the user
st.title("🗺️ Fact Check Posts on Nigerian Politics and Economy 📊verify☑️")

# User-provided input via chat box
if user_input := st.chat_input(placeholder="Ask about Nigerian politics, leadership, or economy:"):
    st.write(f"You asked: {user_input}")

    # Button to generate response
    if st.button("Generate Simple Response"):
        response = simple_response(user_input)
        st.write(response)
