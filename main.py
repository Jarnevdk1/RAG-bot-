import streamlit as st
from lanchain_helper import execute_user_query

st.title("VIVES Exam Regulations Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.markdown("Hello! I am a chatbot that can answer questions about the VIVES exam regulations. How can I assist you today?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query_text := st.chat_input("Ask away!"):
    with st.chat_message("user"):
        st.markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the regulations..."):
                response = execute_user_query(query_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception:
            st.error("The AI model is overloaded right now. Please try again in a moment.")