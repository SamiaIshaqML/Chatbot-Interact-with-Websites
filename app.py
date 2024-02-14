import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Chat With Website", page_icon = ">.<")

st.title("Chat With Website")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage("Hello, I am a bot! How can I help you?"),
    ]

def get_response(user_input):
    return "I dont know"

with st.sidebar:
    st.header("Settings")
    website = st.text_input("Website URL")


user_query = st.chat_input("Type your messages here...")
if user_query is not None and user_query!="":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(user_query))
    st.session_state.chat_history.append(AIMessage(response))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        st.write(response)

with st.sidebar:
    st.write(st.session_state.chat_history)
