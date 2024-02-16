import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Chat With Website", page_icon = ">.<")

st.title("Chat With Website")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
    AIMessage("Hello, I am a bot! How can I help you?"),
    ]

def get_response(user_input):
    return "I dont know"

def get_vector_store(url):
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

with st.sidebar:
    st.header("Settings")
    website = st.text_input("Website URL")

if website is None or website=="":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(website) 

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
