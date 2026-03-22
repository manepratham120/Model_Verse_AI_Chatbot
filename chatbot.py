import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="centered"
)

llm_groq=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
)
st.title("🤖 Generative Multi-Model Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
user_prompt=st.chat_input("Ask chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    
    st.session_state.chat_history.append({"role":"user", "content":user_prompt})
    
    response=llm_groq.invoke(
        input=[{"role":"system","content":"You are helpful assistant"}, *st.session_state.chat_history]
    )
    
    assistance_response=response.content
    st.session_state.chat_history.append({"role":"assistant", "content":assistance_response})

    with st.chat_message("assistant"):
        st.markdown(assistance_response)

