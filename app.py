import streamlit as st
from agent.agent import run_agent

st.set_page_config(
    page_title="RAG Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Study Assistant")
st.caption("Posez vos questions sur les cours de TLA et Mathématiques de l'Ingénieur.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            answer = run_agent(prompt)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})