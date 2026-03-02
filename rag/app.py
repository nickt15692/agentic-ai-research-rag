
# rag/app.py
# Streamlit front-end: simple chat interface with conversation memory and sources.

import streamlit as st
from rag.rag_core import generate_answer

# Configure Streamlit page
st.set_page_config(page_title="ResearchRAG", layout="wide")
st.title("ResearchRAG - Personal Research Paper Assistant")

st.write("Ask questions across your loaded research PDFs (stored in data/papers).")

# Initialize conversation history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Text input for the user's question
user_input = st.text_input("Your question:", "")

# When the user clicks "Ask"
if st.button("Ask") and user_input.strip():
    history = st.session_state.history

    # Call the RAG core to get an answer
    answer, metas, docs = generate_answer(user_input, history)

    # Append to chat history
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": answer})

    st.session_state.history = history
    st.session_state.last_metas = metas
    st.session_state.last_docs = docs

# Display conversation history
st.subheader("Conversation")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        st.markdown(f"**Assistant:** {turn['content']}")

# Display sources used for the last answer
if "last_metas" in st.session_state and st.session_state.last_metas:
    st.subheader("Sources used in last answer")
    for meta, doc in zip(st.session_state.last_metas, st.session_state.last_docs):
        title = meta.get("title", "Unknown paper")
        page = meta.get("page", "?")
        with st.expander(f"{title} (page {page})"):
            st.text(doc)
