import os
from dotenv import load_dotenv
import chromadb
from openai import AzureOpenAI
import streamlit as st

load_dotenv()

# Connect to existing vector database (no ingestion here)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01"
)

st.title("Document Q&A — RAG Assistant")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
question = st.chat_input("Ask a question about the documents")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # RAG: retrieve relevant chunks
    results = collection.query(query_texts=[question], n_results=2)
    context = "\n\n".join(results["documents"][0])

    # RAG: grounded answer from LLM
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer using only the context provided. If the answer is not in the context, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    answer = response.choices[0].message.content

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)