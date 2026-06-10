import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def ingest_document(filepath):
    """Load, chunk and store a document. Run only when documents change."""

    collection = chroma_client.get_or_create_collection(name="documents")

    with open(filepath, "r") as file:
        text = file.read()

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+200])
        start = (start + 200) - 50

    collection.add(
        documents=chunks,
        ids=[f"{filepath}_chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Ingested {len(chunks)} chunks from {filepath}")

ingest_document("document.txt")
ingest_document("document2.txt")