# rag/vector_store.py
# Handles embeddings and talking to the Chroma vector database.

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from rag.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY

# Initialize OpenAI client once
client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts, model_name=EMBEDDING_MODEL):
    """
    Given a list of strings, return a list of embedding vectors.
    Uses OpenAI's embedding API.
    """
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def get_chroma_collection(persist_dir=CHROMA_DIR, name="papers_rag"):
    """
    Create or load a Chroma collection that will store our document embeddings.
    """
    db = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = db.get_collection(name=name)
    except:
        collection = db.create_collection(name=name)

    return collection


def search(query, top_k):
    """
    Perform a similarity search in Chroma:
    - Embed the query
    - Return the top_k most similar chunks
    """
    collection = get_chroma_collection()
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results