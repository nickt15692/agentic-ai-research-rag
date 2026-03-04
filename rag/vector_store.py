# rag/vector_store.py
# Handles embeddings and talking to the Chroma vector database.

import chromadb
from chromadb.config import Settings
from openai import OpenAI



from rag.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY

# Initialize OpenAI client once
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=OPENAI_API_KEY)

# Singleton Chroma client to avoid re-opening the DB on every query
_chroma_client = None
_chroma_collection = None


def embed_texts(texts, model_name=EMBEDDING_MODEL):
    """
    Given a list of strings, return a list of embedding vectors.
    Uses OpenAI's embedding API.
    """
    response = client.embeddings.create(model=model_name, input=texts)
    return [item.embedding for item in response.data]


def get_chroma_collection(persist_dir=CHROMA_DIR, name="papers_rag"):
    """
    Create or load a Chroma collection (cached as a module-level singleton).
    """
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        _chroma_collection = _chroma_client.get_collection(name=name)
    except Exception:
        _chroma_collection = _chroma_client.create_collection(name=name)

    return _chroma_collection


def distance_to_confidence(distance: float) -> float:
    """
    Tuned for re-indexed chunks with pymupdf + smaller chunk size.
    - distance ~0.85 → ~43% (good match)
    - distance ~1.0  → ~33% (acceptable)
    - distance ~1.2+ → ~0%  (poor match)
    """
    return round(max(0.0, 1.0 - (distance / 1.5)), 3)


def get_available_titles() -> list[str]:
    """
    Return a sorted list of unique paper titles stored in the Chroma collection.
    Used to populate the metadata filter UI.
    """
    collection = get_chroma_collection()
    # Fetch all metadatas (no embeddings needed)
    results = collection.get(include=["metadatas"])
    titles = sorted({
        m.get("title", "Unknown")
        for m in results.get("metadatas", [])
        if m
    })
    return titles


def search(query: str, top_k: int, title_filter: list[str] | None = None):
    """
    Perform a similarity search in Chroma.

    Args:
        query:        The user's question string.
        top_k:        How many chunks to retrieve.
        title_filter: Optional list of paper titles to restrict search to.
                      If None or empty, searches across all papers.

    Returns a Chroma results dict augmented with a 'confidence_scores' key:
        {
          "documents": [[...]] ,
          "metadatas": [[...]] ,
          "distances": [[...]] ,
          "confidence_scores": [float, ...]   # one per chunk, 0–1
        }
    """
    collection = get_chroma_collection()
    query_embedding = embed_texts([query])[0]

    # Build a Chroma where-filter if titles are specified
    where_filter = None
    if title_filter:
        if len(title_filter) == 1:
            where_filter = {"title": {"$eq": title_filter[0]}}
        else:
            where_filter = {"title": {"$in": title_filter}}

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # Attach confidence scores derived from L2 distances
    distances = results.get("distances", [[]])[0]
    results["confidence_scores"] = [distance_to_confidence(d) for d in distances]

    print("RAW DISTANCES:", results.get("distances"))
    return results
