# rag/vector_store.py
# Handles embeddings and talking to the Chroma vector database.

import logging
import chromadb
from chromadb.config import Settings
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from rag.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, DISTANCE_NORMALIZATION

logger = logging.getLogger("rag.vector_store")

# Initialize OpenAI client once with timeout
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)

# Singleton Chroma client to avoid re-opening the DB on every query
_chroma_client = None
_chroma_collection = None


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Given a list of strings, return a list of embedding vectors.
    Uses OpenAI's embedding API.

    Raises:
        APIError, APIConnectionError, RateLimitError on API failures.
        ValueError if texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    # OpenAI allows max ~300k tokens per request; batch to stay under the limit.
    BATCH_SIZE = 256
    all_embeddings: list[list[float]] = []

    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=model_name, input=batch)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings
    except RateLimitError:
        logger.error("Rate limited while generating embeddings for %d texts.", len(texts))
        raise
    except APITimeoutError:
        logger.error("Timeout while generating embeddings for %d texts.", len(texts))
        raise
    except (APIError, APIConnectionError) as e:
        logger.error("API error while generating embeddings: %s", e)
        raise


def get_chroma_collection(persist_dir: str = CHROMA_DIR, name: str = "papers_rag"):
    """
    Create or load a Chroma collection (cached as a module-level singleton).

    Raises:
        RuntimeError if Chroma DB cannot be opened (e.g., corrupted or locked).
    """
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    try:
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    except Exception as e:
        logger.error("Failed to open Chroma database at %s: %s", persist_dir, e)
        raise RuntimeError(
            f"Could not open the Chroma vector database at '{persist_dir}'. "
            f"It may be corrupted or locked by another process. Error: {e}"
        ) from e

    try:
        _chroma_collection = _chroma_client.get_collection(name=name)
        logger.info("Loaded existing Chroma collection '%s'.", name)
    except Exception:
        _chroma_collection = _chroma_client.create_collection(name=name)
        logger.info("Created new Chroma collection '%s'.", name)

    return _chroma_collection


def distance_to_confidence(distance: float) -> float:
    """
    Tuned for re-indexed chunks with pymupdf + smaller chunk size.
    - distance ~0.85 -> ~43% (good match)
    - distance ~1.0  -> ~33% (acceptable)
    - distance ~1.2+ -> ~0%  (poor match)
    """
    return round(max(0.0, 1.0 - (distance / DISTANCE_NORMALIZATION)), 3)


def get_available_titles() -> list[str]:
    """
    Return a sorted list of unique paper titles stored in the Chroma collection.
    Used to populate the metadata filter UI.

    Raises:
        RuntimeError if the Chroma collection cannot be loaded.
    """
    collection = get_chroma_collection()
    count = collection.count()
    if count == 0:
        logger.info("Chroma collection is empty — no titles available.")
        return []

    results = collection.get(include=["metadatas"])
    titles = sorted({
        m.get("title", "Unknown")
        for m in results.get("metadatas", [])
        if m
    })
    return titles


def search(query: str, top_k: int, title_filter: list[str] | None = None) -> dict:
    """
    Perform a similarity search in Chroma.

    Args:
        query:        The user's question string.
        top_k:        How many chunks to retrieve.
        title_filter: Optional list of paper titles to restrict search to.
                      If None or empty, searches across all papers.

    Returns a Chroma results dict augmented with a 'confidence_scores' key.

    Raises:
        ValueError if the collection is empty (no documents ingested).
        APIError, RateLimitError etc. if embedding the query fails.
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        raise ValueError(
            "No documents have been indexed yet. "
            "Please run 'python -m rag.ingest' to ingest your PDFs first."
        )

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

    logger.debug("Search distances: %s", distances)
    return results
