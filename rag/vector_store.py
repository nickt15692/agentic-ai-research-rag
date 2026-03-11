# rag/vector_store.py
# Handles embeddings, BM25 keyword search, and talking to the Chroma vector database.

import logging
import re
import chromadb
from chromadb.config import Settings
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from rank_bm25 import BM25Okapi

from rag.config import (
    CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY, DISTANCE_NORMALIZATION,
    VECTOR_WEIGHT, BM25_WEIGHT,
)

logger = logging.getLogger("rag.vector_store")

# Initialize OpenAI client once with timeout
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)

# Singleton Chroma client to avoid re-opening the DB on every query
_chroma_client = None
_chroma_collection = None

# Cached BM25 index
_bm25_index = None
_bm25_doc_count = 0
_bm25_corpus_ids = []
_bm25_corpus_docs = []
_bm25_corpus_metas = []


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
    BATCH_SIZE = 128
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
    Tuned for text-embedding-3-small with pymupdf + smaller chunk size.
    - distance ~0.60 -> ~50% (good match)
    - distance ~0.80 -> ~33% (acceptable)
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


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _get_bm25_index():
    """
    Build (or return cached) BM25 index from all documents in Chroma.
    Rebuilds automatically when the collection size changes.
    """
    global _bm25_index, _bm25_doc_count, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metas

    collection = get_chroma_collection()
    current_count = collection.count()

    if _bm25_index is not None and _bm25_doc_count == current_count:
        return _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metas

    if current_count == 0:
        return None, [], [], []

    logger.info("Building BM25 index over %d chunks...", current_count)
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_corpus_ids = all_data["ids"]
    _bm25_corpus_docs = all_data["documents"]
    _bm25_corpus_metas = all_data["metadatas"]

    tokenized_corpus = [_tokenize(doc) for doc in _bm25_corpus_docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_doc_count = current_count

    logger.info("BM25 index built successfully.")
    return _bm25_index, _bm25_corpus_ids, _bm25_corpus_docs, _bm25_corpus_metas


def _bm25_search(query: str, top_k: int, title_filter: list[str] | None = None) -> dict:
    """
    Perform BM25 keyword search and return results in the same format as Chroma.
    """
    bm25, ids, docs, metas = _get_bm25_index()
    if bm25 is None:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]], "bm25_scores": []}

    tokenized_query = _tokenize(query)
    raw_scores = bm25.get_scores(tokenized_query)

    # Apply title filter
    scored = []
    for i, score in enumerate(raw_scores):
        if title_filter:
            doc_title = metas[i].get("title", "")
            if doc_title not in title_filter:
                continue
        scored.append((i, score))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    result_ids = [ids[i] for i, _ in top]
    result_docs = [docs[i] for i, _ in top]
    result_metas = [metas[i] for i, _ in top]
    result_scores = [s for _, s in top]

    return {
        "ids": [result_ids],
        "documents": [result_docs],
        "metadatas": [result_metas],
        "distances": [[0.0] * len(top)],  # placeholder
        "bm25_scores": result_scores,
    }


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

    vector_results = collection.query(**query_kwargs)

    # --- Hybrid merge: combine vector + BM25 using reciprocal rank fusion ---
    bm25_results = _bm25_search(query, top_k, title_filter)

    # Build rank maps (id -> reciprocal rank score)
    merged_scores: dict[str, float] = {}
    merged_docs: dict[str, str] = {}
    merged_metas: dict[str, dict] = {}
    merged_distances: dict[str, float] = {}

    vec_ids = vector_results.get("ids", [[]])[0]
    vec_docs = vector_results.get("documents", [[]])[0]
    vec_metas = vector_results.get("metadatas", [[]])[0]
    vec_dists = vector_results.get("distances", [[]])[0]

    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(vec_ids, vec_docs, vec_metas, vec_dists)
    ):
        rrf_score = VECTOR_WEIGHT / (rank + 1)
        merged_scores[doc_id] = merged_scores.get(doc_id, 0) + rrf_score
        merged_docs[doc_id] = doc
        merged_metas[doc_id] = meta
        merged_distances[doc_id] = dist

    bm25_ids = bm25_results.get("ids", [[]])[0]
    bm25_docs_list = bm25_results.get("documents", [[]])[0]
    bm25_metas_list = bm25_results.get("metadatas", [[]])[0]

    for rank, (doc_id, doc, meta) in enumerate(
        zip(bm25_ids, bm25_docs_list, bm25_metas_list)
    ):
        rrf_score = BM25_WEIGHT / (rank + 1)
        merged_scores[doc_id] = merged_scores.get(doc_id, 0) + rrf_score
        if doc_id not in merged_docs:
            merged_docs[doc_id] = doc
            merged_metas[doc_id] = meta
            merged_distances[doc_id] = DISTANCE_NORMALIZATION  # no vector distance available

    # Sort by merged RRF score descending, take top_k
    sorted_ids = sorted(merged_scores, key=lambda x: merged_scores[x], reverse=True)[:top_k]

    results = {
        "ids": [sorted_ids],
        "documents": [[merged_docs[i] for i in sorted_ids]],
        "metadatas": [[merged_metas[i] for i in sorted_ids]],
        "distances": [[merged_distances[i] for i in sorted_ids]],
    }

    # Attach confidence scores derived from L2 distances
    distances = results.get("distances", [[]])[0]
    results["confidence_scores"] = [distance_to_confidence(d) for d in distances]

    logger.debug("Hybrid search: %d candidates merged", len(sorted_ids))
    return results
