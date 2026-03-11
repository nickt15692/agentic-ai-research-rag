# rag/reranker.py
# Cross-encoder reranking to improve retrieval precision.

import logging
import math
from sentence_transformers import CrossEncoder
from rag.config import RERANK_MODEL, RERANK_TOP_K

logger = logging.getLogger("rag.reranker")

# Lazy-loaded cross-encoder model
_model = None


def _get_model() -> CrossEncoder:
    """Load the cross-encoder model on first use."""
    global _model
    if _model is None:
        logger.info("Loading cross-encoder model '%s'...", RERANK_MODEL)
        _model = CrossEncoder(RERANK_MODEL)
        logger.info("Cross-encoder model loaded.")
    return _model


def rerank(query: str, results: dict, top_k: int = RERANK_TOP_K) -> dict:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Takes the Chroma-style results dict and returns a new dict with only
    the top_k most relevant chunks according to the cross-encoder.

    Args:
        query:  The user's search query.
        results: Dict with keys 'ids', 'documents', 'metadatas', 'distances',
                 'confidence_scores' — each containing a single inner list.
        top_k:  Number of chunks to keep after reranking.

    Returns:
        A new results dict with the same structure, narrowed to top_k items.
    """
    docs = results["documents"][0]
    if not docs:
        return results

    model = _get_model()

    # Score each (query, document) pair
    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs)

    # Rank by cross-encoder score descending
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked_indices[:top_k]

    ids = results["ids"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Use cross-encoder scores for confidence instead of raw L2 distances.
    # ms-marco-MiniLM-L-6-v2 outputs logits: negative = irrelevant, positive = relevant.
    # Sigmoid maps these to [0, 1] for a proper confidence score.
    reranker_conf = [round(1 / (1 + math.exp(-float(scores[i]))), 3) for i in top_indices]

    return {
        "ids": [[ids[i] for i in top_indices]],
        "documents": [[docs[i] for i in top_indices]],
        "metadatas": [[metas[i] for i in top_indices]],
        "distances": [[dists[i] for i in top_indices]],
        "confidence_scores": reranker_conf,
    }
