# rag/rag_core.py
# Core RAG logic: search + build prompt + call LLM, with conversation memory.

import logging
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from rag.vector_store import search
from rag.config import TOP_K, HISTORY_TURNS, LLM_MODEL, OPENAI_API_KEY, MAX_QUESTION_LENGTH

logger = logging.getLogger("rag.core")

# Initialize OpenAI client with timeout
client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)


def format_history(history: list[dict]) -> str:
    """
    Turn the last N turns of chat history into a text block for the prompt.
    history: list of {"role": "user"/"assistant", "content": str}
    """
    if not history:
        return "No previous conversation."

    lines = []
    for turn in history[-HISTORY_TURNS:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Use the LLM to rewrite the user's question into a more specific,
    self-contained search query before hitting the vector store.
    This improves retrieval for short or vague questions.
    """
    history_text = format_history(history)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    f"Given this conversation:\n{history_text}\n\n"
                    f"Rewrite the following question to be more specific and "
                    f"self-contained for searching academic research papers. "
                    f"Return only the rewritten question, nothing else.\n\n"
                    f"Question: {question}\n\nRewritten question:"
                )
            }],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError:
        logger.warning("Rate limited during query rewrite; using original question.")
        return question
    except APITimeoutError:
        logger.warning("Timeout during query rewrite; using original question.")
        return question
    except (APIError, APIConnectionError) as e:
        logger.warning("API error during query rewrite: %s. Using original question.", e)
        return question


def build_context(results: dict) -> tuple[str, list[dict], list[float]]:
    """
    Convert Chroma search results into a flat text 'context' block
    plus the associated metadata and confidence scores.

    Returns: (context_str, metas, confidence_scores)
    """
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    confidence_scores = results.get("confidence_scores", [None] * len(docs))

    blocks = []
    for doc, meta in zip(docs, metas):
        title = meta.get("title", "Unknown paper")
        page = meta.get("page", "?")
        blocks.append(f"[{title}, page {page}]\n{doc}")

    context = "\n\n---\n\n".join(blocks)
    return context, metas, confidence_scores


def generate_answer(
    question: str,
    history: list[dict],
    title_filter: list[str] | None = None,
) -> tuple[str, list, list, list]:
    """
    Full RAG step:
    - Validate input
    - Rewrite the query for better retrieval
    - Retrieve relevant chunks from Chroma (optionally filtered by paper title)
    - Build a prompt with system msg + history + context + question
    - Call LLM to generate answer

    Args:
        question:      The user's question string.
        history:       List of prior {"role", "content"} turns.
        title_filter:  Optional list of paper titles to restrict retrieval to.

    Returns: (answer_str, metadatas, docs, confidence_scores)
    """
    # 0. Input validation
    if not question or not question.strip():
        return "Please enter a question.", [], [], []

    question = question.strip()
    if len(question) > MAX_QUESTION_LENGTH:
        return (
            f"Your question is too long ({len(question)} characters). "
            f"Please keep it under {MAX_QUESTION_LENGTH} characters.",
            [], [], [],
        )

    # 1. Rewrite query for better vector search recall
    search_query = rewrite_query(question, history)

    # 2. Retrieve relevant chunks
    try:
        results = search(search_query, top_k=TOP_K, title_filter=title_filter)
    except (APIError, APIConnectionError, RateLimitError) as e:
        logger.error("OpenAI API error during search embedding: %s", e)
        return f"OpenAI API error during search: {e}", [], [], []
    except ValueError as e:
        logger.error("Vector store error: %s", e)
        return str(e), [], [], []
    except Exception as e:
        logger.error("Unexpected error during vector search: %s", e, exc_info=True)
        return f"Error while searching the vector store: {e}", [], [], []

    # 3. Handle case where no results are found
    if not results["documents"] or len(results["documents"][0]) == 0:
        return (
            "I couldn't find relevant passages in the loaded papers for that question. "
            "Try different keywords, a broader query, or adjust your paper filter.",
            [],
            [],
            [],
        )

    # 4. Build context from retrieved chunks
    context, metas, confidence_scores = build_context(results)
    history_text = format_history(history)

    system_msg = (
        "You are a research assistant working over a collection of academic PDFs. "
        "Use only the given context to answer questions. When possible, mention "
        "which paper and page support your answer."
    )

    user_msg = f"""
Conversation so far:
{history_text}

Relevant excerpts from the papers:
{context}

User question:
{question}

Answer clearly and concisely. If the context does not contain enough information,
say what is missing or what additional information would be needed.
"""

    # 5. Call the LLM API
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
    except RateLimitError as e:
        logger.error("Rate limited during answer generation: %s", e)
        msg = "The OpenAI API rate limit was exceeded. Please wait a moment and try again."
        return msg, metas, results["documents"][0], confidence_scores
    except APITimeoutError as e:
        logger.error("Timeout during answer generation: %s", e)
        msg = "The request to OpenAI timed out. Please try again."
        return msg, metas, results["documents"][0], confidence_scores
    except (APIError, APIConnectionError) as e:
        logger.error("API error during answer generation: %s", e)
        msg = f"Error while calling the language model: {e}"
        return msg, metas, results["documents"][0], confidence_scores

    answer = response.choices[0].message.content
    return answer, metas, results["documents"][0], confidence_scores
