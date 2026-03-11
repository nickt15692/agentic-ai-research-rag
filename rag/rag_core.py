# rag/rag_core.py
# Core RAG logic: search + build prompt + call LLM, with conversation memory.

from openai import OpenAI
from rag.vector_store import search
from rag.config import TOP_K, HISTORY_TURNS, LLM_MODEL, OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def format_history(history):
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


def rewrite_query(question, history):
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
    except Exception:
        # Fall back to original question if rewrite fails
        return question


def build_context(results):
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


def generate_answer(question, history, title_filter=None):
    """
    Full RAG step:
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
    # 1. Rewrite query for better vector search recall
    search_query = rewrite_query(question, history)

    # 2. Retrieve relevant chunks
    try:
        results = search(search_query, top_k=TOP_K, title_filter=title_filter)
    except Exception as e:
        msg = f"Error while searching the vector store: {e}"
        return msg, [], [], []

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
    except Exception as e:
        msg = f"Error while calling the language model: {e}"
        return msg, metas, results["documents"][0], confidence_scores

    answer = response.choices[0].message.content
    return answer, metas, results["documents"][0], confidence_scores
