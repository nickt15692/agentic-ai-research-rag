# rag/app.py
# Streamlit front-end: chat interface with metadata filtering and confidence scores.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import streamlit as st
from rag.rag_core import generate_answer
from rag.vector_store import get_available_titles
from rag.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, MAX_QUESTION_LENGTH

logger = logging.getLogger("rag.app")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ResearchRAG", layout="wide")
st.title("ResearchRAG - Personal Research Paper Assistant")
st.write("Ask questions across your loaded research PDFs (stored in data/papers).")

# ── Session state init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_metas" not in st.session_state:
    st.session_state.last_metas = []
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

# ── Sidebar: metadata filter ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Filter Papers")
    st.caption("Leave blank to search across all papers.")

    try:
        available_titles = get_available_titles()
    except RuntimeError as e:
        logger.error("Failed to load paper titles: %s", e)
        available_titles = []
        st.error("Could not connect to the vector database. Check the logs for details.")
    except Exception as e:
        logger.error("Unexpected error loading paper titles: %s", e, exc_info=True)
        available_titles = []
        st.warning("Could not load paper titles.")

    if available_titles:
        selected_titles = st.multiselect(
            "Restrict search to:",
            options=available_titles,
            default=[],
            placeholder="All papers",
        )
    else:
        selected_titles = []
        st.info("No papers indexed yet. Run ingest.py first.")

    title_filter = selected_titles if selected_titles else None

    st.divider()
    st.header("⚙️ Confidence threshold")
    min_confidence = st.slider(
        "Hide sources below this score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help=(
            "Confidence is derived from the vector similarity distance. "
            "1.0 = perfect match, 0.0 = very distant. "
            "Sources below this threshold are still used to generate the answer "
            "but will be hidden in the Sources panel."
        ),
    )

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.last_metas = []
        st.session_state.last_docs = []
        st.session_state.last_scores = []
        st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────────
user_input = st.text_input("Your question:", "")

if st.button("Ask") and user_input.strip():
    # Validate input length before sending to the RAG pipeline
    if len(user_input.strip()) > MAX_QUESTION_LENGTH:
        st.error(
            f"Your question is too long ({len(user_input.strip())} characters). "
            f"Please keep it under {MAX_QUESTION_LENGTH} characters."
        )
    else:
        history = st.session_state.history

        with st.spinner("Searching papers and generating answer…"):
            answer, metas, docs, scores = generate_answer(
                user_input, history, title_filter=title_filter
            )

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        st.session_state.history = history
        st.session_state.last_metas = metas
        st.session_state.last_docs = docs
        st.session_state.last_scores = scores

# ── Conversation display ───────────────────────────────────────────────────────
st.subheader("Conversation")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        st.markdown(f"**Assistant:** {turn['content']}")

# ── Sources panel ──────────────────────────────────────────────────────────────
if st.session_state.last_metas:
    st.subheader("Sources used in last answer")

    metas  = st.session_state.last_metas
    docs   = st.session_state.last_docs
    scores = st.session_state.last_scores

    # Pad scores list in case it's shorter than metas (safety)
    scores = list(scores) + [None] * (len(metas) - len(scores))

    shown = 0
    for meta, doc, score in zip(metas, docs, scores):
        if score is not None and score < min_confidence:
            continue  # filtered out by sidebar threshold

        shown += 1
        title = meta.get("title", "Unknown paper")
        page  = meta.get("page", "?")

        # Confidence badge colour
        if score is None:
            badge = ""
        elif score >= CONFIDENCE_HIGH:
            badge = f" 🟢 {score:.0%}"
        elif score >= CONFIDENCE_MEDIUM:
            badge = f" 🟡 {score:.0%}"
        else:
            badge = f" 🔴 {score:.0%}"

        with st.expander(f"{title}  •  page {page}{badge}"):
            if score is not None:
                st.progress(score, text=f"Relevance score: {score:.1%}")
            st.text(doc)

    if shown == 0:
        st.info(
            f"All {len(metas)} source(s) were below the confidence threshold "
            f"({min_confidence:.0%}). Lower the slider in the sidebar to see them."
        )
