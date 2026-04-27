import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rag/ingest.py
# Reads PDFs, splits them into chunks, embeds, and stores in Chroma.

import logging
from pathlib import Path
import fitz  # pymupdf — better layout handling for academic papers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIError, APIConnectionError, RateLimitError

from rag.config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_chroma_collection, embed_texts

logger = logging.getLogger("rag.ingest")

# Sentence-aware splitter — respects paragraph and sentence boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


def extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    """
    Extract text from each page of a PDF using pymupdf.
    Handles two-column academic layouts far better than pypdf.
    Returns a list of (page_number, text) tuples.

    Raises:
        ValueError if the file is not a valid PDF or cannot be opened.
    """
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise ValueError(f"Could not open PDF '{path.name}': {e}") from e

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append((i, text))

    doc.close()
    return pages


def chunk_page_text(
    doc_id: str, title: str, page_num: int, text: str
) -> list[dict]:
    """
    Split a single page's text into overlapping chunks using
    RecursiveCharacterTextSplitter so chunks respect sentence boundaries.
    Returns a list of dicts: {"text": ..., "metadata": {...}}.
    """
    chunks = []
    for chunk_txt in splitter.split_text(text):
        if chunk_txt.strip():
            chunks.append({
                "text": chunk_txt,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "page": page_num
                }
            })
    return chunks


def build_index():
    """
    Main ingestion function:
    - Checks that papers exist
    - Skips already-indexed documents (deduplication by doc_id prefix)
    - Extracts text using pymupdf (with per-file error isolation)
    - Chunks text using sentence-aware splitter
    - Embeds chunks
    - Stores them in Chroma
    """
    if not os.path.isdir(PAPERS_DIR):
        logger.error("Papers directory not found: %s", PAPERS_DIR)
        return

    paper_paths = list(Path(PAPERS_DIR).glob("*.pdf"))
    if not paper_paths:
        logger.warning("No PDFs found in %s. Please add at least one PDF.", PAPERS_DIR)
        return

    collection = get_chroma_collection()

    # Find which doc_ids are already indexed to avoid duplicates
    existing = collection.get(include=["metadatas"])
    indexed_doc_ids = {
        m.get("doc_id")
        for m in existing.get("metadatas", [])
        if m
    }

    all_texts = []
    all_metas = []
    all_ids   = []
    failed_files = []

    logger.info("Found %d PDFs.", len(paper_paths))
    for idx, pdf_path in enumerate(paper_paths):
        title  = pdf_path.stem
        doc_id = f"paper_{idx}"

        if doc_id in indexed_doc_ids:
            logger.info("  Skipping already-indexed: %s", title)
            continue

        # Per-file error isolation: one bad PDF doesn't stop the batch
        try:
            logger.info("  Processing: %s", title)
            pages = extract_pdf_text(pdf_path)

            # Warn if PDF yielded no text (e.g., scanned images without OCR)
            total_text = sum(len(text) for _, text in pages)
            if total_text == 0:
                logger.warning(
                    "  '%s' yielded no extractable text (may be a scanned/image PDF). Skipping.",
                    title,
                )
                failed_files.append((title, "No extractable text"))
                continue

            for page_num, text in pages:
                chunks = chunk_page_text(doc_id, title, page_num, text)
                for i, ch in enumerate(chunks):
                    cid = f"{doc_id}_p{page_num}_c{i}"
                    all_ids.append(cid)
                    all_texts.append(ch["text"])
                    all_metas.append(ch["metadata"])

        except ValueError as e:
            logger.error("  Failed to process '%s': %s", title, e)
            failed_files.append((title, str(e)))
            continue
        except Exception as e:
            logger.error("  Unexpected error processing '%s': %s", title, e, exc_info=True)
            failed_files.append((title, str(e)))
            continue

    if not all_texts:
        logger.info("Nothing new to index.")
        if failed_files:
            logger.warning("The following files failed: %s", failed_files)
        return

    try:
        logger.info("Embedding %d chunks...", len(all_texts))
        embeddings = embed_texts(all_texts)
    except (APIError, APIConnectionError, RateLimitError) as e:
        logger.error("Failed to generate embeddings: %s", e)
        logger.error("No chunks were indexed. Please fix the issue and retry.")
        return

    logger.info("Adding embeddings to Chroma collection...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metas,
        embeddings=embeddings
    )

    logger.info("Indexed %d chunks successfully.", len(all_texts))

    if failed_files:
        logger.warning(
            "%d file(s) failed during ingestion:", len(failed_files)
        )
        for name, reason in failed_files:
            logger.warning("  - %s: %s", name, reason)


if __name__ == "__main__":
    build_index()
