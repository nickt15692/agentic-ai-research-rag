# rag/ingest.py
# Reads PDFs, splits them into chunks, embeds, and stores in Chroma.

from pathlib import Path
import os
from pypdf import PdfReader

from rag.config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_chroma_collection, embed_texts


def extract_pdf_text(path: Path):
    """
    Extract text from each page of a PDF.
    Returns a list of (page_number, text) tuples.
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def chunk_page_text(doc_id, title, page_num, text):
    """
    Split a single page's text into overlapping chunks.
    Returns a list of dicts: {"text": ..., "metadata": {...}}.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_txt = text[start:end]
        if chunk_txt.strip():  # skip empty chunks
            chunks.append({
                "text": chunk_txt,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "page": page_num
                }
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index():
    """
    Main ingestion function:
    - Checks that papers exist
    - Extracts text
    - Chunks text
    - Embeds chunks
    - Stores them in Chroma
    """
    if not os.path.isdir(PAPERS_DIR):
        print(f"Papers directory not found: {PAPERS_DIR}")
        return

    paper_paths = list(Path(PAPERS_DIR).glob("*.pdf"))
    if not paper_paths:
        print("No PDFs found in data/papers. Please add at least one PDF.")
        return

    collection = get_chroma_collection()

    all_texts = []
    all_metas = []
    all_ids = []

    print(f"Found {len(paper_paths)} PDFs.")
    for idx, pdf_path in enumerate(paper_paths):
        title = pdf_path.stem
        doc_id = f"paper_{idx}"

        pages = extract_pdf_text(pdf_path)
        for page_num, text in pages:
            chunks = chunk_page_text(doc_id, title, page_num, text)
            for i, ch in enumerate(chunks):
                cid = f"{doc_id}_p{page_num}_c{i}"
                all_ids.append(cid)
                all_texts.append(ch["text"])
                all_metas.append(ch["metadata"])

    if not all_texts:
        print("No text extracted from PDFs.")
        return

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = embed_texts(all_texts)

    print("Adding embeddings to Chroma collection...")
    collection.add(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metas,
        embeddings=embeddings
    )

    print(f"Indexed {len(all_texts)} chunks from {len(paper_paths)} papers.")


if __name__ == "__main__":
    build_index()
