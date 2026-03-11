import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# rag/ingest.py
# Reads PDFs, splits them into chunks, embeds, and stores in Chroma.

from pathlib import Path
import fitz  # pymupdf — better layout handling for academic papers
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.vector_store import get_chroma_collection, embed_texts

# Sentence-aware splitter — respects paragraph and sentence boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)


def extract_pdf_text(path: Path):
    """
    Extract text from each page of a PDF using pymupdf.
    Handles two-column academic layouts far better than pypdf.
    Returns a list of (page_number, text) tuples.
    """
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append((i, text))
    return pages


def chunk_page_text(doc_id, title, page_num, text):
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
    - Extracts text using pymupdf
    - Chunks text using sentence-aware splitter
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

    print(f"Found {len(paper_paths)} PDFs.")
    for idx, pdf_path in enumerate(paper_paths):
        title  = pdf_path.stem
        doc_id = f"paper_{idx}"

        if doc_id in indexed_doc_ids:
            print(f"  Skipping already-indexed: {title}")
            continue

        print(f"  Processing: {title}")
        pages = extract_pdf_text(pdf_path)
        for page_num, text in pages:
            chunks = chunk_page_text(doc_id, title, page_num, text)
            for i, ch in enumerate(chunks):
                cid = f"{doc_id}_p{page_num}_c{i}"
                all_ids.append(cid)
                all_texts.append(ch["text"])
                all_metas.append(ch["metadata"])

    if not all_texts:
        print("Nothing new to index.")
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

    print(f"Indexed {len(all_texts)} chunks from {len(all_ids)} new chunks.")


if __name__ == "__main__":
    build_index()
