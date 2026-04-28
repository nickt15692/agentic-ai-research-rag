[README.md](https://github.com/user-attachments/files/27148555/README.md)
# Agentic AI Research RAG

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)

A conversational Q&A chatbot for querying academic research papers using Retrieval-Augmented Generation (RAG). Ask questions in natural language and get answers grounded in your paper collection, with source citations and confidence scores.

---

## Features

- **Hybrid search** — combines semantic vector search (70%) and BM25 keyword search (30%) via Reciprocal Rank Fusion for high-recall retrieval
- **Query rewriting** — LLM rewrites vague or ambiguous questions into precise, self-contained queries before retrieval
- **Cross-encoder reranking** — a dedicated reranker model scores and filters 15 initial candidates down to the top 6 most relevant chunks
- **Confidence scoring** — each source is scored 0–1 with visual indicators (green ≥ 0.60, yellow ≥ 0.30, red < 0.30)
- **Paper filtering** — filter queries to specific papers or search across the entire collection
- **Conversation memory** — retains the last 4 turns for multi-turn, context-aware responses
- **Two-column PDF support** — PyMuPDF extracts text accurately from complex academic paper layouts

---

## Architecture

```
PDFs (rag/data/papers/)
        │
        ▼
   ingest.py          ← PyMuPDF extraction → chunking → OpenAI embeddings
        │
        ▼
  ChromaDB (chroma_db/)
        │
        ▼
  vector_store.py     ← Hybrid search: ChromaDB (semantic) + BM25 (keyword) → RRF fusion
        │
        ▼
   reranker.py        ← Cross-encoder reranking: top 15 → top 6
        │
        ▼
   rag_core.py        ← Query rewriting + context assembly + LLM answer generation
        │
        ▼
    app.py            ← Streamlit chat UI with sidebar filters and source panels
```

---

## Project Structure

```
agentic-ai-research-rag/
├── rag/
│   ├── app.py              # Streamlit web UI (chat interface, filters, source display)
│   ├── rag_core.py         # Core RAG pipeline: query rewriting, retrieval, generation
│   ├── vector_store.py     # Hybrid search, embedding management, RRF fusion
│   ├── ingest.py           # PDF ingestion, chunking, and vector database population
│   ├── reranker.py         # Cross-encoder relevance scoring and reranking
│   ├── config.py           # Configuration parameters and environment setup
│   ├── main.py             # CLI entry point for non-interactive use
│   ├── __init__.py
│   ├── data/
│   │   └── papers/         # Place your PDF files here
│   └── chroma_db/          # Vector database (auto-generated, git-ignored)
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- Python 3.9 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nickt15692/agentic-ai-research-rag.git
cd agentic-ai-research-rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env             # or create .env manually
```

Add the following to your `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
RAG_LOG_LEVEL=INFO               # optional, defaults to INFO
```

---

## Usage

### 1. Ingest Research Papers

Place your PDF files in `rag/data/papers/`, then run the ingestion script to build the vector database:

```bash
python -m rag.ingest
```

This extracts text from each PDF, chunks it, generates embeddings, and stores everything in ChromaDB. Already-ingested documents are automatically skipped on subsequent runs.

### 2. Launch the Streamlit App

```bash
streamlit run rag/app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`). Use the sidebar to filter by specific papers or adjust the confidence threshold, then type your question in the chat input.

### 3. CLI Mode

For scripted or non-interactive use:

```bash
python -m rag.main
```

---

## Configuration

Key parameters in `rag/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chunk size | 1,000 chars | Size of each text chunk after splitting |
| Chunk overlap | 200 chars | Overlap between adjacent chunks |
| Initial retrieval count | 15 | Number of candidates fetched before reranking |
| Final retrieval count | 6 | Top-N chunks passed to the LLM as context |
| Vector search weight | 0.70 | Weight given to semantic (embedding) search |
| BM25 search weight | 0.30 | Weight given to keyword (BM25) search |
| Conversation memory | 4 turns | Number of prior conversation turns retained |
| Embedding model | `text-embedding-3-small` | OpenAI embedding model |

---

## How It Works

1. **Ingestion** — PDFs are parsed with PyMuPDF, split into overlapping chunks, and embedded using `text-embedding-3-small`. Chunks and their metadata (document ID, title, page number) are stored in ChromaDB.

2. **Query rewriting** — when a user submits a question, the LLM first rewrites it into a precise, self-contained query to improve retrieval quality.

3. **Hybrid retrieval** — the rewritten query is sent to both the ChromaDB vector store (semantic search) and a BM25 index (keyword search). Results are merged using Reciprocal Rank Fusion.

4. **Reranking** — a cross-encoder model scores all 15 retrieved candidates for true relevance and selects the top 6.

5. **Answer generation** — the top chunks are formatted into a context block, combined with the conversation history and system prompt, and sent to the LLM to produce a grounded answer with source citations.

6. **UI** — the Streamlit interface displays the answer alongside expandable source panels showing paper titles, page numbers, and color-coded confidence scores.

---

## Tech Stack

| Library | Role |
|---------|------|
| `streamlit` | Web UI framework |
| `chromadb` | Vector store for semantic search |
| `openai` | Embeddings (`text-embedding-3-small`) and answer generation |
| `langchain` / `langchain-text-splitters` | LLM orchestration and sentence-aware text splitting |
| `pymupdf` | PDF text extraction with two-column layout support |
| `rank-bm25` | Keyword-based BM25 search |
| `sentence-transformers` | Cross-encoder model for reranking |
| `python-dotenv` | `.env` file loading for API key management |

---

## License

No license has been specified for this project.
