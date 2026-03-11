# rag/config.py
# Central configuration: paths, model names, chunking params.

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# BASE_DIR = folder containing this config.py (rag/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder containing your PDFs
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")

# Folder where Chroma will store its database files
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Folder for saved models, encoders, artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# OpenAI API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model names
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

# Chunking parameters — smaller chunks = more precise retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval / memory settings
TOP_K = 8          # fetch more candidates for better coverage
HISTORY_TURNS = 4

# Ensure directories exist
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
