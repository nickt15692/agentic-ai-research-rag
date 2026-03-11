# rag/config.py
# Central configuration: paths, model names, chunking params.

import os
import logging
from dotenv import load_dotenv

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag")

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

# OpenAI API key from .env — fail fast if missing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Please add it to your .env file or export it as an environment variable."
    )

# Model names
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

# Chunking parameters — smaller chunks = more precise retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval / memory settings
TOP_K = 8          # fetch more candidates for better coverage
HISTORY_TURNS = 4

# Maximum allowed question length (characters) to prevent token overflow
MAX_QUESTION_LENGTH = 2000

# Confidence badge thresholds
CONFIDENCE_HIGH = 0.40
CONFIDENCE_MEDIUM = 0.25

# Distance normalization factor for confidence score calculation
DISTANCE_NORMALIZATION = 1.5

# Ensure directories exist
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
