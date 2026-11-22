"""Central configuration for CampusSage. Phase 2 values; later phases extend this file."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Chunking is measured in characters. Baseline values, tuned later via experiments.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Pages with fewer cleaned characters than this are treated as empty.
MIN_PAGE_CHARS = 20

INDEX_DIR = DATA_DIR / "index"

# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4


# LLM (Ollama)
LLM_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_NUM_CTX = 4096          # prompt + answer must fit; Ollama truncates silently beyond this
LLM_MAX_NEW_TOKENS = 300
LLM_TIMEOUT_S = 300         # CPU inference can be slow
LLM_KEEP_ALIVE = "30m"

# None disables the pre-LLM score gate. Calibrated on labelled data in Phase 5, not guessed.
MIN_RETRIEVAL_SCORE = None