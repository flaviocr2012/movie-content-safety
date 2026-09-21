"""
Configuration for Movie Content Safety Classifier.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit Cloud or environment variables."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError):
        pass
    # Fall back to environment variables (.env)
    return os.getenv(key, default)

# ============ LANGSMITH OBSERVABILITY ============
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "movie-content-safety")

# Apply to environment
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ============ API KEYS ============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ============ PROJECT PATHS ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_PATH, "knowledge_base.csv")
IMDB_MOVIES_PATH = os.path.join(DATA_PATH, "imdb_movies.csv")
TMDB_CACHE_PATH = os.path.join(DATA_PATH, "tmdb_cache.json")
FEEDBACK_PATH = os.path.join(DATA_PATH, "user_feedback.json")

# ============ VALIDATION ============
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in .env file!")

if not TMDB_API_KEY:
    print("⚠️ WARNING: TMDB_API_KEY not found in .env file!")

if not LANGCHAIN_API_KEY:
    print("⚠️ WARNING: LANGCHAIN_API_KEY not found — tracing disabled.")
else:
    print(f"✅ LangSmith tracing enabled (project: {LANGCHAIN_PROJECT})")