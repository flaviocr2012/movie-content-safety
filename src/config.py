"""
Configuration for Movie Content Safety Classifier.
Works both locally (.env) and on Streamlit Cloud (st.secrets).
"""

import os
from dotenv import load_dotenv

# Load .env if it exists (local development only)
load_dotenv()


def get_secret(key: str, default: str = None) -> str:
    """
    Get secret from Streamlit Cloud (st.secrets) or environment (.env).
    Priority:
        1. Streamlit secrets (st.secrets)
        2. Environment variables (.env)
        3. Default value
    """
    # Try Streamlit secrets first (Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError, Exception):
        pass  # Not in Streamlit, or secrets file missing

    # Fall back to environment variables (.env local)
    return os.getenv(key, default)


# ============ LANGSMITH OBSERVABILITY ============
LANGCHAIN_TRACING_V2 = get_secret("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = get_secret("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = get_secret("LANGCHAIN_PROJECT", "movie-content-safety")

os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
os.environ["LANGCHAIN_ENDPOINT"] = str(LANGCHAIN_ENDPOINT)
os.environ["LANGCHAIN_API_KEY"] = str(LANGCHAIN_API_KEY or "")
os.environ["LANGCHAIN_PROJECT"] = str(LANGCHAIN_PROJECT)

# ============ API KEYS ============
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")
TMDB_API_KEY = get_secret("TMDB_API_KEY")

# ============ PROJECT PATHS ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_PATH, "knowledge_base.csv")
IMDB_MOVIES_PATH = os.path.join(DATA_PATH, "imdb_movies.csv")
TMDB_CACHE_PATH = os.path.join(DATA_PATH, "tmdb_cache.json")
FEEDBACK_PATH = os.path.join(DATA_PATH, "user_feedback.json")

# ============ VALIDATION (only warn, don't crash) ============
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found!")

if not TMDB_API_KEY:
    print("⚠️ WARNING: TMDB_API_KEY not found in .env file!")

if not LANGCHAIN_API_KEY:
    print("⚠️ WARNING: LANGCHAIN_API_KEY not found — tracing disabled.")
else:
    print(f"✅ LangSmith tracing enabled (project: {LANGCHAIN_PROJECT})")