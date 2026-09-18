import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ✅ Use a model that actually exists on Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")  # ✅ Updated

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index")
KNOWLEDGE_BASE_PATH = os.path.join(DATA_PATH, "knowledge_base.csv")
IMDB_MOVIES_PATH = os.path.join(DATA_PATH, "imdb_movies.csv")
TMDB_CACHE_PATH = os.path.join(DATA_PATH, "tmdb_cache.json")

# Validate API key
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in .env file!")
    print("Please get your free API key from: https://console.groq.com")
    print("and add it to your .env file as: GROQ_API_KEY=your_key_here")

if not TMDB_API_KEY:
    print("⚠️ WARNING: TMDB_API_KEY not found in .env file!")
    print("   Get your free key at: https://www.themoviedb.org/settings/api")