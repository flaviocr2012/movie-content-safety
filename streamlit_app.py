"""
Streamlit deployment for Movie Content Safety Classifier.
Full Python backend with RAG chain.
"""

import sys
import os

# ✅ CRITICAL: Add src/ to path BEFORE any imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import streamlit as st
import pandas as pd

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Movie Safety Classifier",
    page_icon="🎬",
    layout="wide"
)

# ============ DEBUG BLOCK (TEMPORARY — REMOVE AFTER FIXING) ============
st.write("### 🔍 Debug Info")
try:
    secrets_keys = list(st.secrets.keys())
    st.write(f"**Secrets keys found:** `{secrets_keys}`")
    st.write(f"**Has GROQ_API_KEY:** `{'GROQ_API_KEY' in st.secrets}`")
    st.write(f"**Has GROQ_MODEL:** `{'GROQ_MODEL' in st.secrets}`")
except Exception as e:
    st.error(f"❌ Cannot access st.secrets: `{e}`")
    st.write("**This means the app is not running on Streamlit Cloud OR secrets are not configured.**")

st.write("---")
# ============ END DEBUG BLOCK ============


# ============ IMPORTS (after debug) ============
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


# ============ SECRETS LOADING ============
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, KeyError):
        pass
    return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")

# Debug: Show what was loaded
st.write(f"**GROQ_API_KEY loaded:** `{'Yes' if GROQ_API_KEY else 'No'}`")
st.write(f"**GROQ_MODEL loaded:** `{GROQ_MODEL}`")
st.write("---")

# ============ VALIDATION ============
if not GROQ_API_KEY:
    st.error("❌ **GROQ_API_KEY not configured**")
    st.markdown("""
    ### How to fix this

    **If you're on Streamlit Cloud:**
    1. Go to your app settings (⋮ menu → Settings)
    2. Click **Secrets**
    3. Add your secrets in TOML format (with quotes!):
    ```toml
    GROQ_API_KEY = "your_key_here"
    GROQ_MODEL = "openai/gpt-oss-20b" 
                
""")