"""
Streamlit deployment for Movie Content Safety Classifier.
Full Python backend with RAG chain.
"""

import sys
import os

# Add src/ to path BEFORE any imports
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

LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = get_secret("LANGCHAIN_PROJECT", "movie-content-safety-prod")
LANGCHAIN_TRACING_V2 = get_secret("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT = get_secret("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
    os.environ["LANGCHAIN_ENDPOINT"] = str(LANGCHAIN_ENDPOINT)
    os.environ["LANGCHAIN_API_KEY"] = str(LANGCHAIN_API_KEY)
    os.environ["LANGCHAIN_PROJECT"] = str(LANGCHAIN_PROJECT)

# ============ IMPORTS ============
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ============ VALIDATION ============
if not GROQ_API_KEY:
    st.error("❌ **GROQ_API_KEY not configured**")
    st.markdown("""
    ### How to fix this

    **If you're on Streamlit Cloud:**
    1. Go to your app settings (menu -> Settings)
    2. Click **Secrets**
    3. Add your secrets in TOML format:

    GROQ_API_KEY = "your_key_here"
    GROQ_MODEL = "openai/gpt-oss-20b"
    """)
    st.stop()


# ============ CACHED RESOURCES ============
@st.cache_resource
def load_rag_chain():
    """Load RAG chain (cached across sessions)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0.1,
        max_tokens=500,
        groq_api_key=GROQ_API_KEY
    )

    prompt = PromptTemplate(
        template="""You are a content safety classifier for movies. Your task is to determine if a movie is appropriate for children aged 5-10.

Use the following Q&A pairs as context to guide your decision:

{context}

---
Movie Title: {title}
Movie Overview: {overview}
Movie Genres: {genres}
Movie Rating: {rating}

Important Safety Rules:
1. If the movie genre includes Horror, classify as Not safe for children
2. If the movie genre includes Crime or Thriller with violence, classify as Not safe for children
3. If the movie is rated R, classify as Not safe for children
4. If the movie contains explicit violence, disturbing imagery, or adult themes, classify as Not safe for children
5. If the movie is animated, family-friendly, or rated PG/G, likely Safe for children
6. If the movie contains intense action sequences with frightening creatures or peril, classify as Not safe for children

Based on ALL information above, determine if this movie is appropriate for children.
Be concise and provide:
1. Classification: Safe for children or Not safe for children
2. Explanation: Brief justification

Your response:""",
        input_variables=["context", "title", "overview", "genres", "rating"]
    )

    def get_context(inputs):
        docs = retriever.invoke(inputs["overview"])
        return "\n\n".join([
            f"Q: {d.metadata['question']}\nA: {d.metadata['answer']}"
            for d in docs
        ])

    chain = (
            {
                "context": lambda x: get_context(x),
                "title": lambda x: x["title"],
                "overview": lambda x: x["overview"],
                "genres": lambda x: x["genres"],
                "rating": lambda x: x["rating"]
            }
            | prompt | llm | StrOutputParser()
    )

    return chain


@st.cache_data
def load_movies():
    """Load movies from CSV (cached)."""
    return pd.read_csv("data/imdb_movies.csv").to_dict('records')


# ============ LOAD RESOURCES ============
st.write("### Loading system...")
try:
    movies = load_movies()
    st.write(f"Loaded {len(movies)} movies")
    with st.spinner("Loading RAG chain..."):
        chain = load_rag_chain()
    st.success("System ready!")
except Exception as e:
    st.error(f"Failed to load: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

st.write("---")


# ============ UI ============
st.title("🎬 Movie Content Safety Classifier")
st.markdown("AI-powered RAG system that determines if a movie is appropriate for children aged 5-10")

tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Batch", "🤖 AI Agent", "About"])

# --- Tab 1: Classify ---
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Movie Title", placeholder="e.g., The Lion King")
        overview = st.text_area("Overview", height=150, placeholder="Brief description...")
        genres = st.text_input("Genres", "Unknown")
        rating = st.text_input("Rating", "Unknown")

        if st.button("Classify", type="primary"):
            if title:
                with st.spinner("Classifying..."):
                    try:
                        result = chain.invoke({
                            "title": title,
                            "overview": overview or f"A movie titled {title}.",
                            "genres": genres,
                            "rating": rating
                        })
                        st.success(result)
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a movie title")

    with col2:
        st.markdown("### Example Movies")
        st.markdown("These are examples from the database:")
        for m in movies[:8]:
            st.markdown(f"- {m['title']} ({m.get('year', 'N/A')})")

# --- Tab 2: Batch ---
with tab2:
    st.markdown("### Batch Classification")
    st.markdown("Classify multiple movies from the database at once.")
    limit = st.slider("Number of movies to classify", 1, 20, 5)

    if st.button("Run Batch Classification", type="primary"):
        progress = st.progress(0)
        results = []
        for i, movie in enumerate(movies[:limit]):
            try:
                result = chain.invoke({
                    "title": movie['title'],
                    "overview": movie['overview'],
                    "genres": movie.get('genres', 'Unknown'),
                    "rating": str(movie.get('rating', 'Unknown'))
                })
                emoji = "UNSAFE" if "Not safe" in result else "SAFE"
                results.append(f"[{emoji}] {movie['title']}")
            except Exception as e:
                results.append(f"[ERROR] {movie['title']} - {e}")
            progress.progress((i + 1) / limit)
        st.markdown("\n".join(results))

# --- Tab 3: AI Agent ---
with tab3:
    st.markdown("### 🤖 AI Agent")
    st.markdown("Ask the agent complex questions about movie safety.")

    # Load agent (cached)
    @st.cache_resource
    def load_agent():
        """Load the Movie Safety Agent (cached)."""
        import sys
        import os
        sys.path.insert(0, os.path.abspath("src"))

        from agent import MovieSafetyAgent
        return MovieSafetyAgent()

    try:
        with st.spinner("Loading AI Agent (this takes ~30 seconds)..."):
            agent = load_agent()
        st.success("✅ Agent ready!")
    except Exception as e:
        st.error(f"❌ Failed to load agent: {e}")
        st.stop()

    # Example questions
    st.markdown("#### 💡 Try these questions:")
    example_questions = [
        "What movies do you have?",
        "Is Jurassic Park safe for children?",
        "Show me animated movies",
        "I like fantasy movies. What do you have?",
        "Can you find me a movie like The Lion King?",
    ]

    for q in example_questions:
        if st.button(f"💭 {q}", key=f"agent_q_{q}"):
            st.session_state["agent_question"] = q

    # Input
    question = st.text_area(
        "Your question:",
        value=st.session_state.get("agent_question", ""),
        height=80,
        placeholder="Ask anything about movie safety..."
    )

    if st.button("🤖 Ask Agent", type="primary"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    result = agent.ask(question)
                    st.success(result)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a question")

# --- Tab 4: About ---
with tab4:
    st.markdown("""
    ### About This Project

    An AI-powered movie content safety classifier built with modern LLM tooling.

    #### Tech Stack
    - LangChain - LLM orchestration
    - FAISS - Vector similarity search
    - Groq - Fast LLM inference
    - HuggingFace - Embedding models
    - Streamlit - Web interface

    #### Data
    - 1,132 safety Q&A pairs in the knowledge base
    - 148 movies in the database
    - 56 test cases for evaluation

    #### Links
    - [GitHub Repository](https://github.com/flaviocr2012/movie-content-safety)
    - [LinkedIn](https://www.linkedin.com/in/flavio-rodrigues-7563b631/)
    """)