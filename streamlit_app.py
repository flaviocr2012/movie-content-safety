"""
Streamlit deployment for Movie Content Safety Classifier.
Full Python backend with RAG chain.
"""

import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ============ CONFIG ============
st.set_page_config(page_title="Movie Safety Classifier", page_icon="🎬", layout="wide")

# Get API key from Streamlit secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GROQ_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b")
except Exception:
    st.error("⚠️ GROQ_API_KEY not configured in Streamlit secrets")
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
        template="""You are a content safety classifier for movies...

Use the following context:
{context}

Movie: {title}
Overview: {overview}
Genres: {genres}
Rating: {rating}

Classify as "Safe for children" or "Not safe for children".""",
        input_variables=["context", "title", "overview", "genres", "rating"]
    )

    def get_context(inputs):
        docs = retriever.invoke(inputs["overview"])
        return "\n\n".join([f"Q: {d.metadata['question']}\nA: {d.metadata['answer']}" for d in docs])

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
    return pd.read_csv("data/imdb_movies.csv").to_dict('records')


# ============ UI ============
st.title("🎬 Movie Content Safety Classifier")
st.markdown("*AI-powered RAG system that determines if a movie is appropriate for children aged 5-10*")

chain = load_rag_chain()
movies = load_movies()

tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Batch", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("🎬 Movie Title")
        overview = st.text_area("📝 Overview", height=150)
        genres = st.text_input("🎭 Genres", "Unknown")
        rating = st.text_input("⭐ Rating", "Unknown")

        if st.button("🔍 Classify", type="primary"):
            if title:
                with st.spinner("Classifying..."):
                    result = chain.invoke({
                        "title": title,
                        "overview": overview or f"A movie titled '{title}'.",
                        "genres": genres,
                        "rating": rating
                    })
                    st.success(result)
            else:
                st.warning("Please enter a movie title")

    with col2:
        st.markdown("### 📋 Examples")
        for m in movies[:5]:
            st.markdown(f"- **{m['title']}** ({m.get('year', 'N/A')})")

with tab2:
    limit = st.slider("Number of movies", 1, 20, 5)
    if st.button("📊 Run Batch", type="primary"):
        progress = st.progress(0)
        results = []
        for i, movie in enumerate(movies[:limit]):
            with st.spinner(f"Classifying {movie['title']}..."):
                result = chain.invoke({
                    "title": movie['title'],
                    "overview": movie['overview'],
                    "genres": movie.get('genres', 'Unknown'),
                    "rating": str(movie.get('rating', 'Unknown'))
                })
                emoji = "❌" if "Not safe" in result else "✅"
                results.append(f"{emoji} {movie['title']}")
            progress.progress((i + 1) / limit)
        st.markdown("\n".join(results))

with tab3:
    st.markdown("""
    ### About
    Built with LangChain, FAISS, Groq, and Streamlit.

    - [GitHub](https://github.com/flaviocr2012/movie-content-safety)
    - [LinkedIn](https://www.linkedin.com/in/flavio-rodrigues-7563b631/)
    """)