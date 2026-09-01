import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Import configuration
from config import GROQ_API_KEY, GROQ_MODEL, INDEX_PATH


class RAGChain:
    """
    RAG Chain for movie content safety classification.
    Uses FAISS retriever + Groq LLM to determine if a movie is child-appropriate.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the RAG chain with retriever and LLM.

        Args:
            model_name: Groq model to use (default: from config)
        """
        if model_name is None:
            model_name = GROQ_MODEL

        print("🔄 Initializing RAG Chain...")

        # Load the vector store
        print("🔄 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("🔄 Loading FAISS vector store...")
        self.vector_store = FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # ✅ Initialize Groq LLM with correct model
        print(f"🔄 Initializing Groq LLM with model: {model_name}...")
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,
            max_tokens=500,
            groq_api_key=GROQ_API_KEY
        )

        # Build the chain
        self.chain = self._build_chain()

        print("✅ RAG Chain initialized successfully!")

    def _build_chain(self):
        """Build the RAG chain using LangChain Expression Language (LCEL)."""

        prompt_template = """
        You are a content safety classifier for movies. Your task is to determine if a movie is appropriate for children aged 5-10.
        
        Use the following Q&A pairs as context to guide your decision:
        
        {context}
        
        ---
        Movie Title: {title}
        Movie Overview: {overview}
        
        Based ONLY on the context provided above, determine if this movie is appropriate for children.
        Be concise and provide:
        1. Classification: "Safe for children" or "Not safe for children"
        2. Explanation: Brief justification based on the context
        
        If the context doesn't clearly indicate safety, err on the side of caution and classify as "Not safe for children".
        
        Your response:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "title", "overview"]
        )

        # Define helper function for context retrieval
        def retrieve_context(inputs: dict) -> str:
            query = inputs.get("overview", "")
            if not query:
                return "No context available."

            docs = self.retriever.invoke(query)

            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(
                    f"{i}. Question: {doc.metadata['question']}\n   Answer: {doc.metadata['answer']}"
                )

            return "\n\n".join(context_parts)

        # Build the chain using LCEL
        chain = (
                {
                    "context": lambda x: retrieve_context(x),
                    "title": lambda x: x["title"],
                    "overview": lambda x: x["overview"]
                }
                | prompt
                | self.llm
                | StrOutputParser()
        )

        return chain

    # ✅ This method was missing - now it's here!
    def classify_movie(self, title: str, overview: str) -> str:
        """Classify a movie as safe or not safe for children."""
        print(f"\n🎬 Classifying: {title}")
        print(f"📝 Overview: {overview[:100]}...")

        result = self.chain.invoke({
            "title": title,
            "overview": overview
        })

        return result

    # ✅ This method was also missing - now it's here!
    def classify_batch(self, movies: list[dict]) -> list[dict]:
        """Classify multiple movies in batch."""
        results = []
        for movie in movies:
            result = self.classify_movie(movie['title'], movie['overview'])
            results.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'classification': result
            })
        return results


def main():
    """Test the RAG chain with sample movies."""
    print("=" * 60)
    print("🎬 RAG Chain - Movie Content Safety Classifier (with Groq)")
    print("=" * 60)

    # Check if API key is set
    if not GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not found in .env file!")
        print("Please get your free API key from: https://console.groq.com")
        print("and add it to your .env file: GROQ_API_KEY=your_key_here")
        return

    # Initialize the RAG chain
    try:
        rag = RAGChain()
    except Exception as e:
        print(f"\n❌ Error initializing RAG chain: {e}")
        return

    # Test with sample movies
    test_movies = [
        {
            "title": "The Lion King",
            "overview": "A young lion prince flees his kingdom after the murder of his father and learns about responsibility and friendship."
        },
        {
            "title": "The Dark Knight",
            "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
        },
        {
            "title": "Finding Nemo",
            "overview": "After his son is captured in the Great Barrier Reef and taken to Sydney, a timid clownfish sets out on a journey to bring him home."
        },
        {
            "title": "Pulp Fiction",
            "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption."
        }
    ]

    print("\n📋 Testing with 4 sample movies:")
    print("-" * 60)

    for movie in test_movies:
        try:
            result = rag.classify_movie(movie['title'], movie['overview'])
            print(f"\n📌 Result:\n{result}")
        except Exception as e:
            print(f"\n❌ Error classifying {movie['title']}: {e}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("✅ RAG Chain test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()