"""
LangChain Agent for complex movie queries.
The agent uses the retriever as a tool to answer questions about movie safety.
"""

import os
from typing import List, Dict, Any

# ✅ Correct imports for LangChain 1.3.18
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Import configuration and the RAG chain
from config import GROQ_API_KEY, GROQ_MODEL, IMDB_MOVIES_PATH
from rag_chain import RAGChain
from main import load_movies_from_csv


class MovieSafetyAgent:
    """
    LangChain Agent for answering complex questions about movie safety.
    Uses the retriever as a tool to fetch relevant information.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the agent with a retriever tool.

        Args:
            model_name: Groq model to use (default: from config)
        """
        if model_name is None:
            model_name = GROQ_MODEL

        print("🔄 Initializing Movie Safety Agent...")

        # Initialize the RAG chain (which contains the retriever)
        self.rag = RAGChain(model_name)

        # Initialize the LLM
        print(f"🔄 Initializing LLM with model: {model_name}...")
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,
            max_tokens=1000,
            groq_api_key=GROQ_API_KEY
        )

        # Load movie data for reference
        self.movies = load_movies_from_csv()
        self.movie_titles = [m['title'] for m in self.movies]

        # Create tools using the @tool decorator
        self.tools = self._create_tools()

        # Build the agent
        self.agent = self._build_agent()

        print("✅ Movie Safety Agent initialized successfully!")

    def _create_tools(self):
        """
        Create tools for the agent using the @tool decorator.

        Returns:
            List of tools
        """

        # Tool 1: Retrieve safety information
        @tool
        def safety_knowledge_base(query: str) -> str:
            """
            Retrieves information about what makes a movie safe or unsafe for children.
            Use this when you need to know about safety rules, movie content,
            or specific movie safety information.
            """
            try:
                docs = self.rag.retriever.invoke(query)

                context_parts = []
                for i, doc in enumerate(docs[:5], 1):
                    context_parts.append(
                        f"{i}. Question: {doc.metadata['question']}\n"
                        f"   Answer: {doc.metadata['answer']}"
                    )

                return "\n\n".join(context_parts) if context_parts else "No relevant information found."
            except Exception as e:
                return f"Error retrieving information: {e}"

        # Tool 2: Get movie details from CSV
        @tool
        def movie_details(movie_title: str) -> str:
            """
            Gets detailed information about a specific movie from the database.
            Use this when you need to know details like rating, year, or genres.
            """
            movie_title_lower = movie_title.lower()
            for movie in self.movies:
                if movie['title'].lower() == movie_title_lower:
                    return (f"Title: {movie['title']}\n"
                            f"Overview: {movie['overview']}\n"
                            f"Rating: {movie.get('rating', 'N/A')}\n"
                            f"Year: {movie.get('year', 'N/A')}\n"
                            f"Genres: {movie.get('genres', 'N/A')}")

            return f"Movie '{movie_title}' not found in database."

        # Tool 3: List available movies
        @tool
        def list_movies() -> str:
            """
            Lists all movies available in the database.
            Use this when you need to know what movies are available.
            """
            if not self.movie_titles:
                return "No movies found in database."
            return ", ".join(sorted(self.movie_titles))

        # Tool 4: Filter movies by genre
        @tool
        def filter_movies_by_genre(genre: str) -> str:
            """
            Filters movies by genre (e.g., "Fantasy", "Animation", "Comedy", "Horror").
            Use this when the user asks for movies of a specific genre.
            """
            genre_lower = genre.lower()
            matching = []

            for movie in self.movies:
                movie_genres = movie.get('genres', '').lower()
                if genre_lower in movie_genres:
                    matching.append(f"{movie['title']} ({movie.get('year', 'N/A')})")

            if not matching:
                return f"No movies found with genre '{genre}'."

            return f"Movies with genre '{genre}' ({len(matching)} found):\n" + "\n".join(f"  - {m}" for m in matching)

        return [safety_knowledge_base, movie_details, list_movies, filter_movies_by_genre]

    def _build_agent(self):
        """
        Build the LangChain agent using create_agent.

        Returns:
            CompiledStateGraph agent ready to process queries
        """

        # ✅ Improved system prompt with explicit instruction to always respond
        system_prompt = """You are a helpful AI assistant specialized in movie safety for children.

Your task is to answer questions about whether movies are appropriate for children aged 5-10.

**CRITICAL INSTRUCTION: You MUST ALWAYS provide a final text response to the user.**
- Even if you use tools, you MUST synthesize the results into a clear answer.
- NEVER return an empty response.
- After calling any tool, use its results to answer the user's question.

**Available Tools:**
1. `safety_knowledge_base(query)` - Check safety rules and Q&A pairs
2. `movie_details(movie_title)` - Get details about a specific movie
3. `list_movies()` - List all available movies (use for broad questions)
4. `filter_movies_by_genre(genre)` - Filter movies by genre (use for genre-specific questions)

**When answering questions:**
1. If the user asks about a **specific genre** (fantasy, animation, horror, etc.), use `filter_movies_by_genre`
2. If the user asks about a **specific movie**, use `movie_details`
3. If the user asks about **safety rules**, use `safety_knowledge_base`
4. If the user asks **"what movies do you have"** or similar, use `list_movies`
5. After getting tool results, synthesize a clear, friendly answer in the final response

**Example interactions:**
- User: "I like fantasy movies. What do you have?"
  → Use `filter_movies_by_genre("Fantasy")` → Respond with a formatted list

- User: "Is Jurassic Park safe?"
  → Use `movie_details("Jurassic Park")` and `safety_knowledge_base("Jurassic Park safety")` → Respond with classification

- User: "What movies do you have?"
  → Use `list_movies()` → Respond with a summary (not the full list, just highlights)

**Response format:**
- Use emojis to make responses friendly
- Format movie lists as bullet points
- Always end with a helpful closing or offer to help more
"""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def ask(self, question: str) -> str:
        """
        Ask a question to the agent.

        Args:
            question: The user's question

        Returns:
            The agent's response
        """
        print(f"\n🤖 Question: {question}")
        print("-" * 70)

        try:
            # ✅ Invoke the agent with the question
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            # ✅ Extract the final AI response correctly
            if "messages" in response:
                messages = response["messages"]

                # Iterate from the end to find the last AI message with text content
                for message in reversed(messages):
                    # Get content
                    content = getattr(message, "content", None)

                    # Skip empty content
                    if not content:
                        continue

                    # Handle string content (most common)
                    if isinstance(content, str) and content.strip():
                        # Skip tool calls (they usually start with specific patterns)
                        if not content.strip().startswith('{'):
                            return content.strip()

                    # Handle list content (structured format)
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        if text_parts:
                            return "\n".join(text_parts).strip()

            # Fallback: check for output key
            if "output" in response:
                return response["output"]

            # Last resort: try to stringify
            return "I processed your request but couldn't generate a text response. Please try rephrasing your question."

        except Exception as e:
            return f"Error processing question: {e}"

    def interactive_mode(self):
        """Run the agent in interactive mode."""
        print("=" * 70)
        print("🤖 MOVIE SAFETY AGENT - INTERACTIVE MODE")
        print("=" * 70)
        print("\n📋 Instructions:")
        print("  - Ask questions about movie safety")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Example: 'I like fantasy movies. What do you have?'")
        print("  - Example: 'Can you find me a movie like The Lion King?'")
        print("  - Example: 'What movies are safe for children?'")
        print("  - Example: 'Is Jurassic Park safe?'")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            question = input("💭 Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not question:
                print("⚠️ Please enter a question.")
                continue

            result = self.ask(question)
            print(f"\n📌 Response:\n{result}")


def main():
    """Test the agent with sample questions."""
    agent = MovieSafetyAgent()

    test_questions = [
        "I like fantasy movies. What do you have?",
        "Is Jurassic Park safe for children?",
        "What are the best family movies in the database?",
        "Show me animated movies",
    ]

    print("=" * 70)
    print("🧪 TESTING AGENT WITH SAMPLE QUESTIONS")
    print("=" * 70)

    for question in test_questions:
        result = agent.ask(question)
        print(f"\n📌 Response:\n{result}")
        print("=" * 70)


if __name__ == "__main__":
    agent = MovieSafetyAgent()
    agent.interactive_mode()