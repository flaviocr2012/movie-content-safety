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

        return [safety_knowledge_base, movie_details, list_movies]

    def _build_agent(self):
        """
        Build the LangChain agent using create_agent.

        Returns:
            CompiledStateGraph agent ready to process queries
        """

        # ✅ Create the agent using create_agent with correct parameters
        agent = create_agent(
            model=self.llm,  # ✅ Use 'model' not 'llm'
            tools=self.tools,
            system_prompt=(
                "You are a helpful AI assistant specialized in movie safety for children. "
                "Your task is to answer questions about whether movies are appropriate for children aged 5-10.\n\n"
                "When answering questions about movie safety:\n"
                "1. Use the Safety Knowledge Base to retrieve specific safety rules\n"
                "2. Use Movie Details to get information about specific movies\n"
                "3. Always consider: genres, rating, content, and explicit safety rules\n"
                "4. If you're unsure, err on the side of caution and classify as 'Not safe for children'"
            )
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

            # Extract the response from the agent's output
            if "messages" in response:
                return response["messages"][-1].content
            elif "output" in response:
                return response["output"]
            else:
                return str(response)
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
        print("  - Example: 'Can you find me a movie like The Lion King that is appropriate for a 5-year-old?'")
        print("  - Example: 'What movies are safe for children?'")
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
        "Can you find me a movie like The Lion King that is appropriate for a 5-year-old?",
        "Is Jurassic Park safe for children?",
        "What are the best family movies in the database?",
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