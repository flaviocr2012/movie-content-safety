"""
Movie Safety Agent with Memory.
Extends the base agent with short-term, long-term, and semantic memory.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL, IMDB_MOVIES_PATH
from rag_chain import RAGChain
from main import load_movies_from_csv
from memory import MemoryManager


class MemoryAgent:
    """
    Movie Safety Agent with memory capabilities.
    Remembers conversations, learns preferences, and recalls facts.
    """

    def __init__(self, model_name: str = None, user_id: str = "default"):
        """
        Initialize the agent with memory.

        Args:
            model_name: Groq model to use
            user_id: Unique user identifier for memory persistence
        """
        if model_name is None:
            model_name = GROQ_MODEL

        print("🔄 Initializing Memory Agent...")

        # Memory
        self.memory = MemoryManager(user_id=user_id)
        self.user_id = user_id

        # RAG chain
        self.rag = RAGChain(model_name)

        # LLM
        print(f"🔄 Initializing LLM with model: {model_name}...")
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,
            max_tokens=1000,
            groq_api_key=GROQ_API_KEY
        )

        # Load movies
        self.movies = load_movies_from_csv()
        self.movie_titles = [m['title'] for m in self.movies]

        # Tools
        self.tools = self._create_tools()

        # Build agent
        self.agent = self._build_agent()

        print("✅ Memory Agent initialized successfully!")

    # ============ TOOLS ============

    def _create_tools(self):
        """Create tools for the agent."""

        @tool
        def safety_knowledge_base(query: str) -> str:
            """
            Retrieves information about what makes a movie safe or unsafe for children.
            Use this when you need to know about safety rules or specific movie safety info.
            """
            try:
                docs = self.rag.retriever.invoke(query)
                context_parts = []
                for i, doc in enumerate(docs[:5], 1):
                    context_parts.append(
                        f"{i}. Q: {doc.metadata['question']}\n   A: {doc.metadata['answer']}"
                    )
                return "\n\n".join(context_parts) if context_parts else "No relevant info."
            except Exception as e:
                return f"Error: {e}"

        @tool
        def movie_details(movie_title: str) -> str:
            """
            Gets detailed information about a specific movie.
            Use this when you need details like rating, year, or genres.
            """
            movie_title_lower = movie_title.lower()
            for movie in self.movies:
                if movie['title'].lower() == movie_title_lower:
                    return (f"Title: {movie['title']}\n"
                            f"Overview: {movie['overview']}\n"
                            f"Rating: {movie.get('rating', 'N/A')}\n"
                            f"Year: {movie.get('year', 'N/A')}\n"
                            f"Genres: {movie.get('genres', 'N/A')}")
            return f"Movie '{movie_title}' not found."

        @tool
        def list_movies() -> str:
            """Lists all movies available in the database."""
            if not self.movie_titles:
                return "No movies found."
            return ", ".join(sorted(self.movie_titles))

        @tool
        def filter_movies_by_genre(genre: str) -> str:
            """
            Filters movies by genre (e.g., Fantasy, Animation, Horror).
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
            return f"Movies with genre '{genre}' ({len(matching)}):\n" + "\n".join(f"  - {m}" for m in matching)

        # Memory tools
        @tool
        def remember_preference(key: str, value: str) -> str:
            """
            Remembers a user preference for future conversations.
            Use this when the user explicitly states a preference.
            Example: remember_preference("favorite_genre", "animation")
            """
            self.memory.remember_preference(key, value, source="explicit")
            return f"✅ Remembered: {key} = {value}"

        @tool
        def remember_fact(fact: str, category: str = "general") -> str:
            """
            Remembers a fact about the user.
            Use this for important personal details (e.g., "has a 5-year-old daughter").
            """
            self.memory.remember_fact(fact, category=category)
            return f"✅ Remembered fact: {fact}"

        @tool
        def recall_preferences() -> str:
            """
            Recalls all known user preferences.
            Use this when you need to check what you know about the user.
            """
            prefs = self.memory.long_term.get_all_preferences()
            if not prefs:
                return "No known preferences."
            return "Known preferences:\n" + "\n".join(f"  - {k}: {v}" for k, v in prefs.items())

        return [
            safety_knowledge_base,
            movie_details,
            list_movies,
            filter_movies_by_genre,
            remember_preference,
            remember_fact,
            recall_preferences,
        ]

    # ============ AGENT BUILDER ============

    def _build_agent(self):
        """Build the agent with memory-aware system prompt."""

        system_prompt = """You are a helpful AI assistant specialized in movie safety for children.

Your task is to answer questions about whether movies are appropriate for children aged 5-10.

**CRITICAL INSTRUCTION: You MUST ALWAYS provide a final text response to the user.**
- Even if you use tools, you MUST synthesize the results into a clear answer.
- NEVER return an empty response.

**Available Tools:**
1. `safety_knowledge_base(query)` — Check safety rules
2. `movie_details(movie_title)` — Get details about a movie
3. `list_movies()` — List all available movies
4. `filter_movies_by_genre(genre)` — Filter movies by genre
5. `remember_preference(key, value)` — Save a user preference
6. `remember_fact(fact, category)` — Save a fact about the user
7. `recall_preferences()` — Check what preferences you know

**Using Memory:**
- The user's **preferences** and **facts** are provided in the conversation context
- If the user states a preference (e.g., "I love animated movies"), call `remember_preference`
- If the user shares a fact (e.g., "I have a 5-year-old"), call `remember_fact`
- Reference known preferences when answering (e.g., "Since you like animation, here are...")

**When answering questions:**
1. Consider the user's known preferences (from context)
2. Use the right tool for the query
3. Synthesize results into a friendly, clear response
4. If the user shares something to remember, save it

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

    # ============ ASK ============

    def ask(self, question: str) -> str:
        """
        Ask a question with memory context.

        Args:
            question: User's question

        Returns:
            Agent's response
        """
        print(f"\n🤖 Question: {question}")
        print("-" * 70)

        # Add user message to memory
        self.memory.add_user_message(question)

        # Build memory context
        memory_context = self.memory.build_context()
        print(f"📋 Memory context:\n{memory_context}\n")

        # Build full message with memory context
        enriched_question = question
        if memory_context != "No prior context.":
            enriched_question = (
                f"[CONTEXT FROM MEMORY]\n"
                f"{memory_context}\n"
                f"[END CONTEXT]\n\n"
                f"User question: {question}"
            )

        try:
            # Get conversation history
            history = self.memory.get_conversation_history()

            # Invoke the agent with context
            response = self.agent.invoke(
                {"messages": history + [{"role": "user", "content": enriched_question}]}
            )

            # Extract response
            result = self._extract_response(response)

            # Add assistant message to memory
            self.memory.add_assistant_message(result)

            return result
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(f"❌ {error_msg}")
            return error_msg

    def _extract_response(self, response: Dict) -> str:
        """Extract the final AI response from the agent output."""
        if "messages" in response:
            messages = response["messages"]
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if not content:
                    continue
                if isinstance(content, str) and content.strip():
                    if not content.strip().startswith('{'):
                        return content.strip()
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    if text_parts:
                        return "\n".join(text_parts).strip()

        if "output" in response:
            return response["output"]

        return "I processed your request but couldn't generate a response."

    # ============ INTERACTIVE MODE ============

    def interactive_mode(self):
        """Run the agent in interactive mode with memory."""
        print("=" * 70)
        print("🧠 MOVIE SAFETY AGENT WITH MEMORY")
        print("=" * 70)
        print(f"\n📊 User ID: {self.user_id}")
        print(f"📊 Memory stats: {self.memory.get_stats()}")
        print("\n📋 Instructions:")
        print("  - Ask questions about movie safety")
        print("  - The agent remembers your preferences and facts")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'memory' to see what the agent remembers")
        print("  - Type 'clear' to clear the conversation (keeps preferences)")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            question = input("💭 Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if question.lower() == 'memory':
                print("\n" + "=" * 70)
                print("📊 MEMORY STATE")
                print("=" * 70)
                print(f"Stats: {self.memory.get_stats()}")
                print(f"\nPreferences:\n{self.memory.long_term.get_preferences_context()}")
                print(f"\nFacts:\n{self.memory.semantic.get_facts_context()}")
                print("=" * 70)
                continue

            if question.lower() == 'clear':
                self.memory.clear_short_term()
                print("✅ Conversation history cleared (preferences kept)")
                continue

            if not question:
                print("⚠️ Please enter a question.")
                continue

            result = self.ask(question)
            print(f"\n📌 Response:\n{result}")


# ============ MAIN ============

def main():
    """Test the memory agent."""
    print("=" * 70)
    print("🧪 TESTING MEMORY AGENT")
    print("=" * 70)

    agent = MemoryAgent(user_id="test_user")

    # Test conversation with memory
    test_questions = [
        "I love animated movies",
        "Show me animated movies",
        "Is the first one safe?",
        "I have a 5-year-old daughter",
        "What movies do you recommend?",
    ]

    for question in test_questions:
        result = agent.ask(question)
        print(f"\n📌 Response:\n{result}")
        print("=" * 70)


if __name__ == "__main__":
    agent = MemoryAgent(user_id="default")
    agent.interactive_mode()