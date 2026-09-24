"""
Safety Agent — Specialized in classifying movies as Safe or Not Safe for children.
Uses the RAG chain and knowledge base for accurate classifications.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import re
from typing import Dict, Any, Optional
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL
from rag_chain import RAGChain
from main import load_movies_from_csv


class SafetyAgent:
    """
    Specialized agent for movie safety classification.

    Responsibilities:
    - Determine if a movie is Safe or Not Safe for children aged 5-10
    - Explain the reasoning based on safety rules
    - Use the RAG knowledge base for accurate decisions

    """

    def __init__(self, model_name: str = None):
        """
        Initialize the Safety Agent.

        Args:
            model_name: Groq model to use (default: GROQ_MODEL)
        """
        if model_name is None:
            model_name = GROQ_MODEL

        print("🛡️  Initializing Safety Agent...")

        # RAG chain for safety knowledge
        self.rag = RAGChain(model_name)

        # LLM
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.1,
            max_tokens=500,
            groq_api_key=GROQ_API_KEY
        )

        # Movie database
        self.movies = load_movies_from_csv()
        self.movie_titles = [m['title'] for m in self.movies]

        # Tools
        self.tools = self._create_tools()

        # Build agent
        self.agent = self._build_agent()

        print("✅ Safety Agent ready!")

    def _create_tools(self):
        """Create tools for the Safety Agent."""

        @tool
        def safety_knowledge_base(query: str) -> str:
            """
            Retrieves safety rules and Q&A pairs from the knowledge base.
            Use this to check if a movie is safe or to understand safety rules.
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
            Gets details about a movie: overview, genres, rating, year.
            Use this to gather context before classifying.
            """
            movie_title_lower = movie_title.lower()
            for movie in self.movies:
                if movie['title'].lower() == movie_title_lower:
                    return (f"Title: {movie['title']}\n"
                            f"Overview: {movie['overview']}\n"
                            f"Genres: {movie.get('genres', 'N/A')}\n"
                            f"Rating: {movie.get('rating', 'N/A')}\n"
                            f"Year: {movie.get('year', 'N/A')}")
            return f"Movie '{movie_title}' not found in database."

        return [safety_knowledge_base, movie_details]

    def _build_agent(self):
        """Build the Safety Agent with a focused prompt."""

        system_prompt = """You are a Movie Safety Specialist Agent for children aged 5-10.

YOUR ONLY JOB: Classify a movie as "Safe for children" or "Not safe for children".

CRITICAL OUTPUT RULES:

1. You MUST respond using EXACTLY this format — no deviations:

**Classification:** [Safe for children / Not safe for children]

**Reasoning:**
- [reason 1]
- [reason 2]
- [reason 3]

**Recommendation:** [one sentence for parents]

2. Do NOT use numbered lists. Use dashes (-) for bullets.
3. Do NOT add extra sections or headers.
4. Do NOT use emojis in the response body.
5. Start your final answer with the literal text "**Classification:**"

TOOLS AVAILABLE:
- safety_knowledge_base(query): get safety rules
- movie_details(movie_title): get movie info

PROCESS:
1. Call movie_details to get genres and rating
2. Call safety_knowledge_base for relevant safety rules
3. Classify based on:
   - Horror / Crime / Thriller genres → Not safe
   - Rated R → Not safe
   - Explicit violence or adult themes → Not safe
   - Animated / family-friendly / G-PG → Safe
   - Intense action with scary creatures → Not safe
4. Output ONLY the formatted response above.

RULES:
- Be definitive — no "maybe"
- Be cautious — when unsure, classify as Not safe
- If movie not found, still classify based on the title/genre if available
- NEVER recommend other movies — that's another agent's job

EXAMPLE OUTPUT:

**Classification:** Safe for children

**Reasoning:**
- Animated, family-friendly film
- Rated G with no violence
- Positive messages about family

**Recommendation:** Perfect for children 5-10.
"""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def classify(self, movie_title: str, overview: str = None,
                 genres: str = "Unknown", rating: str = "Unknown") -> str:
        """
        Classify a movie as Safe or Not Safe.

        Args:
            movie_title: Movie title
            overview: Movie overview (optional)
            genres: Movie genres (optional)
            rating: Movie rating (optional)

        Returns:
            Formatted classification response
        """
        print(f"\n🛡️  Safety Agent classifying: {movie_title}")

        # Build the query
        query = f"Is '{movie_title}' safe for children aged 5-10?"
        if genres != "Unknown":
            query += f"\nGenres: {genres}"
        if rating != "Unknown":
            query += f"\nRating: {rating}"
        if overview:
            query += f"\nOverview: {overview[:200]}"

        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            raw = self._extract_response(response)
            return self._format_response(raw, movie_title)
        except Exception as e:
            return f"❌ Error: {e}"

    def _extract_response(self, response: Dict) -> str:
        """Extract the final response from the agent output."""
        if "messages" in response:
            for message in reversed(response["messages"]):
                content = getattr(message, "content", None)
                if not content:
                    continue
                if isinstance(content, str) and content.strip():
                    if not content.strip().startswith('{'):
                        return content.strip()
                elif isinstance(content, list):
                    text_parts = [
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    if text_parts:
                        return "\n".join(text_parts).strip()

        if "output" in response:
            return response["output"]

        return ""

    def _format_response(self, raw: str, movie_title: str) -> str:
        """
        Clean and standardize the response format.
        Fixes indentation, punctuation, and ensures consistent structure.
        """
        if not raw:
            return f"**Classification:** Unable to classify '{movie_title}'\n\n**Reasoning:**\n- No response generated\n\n**Recommendation:** Please try again."

        # Remove leading/trailing whitespace on each line
        lines = [line.strip() for line in raw.split('\n')]

        # Remove empty lines at start/end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        # Detect if response already has the correct format
        has_classification = any('classification:' in line.lower() for line in lines)
        has_reasoning = any('reasoning:' in line.lower() for line in lines)
        has_recommendation = any('recommendation:' in line.lower() for line in lines)

        # If format is missing, wrap it
        if not has_classification:
            # Try to detect safe/not safe from content
            raw_lower = raw.lower()
            if 'not safe' in raw_lower or 'unsafe' in raw_lower:
                classification = "Not safe for children"
            else:
                classification = "Safe for children"

            return (
                f"**Classification:** {classification}\n\n"
                f"**Reasoning:**\n{raw}\n\n"
                f"**Recommendation:** Please review the reasoning above for details."
            )

        # Clean up existing format
        formatted = []
        for line in lines:
            if not line:
                formatted.append("")
                continue

            # Ensure "**X:**" format is clean
            line = re.sub(r'\*\*\s*([^*]+?)\s*:\s*\*\*', r'**\1:** ', line)

            # Fix bullet points (allow -, •, *, etc.)
            if line.startswith(('•', '*', '·')):
                line = '- ' + line[1:].strip()

            formatted.append(line)

        result = "\n".join(formatted)

        # Ensure Reasoning section has bullets
        if has_reasoning and not has_recommendation:
            result += "\n\n**Recommendation:** No specific recommendation available."

        return result.strip()


# ============ TEST ============

def main():
    """Test the Safety Agent."""
    print("=" * 70)
    print("🧪 TESTING SAFETY AGENT")
    print("=" * 70)

    agent = SafetyAgent()

    test_movies = [
        {"title": "Finding Nemo", "genres": "Animation, Adventure, Comedy", "rating": "8.2"},
        {"title": "Jurassic Park", "genres": "Action, Adventure, Sci-Fi", "rating": "8.2"},
        {"title": "The Conjuring", "genres": "Horror, Mystery, Thriller", "rating": "7.5"},
        {"title": "Moana", "genres": "Animation, Adventure, Comedy", "rating": "7.6"},
    ]

    for movie in test_movies:
        print(f"\n{'=' * 70}")
        print(f"🎬 Testing: {movie['title']}")
        print(f"{'=' * 70}")

        result = agent.classify(
            movie_title=movie['title'],
            genres=movie['genres'],
            rating=movie['rating']
        )
        print(f"\n📌 Classification:\n{result}")


if __name__ == "__main__":
    main()