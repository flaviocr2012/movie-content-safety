"""
Lookup Agent — Specialized in retrieving movie information.
Handles movie details, listings, searches, and database queries.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import re
from typing import Dict, List, Optional
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL
from main import load_movies_from_csv


class LookupAgent:
    """
    Specialized agent for movie information retrieval.

    Responsibilities:
    - Get details about a specific movie
    - List movies by genre
    - Search movies by keyword
    - Count movies in categories

    """

    def __init__(self, model_name: str = None):
        """Initialize the Lookup Agent."""
        if model_name is None:
            model_name = GROQ_MODEL

        print("🔍 Initializing Lookup Agent...")

        # LLM (higher temp for natural language)
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.3,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY
        )

        # Load movie database
        self.movies = load_movies_from_csv()
        print(f"📚 Loaded {len(self.movies)} movies")

        # Tools
        self.tools = self._create_tools()

        # Build agent
        self.agent = self._build_agent()

        print("✅ Lookup Agent ready!")

    def _create_tools(self):
        """Create lookup-specific tools."""

        @tool
        def get_movie_details(movie_title: str) -> str:
            """
            Get full details about a specific movie: overview, genres, rating, year.
            Use this when the user asks about a specific movie.
            """
            title_lower = movie_title.lower().strip()
            for movie in self.movies:
                if movie['title'].lower() == title_lower:
                    return (
                        f"Title: {movie['title']}\n"
                        f"Year: {movie.get('year', 'N/A')}\n"
                        f"Genres: {movie.get('genres', 'N/A')}\n"
                        f"Rating: {movie.get('rating', 'N/A')}/10\n"
                        f"Overview: {movie.get('overview', 'N/A')}"
                    )
            # Try partial match
            matches = [m for m in self.movies if title_lower in m['title'].lower()]
            if matches:
                titles = [m['title'] for m in matches[:5]]
                return f"Exact match not found. Did you mean one of these?\n- " + "\n- ".join(titles)
            return f"Movie '{movie_title}' not found in database."

        @tool
        def list_movies_by_genre(genre: str) -> str:
            """
            List all movies matching a specific genre.
            Genres include: Animation, Adventure, Comedy, Action, Drama, Horror, etc.
            Use this when the user asks for movies of a specific genre.
            """
            genre_lower = genre.lower().strip()
            matches = [
                m for m in self.movies
                if genre_lower in m.get('genres', '').lower()
            ]
            if not matches:
                return f"No movies found with genre '{genre}'."
            # Sort alphabetically
            matches.sort(key=lambda m: m['title'])
            lines = [f"Found {len(matches)} movies with genre '{genre}':"]
            for m in matches[:20]:  # Limit to 20 to avoid huge responses
                lines.append(f"- {m['title']} ({m.get('year', 'N/A')})")
            if len(matches) > 20:
                lines.append(f"... and {len(matches) - 20} more")
            return "\n".join(lines)

        @tool
        def search_movies_by_keyword(keyword: str) -> str:
            """
            Search movies whose title or overview contains a keyword.
            Use this for thematic searches (e.g., 'dinosaur', 'space', 'magic').
            """
            keyword_lower = keyword.lower().strip()
            matches = []
            for m in self.movies:
                title = m.get('title', '').lower()
                overview = m.get('overview', '').lower()
                if keyword_lower in title or keyword_lower in overview:
                    matches.append(m)
            if not matches:
                return f"No movies found matching '{keyword}'."
            lines = [f"Found {len(matches)} movies matching '{keyword}':"]
            for m in matches[:15]:
                lines.append(f"- {m['title']} ({m.get('year', 'N/A')}) [{m.get('genres', 'N/A')}]")
            if len(matches) > 15:
                lines.append(f"... and {len(matches) - 15} more")
            return "\n".join(lines)

        @tool
        def count_movies_by_genre(genre: str = None) -> str:
            """
            Count total movies, or count movies in a specific genre.
            Use this when the user asks "how many" questions.
            """
            if genre is None or genre == "":
                return f"Total movies in database: {len(self.movies)}"
            genre_lower = genre.lower().strip()
            count = sum(1 for m in self.movies if genre_lower in m.get('genres', '').lower())
            return f"Movies with genre '{genre}': {count}"

        @tool
        def list_all_movies() -> str:
            """
            List all movie titles in the database.
            Use this ONLY when the user explicitly asks for the full list.
            """
            titles = sorted([m['title'] for m in self.movies])
            lines = [f"All {len(titles)} movies in database:"]
            for t in titles:
                lines.append(f"- {t}")
            return "\n".join(lines)

        return [
            get_movie_details,
            list_movies_by_genre,
            search_movies_by_keyword,
            count_movies_by_genre,
            list_all_movies,
        ]

    def _build_agent(self):
        """Build the Lookup Agent."""

        system_prompt = """You are a Movie Information Specialist.

YOUR ONLY JOB: Retrieve and present information about movies from the database.

CRITICAL OUTPUT RULES:

1. You MUST respond using EXACTLY this format:

**Summary:** [one-sentence answer]

**Details:**
- [detail 1]
- [detail 2]
- [detail 3]

2. Use dashes (-) for bullets. Do NOT use numbered lists.
3. Do NOT use emojis in the response body.
4. Do NOT add extra sections or headers.
5. Start your final answer with the literal text "**Summary:**"

TOOLS AVAILABLE:
- get_movie_details(movie_title): full info about one movie
- list_movies_by_genre(genre): list movies in a genre
- search_movies_by_keyword(keyword): find movies by theme
- count_movies_by_genre(genre): count movies
- list_all_movies(): list everything

PROCESS:
1. Understand what the user wants
2. Pick the right tool
3. Present the information clearly
4. Do NOT classify safety or recommend — just provide information

RULES:
- Be factual — only report what's in the database
- If a movie isn't found, say so clearly
- Limit movie lists to reasonable size (don't dump 148 movies unless asked)
- NEVER classify movies as safe/unsafe — that's another agent's job

EXAMPLE OUTPUT:

**Summary:** Moana is a 2016 animated Disney film.

**Details:**
- Genres: Animation, Adventure, Comedy
- Rating: 7.6/10
- Overview: A young woman uses her navigational talents to set sail for a fabled island.
"""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def lookup(self, query: str) -> str:
        """
        Retrieve movie information.

        Args:
            query: User's lookup question

        Returns:
            Formatted information response
        """
        print(f"\n🔍 Lookup Agent: {query}")

        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            raw = self._extract_response(response)
            return self._format_response(raw)
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

    def _format_response(self, raw: str) -> str:
        """Clean and standardize the response format."""
        if not raw:
            return "**Summary:** No information found.\n\n**Details:**\n- Please try rephrasing your question."

        lines = [line.strip() for line in raw.split('\n')]

        # Remove empty lines at start/end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        has_summary = any('summary:' in line.lower() for line in lines)
        has_details = any('details:' in line.lower() for line in lines)

        if not has_summary:
            # Wrap raw output
            return (
                    f"**Summary:** {raw.split(chr(10))[0][:200]}\n\n"
                    f"**Details:**\n- " + "\n- ".join(lines[1:6])
            )

        # Clean up
        formatted = []
        for line in lines:
            if not line:
                formatted.append("")
                continue

            # Standardize "**X:**" format
            line = re.sub(r'\*\*\s*([^*]+?)\s*:\s*\*\*', r'**\1:** ', line)

            # Fix bullets
            if line.startswith(('•', '*', '·')):
                line = '- ' + line[1:].strip()

            formatted.append(line)

        return "\n".join(formatted).strip()


# ============ TEST ============

def main():
    """Test the Lookup Agent."""
    print("=" * 70)
    print("🧪 TESTING LOOKUP AGENT")
    print("=" * 70)

    agent = LookupAgent()

    test_queries = [
        "Tell me about Moana",
        "Show me all animated movies",
        "Do you have any movies about dinosaurs?",
        "How many horror movies do you have?",
        "What's the plot of Finding Nemo?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"❓ Query: {query}")
        print(f"{'=' * 70}")

        result = agent.lookup(query)
        print(f"\n📌 Response:\n{result}")


if __name__ == "__main__":
    main()