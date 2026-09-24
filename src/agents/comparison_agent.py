"""
Comparison Agent — Specialized in comparing movies side by side.
Handles safety comparisons, age-appropriateness, and multi-criteria analysis.
"""

import sys
import os
import time

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
from rag_chain import RAGChain
from main import load_movies_from_csv


class ComparisonAgent:
    """
    Specialized agent for comparing movies.

    Responsibilities:
    - Compare two or more movies side by side
    - Determine which is safer for children
    - Compare by age-appropriateness
    - Rank movies by safety criteria

    """

    UNSAFE_GENRES = {"horror", "crime", "thriller"}

    def __init__(self, model_name: str = None):
        """Initialize the Comparison Agent."""
        if model_name is None:
            model_name = GROQ_MODEL

        print("⚖️  Initializing Comparison Agent...")

        # RAG chain for safety info
        self.rag = RAGChain(model_name)

        # LLM (low temp for objective comparisons)
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.2,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY
        )

        # Movie database
        self.movies = load_movies_from_csv()
        print(f"📚 Loaded {len(self.movies)} movies")

        # Tools
        self.tools = self._create_tools()

        # Build agent
        self.agent = self._build_agent()

        print("✅ Comparison Agent ready!")

    def _find_movie(self, title: str) -> Optional[Dict]:
        """Find a movie by title (case-insensitive, partial match)."""
        title_lower = title.lower().strip()
        # Exact match
        for m in self.movies:
            if m['title'].lower() == title_lower:
                return m
        # Partial match
        for m in self.movies:
            if title_lower in m['title'].lower():
                return m
        return None

    def _get_safety_signals(self, movie: Dict) -> Dict:
        """Extract safety signals from a movie."""
        genres = movie.get('genres', '').lower()
        return {
            'is_unsafe_genre': any(ug in genres for ug in self.UNSAFE_GENRES),
            'genres': movie.get('genres', 'N/A'),
            'rating': movie.get('rating', 'N/A'),
            'year': movie.get('year', 'N/A'),
            'title': movie.get('title', 'N/A'),
        }

    def _create_tools(self):
        """Create comparison tools."""

        @tool
        def get_movie_info(movie_title: str) -> str:
            """
            Get info about a movie for comparison: title, year, genres, rating.
            Use this to gather data before comparing.
            """
            movie = self._find_movie(movie_title)
            if not movie:
                return f"Movie '{movie_title}' not found in database."
            return (
                f"Title: {movie['title']}\n"
                f"Year: {movie.get('year', 'N/A')}\n"
                f"Genres: {movie.get('genres', 'N/A')}\n"
                f"Rating: {movie.get('rating', 'N/A')}/10\n"
                f"Overview: {movie.get('overview', 'N/A')}"
            )

        @tool
        def get_safety_classification(movie_title: str) -> str:
            """
            Get the safety classification for a movie using the knowledge base.
            Use this to know if each movie is Safe or Not Safe before comparing.
            """
            try:
                docs = self.rag.retriever.invoke(f"{movie_title} safety")
                if not docs:
                    return f"No safety info for '{movie_title}'."
                lines = [f"Safety info for {movie_title}:"]
                for doc in docs[:3]:
                    lines.append(f"- Q: {doc.metadata['question']}")
                    lines.append(f"  A: {doc.metadata['answer']}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error: {e}"

        @tool
        def get_multiple_movies_info(movie_titles: str) -> str:
            """
            Get info for multiple movies at once (comma-separated).
            Use this for efficient batch lookup when comparing 3+ movies.
            Example input: "Toy Story, Shrek, Frozen"
            """
            titles = [t.strip() for t in movie_titles.split(',') if t.strip()]
            if not titles:
                return "No movie titles provided."
            lines = [f"Info for {len(titles)} movies:"]
            for title in titles:
                movie = self._find_movie(title)
                if movie:
                    signals = self._get_safety_signals(movie)
                    lines.append(f"\n**{movie['title']}**")
                    lines.append(f"  Year: {signals['year']}")
                    lines.append(f"  Genres: {signals['genres']}")
                    lines.append(f"  Rating: {signals['rating']}/10")
                    lines.append(f"  Unsafe genres: {signals['is_unsafe_genre']}")
                else:
                    lines.append(f"\n**{title}** — Not found in database")
            return "\n".join(lines)

        @tool
        def compare_by_safety(movie_titles: str) -> str:
            """
            Compare multiple movies by safety signals.
            Returns a ranking from safest to least safe based on genre analysis.
            Use this when comparing 2+ movies.
            """
            titles = [t.strip() for t in movie_titles.split(',') if t.strip()]
            if len(titles) < 2:
                return "Need at least 2 movies to compare."

            results = []
            for title in titles:
                movie = self._find_movie(title)
                if not movie:
                    continue
                signals = self._get_safety_signals(movie)
                # Score: safer = higher
                score = 100
                if signals['is_unsafe_genre']:
                    score -= 50
                # Bonus for animation/family genres
                genres = signals['genres'].lower()
                if 'animation' in genres or 'family' in genres:
                    score += 20
                results.append((signals['title'], score, signals))

            if len(results) < 2:
                return "Not enough valid movies found for comparison."

            # Sort by score (safest first)
            results.sort(key=lambda x: x[1], reverse=True)

            lines = ["Safety comparison (safest to least safe):"]
            for i, (title, score, signals) in enumerate(results, 1):
                lines.append(f"\n{i}. **{title}** (safety score: {score}/120)")
                lines.append(f"   Genres: {signals['genres']}")
                lines.append(f"   Rating: {signals['rating']}/10")
                lines.append(f"   Unsafe genres: {'Yes' if signals['is_unsafe_genre'] else 'No'}")
            return "\n".join(lines)

        return [
            get_movie_info,
            get_safety_classification,
            get_multiple_movies_info,
            compare_by_safety,
        ]

    def _build_agent(self):
        """Build the Comparison Agent."""

        system_prompt = """You are a Movie Comparison Specialist for children aged 5-10.

YOUR ONLY JOB: Compare movies side by side and declare which is safer or more appropriate.

CRITICAL OUTPUT RULES:

1. You MUST respond using EXACTLY this format:

**Verdict:** [clear one-sentence answer, e.g., "Moana is safer than Frozen" or "Both are equally safe"]

**Comparison:**
- Movie A: [brief safety assessment]
- Movie B: [brief safety assessment]

**Reasoning:**
- [reason 1 why one is safer]
- [reason 2 why one is safer]

**Recommendation:** [one sentence for parents]

2. Use DASHES (-) for all bullets.
3. Do NOT use emojis.
4. Start your final answer with the literal text "**Verdict:**"
5. Be DEFINITIVE — always pick a winner or say "both are equally safe".

TOOLS AVAILABLE:
- get_movie_info(movie_title): get one movie's info
- get_safety_classification(movie_title): get safety classification
- get_multiple_movies_info(movie_titles): batch lookup (comma-separated)
- compare_by_safety(movie_titles): ranked safety comparison

PROCESS:

1. Identify all movies mentioned in the query.
2. Use get_multiple_movies_info OR compare_by_safety to gather data.
3. For safety details, call get_safety_classification for each.
4. Compare objectively based on:
   - Genres (Horror/Crime/Thriller = Not safe)
   - Rating (R = Not safe)
   - Animation/Family = likely safe
5. Provide your verdict.

RULES:
- ALWAYS pick a winner or say "both are equally safe"
- NEVER ask for clarification
- NEVER recommend movies — only compare
- Be objective — base comparison on data, not preference
- If a movie isn't in the database, note that clearly

EXAMPLE OUTPUT:

**Verdict:** Moana is slightly safer than Frozen for very young children.

**Comparison:**
- Moana (2016): Animated adventure, G-rated, no scary scenes
- Frozen (2013): Animated musical, PG-rated, one intense snow monster scene

**Reasoning:**
- Moana has no scenes that could scare children under 6
- Frozen has a brief but intense snow monster sequence
- Both are excellent, but Moana edges ahead for sensitive kids

**Recommendation:** For a 5-year-old sensitive to scary scenes, choose Moana. For a 7-10-year-old, both are great.
"""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def compare(self, query: str, max_retries: int = 3, base_wait: int = 15) -> str:
        """
        Compare movies based on the query.

        Args:
            query: User's comparison question
            max_retries: Maximum retry attempts
            base_wait: Base wait between retries (seconds)

        Returns:
            Formatted comparison response
        """
        print(f"\n⚖️  Comparison Agent: {query}")

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                raw = self._extract_response(response)
                print(f"🔍 Raw output length: {len(raw)} chars")

                formatted = self._format_response(raw)
                return formatted

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = base_wait * (attempt + 1)
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    print(f"   Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return f"❌ Error: {e}"

        return "❌ Unable to complete comparison after multiple attempts."

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
        """
        Light cleanup of the response format.
        Fixes the "- **Header:**" asterisk issue without corrupting markdown.
        """
        if not raw:
            return (
                "**Verdict:** Unable to compare.\n\n"
                "**Comparison:**\n- No data available\n\n"
                "**Reasoning:**\n- Please try again\n\n"
                "**Recommendation:** Provide two movie titles to compare."
            )

        lines = [line.rstrip() for line in raw.split('\n')]

        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Clean up the "- *Header:**" asterisk issue
        cleaned_lines = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]

            # Fix: "- *Header:**" → "**Header:**"
            if stripped.startswith('- *') and stripped.endswith('**'):
                header_text = stripped[3:-2].strip()
                line = f"{indent}**{header_text}**"
            # Fix: "*Header:**" → "**Header:**"
            elif stripped.startswith('*') and not stripped.startswith('**') and stripped.endswith('**'):
                header_text = stripped[1:-2].strip()
                line = f"{indent}**{header_text}**"
            # Fix bullet chars
            elif stripped.startswith('•') or stripped.startswith('·'):
                line = f"{indent}- {stripped[1:].strip()}"

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()


# ============ TEST ============

def main():
    """Test the Comparison Agent."""
    print("=" * 70)
    print("🧪 TESTING COMPARISON AGENT")
    print("=" * 70)

    agent = ComparisonAgent()

    test_queries = [
        "Is Frozen safer than Moana?",
        "Compare Toy Story and Shrek for a 5-year-old",
        "Which is safer: Finding Nemo or The Lion King?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"❓ Query: {query}")
        print(f"{'=' * 70}")

        result = agent.compare(query)
        print(f"\n📌 Response:\n{result}")

        # Wait to avoid rate limits
        print("\n⏳ Waiting 30s before next test...")
        time.sleep(30)


if __name__ == "__main__":
    main()