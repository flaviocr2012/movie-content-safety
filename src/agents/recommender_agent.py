"""
Recommender Agent — Specialized in personalized movie recommendations.
Uses memory to personalize suggestions based on user preferences and facts.
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
from memory import MemoryManager


class RecommenderAgent:
    """
    Specialized agent for movie recommendations.

    Responsibilities:
    - Suggest movies based on user preferences
    - Personalize recommendations using memory
    - Filter by genre, age, mood
    - Explain why each movie is recommended
    """

    # Known safe genres for children 5-10
    SAFE_GENRES = {"animation", "family", "adventure", "comedy", "fantasy"}

    # Genres to avoid recommending
    UNSAFE_GENRES = {"horror", "crime", "thriller"}

    def __init__(self, model_name: str = None, user_id: str = "default"):
        """Initialize the Recommender Agent."""
        if model_name is None:
            model_name = GROQ_MODEL

        print("💡 Initializing Recommender Agent...")

        # Memory
        self.memory = MemoryManager(user_id=user_id)
        self.user_id = user_id

        # RAG chain for safety filtering
        self.rag = RAGChain(model_name)

        # LLM (higher temp for creative recommendations)
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.5,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY
        )

        # Load movies
        self.movies = load_movies_from_csv()
        print(f"📚 Loaded {len(self.movies)} movies")

        # Tools
        self.tools = self._create_tools()

        # Build agent
        self.agent = self._build_agent()

        print("✅ Recommender Agent ready!")

    def _is_movie_safe(self, movie: Dict) -> bool:
        """Check if a movie is likely safe based on genres."""
        genres = movie.get('genres', '').lower()
        if any(ug in genres for ug in self.UNSAFE_GENRES):
            return False
        return True

    def _create_tools(self):
        """Create recommendation tools."""

        @tool
        def get_user_preferences() -> str:
            """
            Get all known user preferences from memory.
            Use this FIRST to personalize recommendations.
            """
            prefs = self.memory.long_term.get_all_preferences()
            facts = self.memory.semantic.get_facts()
            if not prefs and not facts:
                return "No preferences or facts known about this user yet."
            lines = []
            if prefs:
                lines.append("Known preferences:")
                for k, v in prefs.items():
                    lines.append(f"- {k}: {v}")
            if facts:
                lines.append("Known facts:")
                for f in facts:
                    lines.append(f"- [{f.category}] {f.fact}")
            return "\n".join(lines)

        @tool
        def find_safe_movies_by_genre(genre: str) -> str:
            """
            Find movies in a genre, filtered to only safe ones.
            Use this to get candidate movies for recommendations.
            """
            genre_lower = genre.lower().strip()
            matches = [
                m for m in self.movies
                if genre_lower in m.get('genres', '').lower()
                   and self._is_movie_safe(m)
            ]
            if not matches:
                return f"No safe movies found in genre '{genre}'."
            matches.sort(key=lambda m: float(m.get('rating', 0) or 0), reverse=True)
            lines = [f"Safe movies in '{genre}' (sorted by rating):"]
            for m in matches[:15]:
                lines.append(f"- {m['title']} ({m.get('year', 'N/A')}) — {m.get('rating', 'N/A')}/10")
            return "\n".join(lines)

        @tool
        def find_safe_movies_by_keyword(keyword: str) -> str:
            """
            Search safe movies by keyword (theme, mood, etc.).
            Use this when the user's request is thematic (e.g., 'friendship', 'space').
            """
            keyword_lower = keyword.lower().strip()
            matches = []
            for m in self.movies:
                if not self._is_movie_safe(m):
                    continue
                title = m.get('title', '').lower()
                overview = m.get('overview', '').lower()
                if keyword_lower in title or keyword_lower in overview:
                    matches.append(m)
            if not matches:
                return f"No safe movies found matching '{keyword}'."
            matches.sort(key=lambda m: float(m.get('rating', 0) or 0), reverse=True)
            lines = [f"Safe movies matching '{keyword}':"]
            for m in matches[:10]:
                lines.append(f"- {m['title']} ({m.get('year', 'N/A')}) — {m.get('rating', 'N/A')}/10")
            return "\n".join(lines)

        @tool
        def find_top_rated_safe_movies(count: int = 10) -> str:
            """
            Get the top-rated safe movies overall.
            Use this when the user has no specific genre or theme.
            """
            safe_movies = [m for m in self.movies if self._is_movie_safe(m)]
            safe_movies.sort(key=lambda m: float(m.get('rating', 0) or 0), reverse=True)
            lines = [f"Top {count} safe movies overall:"]
            for m in safe_movies[:count]:
                lines.append(f"- {m['title']} ({m.get('year', 'N/A')}) — {m.get('rating', 'N/A')}/10 [{m.get('genres', 'N/A')}]")
            return "\n".join(lines)

        @tool
        def verify_movie_safety(movie_title: str) -> str:
            """
            Verify a movie is safe using the knowledge base.
            Use this as a final check before recommending.
            """
            try:
                docs = self.rag.retriever.invoke(f"{movie_title} safety")
                if not docs:
                    return f"No safety info found for {movie_title}."
                lines = [f"Safety info for {movie_title}:"]
                for doc in docs[:3]:
                    lines.append(f"- Q: {doc.metadata['question']}")
                    lines.append(f"  A: {doc.metadata['answer']}")
                return "\n".join(lines)
            except Exception as e:
                return f"Error: {e}"

        @tool
        def remember_preference(key: str, value: str) -> str:
            """
            Save a user preference for future recommendations.
            Use this when the user reveals a preference.
            """
            self.memory.remember_preference(key, value, source="explicit")
            return f"✅ Remembered: {key} = {value}"

        return [
            get_user_preferences,
            find_safe_movies_by_genre,
            find_safe_movies_by_keyword,
            find_top_rated_safe_movies,
            verify_movie_safety,
            remember_preference,
        ]

    def _build_agent(self):
        """Build the Recommender Agent."""

        system_prompt = """You are a Movie Recommendation Specialist for children aged 5-10.

YOUR ONLY JOB: Suggest movies tailored to the user's preferences.

CRITICAL OUTPUT RULES:

1. You MUST respond using EXACTLY this format:

**Top Recommendations:**
1. [Movie Title (Year)] — [one-sentence why]
2. [Movie Title (Year)] — [one-sentence why]
3. [Movie Title (Year)] — [one-sentence why]

**Why These:**
- [reason 1]
- [reason 2]

**Next Step:** [one suggestion for the user]

2. Use NUMBERS (1, 2, 3) for the recommendations list.
3. Use DASHES (-) for the "Why These" bullets.
4. Do NOT use emojis.
5. Start your final answer with the literal text "**Top Recommendations:**"

TOOLS AVAILABLE:
- get_user_preferences(): check what you know about the user
- find_safe_movies_by_genre(genre): get safe movies in a genre
- find_safe_movies_by_keyword(keyword): thematic safe movie search
- find_top_rated_safe_movies(count): best safe movies overall
- verify_movie_safety(movie_title): final safety check
- remember_preference(key, value): save a preference

PROCESS:
1. Call get_user_preferences() FIRST — always check what you know
2. Based on preferences, pick the right search tool
3. Get 3-5 candidate movies
4. Recommend 3 movies with brief reasons
5. NEVER recommend Horror, Crime, or Thriller genres
6. If the user reveals a new preference, save it

RULES:
- ALWAYS personalize based on known preferences
- ALWAYS recommend at least 3 movies
- NEVER recommend unsafe genres (Horror, Crime, Thriller)
- Be enthusiastic but concise
- If no preferences are known, ask a follow-up question OR recommend top-rated safe movies
- NEVER classify movies — only recommend

EXAMPLE OUTPUT:

**Top Recommendations:**
1. Moana (2016) — Animated adventure with a brave heroine
2. Inside Out (2015) — Emotional intelligence through colorful characters
3. Coco (2017) — Family, music, and Mexican culture

**Why These:**
- All are animated, family-friendly, rated G/PG
- Strong positive messages about family and courage
- Age-appropriate for children 5-10

**Next Step:** Want more like these? Just ask for a specific theme!
"""

        agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )

        return agent

    def recommend(self, query: str, max_retries: int = 3, base_wait: int = 15) -> str:
        """
        Generate personalized movie recommendations with automatic retry on rate limits.

        Args:
            query: User's recommendation request
            max_retries: Maximum number of retry attempts
            base_wait: Base wait time between retries (seconds)

        Returns:
            Formatted recommendation response
        """
        print(f"\n💡 Recommender Agent: {query}")

        # Add to memory
        self.memory.add_user_message(query)

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                raw = self._extract_response(response)
                formatted = self._format_response(raw)

                # Add response to memory
                self.memory.add_assistant_message(formatted)

                return formatted

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = base_wait * (attempt + 1)
                    print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries})")
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                return f"❌ Error: {e}"

        return "❌ Max retries exceeded. Please wait 60 seconds and try again."

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
            return "**Top Recommendations:**\n1. No recommendations available\n\n**Why These:**\n- Please try again\n\n**Next Step:** Ask for a specific genre or theme."

        lines = [line.strip() for line in raw.split('\n')]

        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()

        has_recs = any('recommendation' in line.lower() for line in lines)
        has_why = any('why' in line.lower() for line in lines)

        if not has_recs:
            return (
                f"**Top Recommendations:**\n{raw}\n\n"
                f"**Why These:**\n- Curated based on your request\n\n"
                f"**Next Step:** Ask for more specific recommendations."
            )

        formatted = []
        for line in lines:
            if not line:
                formatted.append("")
                continue

            # Standardize headers
            line = re.sub(r'\*\*\s*([^*]+?)\s*:\s*\*\*', r'**\1:** ', line)

            # Fix bullets
            if line.startswith(('•', '*', '·')):
                line = '- ' + line[1:].strip()

            formatted.append(line)

        # Ensure Next Step exists
        result = "\n".join(formatted)
        if not has_why:
            result += "\n\n**Why These:**\n- Curated for age-appropriate viewing"
        if 'next step' not in result.lower():
            result += "\n\n**Next Step:** Want more suggestions? Just ask!"

        return result.strip()


# ============ TEST ============

def main():
    """Test the Recommender Agent with delays to avoid rate limits."""
    print("=" * 70)
    print("🧪 TESTING RECOMMENDER AGENT")
    print("=" * 70)

    agent = RecommenderAgent(user_id="test_recommender")

    # Test 1: No preferences known
    print("\n" + "=" * 70)
    print("TEST 1: No preferences (cold start)")
    print("=" * 70)
    result = agent.recommend("What should I watch with my kids?")
    print(f"\n📌 Response:\n{result}")

    # Wait to avoid rate limits
    print("\n⏳ Waiting 30s before next test (rate limit prevention)...")
    time.sleep(30)

    # Test 2: Explicit preference
    print("\n" + "=" * 70)
    print("TEST 2: User reveals preference")
    print("=" * 70)
    result = agent.recommend("I love animated movies with strong female leads")
    print(f"\n📌 Response:\n{result}")

    # Wait to avoid rate limits
    print("\n⏳ Waiting 30s before next test...")
    time.sleep(30)

    # Test 3: Personalized recommendation
    print("\n" + "=" * 70)
    print("TEST 3: Personalized recommendation")
    print("=" * 70)
    result = agent.recommend("Recommend something for me")
    print(f"\n📌 Response:\n{result}")


if __name__ == "__main__":
    main()