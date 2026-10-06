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

    SAFE_GENRES = {"animation", "family", "adventure", "comedy", "fantasy"}
    UNSAFE_GENRES = {"horror", "crime", "thriller"}

    def __init__(self, model_name: str = None, user_id: str = "default"):
        """Initialize the Recommender Agent."""
        if model_name is None:
            model_name = GROQ_MODEL

        print("💡 Initializing Recommender Agent...")

        self.memory = MemoryManager(user_id=user_id)
        self.user_id = user_id

        self.rag = RAGChain(model_name)

        # ✅ OPTIMIZATION: Lower max_tokens from 800 → 400 for faster responses
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.5,
            max_tokens=400,  # Reduced from 800
            groq_api_key=GROQ_API_KEY
        )

        self.movies = load_movies_from_csv()
        print(f"📚 Loaded {len(self.movies)} movies")

        self.tools = self._create_tools()
        self.agent = self._build_agent()

        print("✅ Recommender Agent ready!")

    def _is_movie_safe(self, movie: Dict) -> bool:
        """Check if a movie is likely safe based on genres."""
        genres = movie.get('genres', '').lower()
        if any(ug in genres for ug in self.UNSAFE_GENRES):
            return False
        return True

    def _find_safe_movies(self, genre: str = None, keyword: str = None,
                          min_rating: float = 0.0, limit: int = 15) -> List[Dict]:
        """
        Unified movie search — filters by genre AND/OR keyword.
        This replaces the separate find_safe_movies_by_genre and
        find_safe_movies_by_keyword tools.
        """
        matches = []
        for m in self.movies:
            if not self._is_movie_safe(m):
                continue

            # Genre filter
            if genre:
                genre_lower = genre.lower().strip()
                if genre_lower not in m.get('genres', '').lower():
                    continue

            # Keyword filter
            if keyword:
                keyword_lower = keyword.lower().strip()
                title = m.get('title', '').lower()
                overview = m.get('overview', '').lower()
                if keyword_lower not in title and keyword_lower not in overview:
                    continue

            # Rating filter
            try:
                rating = float(m.get('rating', 0) or 0)
                if rating < min_rating:
                    continue
            except (ValueError, TypeError):
                pass

            matches.append(m)

        # Sort by rating (highest first)
        matches.sort(key=lambda m: float(m.get('rating', 0) or 0), reverse=True)
        return matches[:limit]

    def _create_tools(self):
        """
        Create recommendation tools.

        ✅ OPTIMIZATION: Reduced from 6 tools to 4.
        Merged find_safe_movies_by_genre + find_safe_movies_by_keyword into
        a single find_safe_movies tool.
        """

        @tool
        def get_user_preferences() -> str:
            """
            Get all known user preferences and facts from memory.
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
        def find_safe_movies(genre: str = "", keyword: str = "", min_rating: float = 0.0) -> str:
            """
            Find safe movies matching optional criteria (genre, keyword, min_rating).
            Both genre and keyword are optional — provide either, both, or neither.

            Args:
                genre: Optional genre filter (e.g., "Animation", "Comedy")
                keyword: Optional keyword filter (e.g., "animals", "space", "friendship")
                min_rating: Optional minimum rating (0-10)

            Examples:
                - find_safe_movies(genre="Animation")
                - find_safe_movies(keyword="animals")
                - find_safe_movies(genre="Animation", keyword="animals")
                - find_safe_movies(min_rating=8.0)
            """
            genre_param = genre if genre else None
            keyword_param = keyword if keyword else None

            matches = self._find_safe_movies(
                genre=genre_param,
                keyword=keyword_param,
                min_rating=min_rating,
                limit=15
            )

            if not matches:
                # Build a helpful message
                criteria = []
                if genre:
                    criteria.append(f"genre='{genre}'")
                if keyword:
                    criteria.append(f"keyword='{keyword}'")
                if min_rating > 0:
                    criteria.append(f"min_rating={min_rating}")
                criteria_str = ", ".join(criteria) if criteria else "no criteria"
                return f"No safe movies found with {criteria_str}."

            lines = [f"Safe movies ({len(matches)} found):"]
            for m in matches:
                lines.append(
                    f"- {m['title']} ({m.get('year', 'N/A')}) — "
                    f"{m.get('rating', 'N/A')}/10 [{m.get('genres', 'N/A')}]"
                )
            return "\n".join(lines)

        @tool
        def find_top_rated_safe_movies(count: int = 10) -> str:
            """
            Get the top-rated safe movies overall.
            Use this when the user has no specific genre or theme.
            """
            matches = self._find_safe_movies(min_rating=0.0, limit=count)
            lines = [f"Top {count} safe movies overall:"]
            for m in matches:
                lines.append(
                    f"- {m['title']} ({m.get('year', 'N/A')}) — "
                    f"{m.get('rating', 'N/A')}/10 [{m.get('genres', 'N/A')}]"
                )
            return "\n".join(lines)

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
            find_safe_movies,
            find_top_rated_safe_movies,
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
- find_safe_movies(genre, keyword, min_rating): unified search by criteria
- find_top_rated_safe_movies(count): best safe movies overall
- remember_preference(key, value): save a preference

PROCESS (FOLLOW EXACTLY):

STEP 1: ALWAYS call get_user_preferences() first.

STEP 2: Based on the user's request AND preferences, call the right search tool:
        - If user mentions a genre → find_safe_movies(genre="...")
        - If user mentions a theme → find_safe_movies(keyword="...")
        - If user mentions BOTH → find_safe_movies(genre="...", keyword="...")
        - If user has no specific request → find_top_rated_safe_movies(10)

STEP 3: You MUST receive tool results BEFORE writing the final answer.

STEP 4: Pick the TOP 3 movies from the tool results.

CRITICAL: DO NOT write a response until you have called at least 2 tools.

RULES:
- ALWAYS personalize based on known preferences
- ALWAYS recommend at least 3 movies from REAL tool results
- NEVER recommend Horror, Crime, or Thriller genres
- Be enthusiastic but concise
- If the user reveals a new preference, save it
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
        """Generate personalized movie recommendations with automatic retry."""
        print(f"\n💡 Recommender Agent: {query}")
        self.memory.add_user_message(query)

        for attempt in range(max_retries):
            try:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]}
                )
                raw = self._extract_response(response)
                print(f"🔍 Raw output length: {len(raw)} chars")

                formatted = self._format_response(raw)
                self.memory.add_assistant_message(formatted)
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

        return "❌ Unable to generate recommendations after multiple attempts."

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
        """Light cleanup of the response format."""
        if not raw:
            return (
                "**Top Recommendations:**\n"
                "1. No recommendations available\n\n"
                "**Why These:**\n- Please try again\n\n"
                "**Next Step:** Ask for a specific genre or theme."
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
    """Test the Recommender Agent."""
    print("=" * 70)
    print("🧪 TESTING RECOMMENDER AGENT")
    print("=" * 70)

    agent = RecommenderAgent(user_id="test_recommender")

    test_queries = [
        "What should I watch with my kids?",
        "I love animated movies with strong female leads",
        "Do you have action movies with animals?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"❓ Query: {query}")
        print(f"{'=' * 70}")

        result = agent.recommend(query)
        print(f"\n📌 Response:\n{result}")

        # Rate limit prevention
        print("\n⏳ Waiting 20s before next test...")
        time.sleep(20)


if __name__ == "__main__":
    main()