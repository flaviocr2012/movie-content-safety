"""
Orchestrator Agent — Routes user queries to the right specialist agent.
Uses LLM-based routing with context awareness for robustness.
"""

import sys
import os
import time
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from typing import Dict, Optional, Tuple, List
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from config import GROQ_API_KEY, GROQ_MODEL
from agents.safety_agent import SafetyAgent
from agents.lookup_agent import LookupAgent
from agents.recommender_agent import RecommenderAgent
from agents.comparison_agent import ComparisonAgent


class Orchestrator:
    """
    Orchestrator Agent — Routes queries to specialized agents using LLM.

    Features:
    - LLM-based intent classification (no regex maintenance)
    - Context-aware routing (uses conversation history)
    - Retry logic for ambiguous queries
    - Automatic fallback to safe defaults
    """

    INTENT_DEFINITIONS = {
        "safety": (
            "Asks whether a movie is safe/appropriate for children. "
            "Examples: 'Is Jurassic Park safe?', 'Can my 5-year-old watch this?', "
            "'Is it appropriate?', 'Is that safe for kids?'"
        ),
        "lookup": (
            "Asks for information about a specific movie: plot, genres, rating, year, "
            "or whether a movie exists in the database. "
            "Examples: 'Tell me about Moana', 'Do you have Batman?', "
            "'What is the plot of Frozen?', 'Which movie is that?', 'How many movies do you have?'"
        ),
        "recommend": (
            "Asks for movie suggestions or recommendations, especially when filtering by "
            "MULTIPLE criteria (genre + theme + age, etc.). "
            "Examples: 'What should I watch?', 'Recommend a comedy', "
            "'I want family movies with animals', 'Do you have action movies with animals?'"
        ),
        "comparison": (
            "Compares two or more movies. "
            "Examples: 'Is Frozen safer than Moana?', 'Compare Toy Story and Shrek', "
            "'Which is better?'"
        )
    }

    def __init__(self, model_name: str = None, user_id: str = "default",
                 preload_agents: bool = True):
        """Initialize the Orchestrator and all specialist agents."""
        if model_name is None:
            model_name = GROQ_MODEL

        print("🧠 Initializing Orchestrator...")

        self.user_id = user_id

        # LLM for intent classification
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.0,
            max_tokens=50,
            groq_api_key=GROQ_API_KEY
        )

        self.intent_chain = self._build_intent_chain()
        self._agents = {}

        if preload_agents:
            print("📦 Preloading specialist agents...")
            self._preload_agents()
            print(f"✅ Preloaded {len(self._agents)} agents!")
        else:
            print("📦 Orchestrator ready! Specialists will be loaded on demand.")

    def _preload_agents(self):
        """Preload all specialist agents at startup."""
        for name in ["safety", "lookup", "recommend", "comparison"]:
            print(f"  → Loading {name} agent...")
            self._get_agent(name)

    def _get_agent(self, name: str):
        """Load specialists on demand."""
        if name not in self._agents:
            print(f"🔄 Loading {name} agent...")
            if name == "safety":
                self._agents[name] = SafetyAgent()
            elif name == "lookup":
                self._agents[name] = LookupAgent()
            elif name == "recommend":
                self._agents[name] = RecommenderAgent(user_id=self.user_id)
            elif name == "comparison":
                self._agents[name] = ComparisonAgent()
        return self._agents[name]

    # ============ INTENT CLASSIFICATION (LLM + CONTEXT) ============

    def _build_intent_chain(self):
        """Build the LLM chain for intent classification with context support."""
        prompt = PromptTemplate(
            template="""You are an intent classifier for a movie safety assistant.

Classify the user's query into EXACTLY ONE of these categories:

{intents}

**Conversation Context:**
{context}

**User Query:**
{query}

**Instructions:**
1. Read the CONTEXT first — it may explain pronouns like "it", "that", "this movie"
2. Read the USER QUERY
3. If the query refers to a previously mentioned movie, use that context
4. Respond with ONLY ONE word: safety, lookup, recommend, or comparison
5. NO explanations, NO punctuation, NO extra words

**Your answer (one word only):**""",
            input_variables=["query", "intents", "context"]
        )

        chain = prompt | self.llm
        return chain

    def _format_context(self, conversation_history: Optional[List[Dict]]) -> str:
        """Format conversation history as a string for the prompt."""
        if not conversation_history:
            return "No prior context."

        recent = conversation_history[-4:]  # Last 4 turns
        context_lines = []
        for msg in recent:
            role = msg.get('role', 'user').capitalize()
            content = msg.get('content', '')[:150]
            if content:
                context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines) if context_lines else "No prior context."

    def _clean_intent(self, raw_intent: str) -> str:
        """Clean the LLM output to extract just the intent name."""
        # Remove all non-letter characters and lowercase
        cleaned = re.sub(r'[^a-z]', '', raw_intent.lower())
        return cleaned

    def classify_intent(self, query: str,
                        conversation_history: Optional[List[Dict]] = None) -> Tuple[str, str]:
        """
        Classify query intent using LLM with context awareness.

        Args:
            query: User's query
            conversation_history: Optional list of {role, content} dicts

        Returns:
            Tuple of (intent, method)
        """
        # Build context from history
        context = self._format_context(conversation_history)

        # Format intents
        intents_text = "\n\n".join([
            f"- **{name}**: {desc}"
            for name, desc in self.INTENT_DEFINITIONS.items()
        ])

        # First attempt
        try:
            response = self.intent_chain.invoke({
                "query": query,
                "intents": intents_text,
                "context": context
            })
            raw = response.content.strip()
            intent = self._clean_intent(raw)

            if intent in self.INTENT_DEFINITIONS:
                return intent, "llm"

            # Try to extract a valid intent from the response
            for valid_intent in self.INTENT_DEFINITIONS:
                if valid_intent in intent:
                    return valid_intent, "llm"

            print(f"⚠️  Invalid intent: '{raw}' — retrying with stricter prompt...")

        except Exception as e:
            print(f"❌ First attempt failed: {e}")

        # Retry with a stricter prompt
        try:
            retry_response = self.intent_chain.invoke({
                "query": query,
                "intents": intents_text,
                "context": context + "\n\n**REMINDER: Answer with ONE word only: safety, lookup, recommend, or comparison**"
            })
            raw_retry = retry_response.content.strip()
            intent_retry = self._clean_intent(raw_retry)

            if intent_retry in self.INTENT_DEFINITIONS:
                return intent_retry, "llm-retry"

            print(f"⚠️  Retry also failed: '{raw_retry}' — using fallback")

        except Exception as e:
            print(f"❌ Retry failed: {e}")

        # Final fallback — context-aware default
        if context != "No prior context.":
            # If there's context, check if previous intent was safety-related
            context_lower = context.lower()
            if any(kw in context_lower for kw in ["safe", "appropriate", "kid", "child"]):
                return "safety", "fallback-context"
            # Default to lookup when we have context (likely follow-up)
            return "lookup", "fallback-context"

        # No context → default to lookup
        return "lookup", "fallback"

    # ============ ROUTING ============

    def route(self, query: str,
              conversation_history: Optional[List[Dict]] = None) -> str:
        """Route the query to the right specialist agent."""
        print(f"\n🧠 Orchestrator received: {query}")

        intent, method = self.classify_intent(query, conversation_history)
        print(f"🎯 Intent: {intent} (via {method})")

        return self._dispatch(query, intent)

    def _dispatch(self, query: str, intent: str) -> str:
        """Dispatch the query to the appropriate agent."""
        try:
            if intent == "safety":
                agent = self._get_agent("safety")
                return agent.classify(movie_title=query)

            elif intent == "lookup":
                agent = self._get_agent("lookup")
                return agent.lookup(query)

            elif intent == "recommend":
                agent = self._get_agent("recommend")
                return agent.recommend(query)

            elif intent == "comparison":
                agent = self._get_agent("comparison")
                return agent.compare(query)

            else:
                return f"⚠️  Unknown intent: {intent}"

        except Exception as e:
            return f"❌ Orchestrator error: {e}"

    # ============ INTERACTIVE MODE ============

    def interactive_mode(self):
        """Run the orchestrator in interactive mode."""
        print("=" * 70)
        print("🧠 MULTI-AGENT ORCHESTRATOR - INTERACTIVE MODE")
        print("=" * 70)
        print(f"\n📊 User ID: {self.user_id}")
        print("\n📋 Available commands:")
        print("  - Ask any question about movies")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'agents' to see loaded agents")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            query = input("💭 Your query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'agents':
                print(f"\n📦 Loaded agents: {list(self._agents.keys())}")
                continue

            if not query:
                print("⚠️  Please enter a query.")
                continue

            result = self.route(query)
            print(f"\n📌 Response:\n{result}")


# ============ TEST ============

def main():
    """Test the Orchestrator with various query types."""
    print("=" * 70)
    print("🧪 TESTING ORCHESTRATOR (LLM-based routing)")
    print("=" * 70)

    orchestrator = Orchestrator(user_id="test_orchestrator", preload_agents=True)

    # Test 1: Basic routing
    print("\n📋 Test 1: Basic routing (no context)")
    queries = [
        "Is Jurassic Park safe for children?",
        "Tell me about Moana",
        "What should I watch with my kids?",
        "Is Frozen safer than Moana?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"❓ Query: {query}")
        print(f"{'=' * 70}")
        result = orchestrator.route(query)
        print(f"\n📌 Response:\n{result}")
        time.sleep(20)

    # Test 2: Context-aware routing
    print("\n\n📋 Test 2: Context-aware routing")
    conversation = [
        {"role": "user", "content": "tell me about Spider-Man: Into the Spider-Verse"},
        {"role": "assistant", "content": "Spider-Man: Into the Spider-Verse (2018) is an animated adventure..."},
    ]

    follow_up = "is it safe for children?"
    print(f"\n{'=' * 70}")
    print(f"📖 Context: {conversation[-1]['content'][:80]}...")
    print(f"❓ Follow-up: {follow_up}")
    print(f"{'=' * 70}")

    result = orchestrator.route(follow_up, conversation_history=conversation)
    print(f"\n📌 Response:\n{result}")


if __name__ == "__main__":
    main()