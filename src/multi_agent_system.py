"""
Multi-Agent System — Unified interface for the Movie Safety Classifier.
Wraps the Orchestrator with session management, statistics, and clean APIs.
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from agents.orchestrator import Orchestrator
from memory import MemoryManager


# ============ DATA CLASSES ============

@dataclass
class QueryRecord:
    """A single query record."""
    query: str
    intent: str
    response: str
    latency_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None


@dataclass
class SessionStats:
    """Statistics for a session."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    intents: Dict[str, int] = field(default_factory=dict)
    total_latency: float = 0.0
    first_query_at: Optional[str] = None
    last_query_at: Optional[str] = None

    def record(self, query_record: QueryRecord):
        """Update stats with a new query record."""
        self.total_queries += 1
        if query_record.error:
            self.failed_queries += 1
        else:
            self.successful_queries += 1

        self.intents[query_record.intent] = self.intents.get(query_record.intent, 0) + 1
        self.total_latency += query_record.latency_seconds

        if self.first_query_at is None:
            self.first_query_at = query_record.timestamp
        self.last_query_at = query_record.timestamp

    @property
    def average_latency(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_latency / self.total_queries

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100


# ============ MULTI-AGENT SYSTEM ============

class MultiAgentSystem:
    """
    Unified interface for the Movie Safety Multi-Agent System.

    Features:
    - Single entry point for all queries
    - Session management (per user)
    - Context-aware routing (uses conversation history)
    - Statistics tracking
    - Memory integration
    - Error handling with graceful fallback
    """

    def __init__(self, user_id: str = "default", model_name: str = None):
        """Initialize the Multi-Agent System."""
        print("=" * 70)
        print("🚀 Initializing Multi-Agent System")
        print("=" * 70)

        self.user_id = user_id
        self.session_start = datetime.now().isoformat()

        # Initialize memory
        print("🧠 Initializing shared memory...")
        self.memory = MemoryManager(user_id=user_id)

        # Initialize orchestrator
        print("🧠 Initializing orchestrator...")
        self.orchestrator = Orchestrator(model_name=model_name, user_id=user_id)

        # Session stats
        self.stats = SessionStats()
        self.query_history: List[QueryRecord] = []

        print(f"✅ Multi-Agent System ready for user: {user_id}")
        print("=" * 70)

    # ============ MAIN API ============

    def ask(self, query: str) -> str:
        """Main API — ask any question to the multi-agent system."""
        if not query or not query.strip():
            return "⚠️ Please enter a question."

        query = query.strip()
        start_time = time.time()

        print(f"\n{'=' * 70}")
        print(f"🎬 Multi-Agent System processing query")
        print(f"👤 User: {self.user_id}")
        print(f"❓ Query: {query}")
        print(f"{'=' * 70}")

        # Get conversation history BEFORE adding the current message
        conversation_history = self.memory.get_conversation_history()

        # Add user message to memory
        self.memory.add_user_message(query)

        # Classify intent ONCE with context
        error = None
        response = ""
        intent = "unknown"

        try:
            # ✅ Pass conversation history for context-aware routing
            intent, method = self.orchestrator.classify_intent(
                query,
                conversation_history=conversation_history
            )
            print(f"🎯 Intent: {intent} (via {method})")

            # Dispatch based on intent
            response = self.orchestrator._dispatch(query, intent)

        except Exception as e:
            error = str(e)
            response = f"❌ System error: {e}"
            print(f"❌ {response}")

        # Calculate latency
        latency = time.time() - start_time

        # Record stats
        record = QueryRecord(
            query=query,
            intent=intent,
            response=response,
            latency_seconds=latency,
            error=error
        )
        self.query_history.append(record)
        self.stats.record(record)

        # Add assistant response to memory
        self.memory.add_assistant_message(response)

        print(f"⏱️  Latency: {latency:.2f}s")
        print(f"✅ Done")

        return response

    def ask_batch(self, queries: List[str], delay_seconds: int = 30) -> List[Dict]:
        """Process multiple queries with automatic rate limit handling."""
        results = []
        total = len(queries)

        print(f"\n🔁 Processing {total} queries in batch...")
        print(f"⏳ Delay between queries: {delay_seconds}s")

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{total}] {query}")
            response = self.ask(query)

            results.append({
                'query': query,
                'response': response,
                'index': i
            })

            if i < total and delay_seconds > 0:
                print(f"⏳ Waiting {delay_seconds}s...")
                time.sleep(delay_seconds)

        return results

    # ============ SESSION MANAGEMENT ============

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            'user_id': self.user_id,
            'session_start': self.session_start,
            'total_queries': self.stats.total_queries,
            'successful_queries': self.stats.successful_queries,
            'failed_queries': self.stats.failed_queries,
            'success_rate': f"{self.stats.success_rate:.1f}%",
            'average_latency_seconds': f"{self.stats.average_latency:.2f}s",
            'intents': self.stats.intents,
            'first_query_at': self.stats.first_query_at,
            'last_query_at': self.stats.last_query_at,
        }

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history."""
        recent = self.query_history[-limit:]
        return [
            {
                'query': r.query,
                'intent': r.intent,
                'latency_seconds': round(r.latency_seconds, 2),
                'timestamp': r.timestamp,
                'error': r.error,
            }
            for r in recent
        ]

    def get_memory(self) -> Dict[str, Any]:
        """Get current memory state."""
        return {
            'short_term_turns': len(self.memory.short_term),
            'long_term_preferences': self.memory.long_term.get_all_preferences(),
            'semantic_facts': [
                {'fact': f.fact, 'category': f.category}
                for f in self.memory.semantic.get_facts()
            ],
        }

    def clear_conversation(self):
        """Clear short-term memory (keeps preferences and facts)."""
        self.memory.clear_short_term()
        print("✅ Conversation history cleared")

    def reset_session(self):
        """Reset all session state (stats, history)."""
        self.stats = SessionStats()
        self.query_history = []
        self.session_start = datetime.now().isoformat()
        print("✅ Session reset")

    # ============ INTERACTIVE MODE ============

    def interactive_mode(self):
        """Run the multi-agent system in interactive mode."""
        print("\n" + "=" * 70)
        print("🎬 MOVIE SAFETY MULTI-AGENT SYSTEM - INTERACTIVE MODE")
        print("=" * 70)
        print(f"👤 User: {self.user_id}")
        print(f"🧠 Agents: Safety, Lookup, Recommender, Comparison")
        print("\n📋 Commands:")
        print("  • Ask any movie question")
        print("  • 'stats' — show session statistics")
        print("  • 'history' — show recent queries")
        print("  • 'memory' — show what I remember about you")
        print("  • 'clear' — clear conversation (keeps preferences)")
        print("  • 'quit' or 'exit' — end session")
        print("=" * 70)

        while True:
            print("\n" + "-" * 70)
            query = input("💭 Your query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                self._print_session_summary()
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'stats':
                self._print_stats()
                continue

            if query.lower() == 'history':
                self._print_history()
                continue

            if query.lower() == 'memory':
                self._print_memory()
                continue

            if query.lower() == 'clear':
                self.clear_conversation()
                continue

            if not query:
                print("⚠️  Please enter a question.")
                continue

            response = self.ask(query)
            print(f"\n📌 Response:\n{response}")

    # ============ PRIVATE HELPERS ============

    def _print_stats(self):
        """Print session stats."""
        stats = self.get_stats()
        print("\n" + "=" * 70)
        print("📊 SESSION STATISTICS")
        print("=" * 70)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 70)

    def _print_history(self):
        """Print query history."""
        history = self.get_history(limit=10)
        print("\n" + "=" * 70)
        print("📜 RECENT QUERIES")
        print("=" * 70)
        for h in history:
            status = "❌" if h['error'] else "✅"
            print(f"  {status} [{h['intent']}] {h['query'][:60]} ({h['latency_seconds']}s)")
        print("=" * 70)

    def _print_memory(self):
        """Print current memory."""
        mem = self.get_memory()
        print("\n" + "=" * 70)
        print("🧠 MEMORY STATE")
        print("=" * 70)
        print(f"  Short-term turns: {mem['short_term_turns']}")
        print(f"\n  Long-term preferences:")
        if mem['long_term_preferences']:
            for k, v in mem['long_term_preferences'].items():
                print(f"    - {k}: {v}")
        else:
            print("    (none)")
        print(f"\n  Semantic facts:")
        if mem['semantic_facts']:
            for f in mem['semantic_facts']:
                print(f"    - [{f['category']}] {f['fact']}")
        else:
            print("    (none)")
        print("=" * 70)

    def _print_session_summary(self):
        """Print session summary at exit."""
        stats = self.get_stats()
        print("\n" + "=" * 70)
        print("📊 SESSION SUMMARY")
        print("=" * 70)
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Success rate: {stats['success_rate']}")
        print(f"  Average latency: {stats['average_latency_seconds']}")
        print(f"  Intents used: {stats['intents']}")
        print("=" * 70)


if __name__ == "__main__":
    system = MultiAgentSystem(user_id="default")
    system.interactive_mode()