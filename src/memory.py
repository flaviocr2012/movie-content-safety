"""
Memory system for the Movie Safety Agent.
Supports short-term (conversation), long-term (preferences), and semantic (facts) memory.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import deque

from config import DATA_PATH


# ============ DATA CLASSES ============

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str          # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """A long-term preference for a user."""
    key: str           # e.g., "favorite_genre"
    value: str         # e.g., "animation"
    confidence: float  # 0.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "inferred"  # "explicit" or "inferred"


@dataclass
class UserFact:
    """A semantic fact about the user."""
    fact: str          # e.g., "User has a 5-year-old daughter"
    category: str      # e.g., "family", "preferences", "context"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0


# ============ SHORT-TERM MEMORY ============

class ShortTermMemory:
    """
    Remembers the current conversation (last N turns).
    In-memory only — cleared between sessions.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: deque = deque(maxlen=max_turns * 2)  # user + assistant pairs

    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.turns.append(turn)

    def get_recent_context(self, n: int = 5) -> str:
        """Get the last N turns as a formatted string."""
        recent = list(self.turns)[-n:]
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in chat format for LLM."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]

    def clear(self):
        """Clear the conversation history."""
        self.turns.clear()

    def __len__(self):
        return len(self.turns)


# ============ LONG-TERM MEMORY ============

class LongTermMemory:
    """
    Remembers user preferences across sessions.
    Persisted to a JSON file.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.file_path = os.path.join(DATA_PATH, f"memory_{user_id}.json")
        self.preferences: Dict[str, UserPreference] = {}
        self._load()

    def _load(self):
        """Load preferences from disk."""
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                for key, pref_data in data.get("preferences", {}).items():
                    self.preferences[key] = UserPreference(**pref_data)
        except Exception as e:
            print(f"⚠️ Failed to load memory: {e}")

    def _save(self):
        """Save preferences to disk."""
        os.makedirs(DATA_PATH, exist_ok=True)
        data = {
            "user_id": self.user_id,
            "updated_at": datetime.now().isoformat(),
            "preferences": {
                k: asdict(v) for k, v in self.preferences.items()
            }
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def set_preference(self, key: str, value: str, confidence: float = 1.0,
                       source: str = "explicit"):
        """Set a user preference."""
        self.preferences[key] = UserPreference(
            key=key,
            value=value,
            confidence=confidence,
            source=source
        )
        self._save()

    def get_preference(self, key: str) -> Optional[str]:
        """Get a preference value."""
        pref = self.preferences.get(key)
        return pref.value if pref else None

    def get_all_preferences(self) -> Dict[str, str]:
        """Get all preferences as a simple dict."""
        return {k: v.value for k, v in self.preferences.items()}

    def get_preferences_context(self) -> str:
        """Format preferences for the LLM context."""
        if not self.preferences:
            return "No known preferences."

        lines = ["Known user preferences:"]
        for key, pref in self.preferences.items():
            lines.append(f"- {key.replace('_', ' ')}: {pref.value} (confidence: {pref.confidence:.1f})")
        return "\n".join(lines)

    def clear(self):
        """Clear all preferences."""
        self.preferences.clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


# ============ SEMANTIC MEMORY ============

class SemanticMemory:
    """
    Remembers facts about the user.
    Stored in JSON for simplicity (can be upgraded to vector search).
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.file_path = os.path.join(DATA_PATH, f"semantic_{user_id}.json")
        self.facts: List[UserFact] = []
        self._load()

    def _load(self):
        """Load facts from disk."""
        if not os.path.exists(self.file_path):
            return
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                self.facts = [UserFact(**fact) for fact in data.get("facts", [])]
        except Exception as e:
            print(f"⚠️ Failed to load semantic memory: {e}")

    def _save(self):
        """Save facts to disk."""
        os.makedirs(DATA_PATH, exist_ok=True)
        data = {
            "user_id": self.user_id,
            "updated_at": datetime.now().isoformat(),
            "facts": [asdict(fact) for fact in self.facts]
        }
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_fact(self, fact: str, category: str = "general", confidence: float = 1.0):
        """Add a new fact about the user."""
        # Avoid duplicates
        for existing in self.facts:
            if existing.fact.lower() == fact.lower():
                return

        self.facts.append(UserFact(
            fact=fact,
            category=category,
            confidence=confidence
        ))
        self._save()

    def get_facts(self, category: Optional[str] = None) -> List[UserFact]:
        """Get facts, optionally filtered by category."""
        if category:
            return [f for f in self.facts if f.category == category]
        return self.facts

    def get_facts_context(self) -> str:
        """Format facts for the LLM context."""
        if not self.facts:
            return "No known facts."

        lines = ["Known facts about the user:"]
        for fact in self.facts:
            lines.append(f"- [{fact.category}] {fact.fact}")
        return "\n".join(lines)

    def clear(self):
        """Clear all facts."""
        self.facts.clear()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


# ============ UNIFIED MEMORY MANAGER ============

class MemoryManager:
    """
    Unified interface to all memory types.
    This is what the agent interacts with.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.short_term = ShortTermMemory(max_turns=10)
        self.long_term = LongTermMemory(user_id=user_id)
        self.semantic = SemanticMemory(user_id=user_id)
        print(f"✅ Memory initialized for user: {user_id}")

    # ---- Short-term ----

    def add_user_message(self, content: str, metadata: Dict = None):
        """Add a user message to conversation."""
        self.short_term.add_turn("user", content, metadata)

    def add_assistant_message(self, content: str, metadata: Dict = None):
        """Add an assistant message to conversation."""
        self.short_term.add_turn("assistant", content, metadata)

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.short_term.get_messages()

    # ---- Long-term ----

    def remember_preference(self, key: str, value: str, source: str = "explicit"):
        """Remember a user preference."""
        self.long_term.set_preference(key, value, source=source)

    def recall_preference(self, key: str) -> Optional[str]:
        """Recall a preference."""
        return self.long_term.get_preference(key)

    # ---- Semantic ----

    def remember_fact(self, fact: str, category: str = "general"):
        """Remember a fact."""
        self.semantic.add_fact(fact, category)

    def get_facts(self, category: Optional[str] = None) -> List[UserFact]:
        """Get facts."""
        return self.semantic.get_facts(category)

    # ---- Context for LLM ----

    def build_context(self) -> str:
        """
        Build a complete context string for the agent.
        Combines all memory types.
        """
        parts = []

        # Long-term preferences
        prefs = self.long_term.get_preferences_context()
        if prefs != "No known preferences.":
            parts.append(prefs)

        # Semantic facts
        facts = self.semantic.get_facts_context()
        if facts != "No known facts.":
            parts.append(facts)

        # Recent conversation
        if len(self.short_term) > 0:
            parts.append(f"Recent conversation:\n{self.short_term.get_recent_context(n=3)}")

        return "\n\n".join(parts) if parts else "No prior context."

    # ---- Utilities ----

    def clear_short_term(self):
        """Clear only short-term memory."""
        self.short_term.clear()

    def clear_all(self):
        """Clear everything."""
        self.short_term.clear()
        self.long_term.clear()
        self.semantic.clear()

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "user_id": self.user_id,
            "short_term_turns": len(self.short_term),
            "long_term_preferences": len(self.long_term.preferences),
            "semantic_facts": len(self.semantic.facts),
        }


# ============ TEST ============

def main():
    """Test the memory system."""
    print("=" * 60)
    print("🧠 MEMORY SYSTEM TEST")
    print("=" * 60)

    # Create memory for a user
    memory = MemoryManager(user_id="test_user")

    # Simulate a conversation
    print("\n📝 Simulating conversation...")
    memory.add_user_message("I like animated movies")
    memory.remember_preference("favorite_genre", "animation", source="explicit")
    memory.remember_fact("User has a 5-year-old daughter", category="family")

    memory.add_assistant_message("Great! I'll remember that.")
    memory.add_user_message("Is Finding Nemo safe?")

    # Build context
    print("\n" + "=" * 60)
    print("📋 MEMORY CONTEXT FOR LLM")
    print("=" * 60)
    print(memory.build_context())

    # Show stats
    print("\n" + "=" * 60)
    print("📊 MEMORY STATS")
    print("=" * 60)
    for key, value in memory.get_stats().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("✅ Memory system test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()