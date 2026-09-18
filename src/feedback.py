"""
User Feedback Loop for Movie Content Safety Classifier.
Collects user feedback, stores it, and enables knowledge base updates.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "data", "user_feedback.json")


@dataclass
class FeedbackEntry:
    """Represents a single feedback entry."""
    movie_title: str
    movie_overview: str
    genres: str
    rating: str
    system_classification: str  # What the system said
    user_feedback: str  # "correct" or "wrong"
    correct_classification: Optional[str] = None  # What the user says is correct
    user_comment: Optional[str] = None  # Optional comment
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class FeedbackManager:
    """
    Manages user feedback for the Movie Safety Classifier.
    """

    def __init__(self, feedback_path: str = FEEDBACK_PATH):
        """
        Initialize the feedback manager.

        Args:
            feedback_path: Path to the feedback JSON file
        """
        self.feedback_path = feedback_path
        self.feedback_entries: List[FeedbackEntry] = []
        self._load_feedback()

    def _load_feedback(self) -> None:
        """Load existing feedback from disk."""
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path, "r") as f:
                    data = json.load(f)
                    self.feedback_entries = [
                        FeedbackEntry(**entry) for entry in data
                    ]
                print(f"✅ Loaded {len(self.feedback_entries)} feedback entries")
            except Exception as e:
                print(f"⚠️ Could not load feedback: {e}")
                self.feedback_entries = []
        else:
            print("📝 No existing feedback found. Starting fresh.")
            self.feedback_entries = []

    def _save_feedback(self) -> None:
        """Save feedback to disk."""
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        with open(self.feedback_path, "w") as f:
            json.dump(
                [asdict(entry) for entry in self.feedback_entries],
                f,
                indent=2
            )
        print(f"💾 Saved {len(self.feedback_entries)} feedback entries")

    def add_feedback(
            self,
            movie_title: str,
            movie_overview: str,
            genres: str,
            rating: str,
            system_classification: str,
            user_feedback: str,
            correct_classification: Optional[str] = None,
            user_comment: Optional[str] = None
    ) -> FeedbackEntry:
        """
        Add a new feedback entry.

        Args:
            movie_title: Movie title
            movie_overview: Movie overview
            genres: Movie genres
            rating: Movie rating
            system_classification: What the system said
            user_feedback: "correct" or "wrong"
            correct_classification: What the user says is correct (if wrong)
            user_comment: Optional comment from user

        Returns:
            The created FeedbackEntry
        """
        entry = FeedbackEntry(
            movie_title=movie_title,
            movie_overview=movie_overview,
            genres=genres,
            rating=rating,
            system_classification=system_classification,
            user_feedback=user_feedback,
            correct_classification=correct_classification,
            user_comment=user_comment
        )
        self.feedback_entries.append(entry)
        self._save_feedback()
        return entry

    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        total = len(self.feedback_entries)
        correct = sum(1 for e in self.feedback_entries if e.user_feedback == "correct")
        wrong = total - correct

        return {
            "total": total,
            "correct": correct,
            "wrong": wrong,
            "accuracy": (correct / total * 100) if total > 0 else 0
        }

    def get_wrong_feedback(self) -> List[FeedbackEntry]:
        """Return only feedback where the user said it was wrong."""
        return [e for e in self.feedback_entries if e.user_feedback == "wrong"]

    def generate_report(self) -> str:
        """Generate a human-readable feedback report."""
        stats = self.get_stats()
        report = "=" * 70 + "\n"
        report += "📊 USER FEEDBACK REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 70 + "\n\n"

        report += f"Total Feedback: {stats['total']}\n"
        report += f"✅ Correct: {stats['correct']}\n"
        report += f"❌ Wrong: {stats['wrong']}\n"
        report += f"📈 Accuracy: {stats['accuracy']:.1f}%\n\n"

        if stats['wrong'] > 0:
            report += "❌ WRONG CLASSIFICATIONS:\n"
            for entry in self.get_wrong_feedback():
                report += f"\n  🎬 {entry.movie_title}\n"
                report += f"     System said: {entry.system_classification[:100]}...\n"
                report += f"     User says: {entry.correct_classification}\n"
                if entry.user_comment:
                    report += f"     Comment: {entry.user_comment}\n"

        return report


def main():
    """Test the feedback manager."""
    print("=" * 60)
    print("🎬 USER FEEDBACK MANAGER - TEST")
    print("=" * 60)

    manager = FeedbackManager()

    # Simulate adding feedback
    manager.add_feedback(
        movie_title="Jurassic Park",
        movie_overview="A paleontologist must protect kids after the park's cloned dinosaurs run loose.",
        genres="Action, Adventure, Sci-Fi",
        rating="8.2",
        system_classification="Safe for children",
        user_feedback="wrong",
        correct_classification="Not safe for children",
        user_comment="Too scary for young children"
    )

    manager.add_feedback(
        movie_title="The Lion King",
        movie_overview="A young lion prince flees his kingdom after the murder of his father.",
        genres="Animation, Adventure, Drama",
        rating="8.5",
        system_classification="Safe for children",
        user_feedback="correct",
        user_comment="Great family movie"
    )

    # Print report
    print("\n" + manager.generate_report())


if __name__ == "__main__":
    main()