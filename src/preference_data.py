"""
Preference Data Generator for DPO Training.
Converts user feedback into preference pairs (chosen vs. rejected).
"""

import json
import os
from typing import List, Dict
from dataclasses import dataclass, asdict

from config import FEEDBACK_PATH, DATA_PATH


@dataclass
class PreferencePair:
    """A single preference pair for DPO training."""
    prompt: str           # The input (movie details)
    chosen: str           # The preferred response
    rejected: str         # The rejected response
    source: str           # "user_feedback" or "synthetic"


class PreferenceDataGenerator:
    """
    Converts user feedback into DPO preference pairs.
    """

    def __init__(self, feedback_path: str = FEEDBACK_PATH):
        self.feedback_path = feedback_path
        self.pairs: List[PreferencePair] = []

    def _format_prompt(self, entry: Dict) -> str:
        """Format the input prompt from movie details."""
        return (
            f"Movie: {entry['movie_title']}\n"
            f"Overview: {entry['movie_overview']}\n"
            f"Genres: {entry['genres']}\n"
            f"Rating: {entry['rating']}\n\n"
            f"Is this movie appropriate for children aged 5-10?"
        )

    def generate_from_feedback(self) -> List[PreferencePair]:
        """Generate preference pairs from user feedback."""
        if not os.path.exists(self.feedback_path):
            print(f"⚠️ No feedback file found at {self.feedback_path}")
            return []

        with open(self.feedback_path, "r") as f:
            feedback = json.load(f)

        pairs = []
        for entry in feedback:
            # Only process "wrong" feedback
            if entry.get("user_feedback") != "wrong":
                continue

            correct_classification = entry.get("correct_classification")
            if not correct_classification:
                continue

            prompt = self._format_prompt(entry)

            # Chosen = correct response (from user)
            chosen = (
                f"Classification: {correct_classification}\n"
                f"Explanation: Based on user feedback, this movie is "
                f"{correct_classification.lower()} for children."
            )

            # Rejected = the system's wrong response
            rejected = entry.get("system_classification", "")

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source="user_feedback"
            ))

        self.pairs = pairs
        print(f"✅ Generated {len(pairs)} preference pairs from user feedback")
        return pairs

    def generate_synthetic_pairs(self) -> List[PreferencePair]:
        """
        Generate synthetic preference pairs for known edge cases.
        Useful when you don't have enough user feedback yet.
        """
        synthetic = [
            # Example 1: Horror should never be safe
            PreferencePair(
                prompt=(
                    "Movie: The Conjuring\n"
                    "Overview: Paranormal investigators help a family terrorized by a dark presence.\n"
                    "Genres: Horror, Mystery, Thriller\n"
                    "Rating: 7.5\n\n"
                    "Is this movie appropriate for children aged 5-10?"
                ),
                chosen=(
                    "Classification: Not safe for children\n"
                    "Explanation: This movie is in the Horror genre and contains "
                    "frightening supernatural imagery, which is not appropriate for children."
                ),
                rejected=(
                    "Classification: Safe for children\n"
                    "Explanation: The movie has a high rating and no explicit content."
                ),
                source="synthetic"
            ),
            # Example 2: Animated should be safe
            PreferencePair(
                prompt=(
                    "Movie: Finding Nemo\n"
                    "Overview: A clownfish sets out on a journey to find his son.\n"
                    "Genres: Animation, Adventure, Comedy\n"
                    "Rating: 8.2\n\n"
                    "Is this movie appropriate for children aged 5-10?"
                ),
                chosen=(
                    "Classification: Safe for children\n"
                    "Explanation: This is an animated, family-friendly film with "
                    "positive messages about family and friendship."
                ),
                rejected=(
                    "Classification: Not safe for children\n"
                    "Explanation: The movie has some intense scenes."
                ),
                source="synthetic"
            ),
            # Example 3: Action with intensity should be unsafe
            PreferencePair(
                prompt=(
                    "Movie: Jurassic Park\n"
                    "Overview: A paleontologist must protect kids after the park's cloned dinosaurs run loose.\n"
                    "Genres: Action, Adventure, Sci-Fi\n"
                    "Rating: 8.2\n\n"
                    "Is this movie appropriate for children aged 5-10?"
                ),
                chosen=(
                    "Classification: Not safe for children\n"
                    "Explanation: Despite being a classic, Jurassic Park contains "
                    "intense dinosaur attacks and frightening imagery that may scare young children."
                ),
                rejected=(
                    "Classification: Safe for children\n"
                    "Explanation: It's a popular adventure movie."
                ),
                source="synthetic"
            ),
        ]

        print(f"✅ Generated {len(synthetic)} synthetic preference pairs")
        return synthetic

    def save_pairs(self, output_path: str = None) -> str:
        """Save pairs to JSONL format for DPO training."""
        if output_path is None:
            output_path = os.path.join(DATA_PATH, "preference_pairs.jsonl")

        all_pairs = self.generate_from_feedback() + self.generate_synthetic_pairs()

        with open(output_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(asdict(pair)) + "\n")

        print(f"💾 Saved {len(all_pairs)} preference pairs to {output_path}")
        return output_path

    def get_stats(self) -> Dict:
        """Get statistics about preference pairs."""
        from_feedback = len([p for p in self.pairs if p.source == "user_feedback"])
        synthetic = len([p for p in self.pairs if p.source == "synthetic"])

        return {
            "total": len(self.pairs),
            "from_feedback": from_feedback,
            "synthetic": synthetic,
        }


def main():
    """Generate preference pairs."""
    print("=" * 60)
    print("🔄 GENERATING PREFERENCE DATA FOR DPO")
    print("=" * 60)

    generator = PreferenceDataGenerator()
    output = generator.save_pairs()

    stats = generator.get_stats()
    print(f"\n📊 Stats:")
    print(f"  Total pairs: {stats['total']}")
    print(f"  From user feedback: {stats['from_feedback']}")
    print(f"  Synthetic: {stats['synthetic']}")


if __name__ == "__main__":
    main()