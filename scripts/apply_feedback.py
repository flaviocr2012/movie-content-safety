"""
Script to apply approved user feedback to the knowledge base.
Run this after reviewing the feedback report.
"""

import json
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from feedback import FeedbackManager


def apply_feedback_to_kb():
    """Apply wrong-feedback corrections to the knowledge base."""
    manager = FeedbackManager()
    wrong_feedback = manager.get_wrong_feedback()

    if not wrong_feedback:
        print("✅ No wrong feedback to apply.")
        return

    # Load existing knowledge base
    kb_path = os.path.join(PROJECT_ROOT, "data", "knowledge_base.csv")
    entries = []

    with open(kb_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        entries = list(reader)

    # Add corrections
    added = 0
    for entry in wrong_feedback:
        title = entry.movie_title
        correct = entry.correct_classification

        if not correct:
            continue

        # Add new entries based on feedback
        new_entries = [
            [f"Is {title} appropriate for children?", correct.replace(" for children", "")],
            [f"Does {title} contain disturbing content?", "No" if "Safe" in correct else "Yes"],
        ]

        for new_entry in new_entries:
            if new_entry not in entries:
                entries.append(new_entry)
                added += 1

    # Save updated knowledge base
    with open(kb_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(entries)

    print(f"✅ Added {added} new entries to knowledge base")
    print(f"📁 Total entries: {len(entries)}")


if __name__ == "__main__":
    apply_feedback_to_kb()