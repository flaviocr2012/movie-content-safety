"""
Create a LangSmith dataset from the eval test suite.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from langsmith import Client
from config import LANGCHAIN_API_KEY
from evals import MovieSafetyEvaluator


def create_dataset():
    """Create a LangSmith dataset from the eval test suite."""
    if not LANGCHAIN_API_KEY:
        print("❌ LANGCHAIN_API_KEY not set. Cannot create dataset.")
        return

    client = Client()
    evaluator = MovieSafetyEvaluator()

    dataset_name = "movie-safety-eval"

    # Check if dataset exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        print(f"⚠️ Dataset '{dataset_name}' already exists. Deleting...")
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Test suite for Movie Content Safety Classifier (55 movies)"
    )

    # Add examples
    for test_case in evaluator.test_suite:
        client.create_example(
            inputs={
                "title": test_case.title,
                "overview": test_case.overview,
                "genres": test_case.genres,
                "rating": test_case.rating
            },
            outputs={
                "expected": test_case.expected,
                "reason": test_case.reason
            },
            dataset_id=dataset.id
        )

    print(f"✅ Created dataset '{dataset_name}' with {len(evaluator.test_suite)} examples")


if __name__ == "__main__":
    create_dataset()