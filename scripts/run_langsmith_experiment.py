"""
Run a LangSmith experiment to evaluate the RAG chain against the test dataset.
"""

import os
import sys

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from langsmith import Client
from langsmith.evaluation import evaluate
from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
from rag_chain import RAGChain


def predict(inputs: dict) -> dict:
    """
    Predict function for the experiment.
    This is called for each example in the dataset.
    """
    rag = predict.rag  # Cached RAG chain
    result = rag.classify_movie(
        inputs["title"],
        inputs["overview"],
        inputs.get("genres", "Unknown"),
        inputs.get("rating", "Unknown")
    )
    return {"classification": result}


def exact_match_evaluator(run, example) -> dict:
    """
    Evaluator that checks if the classification matches the expected result.
    Returns a score of 1.0 (pass) or 0.0 (fail).
    """
    expected = example.outputs.get("expected", "").lower()
    actual = run.outputs.get("classification", "").lower()

    # Check if expected substring is in actual
    passed = expected in actual

    return {
        "key": "exact_match",
        "score": 1.0 if passed else 0.0,
        "comment": f"Expected: {expected} | Got: {actual[:150]}"
    }


def run_experiment():
    """Run the LangSmith experiment."""
    if not LANGCHAIN_API_KEY:
        print("❌ LANGCHAIN_API_KEY not set.")
        return

    print("=" * 60)
    print("🧪 RUNNING LANGSMITH EXPERIMENT")
    print("=" * 60)

    client = Client()

    # Initialize RAG chain ONCE
    print("\n🔄 Initializing RAG chain...")
    rag = RAGChain()
    predict.rag = rag  # Cache for predict function

    # Run experiment
    print("\n🧪 Running experiment against 'movie-safety-eval' dataset...")
    print("⏳ This will classify all 56 movies. Please wait...\n")

    results = evaluate(
        predict,
        data="movie-safety-eval",
        evaluators=[exact_match_evaluator],
        experiment_prefix="rag-chain-v1",
        metadata={
            "model": "openai/gpt-oss-20b",
            "chain": "RAG + Groq",
            "version": "1.0",
            "knowledge_base_size": "1,132 entries"
        }
    )

    print("\n" + "=" * 60)
    print("✅ Experiment complete!")
    print("=" * 60)
    print(f"\n🔗 View results at:")
    print(f"   https://smith.langchain.com/projects/{LANGCHAIN_PROJECT}")


if __name__ == "__main__":
    run_experiment()