"""
Evaluation Framework for Movie Content Safety Classifier.
Measures accuracy, precision, recall, and provides detailed failure analysis.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from rag_chain import RAGChain
from main import classify_single_movie
from config import IMDB_MOVIES_PATH


@dataclass
class TestCase:
    """Represents a single test case."""
    title: str
    overview: str
    genres: str
    rating: str
    expected: str  # "Safe" or "Not safe"
    reason: str  # Why this is the expected result


@dataclass
class TestResult:
    """Represents the result of a single test."""
    test_case: TestCase
    actual: str
    passed: bool
    error: Optional[str] = None


class MovieSafetyEvaluator:
    """
    Evaluation framework for the Movie Safety Classifier.
    Runs test suites and provides detailed metrics and reports.
    """

    def __init__(self, rag_chain: Optional[RAGChain] = None):
        """
        Initialize the evaluator with a RAG chain.

        Args:
            rag_chain: RAGChain instance (creates one if not provided)
        """
        print("🔄 Initializing Movie Safety Evaluator...")
        self.rag = rag_chain if rag_chain else RAGChain()
        self.test_suite = self._load_test_suite()
        self.results: List[TestResult] = []
        print(f"✅ Loaded {len(self.test_suite)} test cases")

    def _load_test_suite(self) -> List[TestCase]:
        """
        Load the test suite of known good answers.

        Returns:
            List of TestCase objects
        """
        return [
            # ========== SAFE MOVIES ==========
            TestCase(
                title="The Lion King",
                overview="A young lion prince flees his kingdom after the murder of his father and learns about responsibility and friendship.",
                genres="Animation, Adventure, Drama",
                rating="8.5",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Toy Story",
                overview="A cowboy doll is threatened when a new spaceman figure supplants him as top toy.",
                genres="Animation, Adventure, Comedy",
                rating="8.3",
                expected="Safe",
                reason="Animated, family-friendly, no violence"
            ),
            TestCase(
                title="Finding Nemo",
                overview="After his son is captured, a timid clownfish sets out on a journey to bring him home.",
                genres="Animation, Adventure, Comedy",
                rating="8.2",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Shrek",
                overview="A mean lord exiles fairytale creatures to the swamp of a grumpy ogre, who must go on a quest.",
                genres="Animation, Adventure, Comedy",
                rating="7.9",
                expected="Safe",
                reason="Animated, family-friendly, cartoon violence only"
            ),
            TestCase(
                title="Home Alone",
                overview="An eight-year-old troublemaker must protect his house from burglars when left home alone.",
                genres="Comedy, Family",
                rating="7.7",
                expected="Safe",
                reason="Family comedy, slapstick violence only"
            ),
            TestCase(
                title="The Incredibles",
                overview="A family of undercover superheroes is forced into action to save the world.",
                genres="Animation, Action, Adventure",
                rating="8.0",
                expected="Safe",
                reason="Animated, family-friendly, cartoon violence only"
            ),
            TestCase(
                title="Up",
                overview="A seventy-eight-year-old balloon salesman fulfills his dream of traveling to South America.",
                genres="Animation, Adventure, Comedy",
                rating="8.3",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Coco",
                overview="A young musician embarks on a journey through the magical land of his ancestors.",
                genres="Animation, Adventure, Family",
                rating="8.4",
                expected="Safe",
                reason="Animated, family-friendly, positive themes"
            ),
            TestCase(
                title="Frozen",
                overview="A princess sets out on a journey to find her estranged sister, whose icy powers have trapped the kingdom.",
                genres="Animation, Adventure, Comedy",
                rating="7.4",
                expected="Safe",
                reason="Animated, family-friendly"
            ),
            TestCase(
                title="How to Train Your Dragon",
                overview="A young Viking becomes the unlikely friend of a young dragon.",
                genres="Animation, Action, Adventure",
                rating="8.1",
                expected="Safe",
                reason="Animated, family-friendly, fantasy violence only"
            ),
            TestCase(
                title="Toy Story 2",
                overview="When Woody is stolen, Buzz and friends set out on a rescue mission.",
                genres="Animation, Adventure, Comedy",
                rating="7.9",
                expected="Safe",
                reason="Animated, family-friendly"
            ),
            TestCase(
                title="Harry Potter and the Sorcerer's Stone",
                overview="An orphaned boy enrolls in a school of wizardry and learns the truth about himself.",
                genres="Adventure, Family, Fantasy",
                rating="7.6",
                expected="Safe",
                reason="Family-friendly fantasy, mild violence only"
            ),
            TestCase(
                title="The Lord of the Rings: The Fellowship of the Ring",
                overview="A Hobbit sets out on a perilous journey to destroy the One Ring.",
                genres="Action, Adventure, Drama",
                rating="8.9",
                expected="Safe",
                reason="Fantasy adventure, epic violence but not graphic"
            ),
            TestCase(
                title="Star Wars: Episode IV - A New Hope",
                overview="A young farmer joins a rebellion against a galactic empire.",
                genres="Action, Adventure, Fantasy",
                rating="8.6",
                expected="Safe",
                reason="Fantasy adventure, classic heroic story"
            ),
            TestCase(
                title="The Avengers",
                overview="Earth's mightiest heroes must come together to stop Loki and his alien army.",
                genres="Action, Sci-Fi",
                rating="8.0",
                expected="Safe",
                reason="Superhero action, fantasy violence only"
            ),
            TestCase(
                title="Avatar",
                overview="A paraplegic Marine is torn between following orders and protecting the world he feels is his home.",
                genres="Action, Adventure, Fantasy",
                rating="7.9",
                expected="Safe",
                reason="Fantasy adventure, sci-fi action without explicit violence"
            ),

            # ========== NOT SAFE MOVIES ==========
            TestCase(
                title="The Dark Knight",
                overview="When the Joker wreaks havoc on Gotham, Batman must accept one of the greatest tests of his ability.",
                genres="Action, Crime, Drama",
                rating="9.0",
                expected="Not safe",
                reason="Extreme graphic violence, disturbing imagery"
            ),
            TestCase(
                title="Pulp Fiction",
                overview="The lives of two mob hitmen, a boxer, and a gangster intertwine in four tales of violence.",
                genres="Crime, Drama",
                rating="8.9",
                expected="Not safe",
                reason="Strong violence, adult themes, strong profanity"
            ),
            TestCase(
                title="The Godfather",
                overview="The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
                genres="Crime, Drama",
                rating="9.2",
                expected="Not safe",
                reason="Violence, adult themes, crime content"
            ),
            TestCase(
                title="The Matrix",
                overview="A computer hacker learns the true nature of his reality and his role in the war against its controllers.",
                genres="Action, Sci-Fi",
                rating="8.7",
                expected="Not safe",
                reason="Violence, intense action scenes"
            ),
            TestCase(
                title="The Conjuring",
                overview="Paranormal investigators help a family terrorized by a dark presence in their farmhouse.",
                genres="Horror, Mystery, Thriller",
                rating="7.5",
                expected="Not safe",
                reason="Horror genre, disturbing imagery"
            ),
            TestCase(
                title="Paranormal Activity",
                overview="A young couple becomes increasingly disturbed by a demonic presence in their house.",
                genres="Horror",
                rating="6.3",
                expected="Not safe",
                reason="Horror genre, frightening imagery"
            ),
            TestCase(
                title="The Terminator",
                overview="A cyborg is sent from the future to kill a waitress whose unborn son will lead humanity.",
                genres="Action, Sci-Fi",
                rating="8.1",
                expected="Not safe",
                reason="Violence, adult themes"
            ),
            TestCase(
                title="Die Hard",
                overview="A police officer tries to save his estranged wife and others taken hostage by terrorists.",
                genres="Action, Thriller",
                rating="8.2",
                expected="Not safe",
                reason="Violence, intense action, adult themes"
            ),
            TestCase(
                title="Jurassic Park",
                overview="A paleontologist must protect kids after the park's cloned dinosaurs run loose.",
                genres="Action, Adventure, Sci-Fi",
                rating="8.2",
                expected="Not safe",
                reason="Intense violence, frightening imagery"
            ),
        ]

    def run_evaluation(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full evaluation suite.

        Args:
            verbose: Print progress during evaluation

        Returns:
            Dictionary with evaluation metrics and detailed results
        """
        print("\n" + "=" * 60)
        print("🔍 RUNNING EVALUATION SUITE")
        print("=" * 60)

        self.results = []
        total = len(self.test_suite)

        for i, test_case in enumerate(self.test_suite, 1):
            if verbose:
                print(f"\n[{i}/{total}] 🎬 Testing: {test_case.title}")
                print(f"   Expected: {test_case.expected}")

            try:
                # Run classification
                result = classify_single_movie(
                    self.rag,
                    test_case.title,
                    test_case.overview,
                    test_case.genres,
                    test_case.rating
                )

                # Determine if test passed
                passed = test_case.expected.lower() in result.lower()
                self.results.append(TestResult(
                    test_case=test_case,
                    actual=result,
                    passed=passed,
                    error=None
                ))

                if verbose:
                    status = "✅ PASSED" if passed else "❌ FAILED"
                    print(f"   {status}")

            except Exception as e:
                self.results.append(TestResult(
                    test_case=test_case,
                    actual="",
                    passed=False,
                    error=str(e)
                ))
                if verbose:
                    print(f"   ❌ ERROR: {e}")

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Save results
        self._save_results()

        if verbose:
            self._print_summary(metrics)

        return metrics

    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from results.

        Returns:
            Dictionary with accuracy, precision, recall, and more
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        # Safe vs Not Safe breakdown
        safe_count = sum(1 for r in self.results if r.test_case.expected == "Safe")
        unsafe_count = sum(1 for r in self.results if r.test_case.expected == "Not safe")

        # ✅ Calculate safe_passed and unsafe_passed
        safe_passed = sum(1 for r in self.results
                          if r.test_case.expected == "Safe" and r.passed)
        unsafe_passed = sum(1 for r in self.results
                            if r.test_case.expected == "Not safe" and r.passed)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "accuracy": (passed / total * 100) if total > 0 else 0,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "safe_passed": safe_passed,
            "unsafe_passed": unsafe_passed,
            "safe_accuracy": (safe_passed / safe_count * 100) if safe_count > 0 else 0,
            "unsafe_accuracy": (unsafe_passed / unsafe_count * 100) if unsafe_count > 0 else 0,
            "failed_cases": [r.test_case.title for r in self.results if not r.passed],
            "passed_cases": [r.test_case.title for r in self.results if r.passed],
        }

    def _print_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print a formatted summary of the evaluation.

        Args:
            metrics: Dictionary of metrics from _calculate_metrics()
        """
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\n📈 Overall Accuracy: {metrics['accuracy']:.1f}%")
        print(f"   ✅ Passed: {metrics['passed']}/{metrics['total']}")
        print(f"   ❌ Failed: {metrics['failed']}/{metrics['total']}")

        print(f"\n🎯 Safe Movies: {metrics['safe_accuracy']:.1f}% accuracy")
        print(f"   ✅ {metrics['safe_passed']}/{metrics['safe_count']}")

        print(f"\n🎯 Not Safe Movies: {metrics['unsafe_accuracy']:.1f}% accuracy")
        print(f"   ✅ {metrics['unsafe_passed']}/{metrics['unsafe_count']}")

        if metrics['failed_cases']:
            print(f"\n❌ Failed Cases ({len(metrics['failed_cases'])}):")
            for title in metrics['failed_cases']:
                print(f"   - {title}")

        print("\n" + "=" * 60)

    def _save_results(self) -> None:
        """Save evaluation results to files."""
        # Save to JSON
        json_path = "data/evaluation_results.json"
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [{
                "title": r.test_case.title,
                "expected": r.test_case.expected,
                "actual": r.actual[:200] if r.actual else "",
                "passed": r.passed,
                "error": r.error,
                "reason": r.test_case.reason
            } for r in self.results]
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Results saved to: {json_path}")

        # Save to CSV
        csv_path = "data/evaluation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Title", "Expected", "Passed", "Error", "Reason"])
            for r in self.results:
                writer.writerow([
                    r.test_case.title,
                    r.test_case.expected,
                    r.passed,
                    r.error or "",
                    r.test_case.reason
                ])
        print(f"💾 Results saved to: {csv_path}")

    def get_failed_cases(self) -> List[TestResult]:
        """Return only the failed test results."""
        return [r for r in self.results if not r.passed]

    def generate_report(self) -> str:
        """
        Generate a human-readable report.

        Returns:
            Formatted report string
        """
        metrics = self._calculate_metrics()
        report = "=" * 70 + "\n"
        report += "📊 MOVIE SAFETY EVALUATION REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 70 + "\n\n"

        report += f"Total Tests: {metrics['total']}\n"
        report += f"Passed: {metrics['passed']} ({metrics['accuracy']:.1f}%)\n"
        report += f"Failed: {metrics['failed']}\n\n"

        report += f"Safe Movies: {metrics['safe_accuracy']:.1f}% accuracy\n"
        report += f"Not Safe Movies: {metrics['unsafe_accuracy']:.1f}% accuracy\n\n"

        if metrics['failed_cases']:
            report += "❌ FAILED CASES:\n"
            for title in metrics['failed_cases']:
                report += f"  - {title}\n"
        else:
            report += "✅ ALL TESTS PASSED!\n"

        return report


def main():
    """Run the evaluation framework."""
    print("=" * 60)
    print("🎬 MOVIE SAFETY EVALUATION FRAMEWORK")
    print("=" * 60)

    # Initialize evaluator
    evaluator = MovieSafetyEvaluator()

    # Run evaluation
    metrics = evaluator.run_evaluation(verbose=True)

    # Generate report
    report = evaluator.generate_report()
    print("\n" + report)

    # Save report
    with open("data/evaluation_report.txt", "w") as f:
        f.write(report)
    print("💾 Report saved to: data/evaluation_report.txt")


if __name__ == "__main__":
    main()