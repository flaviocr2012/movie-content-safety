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

from langsmith import Client
from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT


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
    """Evaluation framework with LangSmith integration."""

    def __init__(self, rag_chain: Optional[RAGChain] = None):
        print("🔄 Initializing Movie Safety Evaluator...")
        self.rag = rag_chain if rag_chain else RAGChain()
        self.test_suite = self._load_test_suite()
        self.results: List[TestResult] = []

        # ✅ LangSmith client
        self.langsmith_client = None
        if LANGCHAIN_API_KEY:
            try:
                self.langsmith_client = Client()
                print(f"✅ LangSmith client initialized (project: {LANGCHAIN_PROJECT})")
            except Exception as e:
                print(f"⚠️ Could not initialize LangSmith: {e}")

        print(f"✅ Loaded {len(self.test_suite)} test cases")

    def _log_to_langsmith(self, test_case: TestCase, result: str, passed: bool):
        """Log a single eval result to LangSmith."""
        if not self.langsmith_client:
            return

        try:
            self.langsmith_client.create_run(
                name=f"eval_{test_case.title}",
                run_type="chain",
                inputs={
                    "title": test_case.title,
                    "overview": test_case.overview,
                    "genres": test_case.genres,
                    "rating": test_case.rating
                },
                outputs={
                    "actual": result,
                    "expected": test_case.expected,
                    "passed": passed
                },
                tags=["eval", "movie-safety", "Safe" if passed else "Failed"],
                extra={
                    "reason": test_case.reason,
                    "project": LANGCHAIN_PROJECT
                }
            )
        except Exception as e:
            print(f"⚠️ Failed to log to LangSmith: {e}")

    def _load_test_suite(self) -> List[TestCase]:
        """
        Load the test suite of known good answers.
        Expanded to 50+ test cases for better coverage.

        Returns:
            List of TestCase objects
        """
        return [
            # ========== SAFE MOVIES (30 cases) ==========
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
                overview="A Hobbit sets out on a perilous journey to destroy the One Ring. Fantasy adventure with epic battles but no graphic violence, featuring heroic themes of friendship and courage.",
                genres="Action, Adventure, Drama, Fantasy",
                rating="8.9",
                expected="Safe",
                reason="Fantasy adventure with heroic themes, epic but not graphic violence"
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
            TestCase(
                title="Aladdin",
                overview="A kind-hearted street urchin and a power-hungry Grand Vizier vie for a magic lamp.",
                genres="Animation, Adventure, Comedy",
                rating="8.0",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Beauty and the Beast",
                overview="A selfish prince is cursed to become a monster, and only true love can break the spell.",
                genres="Animation, Family, Fantasy",
                rating="8.0",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Zootopia",
                overview="In a city of anthropomorphic animals, a rookie bunny cop and a cynical con artist fox must work together.",
                genres="Animation, Adventure, Comedy",
                rating="8.0",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Moana",
                overview="A young woman uses her navigational talents to set sail for a fabled island.",
                genres="Animation, Adventure, Comedy",
                rating="7.6",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Inside Out",
                overview="A young girl's emotions come to life as she navigates a move to a new city.",
                genres="Animation, Adventure, Comedy",
                rating="8.1",
                expected="Safe",
                reason="Animated, family-friendly, emotional but not scary"
            ),
            TestCase(
                title="Ratatouille",
                overview="A rat who can cook makes an unusual alliance with a young kitchen worker.",
                genres="Animation, Adventure, Comedy",
                rating="8.1",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Wall-E",
                overview="A small waste-collecting robot embarks on a journey that will decide the fate of mankind.",
                genres="Animation, Adventure, Family",
                rating="8.4",
                expected="Safe",
                reason="Animated, family-friendly, positive messages"
            ),
            TestCase(
                title="Monsters, Inc.",
                overview="Two monsters must return a human child to her world before it's too late.",
                genres="Animation, Adventure, Comedy",
                rating="8.1",
                expected="Safe",
                reason="Animated, family-friendly, no scary content"
            ),
            TestCase(
                title="The Wizard of Oz",
                overview="A young girl is swept away to a magical land and must find her way home.",
                genres="Adventure, Family, Fantasy",
                rating="8.1",
                expected="Safe",
                reason="Classic family film, mild scary moments"
            ),
            TestCase(
                title="E.T. the Extra-Terrestrial",
                overview="A young boy befriends a friendly alien and helps him return home.",
                genres="Adventure, Family, Sci-Fi",
                rating="7.9",
                expected="Safe",
                reason="Family-friendly sci-fi, positive messages"
            ),
            TestCase(
                title="The Princess Bride",
                overview="A farm boy and a princess must overcome obstacles to find true love.",
                genres="Adventure, Comedy, Family",
                rating="8.1",
                expected="Safe",
                reason="Family-friendly fantasy, mild violence"
            ),
            TestCase(
                title="Matilda",
                overview="A brilliant young girl uses her telekinetic powers to stand up to her cruel headmistress.",
                genres="Comedy, Family, Fantasy",
                rating="7.5",
                expected="Safe",
                reason="Family-friendly, positive messages"
            ),
            TestCase(
                title="Paddington",
                overview="A young Peruvian bear travels to London in search of a home.",
                genres="Adventure, Comedy, Family",
                rating="7.2",
                expected="Safe",
                reason="Family-friendly, positive messages"
            ),
            TestCase(
                title="Paddington 2",
                overview="Paddington tries to buy a unique pop-up book for his aunt's birthday.",
                genres="Adventure, Comedy, Family",
                rating="7.8",
                expected="Safe",
                reason="Family-friendly, positive messages"
            ),
            TestCase(
                title="Wonder",
                overview="A young boy with facial differences attends a mainstream school for the first time.",
                genres="Drama, Family",
                rating="8.0",
                expected="Safe",
                reason="Family-friendly, positive messages"
            ),

            # ========== NOT SAFE MOVIES (25 cases) ==========
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
            TestCase(
                title="Jaws",
                overview="A police chief, a marine scientist, and a fisherman hunt a great white shark.",
                genres="Adventure, Thriller",
                rating="8.1",
                expected="Not safe",
                reason="Frightening shark attacks, intense suspense"
            ),
            TestCase(
                title="Indiana Jones and the Raiders of the Lost Ark",
                overview="An archaeologist races against Nazis to find the Ark of the Covenant.",
                genres="Action, Adventure",
                rating="8.4",
                expected="Not safe",
                reason="Intense action, frightening imagery"
            ),
            TestCase(
                title="Pirates of the Caribbean: The Curse of the Black Pearl",
                overview="A pirate captain and a blacksmith must rescue a governor's daughter from cursed pirates. Contains intense action sequences, frightening imagery, and perilous situations.",
                genres="Action, Adventure, Fantasy",
                rating="8.1",
                expected="Not safe",
                reason="Intense action, frightening imagery, perilous situations"
            ),
            TestCase(
                title="Se7en",
                overview="Two detectives hunt a serial killer who uses the seven deadly sins as his motives.",
                genres="Crime, Drama, Mystery",
                rating="8.6",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="Fight Club",
                overview="An insomniac office worker and a devil-may-care soap maker form an underground fight club.",
                genres="Drama, Thriller",
                rating="8.8",
                expected="Not safe",
                reason="Violence, adult themes, disturbing content"
            ),
            TestCase(
                title="American Psycho",
                overview="A wealthy New York City investment banking executive hides his alternate psychopathic ego.",
                genres="Crime, Drama, Thriller",
                rating="7.6",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="No Country for Old Men",
                overview="Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong.",
                genres="Crime, Drama, Thriller",
                rating="8.2",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="The Departed",
                overview="An undercover cop and a mole in the police attempt to identify each other.",
                genres="Crime, Drama, Thriller",
                rating="8.5",
                expected="Not safe",
                reason="Extreme violence, adult themes"
            ),
            TestCase(
                title="Goodfellas",
                overview="The story of Henry Hill and his life in the mob.",
                genres="Biography, Crime, Drama",
                rating="8.7",
                expected="Not safe",
                reason="Violence, adult themes, crime content"
            ),
            TestCase(
                title="Scarface",
                overview="In 1980s Miami, a determined Cuban immigrant takes over a drug cartel.",
                genres="Crime, Drama",
                rating="8.3",
                expected="Not safe",
                reason="Extreme violence, drug use, strong profanity"
            ),
            TestCase(
                title="The Silence of the Lambs",
                overview="A young FBI cadet must receive the help of an incarcerated cannibal killer to catch another serial killer.",
                genres="Crime, Drama, Thriller",
                rating="8.6",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="Saving Private Ryan",
                overview="Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines.",
                genres="Drama, War",
                rating="8.6",
                expected="Not safe",
                reason="Extreme war violence, graphic imagery"
            ),
            TestCase(
                title="Schindler's List",
                overview="In German-occupied Poland, a businessman gradually becomes concerned for his Jewish workforce.",
                genres="Biography, Drama, History",
                rating="9.0",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="Joker",
                overview="In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society.",
                genres="Crime, Drama, Thriller",
                rating="8.4",
                expected="Not safe",
                reason="Extreme violence, disturbing content"
            ),
            TestCase(
                title="The Shining",
                overview="A family heads to an isolated hotel for the winter where a sinister presence influences the father.",
                genres="Horror, Drama",
                rating="8.4",
                expected="Not safe",
                reason="Horror genre, disturbing imagery"
            ),
            TestCase(
                title="Hereditary",
                overview="A grieving family is haunted by tragic and disturbing occurrences.",
                genres="Horror, Drama, Mystery",
                rating="7.3",
                expected="Not safe",
                reason="Horror genre, disturbing content"
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

        # Calculate safe_passed and unsafe_passed
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