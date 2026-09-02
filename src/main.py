#!/usr/bin/env python
"""
Main application for Movie Content Safety Classifier.
Orchestrates the RAG chain to classify movies as safe or not safe for children.
"""
import sys
import argparse
import pandas as pd
from typing import List, Dict, Optional

# Import the RAG chain and configuration
from rag_chain import RAGChain
from config import GROQ_API_KEY, GROQ_MODEL, IMDB_MOVIES_PATH


def load_movies_from_csv(file_path: str = IMDB_MOVIES_PATH) -> List[Dict[str, str]]:
    """
    Load movies from the IMDB CSV file.

    Args:
        file_path: Path to the IMDB movies CSV file

    Returns:
        List of dictionaries with 'title', 'overview', 'rating', 'year', 'genres'
    """
    try:
        df = pd.read_csv(file_path)
        movies = df.to_dict('records')
        print(f"✅ Loaded {len(movies)} movies from {file_path}")
        return movies
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading movies: {e}")
        return []


def classify_single_movie(rag: RAGChain, title: str, overview: str = None,
                          genres: str = "Unknown", rating: str = "Unknown") -> str:
    """
    Classify a single movie with optional genres and rating.

    Args:
        rag: Initialized RAGChain instance
        title: Movie title
        overview: Movie overview
        genres: Movie genres (e.g., "Animation, Adventure")
        rating: Movie rating (e.g., "8.5")

    Returns:
        Classification result
    """
    if not overview:
        overview = f"A movie titled '{title}'."

    # Pass all parameters to the RAG chain
    return rag.classify_movie(title, overview, genres, rating)


def classify_movies_batch(rag: RAGChain, movies: List[Dict[str, str]],
                          limit: int = None) -> List[Dict[str, str]]:
    """
    Classify multiple movies in batch.

    Args:
        rag: Initialized RAGChain instance
        movies: List of movie dictionaries
        limit: Maximum number of movies to classify (optional)

    Returns:
        List of dictionaries with classification results
    """
    if limit:
        movies = movies[:limit]

    results = []
    total = len(movies)

    print(f"\n📋 Classifying {total} movies...")
    print("-" * 60)

    for i, movie in enumerate(movies, 1):
        title = movie.get('title', 'Unknown')
        overview = movie.get('overview', '')
        genres = movie.get('genres', 'Unknown')
        rating = movie.get('rating', 'Unknown')

        print(f"\n[{i}/{total}] 🎬 {title}")

        try:
            # Pass all parameters to classify_single_movie
            result = classify_single_movie(rag, title, overview, genres, rating)
            results.append({
                'title': title,
                'overview': overview,
                'rating': rating,
                'year': movie.get('year', 'N/A'),
                'genres': genres,
                'classification': result
            })
        except Exception as e:
            print(f"❌ Error classifying {title}: {e}")
            results.append({
                'title': title,
                'overview': overview,
                'rating': rating,
                'year': movie.get('year', 'N/A'),
                'genres': genres,
                'classification': f"Error: {e}"
            })

    return results


def print_classification_summary(results: List[Dict[str, str]]) -> None:
    """
    Print a summary of classification results.

    Args:
        results: List of classification results
    """
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)

    safe_count = 0
    unsafe_count = 0
    error_count = 0

    for result in results:
        classification = result.get('classification', '')
        title = result.get('title', 'Unknown')

        if 'Safe' in classification or 'safe' in classification:
            safe_count += 1
            status = "✅ SAFE"
        elif 'Not safe' in classification or 'not safe' in classification:
            unsafe_count += 1
            status = "❌ UNSAFE"
        elif 'Error' in classification:
            error_count += 1
            status = "⚠️ ERROR"
        else:
            error_count += 1
            status = "⚠️ UNKNOWN"

        print(f"{status}  {title}")

    print("-" * 70)
    print(f"✅ Safe for children:   {safe_count}")
    print(f"❌ Not safe:            {unsafe_count}")
    print(f"⚠️ Errors:              {error_count}")
    print("=" * 70)


def interactive_mode():
    """Run the application in interactive mode with CSV lookup."""
    print("=" * 70)
    print("🎬 MOVIE CONTENT SAFETY CLASSIFIER - INTERACTIVE MODE")
    print("=" * 70)

    # Check API key
    if not GROQ_API_KEY:
        print("\n❌ ERROR: GROQ_API_KEY not found in .env file!")
        print("Please get your free API key from: https://console.groq.com")
        return

    # Load movies from CSV for reference
    print("\n🔄 Loading movie database...")
    movies_dict = {}
    try:
        df = pd.read_csv(IMDB_MOVIES_PATH)
        for _, row in df.iterrows():
            movies_dict[row['title'].lower()] = {
                'overview': row['overview'],
                'year': row.get('year', 'N/A'),
                'rating': row.get('rating', 'N/A'),
                'genres': row.get('genres', 'N/A')
            }
        print(f"✅ Loaded {len(movies_dict)} movie overviews from CSV")
    except Exception as e:
        print(f"⚠️ Could not load movie overviews: {e}")

    # Initialize the RAG chain
    print("\n🔄 Initializing RAG chain...")
    try:
        rag = RAGChain()
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {e}")
        return

    print("\n" + "-" * 70)
    print("📋 INSTRUCTIONS:")
    print("  - Enter a movie title and overview")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'example' for a sample classification")
    print("  - Type 'batch' to classify all movies from CSV")
    print("-" * 70)

    while True:
        print("\n" + "-" * 70)

        # Get user input
        command = input("🎬 Enter movie title (or command): ").strip()

        if command.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        if command.lower() == 'example':
            # Sample classification
            sample_title = "The Lion King"
            sample_overview = "A young lion prince flees his kingdom after the murder of his father and learns about responsibility and friendship."
            sample_genres = "Animation, Adventure, Drama"
            sample_rating = "8.5"
            print(f"\n📝 Using sample: {sample_title}")
            print(f"📖 Overview: {sample_overview}")
            print(f"📝 Genres: {sample_genres} | Rating: {sample_rating}")

            try:
                result = classify_single_movie(rag, sample_title, sample_overview, sample_genres, sample_rating)
                print(f"\n📌 Result:\n{result}")
            except Exception as e:
                print(f"❌ Error: {e}")
            continue

        if command.lower() == 'batch':
            # Classify all movies from CSV
            movies = load_movies_from_csv()
            if not movies:
                print("❌ No movies found in CSV file.")
                continue

            try:
                limit_input = input("📊 How many movies to classify? (default: all): ").strip()
                limit = int(limit_input) if limit_input else None
            except ValueError:
                limit = None

            results = classify_movies_batch(rag, movies, limit)
            print_classification_summary(results)
            continue

        if not command:
            print("⚠️ Please enter a movie title or command.")
            continue

        # ✅ ALWAYS try to load from CSV first
        if command.lower() in movies_dict:
            print(f"📖 Found '{command}' in CSV database!")
            overview = movies_dict[command.lower()]['overview']
            year = movies_dict[command.lower()]['year']
            rating = movies_dict[command.lower()]['rating']
            genres = movies_dict[command.lower()]['genres']
            print(f"📝 Year: {year} | Rating: {rating} | Genres: {genres}")
            print(f"📝 Overview: {overview[:150]}..." if len(overview) > 150 else f"📝 Overview: {overview}")

            # ✅ Use the real overview without asking the user
            result = classify_single_movie(rag, command, overview, genres, rating)
            print(f"\n📌 Result:\n{result}")
        else:
            # Movie not found in CSV - ask user for details
            print(f"⚠️ '{command}' not found in database.")
            overview = input("📝 Enter movie overview: ").strip()
            if not overview:
                print("⚠️ Overview is required for classification. Please try again.")
                continue

            genres = input("📝 Enter movie genres (or press Enter to skip): ").strip() or "Unknown"
            rating = input("📝 Enter movie rating (or press Enter to skip): ").strip() or "Unknown"

            result = classify_single_movie(rag, command, overview, genres, rating)
            print(f"\n📌 Result:\n{result}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Movie Content Safety Classifier using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive
  python main.py --batch --limit 10
  python main.py --title "The Lion King" --overview "A young lion prince..."
        """
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Classify movies in batch mode (from CSV)'
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of movies to classify in batch mode'
    )

    parser.add_argument(
        '--title', '-t',
        type=str,
        help='Movie title for single classification'
    )

    parser.add_argument(
        '--overview', '-o',
        type=str,
        help='Movie overview for single classification'
    )

    parser.add_argument(
        '--genres', '-g',
        type=str,
        default="Unknown",
        help='Movie genres for single classification'
    )

    parser.add_argument(
        '--rating', '-r',
        type=str,
        default="Unknown",
        help='Movie rating for single classification'
    )

    args = parser.parse_args()

    # Handle interactive mode
    if args.interactive or len(sys.argv) == 1:
        interactive_mode()
        return

    # Check API key
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not found in .env file!")
        print("Please get your free API key from: https://console.groq.com")
        return

    # Initialize RAG chain
    print("🔄 Initializing RAG chain...")
    try:
        rag = RAGChain()
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {e}")
        return

    # Handle batch mode
    if args.batch:
        movies = load_movies_from_csv()
        if not movies:
            print("❌ No movies found in CSV file.")
            return

        results = classify_movies_batch(rag, movies, args.limit)
        print_classification_summary(results)
        return

    # Handle single classification
    if args.title:
        if not args.overview:
            args.overview = f"A movie titled '{args.title}'."
            print(f"📖 Using auto-generated overview: {args.overview}")

        try:
            result = classify_single_movie(rag, args.title, args.overview, args.genres, args.rating)
            print(f"\n📌 Result:\n{result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        return

    # If no arguments, show help
    parser.print_help()


if __name__ == "__main__":
    main()