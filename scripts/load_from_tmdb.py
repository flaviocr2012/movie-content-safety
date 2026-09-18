"""
Script to load movies from TMDB API and save them to CSV.
Replaces the hardcoded movie list with real data from TMDB.
"""

import os
import csv
import sys

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from tmdb_client import TMDBClient
from config import IMDB_MOVIES_PATH


def load_movies_from_tmdb():
    """
    Load movies from TMDB and save to CSV.
    Combines popular, family, and top-rated movies.
    """
    print("=" * 60)
    print("🎬 LOADING MOVIES FROM TMDB API")
    print("=" * 60)

    client = TMDBClient()

    # Fetch movies from multiple endpoints
    print("\n📥 Fetching movies...")
    popular = client.get_popular_movies(pages=5)      # ~100 movies
    family = client.get_family_movies(pages=3)        # ~60 movies
    top_rated = client.get_top_rated_movies(pages=2)  # ~40 movies

    # Combine and deduplicate
    all_movies = popular + family + top_rated
    seen_titles = set()
    unique_movies = []

    for movie in all_movies:
        title_key = movie["title"].lower()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_movies.append(movie)

    print(f"\n✅ Total unique movies: {len(unique_movies)}")

    # Save to CSV
    with open(IMDB_MOVIES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "overview", "rating", "year", "genres"]
        )
        writer.writeheader()
        writer.writerows(unique_movies)

    print(f"💾 Saved to: {IMDB_MOVIES_PATH}")
    print(f"📊 Total movies: {len(unique_movies)}")

    # Show sample
    print("\n📋 Sample of loaded movies:")
    for movie in unique_movies[:5]:
        print(f"  🎬 {movie['title']} ({movie['year']}) - {movie['genres']}")


if __name__ == "__main__":
    load_movies_from_tmdb()