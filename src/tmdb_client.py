"""
TMDB API Client for Movie Content Safety Classifier.
Fetches movie data from The Movie Database (TMDB) with local caching.
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from config import TMDB_API_KEY, TMDB_CACHE_PATH


class TMDBClient:
    """
    Client for The Movie Database (TMDB) API.
    Includes local caching to minimize API calls.
    """

    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    CACHE_EXPIRY_DAYS = 7

    # TMDB Genre ID → Name mapping
    GENRE_MAP = {
        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        14: "Fantasy",
        36: "History",
        27: "Horror",
        10402: "Music",
        9648: "Mystery",
        10749: "Romance",
        878: "Sci-Fi",
        10770: "TV Movie",
        53: "Thriller",
        10752: "War",
        37: "Western",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TMDB client.

        Args:
            api_key: TMDB API key (defaults to config value)
        """
        self.api_key = api_key or TMDB_API_KEY
        if not self.api_key:
            raise ValueError("TMDB_API_KEY is required")

        self.cache = self._load_cache()
        print(f"✅ TMDB Client initialized (cache: {len(self.cache)} entries)")

    def _load_cache(self) -> Dict:
        """Load the cache from disk."""
        if os.path.exists(TMDB_CACHE_PATH):
            try:
                with open(TMDB_CACHE_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load cache: {e}")
        return {}

    def _save_cache(self) -> None:
        """Save the cache to disk."""
        os.makedirs(os.path.dirname(TMDB_CACHE_PATH), exist_ok=True)
        with open(TMDB_CACHE_PATH, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self.cache:
            return False

        entry = self.cache[cache_key]
        cached_at = datetime.fromisoformat(entry.get("cached_at", "2000-01-01"))
        expiry = timedelta(days=self.CACHE_EXPIRY_DAYS)

        return datetime.now() - cached_at < expiry

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the TMDB API.

        Args:
            endpoint: API endpoint (e.g., "movie/popular")
            params: Query parameters

        Returns:
            API response as dictionary
        """
        if params is None:
            params = {}

        params["api_key"] = self.api_key
        params["language"] = "en-US"

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ TMDB API error: {e}")
            return {}

    def _format_movie(self, movie_data: Dict) -> Dict:
        """
        Format TMDB movie data to our internal format.

        Args:
            movie_data: Raw TMDB movie data

        Returns:
            Formatted movie dictionary
        """
        # Get genre names from IDs
        genre_ids = movie_data.get("genre_ids", [])
        if not genre_ids and "genres" in movie_data:
            genre_ids = [g["id"] for g in movie_data["genres"]]

        genres = ", ".join([
            self.GENRE_MAP.get(gid, "Unknown")
            for gid in genre_ids
            if gid in self.GENRE_MAP
        ])

        # Get release year
        release_date = movie_data.get("release_date", "")
        year = release_date[:4] if release_date else "N/A"

        return {
            "title": movie_data.get("title", "Unknown"),
            "overview": movie_data.get("overview", "No overview available."),
            "rating": str(round(movie_data.get("vote_average", 0.0), 1)),
            "year": year,
            "genres": genres if genres else "Unknown",
        }

    def get_popular_movies(self, pages: int = 5) -> List[Dict]:
        """
        Fetch popular movies from TMDB.

        Args:
            pages: Number of pages to fetch (20 movies per page)

        Returns:
            List of formatted movie dictionaries
        """
        cache_key = f"popular_{pages}"

        if self._is_cache_valid(cache_key):
            print(f"📦 Using cached data for {cache_key}")
            return self.cache[cache_key]["data"]

        print(f"🌐 Fetching {pages} pages of popular movies from TMDB...")
        movies = []

        for page in range(1, pages + 1):
            data = self._make_request("movie/popular", {"page": page})
            results = data.get("results", [])

            for movie in results:
                movies.append(self._format_movie(movie))

            # Rate limit: TMDB allows ~40 requests per 10 seconds
            time.sleep(0.25)

        # Cache results
        self.cache[cache_key] = {
            "cached_at": datetime.now().isoformat(),
            "data": movies,
        }
        self._save_cache()

        print(f"✅ Fetched {len(movies)} movies from TMDB")
        return movies

    def get_top_rated_movies(self, pages: int = 5) -> List[Dict]:
        """Fetch top-rated movies from TMDB."""
        cache_key = f"top_rated_{pages}"

        if self._is_cache_valid(cache_key):
            print(f"📦 Using cached data for {cache_key}")
            return self.cache[cache_key]["data"]

        print(f"🌐 Fetching {pages} pages of top-rated movies from TMDB...")
        movies = []

        for page in range(1, pages + 1):
            data = self._make_request("movie/top_rated", {"page": page})
            results = data.get("results", [])

            for movie in results:
                movies.append(self._format_movie(movie))

            time.sleep(0.25)

        self.cache[cache_key] = {
            "cached_at": datetime.now().isoformat(),
            "data": movies,
        }
        self._save_cache()

        print(f"✅ Fetched {len(movies)} top-rated movies from TMDB")
        return movies

    def get_family_movies(self, pages: int = 3) -> List[Dict]:
        """
        Fetch family-friendly movies from TMDB.
        Filters by family genres.

        Returns:
            List of family-friendly movie dictionaries
        """
        cache_key = f"family_{pages}"

        if self._is_cache_valid(cache_key):
            print(f"📦 Using cached data for {cache_key}")
            return self.cache[cache_key]["data"]

        print(f"🌐 Fetching family movies from TMDB...")
        movies = []

        for page in range(1, pages + 1):
            # Discover endpoint with family-friendly filters
            data = self._make_request("discover/movie", {
                "page": page,
                "with_genres": "10751|16|35",  # Family, Animation, Comedy
                "sort_by": "popularity.desc",
                "vote_average.gte": 6.0,
            })
            results = data.get("results", [])

            for movie in results:
                movies.append(self._format_movie(movie))

            time.sleep(0.25)

        self.cache[cache_key] = {
            "cached_at": datetime.now().isoformat(),
            "data": movies,
        }
        self._save_cache()

        print(f"✅ Fetched {len(movies)} family movies from TMDB")
        return movies

    def search_movie(self, query: str) -> Optional[Dict]:
        """
        Search for a specific movie by title.

        Args:
            query: Movie title to search for

        Returns:
            Formatted movie dictionary or None if not found
        """
        cache_key = f"search_{query.lower()}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]

        data = self._make_request("search/movie", {"query": query})
        results = data.get("results", [])

        if not results:
            return None

        movie = self._format_movie(results[0])

        self.cache[cache_key] = {
            "cached_at": datetime.now().isoformat(),
            "data": movie,
        }
        self._save_cache()

        return movie

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information about a movie.

        Args:
            movie_id: TMDB movie ID

        Returns:
            Formatted movie dictionary with full details
        """
        cache_key = f"details_{movie_id}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]

        data = self._make_request(f"movie/{movie_id}")
        if not data:
            return None

        movie = self._format_movie(data)
        movie["tmdb_id"] = movie_id
        movie["runtime"] = data.get("runtime", 0)
        movie["tagline"] = data.get("tagline", "")
        movie["poster_path"] = data.get("poster_path", "")

        self.cache[cache_key] = {
            "cached_at": datetime.now().isoformat(),
            "data": movie,
        }
        self._save_cache()

        return movie


def main():
    """Test the TMDB client."""
    print("=" * 60)
    print("🎬 TMDB CLIENT TEST")
    print("=" * 60)

    client = TMDBClient()

    # Test 1: Popular movies
    print("\n📊 Test 1: Popular Movies")
    popular = client.get_popular_movies(pages=1)
    for movie in popular[:3]:
        print(f"  🎬 {movie['title']} ({movie['year']}) - {movie['genres']}")

    # Test 2: Family movies
    print("\n📊 Test 2: Family Movies")
    family = client.get_family_movies(pages=1)
    for movie in family[:3]:
        print(f"  👨‍👩‍👧 {movie['title']} ({movie['year']}) - {movie['genres']}")

    # Test 3: Search
    print("\n📊 Test 3: Search for 'Lion King'")
    result = client.search_movie("Lion King")
    if result:
        print(f"  🦁 {result['title']} ({result['year']}) - {result['genres']}")

    print("\n" + "=" * 60)
    print("✅ TMDB Client test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()