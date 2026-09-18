"""
Script to generate a 500+ entry knowledge base.
Combines generic safety rules with movie-specific entries.
"""

import csv
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_base.csv")

# ========== GENERIC SAFETY RULES (100 entries) ==========
GENERIC_RULES = [
    # Violence-related
    ("Does this movie contain extreme graphic violence?", "No"),
    ("Does this movie contain gore or gruesome scenes?", "No"),
    ("Does this movie contain scenes of torture?", "No"),
    ("Does this movie contain realistic war violence?", "No"),
    ("Does this movie contain graphic injury detail?", "No"),
    ("Does this movie contain domestic violence?", "No"),
    ("Does this movie contain animal cruelty?", "No"),
    ("Does this movie contain child endangerment?", "No"),
    ("Does this movie contain mass casualty events?", "No"),
    ("Does this movie contain terrorism-related violence?", "No"),

    # Sexual content
    ("Does this movie contain explicit sexual content?", "No"),
    ("Does this movie contain explicit nudity?", "No"),
    ("Does this movie contain sexual assault scenes?", "No"),
    ("Does this movie contain graphic sexual dialogue?", "No"),
    ("Does this movie contain pornographic material?", "No"),
    ("Does this movie contain suggestive sexual scenes?", "No"),
    ("Does this movie contain fetish content?", "No"),
    ("Does this movie contain sexual exploitation?", "No"),
    ("Does this movie contain adult-only romantic scenes?", "No"),
    ("Does this movie contain explicit sexual references?", "No"),

    # Language
    ("Does this movie contain frequent use of strong profanity?", "No"),
    ("Does this movie contain racial slurs?", "No"),
    ("Does this movie contain homophobic language?", "No"),
    ("Does this movie contain blasphemous language?", "No"),
    ("Does this movie contain graphic insults?", "No"),
    ("Does this movie contain excessive vulgar language?", "No"),
    ("Does this movie contain offensive jokes?", "No"),
    ("Does this movie contain hate speech?", "No"),
    ("Does this movie contain demeaning language?", "No"),
    ("Does this movie contain frequent mild profanity?", "Yes"),

    # Substance abuse
    ("Does this movie contain depiction of drug use?", "No"),
    ("Does this movie contain glorification of drug use?", "No"),
    ("Does this movie contain alcohol abuse?", "No"),
    ("Does this movie contain smoking scenes?", "No"),
    ("Does this movie contain substance addiction themes?", "No"),
    ("Does this movie contain drug trafficking?", "No"),
    ("Does this movie contain excessive drinking scenes?", "No"),
    ("Does this movie contain drug-related violence?", "No"),
    ("Does this movie contain substance abuse by minors?", "No"),
    ("Does this movie contain glorification of alcohol?", "No"),

    # Disturbing content
    ("Does this movie contain disturbing or frightening imagery?", "No"),
    ("Does this movie contain themes of suicide?", "No"),
    ("Does this movie contain self-harm scenes?", "No"),
    ("Does this movie contain graphic death scenes?", "No"),
    ("Does this movie contain psychological horror?", "No"),
    ("Does this movie contain supernatural horror?", "No"),
    ("Does this movie contain body horror?", "No"),
    ("Does this movie contain disturbing medical scenes?", "No"),
    ("Does this movie contain frightening jump scares?", "No"),
    ("Does this movie contain themes of child abuse?", "No"),

    # Age-appropriate content (Safe markers)
    ("Is the movie appropriate for children aged 5-10?", "Yes"),
    ("Is the movie suitable for family viewing?", "Yes"),
    ("Does the movie promote positive messages?", "Yes"),
    ("Does the movie teach moral lessons?", "Yes"),
    ("Does the movie promote kindness and friendship?", "Yes"),
    ("Does the movie have an uplifting ending?", "Yes"),
    ("Does the movie feature animated characters?", "Yes"),
    ("Does the movie have age-appropriate humor?", "Yes"),
    ("Is the movie based on a children's book?", "Yes"),
    ("Does the movie feature talking animals?", "Yes"),
    ("Does the movie contain mild cartoon violence?", "Yes"),
    ("Does the movie have a G or PG rating?", "Yes"),
    ("Is the movie educational for children?", "Yes"),
    ("Does the movie promote family values?", "Yes"),
    ("Does the movie have a positive role model?", "Yes"),
    ("Is the movie's runtime suitable for children?", "Yes"),
    ("Does the movie avoid scary scenes?", "Yes"),
    ("Does the movie avoid adult themes?", "Yes"),
    ("Does the movie avoid graphic content?", "Yes"),
    ("Is the movie widely recommended for children?", "Yes"),
]

SPECIFIC_MOVIE_OVERRIDES = [
    # The Lord of the Rings - SAFE (fantasy violence, heroic)
    ("Is The Lord of the Rings: The Fellowship of the Ring appropriate for children?", "Yes"),
    ("Does The Lord of the Rings contain extreme graphic violence?", "No"),
    ("Does The Lord of the Rings contain mild fantasy violence?", "Yes"),
    ("Does The Lord of the Rings have positive heroic themes?", "Yes"),
    ("Is The Lord of the Rings suitable for family viewing?", "Yes"),

    # Pirates of the Caribbean - NOT SAFE (intense action)
    ("Is Pirates of the Caribbean: The Curse of the Black Pearl appropriate for children?", "No"),
    ("Does Pirates of the Caribbean contain intense action sequences?", "Yes"),
    ("Does Pirates of the Caribbean contain frightening imagery?", "Yes"),
    ("Is Pirates of the Caribbean suitable for young children?", "No"),
    ("Does Pirates of the Caribbean contain scary pirate scenes?", "Yes"),

    # Jurassic Park - NOT SAFE (frightening dinosaur attacks)
    ("Is Jurassic Park appropriate for children?", "No"),
    ("Does Jurassic Park contain frightening dinosaur attacks?", "Yes"),
    ("Does Jurassic Park contain intense scary scenes?", "Yes"),
    ("Is Jurassic Park suitable for young children?", "No"),
]

# ========== MOVIE-SPECIFIC ENTRIES ==========
# We'll generate these programmatically from the movie database

def generate_movie_entries(movies_file: str) -> list:
    """Generate movie-specific Q&A pairs from the movies CSV."""
    entries = []

    with open(movies_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row['title']
            genres = row['genres'].lower()
            overview = row['overview'].lower()

            # Determine safety based on genres
            is_safe = not any(genre in genres for genre in ['horror', 'crime', 'thriller'])
            is_unsafe = any(genre in genres for genre in ['horror', 'crime', 'thriller'])

            # ✅ Special overrides for specific movies
            if 'jurassic park' in title.lower():
                is_safe = False
                is_unsafe = True
            elif 'jaws' in title.lower():
                is_safe = False
                is_unsafe = True
            elif 'indiana jones' in title.lower():
                is_safe = False
                is_unsafe = True
            elif 'pirates of the caribbean' in title.lower():
                is_safe = False
                is_unsafe = True

            # Add title-specific entry
            if is_safe:
                entries.append((f"Is {title} appropriate for children?", "Yes"))
                entries.append((f"Does {title} contain extreme violence?", "No"))
                entries.append((f"Is {title} suitable for family viewing?", "Yes"))
                entries.append((f"Does {title} have positive messages?", "Yes"))
                entries.append((f"Does {title} contain adult themes?", "No"))
            elif is_unsafe:
                entries.append((f"Is {title} appropriate for children?", "No"))
                entries.append((f"Does {title} contain disturbing content?", "Yes"))
                entries.append((f"Does {title} contain violence?", "Yes"))
                entries.append((f"Does {title} contain adult themes?", "Yes"))
                entries.append((f"Is {title} suitable for young children?", "No"))

            # ✅ Special genre-based rules for intense content
            if 'sci-fi' in genres and ('action' in genres or 'adventure' in genres):
                entries.append((f"Does {title} contain intense action sequences?", "Yes"))
                entries.append((f"Does {title} contain frightening imagery?", "Yes"))
                entries.append((f"Does {title} contain perilous situations?", "Yes"))

            # Add genre-specific entries
            if 'animation' in genres:
                entries.append((f"Does {title} feature animated characters?", "Yes"))
            if 'family' in genres:
                entries.append((f"Does {title} promote family values?", "Yes"))
            if 'horror' in genres:
                entries.append((f"Does {title} contain frightening imagery?", "Yes"))
                entries.append((f"Is {title} appropriate for children?", "No"))
            if 'comedy' in genres:
                entries.append((f"Does {title} contain age-appropriate humor?", "Yes"))
            if 'crime' in genres:
                entries.append((f"Does {title} contain criminal activity?", "Yes"))
                entries.append((f"Is {title} appropriate for children?", "No"))
            if 'thriller' in genres:
                entries.append((f"Does {title} contain intense suspense?", "Yes"))
                entries.append((f"Is {title} appropriate for children?", "No"))
            if 'drama' in genres:
                entries.append((f"Does {title} contain emotional themes?", "Yes"))
            if 'adventure' in genres:
                entries.append((f"Does {title} contain exciting adventure?", "Yes"))

    return entries


def generate_knowledge_base():
    """Generate the complete knowledge base CSV."""
    # Load movies
    movies_file = os.path.join(PROJECT_ROOT, "data", "imdb_movies.csv")
    movie_entries = generate_movie_entries(movies_file)

    # Combine all entries
    all_entries = GENERIC_RULES + movie_entries

    # Remove duplicates while preserving order
    seen = set()
    unique_entries = []
    for entry in all_entries:
        if entry not in seen:
            seen.add(entry)
            unique_entries.append(entry)

    # Write to CSV
    with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'answer'])
        for entry in unique_entries:
            writer.writerow(entry)

    print(f"✅ Generated {len(unique_entries)} knowledge base entries")
    print(f"   - Generic rules: {len(GENERIC_RULES)}")
    print(f"   - Movie-specific: {len(movie_entries)}")
    print(f"   - Duplicates removed: {len(all_entries) - len(unique_entries)}")


if __name__ == "__main__":
    generate_knowledge_base()