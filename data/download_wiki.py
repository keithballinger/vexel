#!/usr/bin/env python3
"""Download Wikipedia articles for training data."""

import urllib.request
import urllib.parse
import json
import os
import time

# Diverse topics for varied training data
ARTICLES = [
    # Science & Technology
    "Computer", "Artificial_intelligence", "Machine_learning", "Neural_network",
    "Programming_language", "Algorithm", "Data_structure", "Operating_system",
    "Internet", "World_Wide_Web", "Software_engineering", "Computer_science",
    "Mathematics", "Physics", "Chemistry", "Biology", "Quantum_mechanics",
    "Theory_of_relativity", "Evolution", "DNA", "Cell_(biology)", "Atom",

    # History
    "World_War_II", "World_War_I", "Ancient_Rome", "Ancient_Greece",
    "Renaissance", "Industrial_Revolution", "French_Revolution", "American_Revolution",
    "Cold_War", "Roman_Empire", "Byzantine_Empire", "Medieval_history",

    # Geography & Nature
    "Earth", "Solar_System", "Galaxy", "Universe", "Climate_change",
    "Ocean", "Mountain", "River", "Forest", "Desert", "Ecosystem",

    # Arts & Culture
    "Music", "Art", "Literature", "Philosophy", "Architecture",
    "Film", "Theatre", "Dance", "Poetry", "Novel",

    # People
    "Albert_Einstein", "Isaac_Newton", "Charles_Darwin", "Marie_Curie",
    "Leonardo_da_Vinci", "William_Shakespeare", "Aristotle", "Plato",

    # Society
    "Democracy", "Economics", "Psychology", "Sociology", "Language",
    "Religion", "Education", "Law", "Medicine", "Agriculture",

    # More tech
    "Compiler", "Database", "Cryptography", "Computer_network", "CPU",
    "GPU", "Memory_(computing)", "Linux", "Unix", "Python_(programming_language)",
    "JavaScript", "HTML", "HTTP", "API", "Cloud_computing",
]

def fetch_article(title):
    """Fetch article text from Wikipedia API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
    }

    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"

    try:
        req = urllib.request.Request(full_url, headers={"User-Agent": "WikiDownloader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    return page_data.get("extract", "")
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
    return ""

def main():
    output_file = "wikipedia_corpus.txt"

    print(f"Downloading {len(ARTICLES)} Wikipedia articles...")

    all_text = []
    for i, title in enumerate(ARTICLES):
        print(f"[{i+1}/{len(ARTICLES)}] Fetching {title}...")
        text = fetch_article(title)
        if text:
            # Add article with separator
            all_text.append(f"=== {title.replace('_', ' ')} ===\n\n{text}\n\n")
            print(f"  Got {len(text)} characters")
        time.sleep(0.5)  # Rate limit

    # Write combined corpus
    corpus = "\n".join(all_text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nSaved {len(corpus):,} characters to {output_file}")
    print(f"Total articles: {len(all_text)}")

if __name__ == "__main__":
    main()
