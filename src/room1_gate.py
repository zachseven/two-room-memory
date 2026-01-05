"""
Two-Room Memory Architecture - Room 1 Gate
Core triviality filter with unidirectional flow to Room 2

Uses TF-IDF for baseline testing. For production, swap to sentence-transformers
with: model = SentenceTransformer('all-MiniLM-L6-v2')
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Triviality archetype - canonical examples of non-relational exchanges
TRIVIAL_EXAMPLES = [
    # Encyclopedic queries
    "What color are ladybugs",
    "How many feet in a mile",
    "Define photosynthesis",
    "What's the capital of France",
    "How do you spell necessary",
    "What time is it in Tokyo",
    "What's 12 times 15",
    "How many planets are there",
    "What year did World War 2 end",
    "What's the boiling point of water",
    
    # Weather/environment small talk
    "What's the weather like",
    "It's really hot today",
    "Looks like it might rain",
    
    # Random observations / idle chatter
    "Did you know octopuses have three hearts",
    "Random fact about penguins",
    "Clouds look like animals sometimes",
    "They should make more purple legos",
    "Doors open outward because of fire code",
    "That's an interesting building",
    "I like the color blue",
    "Coffee tastes better in the morning",
    
    # Generic queries with no personal stakes
    "What's a good movie to watch",
    "How do I make pasta",
    "What does this word mean",
    "Can you explain quantum physics",
    "Tell me a joke",
    "What's trending today",
]

# Build triviality archetype using TF-IDF
print("Building triviality archetype...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
trivial_matrix = vectorizer.fit_transform(TRIVIAL_EXAMPLES)
A_t = np.asarray(trivial_matrix.mean(axis=0)).flatten()  # centroid
print(f"Archetype built from {len(TRIVIAL_EXAMPLES)} examples, vocabulary size: {len(vectorizer.vocabulary_)}")

# Room 2 storage path
ROOM2_PATH = Path(__file__).parent / "room2.json"


def triviality_score(exchange: str) -> float:
    """Compute cosine similarity between exchange and triviality archetype"""
    embedding = vectorizer.transform([exchange]).toarray().flatten()
    
    # Handle zero vector (no matching terms)
    if np.linalg.norm(embedding) == 0:
        return 0.0  # No overlap with trivial vocabulary = probably not trivial
    
    similarity = np.dot(embedding, A_t) / (np.linalg.norm(embedding) * np.linalg.norm(A_t))
    return float(similarity)


def should_persist(exchange: str, threshold: float = 0.72) -> bool:
    """Gate decision: persist to Room 2 if NOT trivial"""
    return triviality_score(exchange) < threshold


def persist(exchange: str, category: Optional[str] = None, metadata: Optional[dict] = None):
    """Write exchange to Room 2 (JSON store)"""
    room2 = json.loads(ROOM2_PATH.read_text()) if ROOM2_PATH.exists() else []
    
    entry = {
        "text": exchange,
        "timestamp": datetime.now().isoformat(),
        "category": category,
        **(metadata or {})
    }
    
    room2.append(entry)
    ROOM2_PATH.write_text(json.dumps(room2, indent=2))
    return entry


def process_exchange(exchange: str, threshold: float = 0.72, auto_persist: bool = True) -> dict:
    """
    Main gate function: evaluate exchange and route accordingly
    Returns decision info for logging/debugging
    """
    score = triviality_score(exchange)
    persist_decision = score < threshold
    
    result = {
        "exchange": exchange,
        "score": round(score, 4),
        "threshold": threshold,
        "decision": "PERSIST" if persist_decision else "FLUSH"
    }
    
    if persist_decision and auto_persist:
        persist(exchange)
        result["persisted"] = True
    
    return result


def get_room2_contents() -> list:
    """Retrieve all Room 2 entries"""
    if ROOM2_PATH.exists():
        return json.loads(ROOM2_PATH.read_text())
    return []


def clear_room2():
    """Reset Room 2 (for testing)"""
    if ROOM2_PATH.exists():
        ROOM2_PATH.unlink()


# Quick test
if __name__ == "__main__":
    test_exchanges = [
        "they should have more purple legos",
        "im worried about my exam tomorrow",
        "my mother has NPD and i was always the scapegoat",
        "did you ever notice that clouds can look like animals?",
        "the president's understanding of constitutional law concerns me",
        "doors always open outward due to fire code",
        "i've been sober for 6 months",
        "what's the weather like",
        "my dad died yesterday",
        "i have ADHD and it affects how i work",
        "snowflakes are cool",
        "i'm shipping my game in January",
    ]
    
    print("\n" + "="*60)
    print("TRIVIALITY GATE TEST")
    print("="*60)
    
    for exchange in test_exchanges:
        result = process_exchange(exchange, auto_persist=False)
        marker = "→ ROOM 2" if result["decision"] == "PERSIST" else "  (flush)"
        print(f"{result['score']:.3f} {marker}: {exchange}")
