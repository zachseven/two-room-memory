"""
Two-Room Memory Architecture - Classifier Gate
Trained on labeled examples instead of archetype similarity
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import pickle

# Training data: (exchange, label)
# 0 = flush (trivial), 1 = persist (meaningful)
TRAINING_DATA = [
    # === FLUSH (trivial, encyclopedic, small talk) ===
    ("what color are ladybugs", 0),
    ("how many feet in a mile", 0),
    ("what's the capital of France", 0),
    ("define photosynthesis", 0),
    ("what time is it in Tokyo", 0),
    ("how do you spell necessary", 0),
    ("what's 12 times 15", 0),
    ("tell me a random fact about penguins", 0),
    ("did you know octopuses have three hearts", 0),
    ("what's the boiling point of water", 0),
    ("how many planets are in the solar system", 0),
    ("what year did the titanic sink", 0),
    ("they should make more purple legos", 0),
    ("clouds look like animals sometimes", 0),
    ("doors open outward because of fire code", 0),
    ("that's an interesting building", 0),
    ("coffee tastes better in the morning", 0),
    ("it's really hot today", 0),
    ("looks like it might rain", 0),
    ("what's a good movie to watch", 0),
    ("how do I make pasta", 0),
    ("tell me a joke", 0),
    ("snowflakes are cool", 0),
    ("the sky is really blue today", 0),
    ("I like pizza", 0),
    ("what does serendipity mean", 0),
    ("how do airplanes fly", 0),
    ("why is the ocean salty", 0),
    ("what's trending on twitter", 0),
    ("can you explain how magnets work", 0),
    ("the president's understanding of constitutional law concerns me", 0),
    ("i think pineapple belongs on pizza", 0),
    ("What year did World War 1 start", 0),
    ("Who invented the telephone", 0),
    ("What's the capital of Germany", 0),
    ("What's the largest ocean", 0),
    ("Who wrote Romeo and Juliet", 0),
    ("How old is the earth", 0),
    ("That's a nice car", 0),
    ("This food is good", 0),
    ("The traffic is bad today", 0),
    ("How do I tie a tie", 0),
    ("What time does the store close", 0),
    
    # === PERSIST (emotional, personal, formative) ===
    ("my dad died yesterday", 1),
    ("i'm worried about my exam tomorrow", 1),
    ("my mother has NPD and i was always the scapegoat", 1),
    ("i have ADHD and it affects how i work", 1),
    ("i've been sober for 6 months", 1),
    ("i'm getting divorced", 1),
    ("my therapist thinks i have anxiety", 1),
    ("i was diagnosed with depression last year", 1),
    ("my daughter has autism", 1),
    ("i'm pregnant and scared", 1),
    ("i lost my job last week", 1),
    ("my best friend betrayed me", 1),
    ("i'm in an abusive relationship", 1),
    ("i attempted suicide when i was younger", 1),
    ("my son is struggling with addiction", 1),
    ("i grew up in poverty", 1),
    ("i was the first in my family to go to college", 1),
    ("my parents are getting divorced", 1),
    ("i have imposter syndrome at work", 1),
    ("i'm caring for my aging mother", 1),
    ("i struggle with binge eating", 1),
    ("my partner doesn't understand me", 1),
    ("i feel like i'm failing as a parent", 1),
    ("i was bullied throughout school", 1),
    ("i'm estranged from my family", 1),
    
    # Communication / preference
    ("i prefer direct communication", 1),
    ("don't sugarcoat things for me", 1),
    ("i need detailed explanations", 1),
    ("i learn better with examples", 1),
    ("please be patient with me", 1),
    
    # Identity / background
    ("i'm a lawyer", 1),
    ("i went to law school", 1),
    ("i have a PhD in physics", 1),
    ("i'm a recovering alcoholic", 1),
    ("i'm transgender", 1),
    ("english is my second language", 1),
    ("i'm neurodivergent", 1),
    ("i'm a single parent", 1),
    ("i served in the military", 1),
    ("i'm an immigrant", 1),
    
    # Projects / goals
    ("i'm shipping my game in January", 1),
    ("i'm starting my own business", 1),
    ("i'm training for a marathon", 1),
    ("i'm writing a novel", 1),
    ("i'm learning to code", 1),
    ("i'm trying to lose weight", 1),
    ("i'm saving up for a house", 1),
    ("i'm studying for the bar exam", 1),
    
    # Context
    ("i work as a contractor for the VA", 1),
    ("i live in Austin", 1),
    ("i have two kids", 1),
    ("my wife is an esthetician", 1),
    ("i work remotely", 1),
    
    # Edge cases - persist
    ("i've been thinking about death lately", 1),
    ("i can't sleep anymore", 1),
    ("everything feels pointless", 1),
    ("i just feel stuck", 1),
    ("work has been really stressful", 1),
    ("i'm so tired all the time", 1),
    ("nobody really understands me", 1),
    ("i've been having panic attacks", 1),
    ("my childhood was complicated", 1),
    ("i don't trust easily", 1),
    ("i tend to overthink everything", 1),
    ("confrontation makes me shut down", 1),
    ("i have a hard time asking for help", 1),
]

# Initialize
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Prepare data
print("Preparing training data...")
texts = [t[0] for t in TRAINING_DATA]
labels = np.array([t[1] for t in TRAINING_DATA])
embeddings = model.encode(texts)
print(f"Training data: {len(texts)} examples ({sum(labels)} persist, {len(labels) - sum(labels)} flush)")

# Train classifier
print("Training classifier...")
classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
classifier.fit(embeddings, labels)

# Cross-validation score
cv_scores = cross_val_score(classifier, embeddings, labels, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

# Room 2 storage
ROOM2_PATH = Path(__file__).parent / "room2.json"
MODEL_PATH = Path(__file__).parent / "classifier.pkl"

# Save trained classifier
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(classifier, f)
print(f"Classifier saved to {MODEL_PATH}")


def predict(exchange: str) -> tuple[str, float]:
    """Predict flush/persist with confidence score"""
    embedding = model.encode([exchange])
    proba = classifier.predict_proba(embedding)[0]
    prediction = "PERSIST" if proba[1] > 0.5 else "FLUSH"
    confidence = max(proba)
    return prediction, confidence


def should_persist(exchange: str, confidence_threshold: float = 0.5) -> bool:
    """Gate decision: persist to Room 2 if classified as meaningful"""
    embedding = model.encode([exchange])
    proba = classifier.predict_proba(embedding)[0]
    return proba[1] > confidence_threshold


def persist(exchange: str, category: Optional[str] = None, metadata: Optional[dict] = None):
    """Write exchange to Room 2"""
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


def process_exchange(exchange: str, auto_persist: bool = True) -> dict:
    """Main gate function"""
    prediction, confidence = predict(exchange)
    result = {
        "exchange": exchange,
        "decision": prediction,
        "confidence": round(confidence, 3)
    }
    if prediction == "PERSIST" and auto_persist:
        persist(exchange)
        result["persisted"] = True
    return result


def get_room2_contents() -> list:
    if ROOM2_PATH.exists():
        return json.loads(ROOM2_PATH.read_text())
    return []


def clear_room2():
    if ROOM2_PATH.exists():
        ROOM2_PATH.unlink()


# Test
if __name__ == "__main__":
    test_exchanges = [
        # Should flush
        "they should have more purple legos",
        "did you ever notice that clouds can look like animals?",
        "the president's understanding of constitutional law concerns me",
        "doors always open outward due to fire code",
        "what's the weather like",
        "snowflakes are cool",
        "what color are ladybugs",
        "how many feet in a mile",
        # Should persist
        "im worried about my exam tomorrow",
        "my mother has NPD and i was always the scapegoat",
        "i've been sober for 6 months",
        "my dad died yesterday",
        "i have ADHD and it affects how i work",
        "i'm shipping my game in January",
        "i'm writing a novel",
        "i'm learning to code",
        "i just feel stuck",
    ]
    
    print("\n" + "="*60)
    print("CLASSIFIER GATE TEST")
    print("="*60)
    
    for exchange in test_exchanges:
        result = process_exchange(exchange, auto_persist=False)
        marker = "→ ROOM 2" if result["decision"] == "PERSIST" else "  (flush)"
        print(f"{result['confidence']:.2f} {marker}: {exchange}")
