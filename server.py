"""
Two-Room Memory Demo Server
Runs the triviality gate classifier separately from Claude conversation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow browser requests

# Load or train classifier
MODEL_PATH = Path(__file__).parent / "src" / "classifier.pkl"
EMBED_MODEL = None
CLASSIFIER = None

# Training data from classifier_gate.py
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
    ("Stay hydrated", 0),
    ("Bundle up out there", 0),
    ("Don't forget sunscreen", 0),
    ("Heat advisory", 0),
    ("Freeze warning", 0),
    ("Shelter in place", 0),
    ("Travel not recommended", 0),
    ("Stay safe out there", 0),
    ("Check on your neighbors", 0),
    ("Stock up on supplies", 0),
    ("Charge your devices", 0),
    ("Fill up your gas tank", 0),
    ("Get emergency kit ready", 0),
    ("Nice weekend ahead", 0),
    ("Can't dress for this weather", 0),
    ("Got sunburned", 0),
    ("Froze my butt off", 0),
    ("Sweating through my shirt", 0),
    ("The app crashed", 0),
    ("My socks don't match", 0),
    ("I lost a button", 0),
    ("Falls apart fast", 0),
    ("Lasts forever", 0),
    ("Different in person", 0),
    ("Read the reviews first", 0),
    ("Good day for staying inside", 0),
    ("I miss summer already", 0),
    ("I prefer cold to hot", 0),
    ("The scissors are dull", 0),
    ("The brakes are squeaky", 0),
    ("Impulse buy", 0),
    ("Useful", 0),
    ("Just right", 0),
    ("Better than expected", 0),
    ("Worse than expected", 0),
    ("As expected", 0),
    ("Should have researched more", 0),
    ("Never heard of them", 0),
    ("Mainstream", 0),
    ("Hard to find", 0),
    ("Barely works", 0),
    ("Minimalist", 0),
    ("Maximalist", 0),
    ("Practical", 0),
    ("How many teeth do adults have", 0),
    ("Who wrote Pride and Prejudice", 0),
    ("Dogs are more loyal", 0),
    ("I'm a morning person", 0),
    ("I'm a night owl", 0),
    ("Coffee helps me function", 0),
    ("Ranch goes on everything", 0),
    ("Vanilla is not boring", 0),
    ("I like things organized", 0),
    ("Clean space clean mind", 0),
    ("New things work better", 0),
    ("Hard work pays off", 0),
    ("Consistency is important", 0),
    ("Pick your battles", 0),
    ("Communication is key", 0),
    ("Actions speak louder", 0),
    ("Words matter too", 0),
    ("Both", 0),
    ("Neither", 0),
    ("hi", 0),
    ("hello", 0),
    ("hey", 0),
    ("hi there", 0),
    ("good morning", 0),
    ("good afternoon", 0),
    ("how are you", 0),
    ("what's up", 0),
    ("thanks", 0),
    ("thank you", 0),
    ("ok", 0),
    ("okay", 0),
    ("sure", 0),
    ("yes", 0),
    ("no", 0),
    ("maybe", 0),
    ("I guess", 0),
    ("went to a baseball game", 0),
    ("saw a movie", 0),
    ("had lunch", 0),
    ("it was okay", 0),
    ("it was fine", 0),
    ("not bad", 0),

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
    ("i prefer direct communication", 1),
    ("don't sugarcoat things for me", 1),
    ("i need detailed explanations", 1),
    ("i learn better with examples", 1),
    ("please be patient with me", 1),
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
    ("i'm shipping my game in January", 1),
    ("i'm starting my own business", 1),
    ("i'm training for a marathon", 1),
    ("i'm writing a novel", 1),
    ("i'm learning to code", 1),
    ("i'm trying to lose weight", 1),
    ("i'm saving up for a house", 1),
    ("i'm studying for the bar exam", 1),
    ("i work as a contractor for the VA", 1),
    ("i live in Austin", 1),
    ("i have two kids", 1),
    ("my wife is an esthetician", 1),
    ("i work remotely", 1),
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
    ("I don't see the point anymore", 1),
    ("The walls were closing in", 1),
    ("Their best wasn't enough", 1),
    ("Walking on eggshells", 1),
    ("They deny what happened", 1),
    ("Growth looks like change", 1),
    ("Love isn't always enough", 1),
    ("I see the pattern now", 1),
    ("The house is so quiet", 1),
    ("Their chair is empty", 1),
    ("They deserve happy memories", 1),
    ("Queer joy is real", 1),
    ("The world isn't always safe", 1),
    ("I saw things no one should see", 1),
    ("Rock bottom was real", 1),
    ("Flares are unpredictable", 1),
    ("I never know when one will hit", 1),
    ("Some days it wins", 1),
    ("This wasn't planned", 1),
    ("This isn't what I expected", 1),
    ("The tears won't stop", 1),
    ("Bottling everything up", 1),
    ("Triggers are everywhere", 1),
    ("The inner voice is cruel", 1),
    ("I never asked for this", 1),
    ("The meds have side effects", 1),
    ("Finding the right combo is hard", 1),
    ("Everyone else has it together", 1),
    ("Strong meant silent", 1),
    ("I feel seen", 1),
    ("I have to pick sides", 1),
    ("I love both of them", 1),
    ("The house was always tense", 1),
    ("We're not close anymore", 1),
    ("We remember things differently", 1),
    ("I'm running on empty", 1),
    ("Nothing seems to work", 1),
    ("I'm not sure we're compatible", 1),
    ("I don't know if it can be saved", 1),
    ("Healthy love exists", 1),
    ("I want to protect them from everything", 1),
    ("It took time to get here", 1),
    ("I can't be out everywhere", 1),
    ("I don't know if they're safe", 1),
    ("I know what cold feels like", 1),
    ("I look fine on the outside", 1),
    ("Accessibility matters", 1),
    ("One day at a time", 1),
    ("Radiation is exhausting", 1),
    ("Some days are better", 1),
    ("Breaking the cycle is hard", 1),
    ("Gaslighting from family is the worst", 1),
    ("Different not less", 1),
    ("Third culture kid", 1),
    ("The pain is severe", 1),
    ("It affects fertility", 1),
    ("PPD is real", 1),
    ("Just going through the motions", 1),
    ("Mornings are the hardest", 1),
    ("We want different things", 1),
    ("I respect their boundaries", 1),
    ("Fellow vets understand", 1),
    ("Sobriety is a daily choice", 1),
    ("Every day counts", 1),
    ("Managing blood sugar is constant", 1),
    ("Small victories matter", 1),
    ("Golden handcuffs are real", 1),
    ("Underpaid and overworked", 1),
    ("HR protects the company", 1),
    ("People don't understand", 1),
    ("I believe in this idea", 1),
    ("For the first time", 1),
    ("The interest keeps compounding", 1),
    ("The escape is the same", 1),
    ("I made a mistake", 1),
    ("Something that matters to me", 1),
    ("I care", 1),
    ("Being a beginner again", 1),
    ("I'm saving for something important", 1),
    ("The goal matters", 1),
    ("Moving somewhere new", 1),
    ("Starting fresh", 1),
    ("Impact", 1),
    ("Life's too short", 1),
    ("Finally", 1),
    ("my mom loved baseball and she died yesterday", 1),
    ("we went because it was her favorite", 1),
    ("I miss her so much", 1),
]


def load_classifier():
    """Load or train the classifier"""
    global EMBED_MODEL, CLASSIFIER

    print("Loading embedding model...")
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")

    # Try to load saved classifier
    if MODEL_PATH.exists():
        print(f"Loading classifier from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            CLASSIFIER = pickle.load(f)
    else:
        print("Training classifier...")
        texts = [t[0] for t in TRAINING_DATA]
        labels = np.array([t[1] for t in TRAINING_DATA])
        embeddings = EMBED_MODEL.encode(texts)

        CLASSIFIER = LogisticRegression(max_iter=1000, class_weight='balanced')
        CLASSIFIER.fit(embeddings, labels)
        print("Classifier trained.")


@app.route('/classify', methods=['POST'])
def classify():
    """Classify a message as FLUSH or PERSIST"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Get embedding and predict
    embedding = EMBED_MODEL.encode([text])
    proba = CLASSIFIER.predict_proba(embedding)[0]

    decision = 'PERSIST' if proba[1] > 0.5 else 'FLUSH'
    confidence = float(max(proba))

    # Determine category for persisted items (simple heuristic)
    category = None
    if decision == 'PERSIST':
        text_lower = text.lower()
        if any(w in text_lower for w in ['died', 'death', 'passed', 'grief', 'miss', 'lost', 'sad', 'cry', 'tears']):
            category = 'EMPATHY'
        elif any(w in text_lower for w in ['adhd', 'autism', 'anxiety', 'depression', 'neurodivergent', 'disability']):
            category = 'UNDERSTANDING'
        elif any(w in text_lower for w in ['degree', 'phd', 'lawyer', 'doctor', 'engineer', 'expert', 'professional']):
            category = 'RESPECT'
        elif any(w in text_lower for w in ['prefer', 'direct', 'patient', 'explain', 'style']):
            category = 'COMMUNICATION'
        elif any(w in text_lower for w in ['shipping', 'launching', 'deadline', 'project', 'goal']):
            category = 'VOLATILE'
        else:
            category = 'CONTEXT'

    return jsonify({
        'decision': decision,
        'confidence': confidence,
        'category': category
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    load_classifier()
    print("\n" + "="*50)
    print("Two-Room Memory Server")
    print("="*50)
    print("Classifier ready. Starting server on http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
