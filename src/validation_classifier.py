"""
Validation for classifier-based gate
"""

from classifier_gate import predict, TRAINING_DATA
import numpy as np

# Use same test cases as before
TEST_CASES = [
    # === CLEAR FLUSH ===
    ("what color are ladybugs", "flush"),
    ("how many feet in a mile", "flush"),
    ("what's the capital of France", "flush"),
    ("define photosynthesis", "flush"),
    ("what time is it in Tokyo", "flush"),
    ("how do you spell necessary", "flush"),
    ("what's 12 times 15", "flush"),
    ("tell me a random fact about penguins", "flush"),
    ("did you know octopuses have three hearts", "flush"),
    ("what's the boiling point of water", "flush"),
    ("how many planets are in the solar system", "flush"),
    ("what year did the titanic sink", "flush"),
    ("they should make more purple legos", "flush"),
    ("clouds look like animals sometimes", "flush"),
    ("doors open outward because of fire code", "flush"),
    ("that's an interesting building", "flush"),
    ("coffee tastes better in the morning", "flush"),
    ("it's really hot today", "flush"),
    ("looks like it might rain", "flush"),
    ("what's a good movie to watch", "flush"),
    ("how do I make pasta", "flush"),
    ("tell me a joke", "flush"),
    ("snowflakes are cool", "flush"),
    ("the sky is really blue today", "flush"),
    ("I like pizza", "flush"),
    ("what does serendipity mean", "flush"),
    ("how do airplanes fly", "flush"),
    ("why is the ocean salty", "flush"),
    ("what's trending on twitter", "flush"),
    ("can you explain how magnets work", "flush"),
    
    # === CLEAR PERSIST ===
    ("my dad died yesterday", "persist"),
    ("i'm worried about my exam tomorrow", "persist"),
    ("my mother has NPD and i was always the scapegoat", "persist"),
    ("i have ADHD and it affects how i work", "persist"),
    ("i've been sober for 6 months", "persist"),
    ("i'm getting divorced", "persist"),
    ("my therapist thinks i have anxiety", "persist"),
    ("i was diagnosed with depression last year", "persist"),
    ("my daughter has autism", "persist"),
    ("i'm pregnant and scared", "persist"),
    ("i lost my job last week", "persist"),
    ("my best friend betrayed me", "persist"),
    ("i'm in an abusive relationship", "persist"),
    ("i attempted suicide when i was younger", "persist"),
    ("my son is struggling with addiction", "persist"),
    ("i grew up in poverty", "persist"),
    ("i was the first in my family to go to college", "persist"),
    ("my parents are getting divorced", "persist"),
    ("i have imposter syndrome at work", "persist"),
    ("i'm caring for my aging mother", "persist"),
    ("i struggle with binge eating", "persist"),
    ("my partner doesn't understand me", "persist"),
    ("i feel like i'm failing as a parent", "persist"),
    ("i was bullied throughout school", "persist"),
    ("i'm estranged from my family", "persist"),
    
    # Communication / preference
    ("i prefer direct communication", "persist"),
    ("don't sugarcoat things for me", "persist"),
    ("i need detailed explanations", "persist"),
    ("i learn better with examples", "persist"),
    ("please be patient with me", "persist"),
    
    # Identity / background
    ("i'm a lawyer", "persist"),
    ("i went to law school", "persist"),
    ("i have a PhD in physics", "persist"),
    ("i'm a recovering alcoholic", "persist"),
    ("i'm transgender", "persist"),
    ("english is my second language", "persist"),
    ("i'm neurodivergent", "persist"),
    ("i'm a single parent", "persist"),
    ("i served in the military", "persist"),
    ("i'm an immigrant", "persist"),
    
    # Projects / goals
    ("i'm shipping my game in January", "persist"),
    ("i'm starting my own business", "persist"),
    ("i'm training for a marathon", "persist"),
    ("i'm writing a novel", "persist"),
    ("i'm learning to code", "persist"),
    ("i'm trying to lose weight", "persist"),
    ("i'm saving up for a house", "persist"),
    ("i'm studying for the bar exam", "persist"),
    
    # Context
    ("i work as a contractor for the VA", "persist"),
    ("i live in Austin", "persist"),
    ("i have two kids", "persist"),
    ("my wife is an esthetician", "persist"),
    ("i work remotely", "persist"),
    
    # Edge cases
    ("the president's understanding of constitutional law concerns me", "flush"),
    ("i think pineapple belongs on pizza", "flush"),
    ("i've been thinking about death lately", "persist"),
    ("i can't sleep anymore", "persist"),
    ("everything feels pointless", "persist"),
    ("i just feel stuck", "persist"),
    ("work has been really stressful", "persist"),
    ("i'm so tired all the time", "persist"),
    ("nobody really understands me", "persist"),
    ("i've been having panic attacks", "persist"),
    ("my childhood was complicated", "persist"),
    ("i don't trust easily", "persist"),
    ("i tend to overthink everything", "persist"),
    ("confrontation makes me shut down", "persist"),
    ("i have a hard time asking for help", "persist"),
]


def run_validation(verbose: bool = True):
    """Run all test cases"""
    correct = 0
    false_positives = []  # flushed when should persist (BAD)
    false_negatives = []  # persisted when should flush (less bad)
    
    for exchange, expected in TEST_CASES:
        prediction, confidence = predict(exchange)
        predicted = prediction.lower()
        
        if predicted == expected:
            correct += 1
        else:
            if expected == "persist" and predicted == "flush":
                false_positives.append((exchange, confidence))
            else:
                false_negatives.append((exchange, confidence))
    
    accuracy = correct / len(TEST_CASES)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"CLASSIFIER VALIDATION")
        print(f"{'='*60}")
        print(f"Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.1%}")
        print(f"False Positives (lost memories): {len(false_positives)}")
        print(f"False Negatives (noise persisted): {len(false_negatives)}")
        
        if false_positives:
            print(f"\n--- FALSE POSITIVES (should persist, got flushed) ---")
            for ex, conf in false_positives:
                print(f"  {conf:.2f}: {ex}")
        
        if false_negatives:
            print(f"\n--- FALSE NEGATIVES (should flush, got persisted) ---")
            for ex, conf in false_negatives:
                print(f"  {conf:.2f}: {ex}")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(TEST_CASES),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
    }


def test_novel_examples():
    """Test on examples NOT in training data"""
    novel = [
        # Should flush
        ("what's the square root of 144", "flush"),
        ("how tall is the eiffel tower", "flush"),
        ("nice weather we're having", "flush"),
        ("that's a cool hat", "flush"),
        ("who won the superbowl last year", "flush"),
        
        # Should persist
        ("my grandmother raised me", "persist"),
        ("i've been clean for two years", "persist"),
        ("i'm autistic and it affects my relationships", "persist"),
        ("i'm going through a rough patch", "persist"),
        ("my ex was emotionally abusive", "persist"),
        ("i dropped out of college", "persist"),
        ("i'm the black sheep of my family", "persist"),
        ("i've never felt like i belonged", "persist"),
    ]
    
    print(f"\n{'='*60}")
    print("NOVEL EXAMPLES (not in training data)")
    print("="*60)
    
    correct = 0
    for exchange, expected in novel:
        prediction, confidence = predict(exchange)
        predicted = prediction.lower()
        match = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1
        marker = "→ ROOM 2" if predicted == "persist" else "  (flush)"
        print(f"{match} {confidence:.2f} {marker}: {exchange}")
    
    print(f"\nNovel accuracy: {correct}/{len(novel)} = {correct/len(novel):.1%}")


if __name__ == "__main__":
    run_validation()
    test_novel_examples()
