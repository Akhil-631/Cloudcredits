import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import clean_text


# Test cases
test_reviews = [
    "This movie is GREAT!!!",

    "I don't like this movie.",

    "This film is not good at all.",

    "A wonderful production.<br /><br />Amazing acting!",

    "I can't recommend this movie enough.",

    "The movie's storyline was excellent.",

    "   This    movie    is    fantastic.   ",

    "It's not a bad movie.",

    "The actors weren't good.",

    "I wouldn't recommend this film.",

    "This movie &amp; its storyline are amazing.",

    "The actor's performance wasn't impressive.",

    "12345",

    "This movie is the best I've seen in years!",

    None
]


print("\n========== PREPROCESSING TESTS ==========\n")

for index, review in enumerate(test_reviews, start=1):

    cleaned = clean_text(review)

    print(f"Test {index}")
    print("Original:", review)
    print("Cleaned :", cleaned)
    print("-" * 80)

    extra_tests = [
    "very-unassuming performance",
    "watching a light-hearted comedy",
    "well-written and beautifully-directed film",
    "The movie was not-good at all."
]

print("\n========== ADDITIONAL TEXT TESTS ==========\n")

for review in extra_tests:
    print("Original:", review)
    print("Cleaned :", clean_text(review))
    print("-" * 70)