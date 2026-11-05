import re
import html


def clean_text(text):
    """
    Clean an individual movie review.

    Operations:
    1. Handle missing or invalid input.
    2. Decode HTML entities.
    3. Remove HTML tags.
    4. Convert text to lowercase.
    5. Expand common contractions.
    6. Remove unnecessary punctuation.
    7. Normalize whitespace.
    """

    # Handle missing or invalid input
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Expand common contractions
    contractions = {
        "can't": "can not",
        "cannot": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "'s": " is"
    }

    for contraction, replacement in contractions.items():

        if contraction == "'s":
            # Expand "'s" only when it follows a pronoun
            text = re.sub(
                r"\b(it|he|she|that|there|what|who|here|how)'s\b",
                lambda match: match.group(1) + " is",
                text
            )
        else:
            text = text.replace(contraction, replacement)

    # Remove remaining possessive apostrophe-s
    text = re.sub(r"(?<=\w)'s\b", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text