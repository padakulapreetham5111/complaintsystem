"""Text cleaning used before TF-IDF vectorisation."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NON_ALPHA = re.compile(r"[^a-z\s']")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/digits/punctuation, collapse whitespace.

    Stop-words are intentionally kept: words like "not" and "never" carry
    meaning in complaints ("did not receive"), and bigrams capture them.
    """
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
