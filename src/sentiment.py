"""Sentiment analysis (VADER) and support-priority scoring."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

URGENCY_WORDS = {
    "urgent", "urgently", "immediately", "asap", "emergency", "right away",
    "unacceptable", "legal", "lawyer", "fraud", "scam", "police",
    "chargeback", "worst",
}

# Some categories carry more business risk (money / access problems).
CATEGORY_WEIGHT = {
    "Billing": 1.0,
    "Refund": 1.0,
    "Account": 0.8,
    "Technical Support": 0.5,
    "Delivery": 0.5,
    "Product Quality": 0.5,
}


def analyze_sentiment(text: str) -> dict:
    """Return VADER scores plus a human-readable label."""
    scores = _analyzer.polarity_scores(text)
    c = scores["compound"]
    if c <= -0.5:
        label = "Very Negative"
    elif c <= -0.05:
        label = "Negative"
    elif c < 0.05:
        label = "Neutral"
    else:
        label = "Positive"
    return {"compound": round(c, 4), "label": label,
            "neg": scores["neg"], "neu": scores["neu"], "pos": scores["pos"]}


def urgency_hits(text: str) -> list:
    low = text.lower()
    return sorted(w for w in URGENCY_WORDS if w in low)


def compute_priority(text: str, sentiment: dict, category: str) -> dict:
    """Combine sentiment, urgency keywords and category risk into a priority.

    score = negativity (0-5) + urgency keywords (0-3) + category weight (0-1)
    """
    negativity = max(0.0, -sentiment["compound"]) * 5
    hits = urgency_hits(text)
    urgency = min(len(hits), 3)
    score = negativity + urgency + CATEGORY_WEIGHT.get(category, 0.5)

    if score >= 5:
        level = "High"
    elif score >= 2.5:
        level = "Medium"
    else:
        level = "Low"
    return {"level": level, "score": round(score, 2), "urgency_keywords": hits}
