"""End-to-end pipeline: sentiment -> stage 1 -> stage 2 -> priority -> reply."""
from . import classifier
from .config import (CONFIDENCE_THRESHOLD, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH)
from .responder import generate_response, new_ticket_id
from .sentiment import analyze_sentiment, compute_priority


class ComplaintPipeline:
    def __init__(self):
        self.reload()

    def reload(self):
        """(Re)load models from disk - call after retraining."""
        self.stage1 = classifier.load(STAGE1_MODEL_PATH)
        self.stage2 = classifier.load(STAGE2_MODEL_PATH)

    @property
    def taxonomy(self) -> dict:
        return {cat: list(m.classes_) for cat, m in self.stage2.items()}

    def analyze(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Complaint text is empty.")

        sentiment = analyze_sentiment(text)

        category, cat_conf = classifier.predict_with_confidence(self.stage1, text)
        intent, int_conf = None, 0.0
        if category in self.stage2:
            intent, int_conf = classifier.predict_with_confidence(
                self.stage2[category], text)

        escalated = cat_conf < CONFIDENCE_THRESHOLD
        priority = compute_priority(text, sentiment, category)
        if escalated:
            priority["level"] = "High" if priority["level"] != "Low" else "Medium"

        ticket_id = new_ticket_id()
        response = generate_response(None if escalated else intent,
                                     sentiment["label"], priority["level"], ticket_id)
        return {
            "ticket_id": ticket_id,
            "text": text,
            "category": category,
            "category_confidence": round(cat_conf, 3),
            "intent": intent,
            "intent_confidence": round(int_conf, 3),
            "sentiment": sentiment,
            "priority": priority,
            "escalated_to_human": escalated,
            "response": response,
        }
