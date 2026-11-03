"""Automated response generation (sentiment-aware templates)."""
import json
import uuid

from .config import RESPONSES_PATH

_SLA = {"High": "within 2 hours", "Medium": "within 24 hours", "Low": "within 3 business days"}

_OPENINGS = {
    "Very Negative": "I am truly sorry for the frustration this has caused, and I understand how upsetting it is.",
    "Negative": "I am sorry for the trouble you have experienced.",
    "Neutral": "Thank you for reaching out to us.",
    "Positive": "Thank you for contacting us, and for your patience.",
}

ESCALATION_MESSAGE = (
    "Thank you for reaching out. We could not automatically understand this "
    "request, so we have forwarded it to a human agent who will contact you {sla}."
)


def _load_templates() -> dict:
    with open(RESPONSES_PATH, encoding="utf-8") as f:
        return json.load(f)


def new_ticket_id() -> str:
    return "TKT-" + uuid.uuid4().hex[:8].upper()


def generate_response(intent, sentiment_label: str, priority: str, ticket_id: str) -> str:
    """Build the reply. `intent=None` means escalate to a human."""
    sla = _SLA[priority]
    if intent is None:
        return ESCALATION_MESSAGE.format(sla=sla) + f" (Ref: {ticket_id})"
    body = _load_templates().get(
        intent, "We have logged your complaint and our team is working on it.")
    opening = _OPENINGS.get(sentiment_label, _OPENINGS["Neutral"])
    return (f"{opening} {body} Your ticket {ticket_id} has been marked "
            f"{priority} priority and you will hear from us {sla}.")
