import pytest
from src.preprocess import clean_text
from src.sentiment import analyze_sentiment, compute_priority
from src.pipeline import ComplaintPipeline


@pytest.fixture(scope="module")
def pipe():
    return ComplaintPipeline()


def test_clean_text():
    assert clean_text("Visit http://x.com NOW!!! 123") == "visit now"


def test_sentiment_negative_vs_positive():
    assert analyze_sentiment("This is terrible, I hate it")["compound"] < 0
    assert analyze_sentiment("Great service, thank you!")["compound"] > 0


def test_priority_high_for_angry_urgent_billing():
    text = "Worst service ever! Fix this immediately, it is unacceptable!"
    pr = compute_priority(text, analyze_sentiment(text), "Billing")
    assert pr["level"] == "High"


@pytest.mark.parametrize("text,category,intent", [
    ("I was charged twice for the same order", "Billing", "double_charge"),
    ("my package shows delivered but I never received it", "Delivery", "lost_package"),
    ("the app keeps crashing when I open it", "Technical Support", "app_crash"),
    ("I want to cancel my subscription", "Refund", "cancel_order"),
])
def test_two_stage_classification(pipe, text, category, intent):
    r = pipe.analyze(text)
    assert r["category"] == category
    assert r["intent"] == intent
    assert r["ticket_id"] in r["response"]


def test_empty_text_raises(pipe):
    with pytest.raises(ValueError):
        pipe.analyze("   ")


def test_gibberish_escalates(pipe):
    r = pipe.analyze("qwerty zxcvb lorem")
    assert "response" in r  # always returns something sensible
