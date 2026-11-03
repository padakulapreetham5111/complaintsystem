"""Command-line interface.

Examples:
    python main.py "I was charged twice and I need this fixed urgently!"
    python main.py            # interactive mode with feedback prompts
"""
import sys
from src.feedback import record_feedback
from src.pipeline import ComplaintPipeline


def show(r: dict):
    print("\n" + "=" * 62)
    print(f"Ticket    : {r['ticket_id']}")
    print(f"Category  : {r['category']} ({r['category_confidence']:.0%})")
    print(f"Intent    : {r['intent']} ({r['intent_confidence']:.0%})")
    print(f"Sentiment : {r['sentiment']['label']} (compound {r['sentiment']['compound']})")
    p = r["priority"]
    print(f"Priority  : {p['level']} (score {p['score']}, keywords {p['urgency_keywords']})")
    if r["escalated_to_human"]:
        print("Escalated : YES - low confidence, routed to a human agent")
    print("-" * 62)
    print("Response  :", r["response"])
    print("=" * 62)


def ask_feedback(pipe: ComplaintPipeline, r: dict):
    ans = input("Was this helpful? [y/n/skip]: ").strip().lower()
    if ans == "y":
        record_feedback(r["ticket_id"], r["text"], r["category"], r["intent"], True)
        print("Thanks! Feedback saved.")
    elif ans == "n":
        cats = list(pipe.taxonomy)
        print("Categories:", ", ".join(cats))
        cat = input("Correct category: ").strip()
        if cat not in pipe.taxonomy:
            print("Unknown category, feedback skipped.")
            return
        print("Intents:", ", ".join(pipe.taxonomy[cat]))
        intent = input("Correct intent: ").strip()
        if intent not in pipe.taxonomy[cat]:
            print("Unknown intent, feedback skipped.")
            return
        record_feedback(r["ticket_id"], r["text"], r["category"], r["intent"],
                        False, cat, intent)
        print("Correction saved. Run `python retrain.py` to apply it.")


def main():
    pipe = ComplaintPipeline()
    if len(sys.argv) > 1:
        show(pipe.analyze(" ".join(sys.argv[1:])))
        return
    print("AI Complaint Analysis System - type 'quit' to exit.")
    while True:
        text = input("\nEnter complaint: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        r = pipe.analyze(text)
        show(r)
        ask_feedback(pipe, r)


if __name__ == "__main__":
    main()
