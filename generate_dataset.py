"""Generate a synthetic labelled complaint dataset (category + intent).

Replace data/complaints.csv with real data (e.g. the CFPB Consumer Complaint
Database) to use the system in production. Required columns:
    text, category, intent
"""
import random
import pandas as pd
from src.config import DATASET_PATH, RANDOM_STATE

INTENTS = {
    "Billing": {
        "double_charge": [
            "I was charged twice for the same order",
            "my card was billed two times for a single purchase",
            "there is a duplicate charge on my bank statement",
            "you took money from my account twice",
        ],
        "wrong_amount": [
            "the amount charged is higher than the price shown",
            "I was overcharged on my bill",
            "the total on my invoice does not match what I agreed to pay",
            "extra fees were added to my bill without any notice",
        ],
        "invoice_missing": [
            "I never received the invoice for my purchase",
            "I need a copy of my receipt but cannot find it",
            "the invoice for last month is missing from my account",
            "please send me the tax invoice for my order",
        ],
    },
    "Delivery": {
        "late_delivery": [
            "my order has not arrived yet and it is long overdue",
            "the delivery is delayed by over a week",
            "the package was supposed to arrive days ago",
            "shipping is taking far too long",
        ],
        "lost_package": [
            "my package shows delivered but I never received it",
            "the parcel seems to be lost in transit",
            "tracking has not updated for days and my package is missing",
            "the courier says delivered but nothing is at my door",
        ],
        "wrong_item": [
            "I received the wrong item in my order",
            "you sent me a different product than what I ordered",
            "the size and colour delivered is not what I selected",
            "my order contained someone else's items",
        ],
    },
    "Product Quality": {
        "defective_product": [
            "the product stopped working after two days",
            "the device is defective and will not turn on",
            "the item is faulty and keeps malfunctioning",
            "it broke after minimal use",
        ],
        "damaged_on_arrival": [
            "the package arrived crushed and the item is damaged",
            "the product was broken when I opened the box",
            "the item came with cracks and scratches",
            "the screen was shattered on delivery",
        ],
        "not_as_described": [
            "the product looks nothing like the pictures",
            "the item does not match the description on your website",
            "the quality is much lower than advertised",
            "the features listed are missing from the product",
        ],
    },
    "Technical Support": {
        "app_crash": [
            "the app keeps crashing when I open it",
            "the application freezes and closes by itself",
            "the website throws an error every time I submit the form",
            "your software crashes on startup",
        ],
        "login_issue": [
            "I cannot log in to my account even with correct details",
            "the login page keeps showing an error",
            "the verification code never arrives so I cannot sign in",
            "sign in fails with an invalid credentials message",
        ],
        "slow_performance": [
            "the app is extremely slow and laggy",
            "pages take forever to load on your site",
            "the system response time is terrible",
            "everything loads very slowly since the update",
        ],
    },
    "Account": {
        "password_reset": [
            "I forgot my password and the reset email is not coming",
            "the password reset link has expired",
            "I want to change my password but cannot",
            "the reset link does not work",
        ],
        "account_locked": [
            "my account has been locked for no reason",
            "I am locked out after too many attempts",
            "my account was suspended without explanation",
            "access to my account is blocked",
        ],
        "update_details": [
            "I need to change my email address on the account",
            "how can I update my phone number and address",
            "please correct the name on my profile",
            "I want to change my registered details",
        ],
    },
    "Refund": {
        "refund_delay": [
            "I returned the item but have not received my refund",
            "my refund has been pending for weeks",
            "I am still waiting for the money back from my return",
            "the refund was promised in 5 days but nothing arrived",
        ],
        "refund_denied": [
            "my refund request was rejected unfairly",
            "you refused to refund me even though the item was faulty",
            "the return was denied and I do not understand why",
            "I was told I am not eligible for a refund",
        ],
        "cancel_order": [
            "I want to cancel my order immediately",
            "please cancel my subscription",
            "how do I cancel the purchase I just made",
            "I placed an order by mistake and need it cancelled",
        ],
    },
}

OPENERS = ["", "", "Hello,", "Hi team,", "Dear support,", "To whom it may concern,",
           "Hi,", "I am writing because", "Good morning,", "Hey,"]
CLOSERS = [
    "", "", "Please look into it.", "Kindly advise.", "Can you help?",
    "Thanks in advance for your help.",
    "This is unacceptable!", "I am extremely frustrated.",
    "Worst service ever.", "I am very disappointed.",
    "I need this fixed urgently.", "This is an emergency, please act immediately!",
    "I have been a loyal customer for years.", "Please respond asap.",
]


def add_typo(text: str, rng: random.Random, p: float = 0.15) -> str:
    """Randomly drop one character in some words to simulate real typos."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < p:
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1:]
    return " ".join(words)


def build(samples_per_intent: int = 70) -> pd.DataFrame:
    rng = random.Random(RANDOM_STATE)
    rows = set()
    for category, intents in INTENTS.items():
        for intent, phrases in intents.items():
            tries = 0
            count = 0
            while count < samples_per_intent and tries < samples_per_intent * 20:
                tries += 1
                body = rng.choice(phrases)
                text = f"{rng.choice(OPENERS)} {body}. {rng.choice(CLOSERS)}".strip()
                text = add_typo(text, rng)
                row = (text, category, intent)
                if row not in rows:
                    rows.add(row)
                    count += 1
    df = pd.DataFrame(sorted(rows), columns=["text", "category", "intent"])
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows to {DATASET_PATH}")
    print(df["category"].value_counts())
