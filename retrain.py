"""Feedback loop: report feedback stats, then retrain with human corrections."""
from src.feedback import feedback_stats
import train

if __name__ == "__main__":
    print("Feedback so far:", feedback_stats())
    train.main(use_feedback=True)
