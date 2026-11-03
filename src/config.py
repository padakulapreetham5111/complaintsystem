"""Central configuration for paths and thresholds."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "complaints.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
RESPONSES_PATH = DATA_DIR / "responses.json"

STAGE1_MODEL_PATH = MODEL_DIR / "stage1_category.joblib"
STAGE2_MODEL_PATH = MODEL_DIR / "stage2_intents.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# If Stage-1 confidence is below this, the complaint is escalated to a human.
CONFIDENCE_THRESHOLD = 0.40

# Feedback rows are repeated this many times when retraining (they are
# trusted, human-verified labels, so they get more influence).
FEEDBACK_WEIGHT = 5

RANDOM_STATE = 42
