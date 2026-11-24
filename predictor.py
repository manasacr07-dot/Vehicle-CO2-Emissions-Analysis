from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REG_PATH = MODEL_DIR / "co2_regressor.joblib"
CLF_PATH = MODEL_DIR / "co2_classifier.joblib"

def load_models():
    if not REG_PATH.exists() or not CLF_PATH.exists():
        raise FileNotFoundError(
            "Trained models are missing. Run: python train_model.py"
        )
    return joblib.load(REG_PATH), joblib.load(CLF_PATH)

def predict_vehicle(values):
    reg, clf = load_models()
    row = pd.DataFrame([values])
    predicted = float(reg.predict(row)[0])
    tier = str(clf.predict(row)[0])
    return predicted, tier
