from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model.joblib"


def load_model(path: Path = MODEL_PATH):
    """Carrega o modelo treinado."""
    return joblib.load(path)
